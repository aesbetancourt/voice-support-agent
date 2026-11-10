"""Inbound adapter for voice: browser (WebRTC) → STT → Gemini → TTS.

Run with ``screening-agent voice`` or ``python -m screening_agent.adapters.inbound.voice``,
then open http://localhost:7860. The Pipecat development runner serves the
prebuilt browser client and the WebRTC signaling endpoint.
Each browser connection gets its own pipeline, screening and saved record.
The STT and TTS services come from ``speech.py``, chosen by ``STT_PROVIDER`` and ``TTS_PROVIDER``.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from datetime import date

from google.auth.exceptions import GoogleAuthError
from loguru import logger
from pipecat.adapters.llm.schemas.function_schema import FunctionSchema
from pipecat.adapters.llm.schemas.tools_schema import ToolsSchema
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import (
    EndWorkerFrame,
    Frame,
    FunctionCallResultProperties,
    InterruptionFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMMessagesAppendFrame,
    LLMRunFrame,
    LLMTextFrame,
    TranscriptionFrame,
    TTSSpeakFrame,
    TTSUpdateSettingsFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    AssistantTurnStoppedMessage,
    LLMAssistantAggregator,
    LLMContextAggregatorPair,
    LLMUserAggregator,
    LLMUserAggregatorParams,
    UserTurnStoppedMessage,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.processors.frameworks.rtvi.processor import RTVIProcessor
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import create_transport
from pipecat.services.google.vertex.llm import GoogleVertexLLMService
from pipecat.services.llm_service import FunctionCallParams
from pipecat.services.stt_service import STTService
from pipecat.services.tts_service import TTSService
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.turns.user_stop.base_user_turn_stop_strategy import BaseUserTurnStopStrategy
from pipecat.workers.runner import WorkerRunner

from screening_agent.adapters.constants import (
    CLIENT_LANGUAGE_KEY,
    DEVELOPER_ROLE,
    END_REASON_CLOSED,
    GOOGLE_LANGUAGE_CODE_ATTR,
    SERVER_MESSAGE_FRUSTRATION,
    SERVER_MESSAGE_LANGUAGE,
    WEBRTC_TRANSPORT,
)
from screening_agent.adapters.inbound.speech import build_stt, build_tts, tts_language_update
from screening_agent.adapters.outbound.gemini import ThoughtTagStripper, make_client
from screening_agent.application.prompts import (
    agent_note,
    greeting_trigger,
    language_switch_note,
    system_prompt,
)
from screening_agent.application.sentiment import ANALYSIS_TIMEOUT_SECS, SentimentMonitor
from screening_agent.application.tools import TOOL_SPECS, ToolExecutor
from screening_agent.bootstrap import (
    Conversation,
    job_knowledge,
    make_sentiment_analyzer,
    make_summarizer,
    new_conversation,
)
from screening_agent.config import ConfigError, Settings
from screening_agent.domain.models import JobPosting, Language
from screening_agent.domain.validators import language_switch, parse_language


def session_language(runner_args: RunnerArguments, default: Language) -> Language:
    """The language for one session: ``{"language": "es"}`` in the client body, else the default."""
    body = runner_args.body if isinstance(runner_args.body, dict) else {}
    raw = body.get(CLIENT_LANGUAGE_KEY)
    if not raw:
        return default
    try:
        return parse_language(str(raw))
    except ValueError as exc:
        logger.warning(f"{exc} Using {default.value}.")
        return default


def build_tools_schema() -> ToolsSchema:
    return ToolsSchema(
        standard_tools=[
            FunctionSchema(
                name=spec.name,
                description=spec.description,
                properties=spec.properties,
                required=spec.required,
            )
            for spec in TOOL_SPECS
        ]
    )


def make_tool_handler(
    executor: ToolExecutor, on_closing: Callable[[str], None] | None = None
) -> Callable[[FunctionCallParams], Awaitable[None]]:
    """Return a Pipecat function handler that runs every tool through the executor."""

    async def handle(params: FunctionCallParams) -> None:
        outcome = await executor.execute(params.function_name, dict(params.arguments or {}))
        if outcome.closing_message is None:
            await params.result_callback(outcome.payload)
            return
        # The conversation is closed: say the fixed closing message, then end the call.
        # The LLM does not run again, so it cannot add or change anything.
        await params.result_callback(
            outcome.payload, properties=FunctionCallResultProperties(run_llm=False)
        )
        # Record the message here: the call can end before the assistant turn event fires.
        if on_closing is not None:
            on_closing(outcome.closing_message)
        await params.llm.push_frame(TTSSpeakFrame(outcome.closing_message))
        await params.llm.push_frame(EndWorkerFrame(reason=END_REASON_CLOSED))

    return handle


def make_guidance_sender(
    monitor: SentimentMonitor,
    is_closed: Callable[[], bool],
    queue_frames: Callable[[list[Frame]], Awaitable[None]],
    on_reading: Callable[[int], Awaitable[None]] | None = None,
) -> Callable[[int | None, str], Awaitable[None]]:
    """Return a task body: rate one candidate turn, then add any guidance to the LLM context."""

    async def send(turn_index: int | None, candidate_text: str) -> None:
        guidance = await monitor.observe(turn_index, candidate_text)
        if is_closed():
            return
        if on_reading is not None and turn_index is not None:
            try:
                await on_reading(turn_index)
            except Exception as exc:  # noqa: BLE001 - the reading display is optional
                logger.debug(f"Frustration reading not shown: {exc!r}")
        if guidance is None:
            return
        # run_llm=False: the guidance shapes the next reply; it does not trigger a new one.
        message = {"role": DEVELOPER_ROLE, "content": agent_note(guidance)}
        try:
            await queue_frames([LLMMessagesAppendFrame(messages=[message], run_llm=False)])
        except Exception as exc:  # noqa: BLE001 - the call can end while the task runs
            logger.debug(f"Frustration guidance not sent: {exc!r}")

    return send


def language_switch_frames(settings: Settings, language: Language) -> list[Frame]:
    """Frames that move the rest of the pipeline to a new candidate language."""
    note = {"role": DEVELOPER_ROLE, "content": agent_note(language_switch_note(language))}
    return [
        TTSUpdateSettingsFrame(delta=tts_language_update(settings, language)),
        # run_llm=False: the note shapes the reply to the turn that is in progress.
        LLMMessagesAppendFrame(messages=[note], run_llm=False),
    ]


class LanguageSwitcher(FrameProcessor):
    """Follow the language that speech recognition detects in each final transcript.

    It sits between STT and the user aggregator, so the voice change and the note for
    Gemini go downstream before the candidate's words reach the LLM.
    """

    def __init__(
        self,
        settings: Settings,
        language: Language,
        on_switch: Callable[[Language], Awaitable[None]],
    ) -> None:
        super().__init__()
        self.language = language
        self._settings = settings
        self._on_switch = on_switch

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)
        if isinstance(frame, TranscriptionFrame):
            new_language = language_switch(self.language, detected_language(frame), frame.text)
            if new_language is not None:
                self.language = new_language
                for update in language_switch_frames(self._settings, new_language):
                    await self.push_frame(update, direction)
                await self._on_switch(new_language)
        await self.push_frame(frame, direction)


class ThoughtTagFilter(FrameProcessor):
    """Remove ``<thought>`` tags from Gemini's streamed text before TTS speaks it.

    It sits right after the LLM, so the voice, the browser transcript and the saved
    record all get the cleaned text.
    """

    def __init__(self) -> None:
        super().__init__()
        self._stripper = ThoughtTagStripper()

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)
        if isinstance(frame, (LLMFullResponseStartFrame, InterruptionFrame)):
            self._stripper = ThoughtTagStripper()
        elif isinstance(frame, LLMTextFrame):
            frame.text = self._stripper.feed(frame.text)
            if not frame.text:
                return
        elif isinstance(frame, LLMFullResponseEndFrame):
            rest = self._stripper.flush()
            if rest:
                await self.push_frame(LLMTextFrame(rest), direction)
        await self.push_frame(frame, direction)


def detected_language(frame: TranscriptionFrame) -> str | None:
    """The language that STT detected for one utterance, from Google or Deepgram."""
    # Google puts the detected code on its raw result; Deepgram puts it on the frame.
    google_code = getattr(frame.result, GOOGLE_LANGUAGE_CODE_ATTR, None)
    if google_code:
        return str(google_code)
    if frame.language is not None:
        return str(getattr(frame.language, "value", frame.language))
    return None


def build_services(
    settings: Settings, language: Language, job: JobPosting, today: date
) -> tuple[STTService, GoogleVertexLLMService, TTSService]:
    project_id = settings.require_project()
    gemini = settings.gemini
    stt = build_stt(settings, language)
    llm = GoogleVertexLLMService(
        project_id=project_id,
        location=gemini.location,
        settings=GoogleVertexLLMService.Settings(
            model=gemini.model,
            system_instruction=system_prompt("voice", job, today, language, switchable=True),
            temperature=gemini.temperature,
        ),
    )
    tts = build_tts(settings, language)
    return stt, llm, tts


async def run_bot(transport: BaseTransport, runner_args: RunnerArguments) -> None:
    try:
        settings = Settings.from_env()
    except ConfigError as exc:
        logger.error(f"Cannot start the voice bot: {exc}")
        return
    language = session_language(runner_args, settings.app.default_language)
    try:
        client = make_client(settings)
        summarizer = make_summarizer(settings, client)
        sentiment_analyzer = make_sentiment_analyzer(settings, client)
        conversation: Conversation = new_conversation(
            settings, "voice", summarizer, sentiment_analyzer, language
        )
        job, today = conversation.session.job, conversation.session.today
        stt, llm, tts = build_services(settings, language, job, today)
    except (ConfigError, GoogleAuthError) as exc:
        logger.error(
            f"Cannot start the voice bot: {exc}. Set GOOGLE_CLOUD_PROJECT in .env and run "
            "'gcloud auth application-default login'."
        )
        return

    # Build the job search index while Ava greets, so the first question does not wait.
    warm_up = asyncio.create_task(_warm_up_knowledge(settings))
    recorder = conversation.recorder
    logger.info(
        f"Starting voice conversation {recorder.id} in {language.value} "
        f"(STT: {settings.stt.provider.value}, TTS: {settings.tts.provider.value})"
    )

    # The closing message is recorded when the tool runs, because the call can end before
    # the assistant turn event fires. Skip it if that event reports it again later.
    closing_messages: set[str] = set()

    def record_closing(text: str) -> None:
        closing_messages.add(text.strip())
        recorder.add_turn("assistant", text)

    llm.register_function(None, make_tool_handler(conversation.executor, on_closing=record_closing))

    async def on_language_switch(new_language: Language) -> None:
        # Closing messages and the saved record follow the candidate's language.
        conversation.executor.language = new_language
        recorder.language = new_language
        logger.info(f"Candidate language switched to {new_language.value}")
        try:
            await worker.rtvi.send_server_message(
                {"type": SERVER_MESSAGE_LANGUAGE, "language": new_language.value}
            )
        except Exception as exc:  # noqa: BLE001 - the display is optional
            logger.debug(f"Language switch not shown: {exc!r}")

    switcher = LanguageSwitcher(settings, language, on_language_switch)

    context = LLMContext(tools=build_tools_schema())
    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(vad_analyzer=SileroVADAnalyzer()),
    )

    pipeline = Pipeline(
        [
            transport.input(),
            stt,
            switcher,
            user_aggregator,
            llm,
            ThoughtTagFilter(),
            tts,
            transport.output(),
            assistant_aggregator,
        ]
    )
    worker = PipelineWorker(
        pipeline,
        params=PipelineParams(enable_metrics=True, enable_usage_metrics=True),
    )
    runner = WorkerRunner(handle_sigint=runner_args.handle_sigint)
    await runner.add_workers(worker)

    @worker.rtvi.event_handler("on_client_ready")
    async def on_client_ready(rtvi: RTVIProcessor) -> None:
        context.add_message({"role": DEVELOPER_ROLE, "content": greeting_trigger(job, language)})
        await worker.queue_frames([LLMRunFrame()])

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport: BaseTransport, client: object) -> None:
        logger.info(f"Client disconnected from conversation {recorder.id}")
        await runner.cancel()

    async def show_reading(turn_index: int) -> None:
        # The prebuilt client lists server messages in its Events panel.
        reading = recorder.sentiment_at(turn_index)
        if reading is not None:
            await worker.rtvi.send_server_message(
                {
                    "type": SERVER_MESSAGE_FRUSTRATION,
                    "turn": turn_index,
                    **reading.model_dump(mode="json"),
                }
            )

    # Frustration readings run in the background, so they never delay Ava's reply.
    sentiment_tasks: set[asyncio.Task[None]] = set()
    send_guidance = (
        make_guidance_sender(
            conversation.sentiment,
            lambda: conversation.session.is_closed,
            worker.queue_frames,
            on_reading=show_reading,
        )
        if conversation.sentiment
        else None
    )

    @user_aggregator.event_handler("on_user_turn_stopped")
    async def on_user_turn_stopped(
        aggregator: LLMUserAggregator,
        strategy: BaseUserTurnStopStrategy,
        message: UserTurnStoppedMessage,
    ) -> None:
        turn_index = recorder.add_turn("user", message.content or "")
        if send_guidance is not None and message.content:
            task = asyncio.create_task(send_guidance(turn_index, message.content))
            sentiment_tasks.add(task)
            task.add_done_callback(sentiment_tasks.discard)

    @assistant_aggregator.event_handler("on_assistant_turn_stopped")
    async def on_assistant_turn_stopped(
        aggregator: LLMAssistantAggregator, message: AssistantTurnStoppedMessage
    ) -> None:
        if message.content and message.content.strip() not in closing_messages:
            recorder.add_turn("assistant", message.content)

    try:
        await runner.run()
    finally:
        # Let readings in progress finish, so the saved record includes them.
        if sentiment_tasks:
            await asyncio.wait(set(sentiment_tasks), timeout=ANALYSIS_TIMEOUT_SECS)
        warm_up.cancel()
        record = await recorder.finish()
        logger.info(f"Saved conversation {record.id} ({record.status.value}): {recorder.location}")


async def _warm_up_knowledge(settings: Settings) -> None:
    try:
        await job_knowledge(settings).warm_up()
    except Exception as exc:  # noqa: BLE001 - search falls back to the recruiter; never break
        logger.warning(f"Job knowledge index not built at start: {exc!r}")


async def bot(runner_args: RunnerArguments) -> None:
    """Entry point that the Pipecat development runner calls for each session."""
    transport_params = {
        WEBRTC_TRANSPORT: lambda: TransportParams(audio_in_enabled=True, audio_out_enabled=True),
    }
    transport = await create_transport(runner_args, transport_params)
    await run_bot(transport, runner_args)


if __name__ == "__main__":
    from pipecat.runner.run import main

    main()
