"""Inbound adapter for voice: browser (WebRTC) → Google STT → Gemini → Google TTS.

Run with ``support-agent voice`` or ``python -m support_agent.adapters.inbound.voice``,
then open http://localhost:7860. The Pipecat development runner serves the
prebuilt browser client and the WebRTC signaling endpoint.
Each browser connection gets its own pipeline, ticket and saved record.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

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
from pipecat.services.google.stt import GoogleSTTService
from pipecat.services.google.tts import GoogleTTSService
from pipecat.services.google.vertex.llm import GoogleVertexLLMService
from pipecat.services.llm_service import FunctionCallParams
from pipecat.transcriptions.language import Language as PipecatLanguage
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.turns.user_stop.base_user_turn_stop_strategy import BaseUserTurnStopStrategy
from pipecat.workers.runner import WorkerRunner

from support_agent.adapters.outbound.gemini import ThoughtTagStripper, make_client
from support_agent.application.prompts import (
    agent_note,
    greeting_trigger,
    language_switch_note,
    system_prompt,
)
from support_agent.application.sentiment import ANALYSIS_TIMEOUT_SECS, SentimentMonitor
from support_agent.application.tools import TOOL_SPECS, ToolExecutor
from support_agent.bootstrap import (
    Conversation,
    make_sentiment_analyzer,
    make_summarizer,
    new_conversation,
)
from support_agent.config import AGENT_TEMPERATURE, ConfigError, Settings
from support_agent.domain.models import Category, Language
from support_agent.domain.validators import language_switch, parse_language

# One recognizer for every language: Chirp 3 detects the language of each utterance,
# so the customer can switch during the call. It is served from regional locations only.
# latest_long misheard fluent Spanish order numbers ("O R D, uno cero cuatro…" became
# "o Rd 1004"); chirp_3 got them right in both languages. See ADR-0008 and ADR-0009.
STT_MODEL = "chirp_3"
STT_LOCATION = "us"


@dataclass(frozen=True)
class VoiceProfile:
    """Speech settings for one conversation language."""

    stt_language: PipecatLanguage
    tts_voice: str
    # Phrases that speech recognition must favor: the order prefix and the category names.
    phrase_hints: list[dict[str, Any]]


# Phrase boosts: the order prefix most, because a misheard ORD fails the lookup;
# category words least, because Gemini maps them from free speech anyway.
VOICE_PROFILES = {
    Language.EN: VoiceProfile(
        stt_language=PipecatLanguage.EN_US,
        tts_voice="en-US-Chirp3-HD-Charon",
        phrase_hints=[
            {"value": "ORD", "boost": 15.0},
            {"value": "order number", "boost": 10.0},
            *({"value": c.value.replace("_", " "), "boost": 5.0} for c in Category),
        ],
    ),
    Language.ES: VoiceProfile(
        stt_language=PipecatLanguage.ES_US,
        tts_voice="es-US-Chirp3-HD-Charon",
        phrase_hints=[
            {"value": "ORD", "boost": 15.0},
            {"value": "número de pedido", "boost": 10.0},
            *(
                {"value": word, "boost": 5.0}
                for word in ("envío", "factura", "defectuoso", "devolución", "reembolso", "cuenta")
            ),
        ],
    ),
}


def session_language(runner_args: RunnerArguments, default: Language) -> Language:
    """The language for one session: ``{"language": "es"}`` in the client body, else the default."""
    body = runner_args.body if isinstance(runner_args.body, dict) else {}
    raw = body.get("language")
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
        outcome = executor.execute(params.function_name, dict(params.arguments or {}))
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
        await params.llm.push_frame(EndWorkerFrame(reason="conversation closed"))

    return handle


def make_guidance_sender(
    monitor: SentimentMonitor,
    is_closed: Callable[[], bool],
    queue_frames: Callable[[list[Frame]], Awaitable[None]],
    on_reading: Callable[[int], Awaitable[None]] | None = None,
) -> Callable[[int | None, str], Awaitable[None]]:
    """Return a task body: rate one customer turn, then add any guidance to the LLM context."""

    async def send(turn_index: int | None, customer_text: str) -> None:
        guidance = await monitor.observe(turn_index, customer_text)
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
        message = {"role": "developer", "content": agent_note(guidance)}
        try:
            await queue_frames([LLMMessagesAppendFrame(messages=[message], run_llm=False)])
        except Exception as exc:  # noqa: BLE001 - the call can end while the task runs
            logger.debug(f"Frustration guidance not sent: {exc!r}")

    return send


def tts_settings(language: Language) -> GoogleTTSService.Settings:
    profile = VOICE_PROFILES[language]
    # Set the language too: the service defaults its language code to en-US.
    return GoogleTTSService.Settings(voice=profile.tts_voice, language=profile.stt_language)


def stt_languages(language: Language) -> list[PipecatLanguage]:
    """Every supported language, the session language first (Google's primary language)."""
    others = [p.stt_language for lang, p in VOICE_PROFILES.items() if lang is not language]
    return [VOICE_PROFILES[language].stt_language, *others]


def language_switch_frames(language: Language) -> list[Frame]:
    """Frames that move the rest of the pipeline to a new customer language."""
    note = {"role": "developer", "content": agent_note(language_switch_note(language))}
    return [
        TTSUpdateSettingsFrame(delta=tts_settings(language)),
        # run_llm=False: the note shapes the reply to the turn that is in progress.
        LLMMessagesAppendFrame(messages=[note], run_llm=False),
    ]


class LanguageSwitcher(FrameProcessor):
    """Follow the language that speech recognition detects in each final transcript.

    It sits between STT and the user aggregator, so the voice change and the note for
    Gemini go downstream before the customer's words reach the LLM.
    """

    def __init__(
        self,
        language: Language,
        on_switch: Callable[[Language], Awaitable[None]],
    ) -> None:
        super().__init__()
        self.language = language
        self._on_switch = on_switch

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)
        if isinstance(frame, TranscriptionFrame):
            detected = getattr(frame.result, "language_code", None)
            new_language = language_switch(self.language, detected, frame.text)
            if new_language is not None:
                self.language = new_language
                for update in language_switch_frames(new_language):
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


def build_services(
    settings: Settings, language: Language
) -> tuple[GoogleSTTService, GoogleVertexLLMService, GoogleTTSService]:
    project_id = settings.require_project()
    hints = {
        hint["value"]: hint for profile in VOICE_PROFILES.values() for hint in profile.phrase_hints
    }
    adaptation = (
        {"phrase_sets": [{"phrases": list(hints.values())}]} if settings.stt_phrase_hints else None
    )
    stt = GoogleSTTService(
        location=STT_LOCATION,
        settings=GoogleSTTService.Settings(
            languages=stt_languages(language), model=STT_MODEL, adaptation=adaptation
        ),
    )
    llm = GoogleVertexLLMService(
        project_id=project_id,
        location=settings.location,
        settings=GoogleVertexLLMService.Settings(
            model=settings.model,
            system_instruction=system_prompt("voice", language, switchable=True),
            temperature=AGENT_TEMPERATURE,
        ),
    )
    tts = GoogleTTSService(settings=tts_settings(language))
    return stt, llm, tts


async def run_bot(transport: BaseTransport, runner_args: RunnerArguments) -> None:
    settings = Settings.from_env()
    language = session_language(runner_args, settings.default_language)
    try:
        stt, llm, tts = build_services(settings, language)
        client = make_client(settings)
        summarizer = make_summarizer(settings, client)
        sentiment_analyzer = make_sentiment_analyzer(settings, client)
    except (ConfigError, GoogleAuthError) as exc:
        logger.error(
            f"Cannot start the voice bot: {exc}. Set GOOGLE_CLOUD_PROJECT in .env and run "
            "'gcloud auth application-default login'."
        )
        return

    conversation: Conversation = new_conversation(
        settings, "voice", summarizer, sentiment_analyzer, language
    )
    recorder = conversation.recorder
    logger.info(f"Starting voice conversation {recorder.id} in {language.value}")

    # The closing message is recorded when the tool runs, because the call can end before
    # the assistant turn event fires. Skip it if that event reports it again later.
    closing_messages: set[str] = set()

    def record_closing(text: str) -> None:
        closing_messages.add(text.strip())
        recorder.add_turn("assistant", text)

    llm.register_function(None, make_tool_handler(conversation.executor, on_closing=record_closing))

    async def on_language_switch(new_language: Language) -> None:
        # Closing messages and the saved record follow the customer's language.
        conversation.executor.language = new_language
        recorder.language = new_language
        logger.info(f"Customer language switched to {new_language.value}")
        try:
            await worker.rtvi.send_server_message(
                {"type": "language", "language": new_language.value}
            )
        except Exception as exc:  # noqa: BLE001 - the display is optional
            logger.debug(f"Language switch not shown: {exc!r}")

    switcher = LanguageSwitcher(language, on_language_switch)

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
        context.add_message({"role": "developer", "content": greeting_trigger(language)})
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
                {"type": "frustration", "turn": turn_index, **reading.model_dump(mode="json")}
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
        record = await recorder.finish()
        logger.info(f"Saved conversation {record.id} ({record.status.value}): {recorder.location}")


async def bot(runner_args: RunnerArguments) -> None:
    """Entry point that the Pipecat development runner calls for each session."""
    transport_params = {
        "webrtc": lambda: TransportParams(audio_in_enabled=True, audio_out_enabled=True),
    }
    transport = await create_transport(runner_args, transport_params)
    await run_bot(transport, runner_args)


if __name__ == "__main__":
    from pipecat.runner.run import main

    main()
