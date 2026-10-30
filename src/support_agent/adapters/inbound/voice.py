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
    LLMMessagesAppendFrame,
    LLMRunFrame,
    TTSSpeakFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    AssistantTurnStoppedMessage,
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
    UserTurnStoppedMessage,
)
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import create_transport
from pipecat.services.google.stt import GoogleSTTService
from pipecat.services.google.tts import GoogleTTSService
from pipecat.services.google.vertex.llm import GoogleVertexLLMService
from pipecat.services.llm_service import FunctionCallParams
from pipecat.transcriptions.language import Language as PipecatLanguage
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.workers.runner import WorkerRunner

from support_agent.adapters.outbound.gemini import make_client
from support_agent.application.prompts import greeting_trigger, system_prompt
from support_agent.application.sentiment import ANALYSIS_TIMEOUT_SECS, SentimentMonitor
from support_agent.application.tools import TOOL_SPECS, ToolExecutor
from support_agent.bootstrap import (
    Conversation,
    make_sentiment_analyzer,
    make_summarizer,
    new_conversation,
)
from support_agent.config import ConfigError, Settings
from support_agent.domain.models import Category, Language
from support_agent.domain.validators import parse_language


@dataclass(frozen=True)
class VoiceProfile:
    """Speech settings for one conversation language."""

    stt_language: PipecatLanguage
    # STT model and Google Cloud location. Chirp 3 is served from regional locations only.
    stt_model: str
    stt_location: str
    tts_voice: str
    # Phrases that speech recognition must favor: the order prefix and the category names.
    phrase_hints: list[dict[str, Any]]


VOICE_PROFILES = {
    Language.EN: VoiceProfile(
        stt_language=PipecatLanguage.EN_US,
        stt_model="latest_long",
        stt_location="global",
        tts_voice="en-US-Chirp3-HD-Charon",
        phrase_hints=[
            {"value": "ORD", "boost": 15.0},
            {"value": "order number", "boost": 10.0},
            *({"value": c.value.replace("_", " "), "boost": 5.0} for c in Category),
        ],
    ),
    Language.ES: VoiceProfile(
        stt_language=PipecatLanguage.ES_US,
        # latest_long misheard fluent Spanish order numbers ("O R D, uno cero cuatro…"
        # became "o Rd 1004"); chirp_3 got them right. See ADR-0008.
        stt_model="chirp_3",
        stt_location="us",
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


def make_tool_handler(executor: ToolExecutor, on_closing: Callable[[str], None] | None = None):
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
        message = {"role": "developer", "content": guidance}
        try:
            await queue_frames([LLMMessagesAppendFrame(messages=[message], run_llm=False)])
        except Exception as exc:  # noqa: BLE001 - the call can end while the task runs
            logger.debug(f"Frustration guidance not sent: {exc!r}")

    return send


def build_services(settings: Settings, language: Language):
    project_id = settings.require_project()
    profile = VOICE_PROFILES[language]
    adaptation = (
        {"phrase_sets": [{"phrases": profile.phrase_hints}]} if settings.stt_phrase_hints else None
    )
    stt = GoogleSTTService(
        location=profile.stt_location,
        settings=GoogleSTTService.Settings(
            languages=[profile.stt_language], model=profile.stt_model, adaptation=adaptation
        ),
    )
    llm = GoogleVertexLLMService(
        project_id=project_id,
        location=settings.location,
        settings=GoogleVertexLLMService.Settings(
            model=settings.model,
            system_instruction=system_prompt("voice", language),
            temperature=0.4,
        ),
    )
    # Set the language too: the service defaults its language code to en-US.
    tts = GoogleTTSService(
        settings=GoogleTTSService.Settings(voice=profile.tts_voice, language=profile.stt_language)
    )
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

    llm.register_function(
        None,
        make_tool_handler(
            conversation.executor, on_closing=lambda text: recorder.add_turn("assistant", text)
        ),
    )

    context = LLMContext(tools=build_tools_schema())
    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(vad_analyzer=SileroVADAnalyzer()),
    )

    pipeline = Pipeline(
        [
            transport.input(),
            stt,
            user_aggregator,
            llm,
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
    async def on_client_ready(rtvi):
        context.add_message({"role": "developer", "content": greeting_trigger(language)})
        await worker.queue_frames([LLMRunFrame()])

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
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
    async def on_user_turn_stopped(aggregator, strategy, message: UserTurnStoppedMessage):
        turn_index = recorder.add_turn("user", message.content or "")
        if send_guidance is not None and message.content:
            task = asyncio.create_task(send_guidance(turn_index, message.content))
            sentiment_tasks.add(task)
            task.add_done_callback(sentiment_tasks.discard)

    @assistant_aggregator.event_handler("on_assistant_turn_stopped")
    async def on_assistant_turn_stopped(aggregator, message: AssistantTurnStoppedMessage):
        if message.content:
            recorder.add_turn("assistant", message.content)

    try:
        await runner.run()
    finally:
        # Let readings in progress finish, so the saved record includes them.
        if sentiment_tasks:
            await asyncio.wait(set(sentiment_tasks), timeout=ANALYSIS_TIMEOUT_SECS)
        record = await recorder.finish()
        logger.info(f"Saved conversation {record.id} ({record.status.value}): {recorder.location}")


async def bot(runner_args: RunnerArguments):
    """Entry point that the Pipecat development runner calls for each session."""
    transport_params = {
        "webrtc": lambda: TransportParams(audio_in_enabled=True, audio_out_enabled=True),
    }
    transport = await create_transport(runner_args, transport_params)
    await run_bot(transport, runner_args)


if __name__ == "__main__":
    from pipecat.runner.run import main

    main()
