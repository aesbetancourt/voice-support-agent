"""Inbound adapter for voice: browser (WebRTC) → Google STT → Gemini → Google TTS.

Run with ``support-agent voice`` or ``python -m support_agent.adapters.inbound.voice``,
then open http://localhost:7860. The Pipecat development runner serves the
prebuilt browser client and the WebRTC signaling endpoint.
Each browser connection gets its own pipeline, ticket and saved record.
"""

from __future__ import annotations

from collections.abc import Callable

from google.auth.exceptions import GoogleAuthError
from loguru import logger
from pipecat.adapters.llm.schemas.function_schema import FunctionSchema
from pipecat.adapters.llm.schemas.tools_schema import ToolsSchema
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import (
    EndWorkerFrame,
    FunctionCallResultProperties,
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
from pipecat.transcriptions.language import Language
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.workers.runner import WorkerRunner

from support_agent.adapters.outbound.gemini import make_client
from support_agent.application.prompts import GREETING_TRIGGER, system_prompt
from support_agent.application.tools import TOOL_SPECS, ToolExecutor
from support_agent.bootstrap import Conversation, make_summarizer, new_conversation
from support_agent.config import ConfigError, Settings
from support_agent.domain.models import Category

# Phrases that speech recognition must favor: the order prefix and the category names.
_PHRASE_HINTS = [
    {"value": "ORD", "boost": 15.0},
    {"value": "order number", "boost": 10.0},
    *({"value": c.value.replace("_", " "), "boost": 5.0} for c in Category),
]


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


def build_services(settings: Settings):
    project_id = settings.require_project()
    adaptation = (
        {"phrase_sets": [{"phrases": _PHRASE_HINTS}]} if settings.stt_phrase_hints else None
    )
    stt = GoogleSTTService(
        location=settings.stt_location,
        settings=GoogleSTTService.Settings(
            languages=[Language.EN_US], model="latest_long", adaptation=adaptation
        ),
    )
    llm = GoogleVertexLLMService(
        project_id=project_id,
        location=settings.location,
        settings=GoogleVertexLLMService.Settings(
            model=settings.model,
            system_instruction=system_prompt("voice"),
            temperature=0.4,
        ),
    )
    tts = GoogleTTSService(settings=GoogleTTSService.Settings(voice=settings.tts_voice))
    return stt, llm, tts


async def run_bot(transport: BaseTransport, runner_args: RunnerArguments) -> None:
    settings = Settings.from_env()
    try:
        stt, llm, tts = build_services(settings)
        summarizer = make_summarizer(settings, make_client(settings))
    except (ConfigError, GoogleAuthError) as exc:
        logger.error(
            f"Cannot start the voice bot: {exc}. Set GOOGLE_CLOUD_PROJECT in .env and run "
            "'gcloud auth application-default login'."
        )
        return

    conversation: Conversation = new_conversation(settings, "voice", summarizer)
    recorder = conversation.recorder
    logger.info(f"Starting voice conversation {recorder.id}")

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
        context.add_message({"role": "developer", "content": GREETING_TRIGGER})
        await worker.queue_frames([LLMRunFrame()])

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info(f"Client disconnected from conversation {recorder.id}")
        await runner.cancel()

    @user_aggregator.event_handler("on_user_turn_stopped")
    async def on_user_turn_stopped(aggregator, strategy, message: UserTurnStoppedMessage):
        recorder.add_turn("user", message.content)

    @assistant_aggregator.event_handler("on_assistant_turn_stopped")
    async def on_assistant_turn_stopped(aggregator, message: AssistantTurnStoppedMessage):
        if message.content:
            recorder.add_turn("assistant", message.content)

    try:
        await runner.run()
    finally:
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
