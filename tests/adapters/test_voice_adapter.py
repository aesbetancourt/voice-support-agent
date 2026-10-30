"""Voice wiring tests that need no network: tool schema and tool handler."""

from types import SimpleNamespace

import pytest
from pipecat.frames.frames import EndWorkerFrame, LLMMessagesAppendFrame, TTSSpeakFrame

from support_agent.adapters.inbound.voice import (
    VOICE_PROFILES,
    build_tools_schema,
    make_guidance_sender,
    make_tool_handler,
    session_language,
)
from support_agent.application.tools import TOOL_SPECS, ToolExecutor
from support_agent.domain.models import Language


class FakeLLM:
    def __init__(self):
        self.frames = []

    async def push_frame(self, frame, direction=None):
        self.frames.append(frame)


def make_params(llm, name, arguments, results):
    async def result_callback(result, properties=None):
        results.append((result, properties))

    return SimpleNamespace(
        function_name=name, arguments=arguments, llm=llm, result_callback=result_callback
    )


def test_tools_schema_lists_every_tool():
    schema = build_tools_schema()
    assert [f.name for f in schema.standard_tools] == [s.name for s in TOOL_SPECS]


async def test_handler_returns_result_to_llm(session):
    llm, results = FakeLLM(), []
    handler = make_tool_handler(ToolExecutor(session))
    await handler(make_params(llm, "lookup_order", {"order_number": "ORD-104233"}, results))
    assert results[0][0]["ok"]
    assert results[0][1] is None
    assert llm.frames == []


async def test_closing_tool_speaks_fixed_message_and_ends_call(session):
    llm, results, recorded = FakeLLM(), [], []
    handler = make_tool_handler(ToolExecutor(session), on_closing=recorded.append)
    await handler(make_params(llm, "escalate_to_human", {"reason": "asked for a person"}, results))
    assert results[0][1].run_llm is False
    assert recorded == [llm.frames[0].text]
    speak, end = llm.frames
    assert isinstance(speak, TTSSpeakFrame)
    assert "human agent" in speak.text
    assert isinstance(end, EndWorkerFrame)


class FixedMonitor:
    def __init__(self, guidance):
        self.guidance = guidance

    async def observe(self, turn_index, customer_text):
        return self.guidance


async def test_guidance_is_appended_to_context_without_running_the_llm():
    queued = []

    async def queue_frames(frames):
        queued.extend(frames)

    send = make_guidance_sender(FixedMonitor("Acknowledge it."), lambda: False, queue_frames)
    await send(0, "This is the third time!")
    (frame,) = queued
    assert isinstance(frame, LLMMessagesAppendFrame)
    assert frame.run_llm is False
    assert frame.messages == [{"role": "developer", "content": "Acknowledge it."}]


async def test_no_guidance_after_the_call_closed():
    queued = []

    async def queue_frames(frames):
        queued.extend(frames)

    send = make_guidance_sender(FixedMonitor("Acknowledge it."), lambda: True, queue_frames)
    await send(0, "Bye!")
    assert queued == []


async def test_every_reading_is_shown_even_without_guidance():
    shown = []

    async def queue_frames(frames):
        pass

    async def on_reading(turn_index):
        shown.append(turn_index)

    send = make_guidance_sender(FixedMonitor(None), lambda: False, queue_frames, on_reading)
    await send(3, "Thanks, that's all.")
    assert shown == [3]


async def test_a_failed_display_still_sends_the_guidance():
    queued = []

    async def queue_frames(frames):
        queued.extend(frames)

    async def on_reading(turn_index):
        raise RuntimeError("client gone")

    send = make_guidance_sender(
        FixedMonitor("Acknowledge it."), lambda: False, queue_frames, on_reading
    )
    await send(0, "This is the third time!")
    assert len(queued) == 1


@pytest.mark.parametrize("language", list(Language))
def test_every_language_has_a_matching_voice_profile(language):
    profile = VOICE_PROFILES[language]
    # The TTS voice and the STT language must use the same locale, e.g. es-US.
    assert profile.tts_voice.startswith(f"{profile.stt_language.value}-Chirp3-HD-")
    assert profile.stt_language.value.startswith(language.value)
    assert any(hint["value"] == "ORD" for hint in profile.phrase_hints)


@pytest.mark.parametrize(
    ("body", "expected"),
    [
        ({}, Language.EN),
        ({"language": "es"}, Language.ES),
        ({"language": "es-US"}, Language.ES),
        ({"language": "klingon"}, Language.EN),
        (None, Language.EN),
    ],
)
def test_session_language_comes_from_the_client_body(body, expected):
    runner_args = SimpleNamespace(body=body)
    assert session_language(runner_args, default=Language.EN) is expected


def test_spanish_uses_chirp_3_in_a_regional_location():
    profile = VOICE_PROFILES[Language.ES]
    assert (profile.stt_model, profile.stt_location) == ("chirp_3", "us")
