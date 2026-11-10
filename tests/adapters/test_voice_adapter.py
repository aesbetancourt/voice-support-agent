"""Voice wiring tests that need no network: tool schema and tool handler."""

from types import SimpleNamespace

import pytest
from pipecat.frames.frames import (
    EndWorkerFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMMessagesAppendFrame,
    LLMTextFrame,
    TranscriptionFrame,
    TTSSpeakFrame,
    TTSUpdateSettingsFrame,
)
from pipecat.processors.frame_processor import FrameDirection

from screening_agent.adapters.inbound.voice import (
    STT_LOCATION,
    STT_MODEL,
    VOICE_PROFILES,
    LanguageSwitcher,
    ThoughtTagFilter,
    build_tools_schema,
    language_switch_frames,
    make_guidance_sender,
    make_tool_handler,
    session_language,
    stt_languages,
)
from screening_agent.application.prompts import agent_note
from screening_agent.application.tools import TOOL_SPECS
from screening_agent.domain.models import Language


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


async def test_handler_returns_result_to_llm(executor):
    llm, results = FakeLLM(), []
    handler = make_tool_handler(executor)
    await handler(
        make_params(llm, "start_screening", {"candidate_name": "Lucia", "consent": True}, results)
    )
    assert results[0][0]["ok"]
    assert results[0][1] is None
    assert llm.frames == []


async def test_closing_tool_speaks_fixed_message_and_ends_call(executor):
    llm, results, recorded = FakeLLM(), [], []
    handler = make_tool_handler(executor, on_closing=recorded.append)
    await handler(make_params(llm, "escalate_to_human", {"reason": "asked for a person"}, results))
    assert results[0][1].run_llm is False
    assert recorded == [llm.frames[0].text]
    speak, end = llm.frames
    assert isinstance(speak, TTSSpeakFrame)
    assert "recruiter" in speak.text
    assert isinstance(end, EndWorkerFrame)


class FixedMonitor:
    def __init__(self, guidance):
        self.guidance = guidance

    async def observe(self, turn_index, candidate_text):
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
    assert frame.messages == [{"role": "developer", "content": agent_note("Acknowledge it.")}]


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
    assert any(hint["value"] == "Harbor Logistics" for hint in profile.phrase_hints)


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


def test_stt_uses_chirp_3_in_a_regional_location():
    assert (STT_MODEL, STT_LOCATION) == ("chirp_3", "us")


@pytest.mark.parametrize("language", list(Language))
def test_stt_listens_for_every_language_with_the_session_language_first(language):
    languages = stt_languages(language)
    assert languages[0] is VOICE_PROFILES[language].stt_language
    assert set(languages) == {profile.stt_language for profile in VOICE_PROFILES.values()}


def test_a_language_switch_changes_the_voice_and_tells_the_llm():
    tts_update, note = language_switch_frames(Language.ES)
    assert isinstance(tts_update, TTSUpdateSettingsFrame)
    assert tts_update.delta.voice == VOICE_PROFILES[Language.ES].tts_voice
    assert isinstance(note, LLMMessagesAppendFrame)
    assert note.run_llm is False
    assert "speak only Spanish" in note.messages[0]["content"]


def transcription(text, language_code):
    return TranscriptionFrame(
        text, "user", "2026-10-30T00:00:00Z", result=SimpleNamespace(language_code=language_code)
    )


async def run_switcher(frames):
    switches, pushed = [], []

    async def on_switch(language):
        switches.append(language)

    switcher = LanguageSwitcher(Language.EN, on_switch)

    async def push_frame(frame, direction=FrameDirection.DOWNSTREAM):
        pushed.append(frame)

    switcher.push_frame = push_frame
    for frame in frames:
        await switcher.process_frame(frame, FrameDirection.DOWNSTREAM)
    return switcher, switches, pushed


async def test_switcher_follows_the_detected_language_before_the_words():
    words = transcription("Hola, mis audífonos llegaron rotos", "es-US")
    switcher, switches, pushed = await run_switcher([words])
    assert switches == [Language.ES]
    assert switcher.language is Language.ES
    # The voice change and the LLM note go before the candidate's words.
    assert [type(f) for f in pushed] == [
        TTSUpdateSettingsFrame,
        LLMMessagesAppendFrame,
        TranscriptionFrame,
    ]


async def test_switcher_ignores_the_same_language_and_short_replies():
    frames = [
        transcription("My headphones arrived broken", "en-US"),
        transcription("Sí, gracias", "es-US"),
    ]
    switcher, switches, pushed = await run_switcher(frames)
    assert switches == []
    assert switcher.language is Language.EN
    assert all(isinstance(f, TranscriptionFrame) for f in pushed)


async def test_thought_filter_cleans_streamed_llm_text():
    pushed = []
    tag_filter = ThoughtTagFilter()

    async def push_frame(frame, direction=FrameDirection.DOWNSTREAM):
        pushed.append(frame)

    tag_filter.push_frame = push_frame
    frames = [
        LLMFullResponseStartFrame(),
        LLMTextFrame("<thou"),
        LLMTextFrame("ght>\nEntiendo."),
        LLMTextFrame(" ¿Algo más?"),
        LLMFullResponseEndFrame(),
    ]
    for frame in frames:
        await tag_filter.process_frame(frame, FrameDirection.DOWNSTREAM)
    spoken = "".join(f.text for f in pushed if isinstance(f, LLMTextFrame))
    assert spoken == "\nEntiendo. ¿Algo más?"
    assert isinstance(pushed[-1], LLMFullResponseEndFrame)
