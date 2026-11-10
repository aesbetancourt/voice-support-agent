"""End-to-end conversations through the agent loop, with a scripted ChatModel (no network)."""

import asyncio
import json
from pathlib import Path

import pytest
from conftest import TODAY, FakeKnowledge, make_calendar

from screening_agent.adapters.outbound.json_store import JsonConversationStore
from screening_agent.application.agent import TextAgent
from screening_agent.application.messages import retry_message
from screening_agent.application.ports import FunctionCall, ModelReply
from screening_agent.application.recorder import ConversationRecorder
from screening_agent.application.sentiment import SentimentMonitor
from screening_agent.application.summary import SafeSummarizer
from screening_agent.application.tools import ToolExecutor
from screening_agent.domain.models import (
    ConversationStatus,
    FrustrationAction,
    Language,
    Recommendation,
    Sentiment,
    SentimentDraft,
    Stage,
)
from screening_agent.domain.state import ScreeningSession


class ScriptedModel:
    """Returns replies in order and keeps every tool result it receives."""

    def __init__(self, replies: list[ModelReply]):
        self._replies = list(replies)
        self.tool_results: list[dict] = []
        self.instructions: list[str | None] = []

    async def send_text(self, text, instruction=None):
        self.instructions.append(instruction)
        return self._replies.pop(0)

    async def send_tool_results(self, results):
        self.tool_results.extend(payload for _, payload in results)
        return self._replies.pop(0)


class ScriptedAnalyzer:
    """Returns frustration scores in order."""

    def __init__(self, scores):
        self._scores = list(scores)

    async def analyze(self, candidate_text, recent_turns):
        return SentimentDraft(frustration=self._scores.pop(0), reason="candidate is upset")


class FailingSummarizer:
    async def summarize(self, record):
        raise RuntimeError("Vertex AI is down")


def call(name, **arguments):
    return ModelReply(text="", function_calls=[FunctionCall(name=name, arguments=arguments)])


def build(job, tmp_path, replies, frustration_scores=None, language=Language.EN, analyzer=None):
    recorder = None
    session = ScreeningSession(job, TODAY, lambda: recorder.candidate_texts() if recorder else [])
    recorder = ConversationRecorder(
        "text",
        session,
        JsonConversationStore(tmp_path),
        SafeSummarizer(FailingSummarizer()),
        language,
    )
    model = ScriptedModel(replies)
    if frustration_scores is not None:
        analyzer = ScriptedAnalyzer(frustration_scores)
    monitor = SentimentMonitor(analyzer, recorder) if analyzer is not None else None
    executor = ToolExecutor(
        session,
        FakeKnowledge(),
        make_calendar(),
        recorder.add_tool_call,
        language,
    )
    agent = TextAgent(model, executor, recorder, sentiment=monitor)
    return agent, model, session, recorder


@pytest.fixture
def make_agent(job, tmp_path):
    def make(replies, frustration_scores=None, **kwargs):
        return build(job, tmp_path, replies, frustration_scores, **kwargs)

    return make


TEAMWORK = "Last Christmas two trucks came late, so I split the team in pairs and we finished."
SAFETY = "I saw a pallet leaning near the exit, so I stopped and told my supervisor."


async def test_full_screening_is_booked_and_saved(make_agent, tmp_path):
    agent, model, session, recorder = make_agent(
        [
            ModelReply(text="Hi, I am Ava from Harbor Logistics. Is it OK to continue?"),
            call("start_screening", candidate_name="Lucia Gomez", consent=True),
            ModelReply(text="Thanks Lucia! Do you have a permit to work in Spain?"),
            call("record_must_have", requirement="work_permit", answer="yes"),
            ModelReply(text="Great. Which shifts can you work?"),
            call("record_must_have", requirement="shifts", answer="morning, afternoon"),
            call("record_must_have", requirement="start_date", answer="immediately"),
            call("record_must_have", requirement="commute", answer="yes"),
            ModelReply(text="Tell me about a time you worked with a team under pressure."),
            call(
                "record_answer",
                question_id="teamwork",
                summary="Organized the team in pairs to unload late trucks.",
                evidence_quote="I split the team in pairs",
                specificity="specific",
                score=4,
            ),
            ModelReply(text="Nice! Now, tell me about a safety risk you noticed."),
            call(
                "record_answer",
                question_id="safety",
                summary="Stopped work and reported a leaning pallet.",
                evidence_quote="I stopped and told my supervisor",
                specificity="specific",
                score=4,
            ),
            call("get_interview_slots"),
            ModelReply(text="I have Wednesday at ten or at half past twelve. Which one?"),
            call("book_interview", slot_id="20261111-1000"),
        ]
    )

    assert (await agent.greet()).text.startswith("Hi")
    assert "Lucia" in (await agent.respond("Yes, I am Lucia Gomez, go ahead.")).text
    await agent.respond("Yes, I have one.")
    await agent.respond("Mornings and afternoons, I can start now and I drive.")
    await agent.respond(TEAMWORK)
    reply = await agent.respond(SAFETY)
    assert "Which one?" in reply.text
    reply = await agent.respond("Wednesday at ten, please.")
    assert reply.ended
    assert "Wednesday 11 November at 10:00" in reply.text

    record = await recorder.finish()
    assert record.status is ConversationStatus.BOOKED
    assert record.scorecard.recommendation is Recommendation.ADVANCE
    assert record.summary.generated_by == "fallback"
    saved = json.loads((tmp_path / f"{record.id}.json").read_text())
    assert saved["scorecard"]["answers"]["teamwork"]["evidence_quote"] == (
        "I split the team in pairs"
    )
    assert saved["scorecard"]["booking_id"].startswith("INT-")
    assert [c["name"] for c in saved["tool_calls"]][-1] == "book_interview"


async def test_model_cannot_book_before_the_screening(make_agent):
    agent, model, session, _ = make_agent(
        [call("book_interview", slot_id="20261111-1000"), ModelReply(text="First, your name?")]
    )
    reply = await agent.respond("Just book me in.")
    assert not reply.ended
    assert model.tool_results[0]["ok"] is False
    assert session.stage is Stage.CONSENT


async def test_endless_tool_calls_fall_back_to_a_safe_reply(make_agent):
    agent, _, _, _ = make_agent([call("get_interview_slots") for _ in range(10)])
    reply = await agent.respond("hello")
    assert reply.text == retry_message(Language.EN)


async def test_recorder_skips_duplicate_consecutive_turn(make_agent):
    _, _, _, recorder = make_agent([])
    recorder.add_turn("assistant", "Goodbye!")
    recorder.add_turn("assistant", "Goodbye!")
    recorder.add_turn("user", "Goodbye!")
    assert [t.role for t in recorder.build_record().transcript] == ["assistant", "user"]


async def test_dropped_conversation_is_saved_as_abandoned(make_agent):
    agent, _, _, recorder = make_agent([ModelReply(text="Hi!")])
    await agent.greet()
    assert Path(recorder.location).exists()
    record = await recorder.finish()
    assert record.status is ConversationStatus.ABANDONED
    assert "ended the screening before" in record.summary.headline


async def test_frustration_guidance_applies_to_the_next_turn(make_agent):
    agent, model, _, recorder = make_agent(
        [ModelReply(text="Your name?"), ModelReply(text="Sorry."), ModelReply(text="OK.")],
        frustration_scores=[0.9, 0.1],
    )
    await agent.respond("Why does this take so long? I already applied online!")
    await agent.respond("Lucia.")
    await agent.wait_for_reading()
    assert model.instructions[0] is None
    assert "acknowledge" in model.instructions[1]
    turns = recorder.build_record().transcript
    assert turns[0].sentiment.action is FrustrationAction.ACKNOWLEDGE
    assert turns[2].sentiment.action is FrustrationAction.NONE


async def test_summary_fallback_uses_peak_frustration(make_agent):
    agent, _, _, recorder = make_agent(
        [ModelReply(text="Hi."), ModelReply(text="Sorry.")], frustration_scores=[0.6, 0.95]
    )
    await agent.respond("This is slow.")
    await agent.respond("This is ridiculous, I want to talk to a person now!")
    await agent.wait_for_reading()
    record = await recorder.finish()
    assert record.summary.candidate_sentiment is Sentiment.ANGRY


async def test_record_keeps_the_conversation_language(make_agent):
    agent, model, _, recorder = make_agent(
        [ModelReply(text="Hola, soy Ava.")], language=Language.ES
    )
    await agent.greet()
    record = await recorder.finish()
    assert record.language is Language.ES
    assert model.instructions == [None]


class SlowAnalyzer:
    """Holds the reading until the test releases it."""

    def __init__(self):
        self.release = asyncio.Event()

    async def analyze(self, candidate_text, recent_turns):
        await self.release.wait()
        return SentimentDraft(frustration=0.9, reason="candidate is upset")


async def test_reply_does_not_wait_for_the_frustration_reading(make_agent):
    analyzer = SlowAnalyzer()
    agent, _, _, _ = make_agent([ModelReply(text="Sorry about that.")], analyzer=analyzer)
    reply = await agent.respond("This is the third time you call me!")
    assert reply.text == "Sorry about that."
    analyzer.release.set()
    assert "acknowledge" in await agent.wait_for_reading()
