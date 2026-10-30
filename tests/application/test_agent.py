"""End-to-end conversations through the agent loop, with a scripted ChatModel (no network)."""

import json
from pathlib import Path

import pytest

from support_agent.adapters.outbound.json_orders import JsonOrderRepository
from support_agent.adapters.outbound.json_store import JsonConversationStore
from support_agent.application.agent import FALLBACK_REPLY, TextAgent
from support_agent.application.ports import FunctionCall, ModelReply
from support_agent.application.recorder import ConversationRecorder
from support_agent.application.sentiment import SentimentMonitor
from support_agent.application.summary import SafeSummarizer
from support_agent.application.tools import ToolExecutor
from support_agent.domain.models import (
    ConversationStatus,
    FrustrationAction,
    Sentiment,
    SentimentDraft,
    Stage,
)
from support_agent.domain.state import TicketSession


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

    async def analyze(self, customer_text, recent_turns):
        return SentimentDraft(frustration=self._scores.pop(0), reason="customer is upset")


class FailingSummarizer:
    async def summarize(self, record):
        raise RuntimeError("Vertex AI is down")


def call(name, **arguments):
    return ModelReply(text="", function_calls=[FunctionCall(name=name, arguments=arguments)])


@pytest.fixture
def make_agent(orders: JsonOrderRepository, tmp_path):
    def build(replies, frustration_scores=None):
        session = TicketSession(orders.get)
        recorder = ConversationRecorder(
            "text", session, JsonConversationStore(tmp_path), SafeSummarizer(FailingSummarizer())
        )
        model = ScriptedModel(replies)
        monitor = (
            SentimentMonitor(ScriptedAnalyzer(frustration_scores), recorder)
            if frustration_scores is not None
            else None
        )
        executor = ToolExecutor(session, recorder.add_tool_call)
        agent = TextAgent(model, executor, recorder, sentiment=monitor)
        return agent, model, session, recorder

    return build


async def test_full_conversation_is_completed_and_saved(make_agent, tmp_path):
    agent, model, session, recorder = make_agent(
        [
            ModelReply(text="Hi, I am Ava. How can I help?"),
            call("lookup_order", order_number="ORD 104233"),
            ModelReply(text="Thanks Maria. What happened?"),
            call(
                "update_ticket",
                category="product_defect",
                description="The left ear cup arrived cracked.",
                urgency="medium",
            ),
            call("review_ticket"),
            ModelReply(text="Order ORD-104233, product defect, medium urgency. Correct?"),
            call("finalize_ticket"),
        ]
    )

    assert (await agent.greet()).text.startswith("Hi")
    assert "Maria" in (await agent.respond("My order is ORD 104233")).text
    reply = await agent.respond("The left ear cup arrived cracked.")
    assert "Correct?" in reply.text
    reply = await agent.respond("Yes, that is right.")
    assert reply.ended
    assert session.ticket.ticket_id in reply.text

    record = await recorder.finish()
    assert record.status is ConversationStatus.COMPLETED
    assert record.summary.generated_by == "fallback"
    assert [t.role for t in record.transcript] == [
        "assistant",
        "user",
        "assistant",
        "user",
        "assistant",
        "user",
        "assistant",
    ]
    saved = json.loads((tmp_path / f"{record.id}.json").read_text())
    assert saved["ticket"]["order_number"] == "ORD-104233"
    assert [c["name"] for c in saved["tool_calls"]][-1] == "finalize_ticket"


async def test_model_cannot_finalize_without_review(make_agent):
    agent, model, session, _ = make_agent(
        [call("finalize_ticket"), ModelReply(text="Let me collect your details first.")]
    )
    reply = await agent.respond("Just submit it.")
    assert not reply.ended
    assert model.tool_results[0]["ok"] is False
    assert session.stage is Stage.COLLECTING


async def test_endless_tool_calls_fall_back_to_a_safe_reply(make_agent):
    agent, _, _, _ = make_agent([call("review_ticket") for _ in range(10)])
    reply = await agent.respond("hello")
    assert reply.text == FALLBACK_REPLY


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
    assert record.summary.headline.startswith("Conversation ended before")


async def test_frustration_guidance_applies_to_the_next_turn(make_agent):
    agent, model, _, recorder = make_agent(
        [ModelReply(text="How can I help?"), ModelReply(text="Sorry."), ModelReply(text="OK.")],
        frustration_scores=[0.9, 0.1],
    )
    await agent.respond("This is the third time I call about this!")
    await agent.respond("My order is ORD-104233.")
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
    await agent.respond("This is ridiculous, I want my money back now!")
    record = await recorder.finish()
    assert record.summary.customer_sentiment is Sentiment.ANGRY
