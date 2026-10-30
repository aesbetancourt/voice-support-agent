"""End-to-end text conversations with a scripted model instead of Gemini."""

import json

import pytest

from support_agent.models import ConversationStatus, Stage
from support_agent.orders import OrderRepository
from support_agent.recorder import ConversationRecorder
from support_agent.state import TicketSession
from support_agent.storage import ConversationStore
from support_agent.summary import SafeSummarizer
from support_agent.text_agent import FALLBACK_REPLY, FunctionCall, ModelReply, TextAgent
from support_agent.tools import ToolExecutor


class ScriptedModel:
    """Returns replies in order and keeps every tool result it receives."""

    def __init__(self, replies: list[ModelReply]):
        self._replies = list(replies)
        self.tool_results: list[dict] = []

    async def send_text(self, text):
        return self._replies.pop(0)

    async def send_tool_results(self, results):
        self.tool_results.extend(payload for _, payload in results)
        return self._replies.pop(0)


class FailingSummarizer:
    async def summarize(self, record):
        raise RuntimeError("Vertex AI is down")


def call(name, **arguments):
    return ModelReply(text="", function_calls=[FunctionCall(name=name, arguments=arguments)])


@pytest.fixture
def make_agent(orders: OrderRepository, tmp_path):
    def build(replies):
        session = TicketSession(orders)
        recorder = ConversationRecorder(
            "text", session, ConversationStore(tmp_path), SafeSummarizer(FailingSummarizer())
        )
        model = ScriptedModel(replies)
        agent = TextAgent(model, ToolExecutor(session, recorder.add_tool_call), recorder)
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
    assert recorder.path.exists()
    record = await recorder.finish()
    assert record.status is ConversationStatus.ABANDONED
    assert record.summary.headline.startswith("Conversation ended before")
