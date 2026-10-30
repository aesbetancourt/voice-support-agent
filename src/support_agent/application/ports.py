"""Ports: the interfaces the application needs from the outside world.

Outbound adapters in ``support_agent.adapters.outbound`` implement them.
The application depends only on these protocols, never on an adapter.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from support_agent.domain.models import ConversationRecord, Order, SentimentDraft, Summary, Turn


class OrderRepository(Protocol):
    """Finds orders by their canonical number (``ORD-123456``)."""

    def get(self, order_number: str) -> Order | None: ...


class ConversationStore(Protocol):
    """Saves conversation records."""

    def save(self, record: ConversationRecord) -> None: ...

    def location(self, conversation_id: str) -> str:
        """Where the record is saved, for logs and messages."""
        ...


class Summarizer(Protocol):
    """Writes the end-of-conversation summary."""

    async def summarize(self, record: ConversationRecord) -> Summary: ...


class SentimentAnalyzer(Protocol):
    """Rates how frustrated the customer is in their latest message."""

    async def analyze(self, customer_text: str, recent_turns: list[Turn]) -> SentimentDraft: ...


@dataclass(frozen=True)
class FunctionCall:
    """A tool call requested by the chat model."""

    name: str
    arguments: dict[str, Any]
    call_id: str | None = None


@dataclass(frozen=True)
class ModelReply:
    """One chat-model response: text, tool calls, or both."""

    text: str
    function_calls: list[FunctionCall] = field(default_factory=list)


class ChatModel(Protocol):
    """A tool-calling chat model that keeps its own conversation history."""

    async def send_text(self, text: str, instruction: str | None = None) -> ModelReply:
        """Send customer text. ``instruction`` is guidance for the model, not customer speech."""
        ...

    async def send_tool_results(
        self, results: list[tuple[FunctionCall, dict[str, Any]]]
    ) -> ModelReply: ...
