"""Ports: the interfaces the application needs from the outside world.

Outbound adapters in ``screening_agent.adapters.outbound`` implement them.
The application depends only on these protocols, never on an adapter.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from screening_agent.domain.models import (
    ConversationRecord,
    InterviewSlot,
    SentimentDraft,
    Summary,
    Turn,
)


@dataclass(frozen=True)
class KnowledgeHit:
    """One piece of job information that matches a candidate's question."""

    id: str
    text: str
    score: float


class JobKnowledge(Protocol):
    """Finds job information (pay, shifts, location, benefits) for a candidate's question."""

    async def search(self, question: str, limit: int) -> list[KnowledgeHit]:
        """The best matches, best first. Only matches above the adapter's threshold."""
        ...


class InterviewCalendar(Protocol):
    """The recruiting team's free interview times."""

    def free_slots(self, limit: int) -> list[InterviewSlot]: ...

    def book(self, slot_id: str) -> None: ...


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
    """Rates how frustrated the candidate is in their latest message."""

    async def analyze(self, candidate_text: str, recent_turns: list[Turn]) -> SentimentDraft: ...


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
        """Send candidate text. ``instruction`` is guidance for the model, not candidate speech."""
        ...

    async def send_tool_results(
        self, results: list[tuple[FunctionCall, dict[str, Any]]]
    ) -> ModelReply: ...
