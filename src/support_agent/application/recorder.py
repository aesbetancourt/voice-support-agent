"""Records one conversation and saves it through the ``ConversationStore`` port.

Every channel uses this class. It saves after each turn, so a crash or a dropped
call keeps a partial record. ``finish`` adds the summary and saves the final record.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any, Literal

from support_agent.application.ports import ConversationStore, Summarizer
from support_agent.domain.models import (
    ConversationRecord,
    ConversationStatus,
    Stage,
    ToolCallEvent,
    Turn,
    utc_now,
)
from support_agent.domain.state import TicketSession

_STATUS_BY_STAGE = {
    Stage.COMPLETED: ConversationStatus.COMPLETED,
    Stage.ESCALATED: ConversationStatus.ESCALATED,
    Stage.ENDED: ConversationStatus.ABANDONED,
}


class ConversationRecorder:
    def __init__(
        self,
        channel: Literal["text", "voice"],
        session: TicketSession,
        store: ConversationStore,
        summarizer: Summarizer,
    ):
        self.id = f"{utc_now():%Y%m%dT%H%M%S}-{uuid.uuid4().hex[:8]}"
        self.channel = channel
        self.session = session
        self._store = store
        self._summarizer = summarizer
        self._started_at = utc_now()
        self._transcript: list[Turn] = []
        self._tool_calls: list[ToolCallEvent] = []
        self._final: ConversationRecord | None = None

    @property
    def location(self) -> str:
        return self._store.location(self.id)

    def add_turn(self, role: Literal["user", "assistant"], text: str) -> None:
        text = text.strip()
        if not text:
            return
        # The voice channel can report the same closing message twice; keep one copy.
        last = self._transcript[-1] if self._transcript else None
        if last is not None and last.role == role and last.text == text:
            return
        self._transcript.append(Turn(role=role, text=text))
        self._store.save(self.build_record())

    def add_tool_call(self, name: str, arguments: dict[str, Any], result: dict[str, Any]) -> None:
        self._tool_calls.append(ToolCallEvent(name=name, arguments=arguments, result=result))

    def build_record(self, ended_at: datetime | None = None) -> ConversationRecord:
        session = self.session
        if ended_at is None:
            status = ConversationStatus.IN_PROGRESS
        else:
            # A conversation that ends while still open was dropped by the customer.
            status = _STATUS_BY_STAGE.get(session.stage, ConversationStatus.ABANDONED)
        return ConversationRecord(
            id=self.id,
            channel=self.channel,
            status=status,
            stage=session.stage,
            started_at=self._started_at,
            ended_at=ended_at,
            ticket=session.ticket.model_copy(),
            order=session.order,
            escalation_reason=session.escalation_reason,
            transcript=list(self._transcript),
            tool_calls=list(self._tool_calls),
            validation_events=list(session.validation_events),
        )

    async def finish(self) -> ConversationRecord:
        """Add the summary and save the final record. Safe to call more than once."""
        if self._final is not None:
            return self._final
        record = self.build_record(ended_at=utc_now())
        record.summary = await self._summarizer.summarize(record)
        self._store.save(record)
        self._final = record
        return record
