"""Records one conversation and saves it through the ``ConversationStore`` port.

Every channel uses this class. It saves after each turn, so a crash or a dropped
call keeps a partial record. ``finish`` adds the summary and saves the final record.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any, Literal

from screening_agent.application.ports import ConversationStore, Summarizer
from screening_agent.domain.models import (
    ConversationRecord,
    ConversationStatus,
    Language,
    SentimentReading,
    Stage,
    ToolCallEvent,
    Turn,
    utc_now,
)
from screening_agent.domain.state import ScreeningSession

_STATUS_BY_STAGE = {
    Stage.BOOKED: ConversationStatus.BOOKED,
    Stage.NOT_ELIGIBLE: ConversationStatus.NOT_ELIGIBLE,
    Stage.ESCALATED: ConversationStatus.ESCALATED,
    Stage.ENDED: ConversationStatus.ENDED,
}


class ConversationRecorder:
    def __init__(
        self,
        channel: Literal["text", "voice"],
        session: ScreeningSession,
        store: ConversationStore,
        summarizer: Summarizer,
        language: Language = Language.EN,
    ) -> None:
        self.id = f"{utc_now():%Y%m%dT%H%M%S}-{uuid.uuid4().hex[:8]}"
        self.channel = channel
        self.language = language
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

    def add_turn(self, role: Literal["user", "assistant"], text: str) -> int | None:
        """Add a turn and save. Returns the turn's index, or None if it was skipped."""
        text = text.strip()
        if not text:
            return None
        # The voice channel can report the same closing message twice; keep one copy.
        last = self._transcript[-1] if self._transcript else None
        if last is not None and last.role == role and last.text == text:
            return None
        self._transcript.append(Turn(role=role, text=text))
        self._store.save(self.build_record())
        return len(self._transcript) - 1

    def candidate_texts(self) -> list[str]:
        """Everything the candidate said, for the evidence quote check."""
        return [turn.text for turn in self._transcript if turn.role == "user"]

    def turns_before(self, index: int, limit: int) -> list[Turn]:
        return list(self._transcript[max(0, index - limit) : index])

    def sentiment_at(self, index: int) -> SentimentReading | None:
        return self._transcript[index].sentiment

    def attach_sentiment(self, index: int, reading: SentimentReading) -> None:
        self._transcript[index].sentiment = reading
        if self._final is None:
            self._store.save(self.build_record())

    def add_tool_call(self, name: str, arguments: dict[str, Any], result: dict[str, Any]) -> None:
        self._tool_calls.append(ToolCallEvent(name=name, arguments=arguments, result=result))

    def build_record(self, ended_at: datetime | None = None) -> ConversationRecord:
        session = self.session
        if ended_at is None:
            status = ConversationStatus.IN_PROGRESS
        else:
            # A conversation that ends while still open was dropped by the candidate.
            status = _STATUS_BY_STAGE.get(session.stage, ConversationStatus.ABANDONED)
        return ConversationRecord(
            id=self.id,
            channel=self.channel,
            language=self.language,
            status=status,
            stage=session.stage,
            started_at=self._started_at,
            ended_at=ended_at,
            job_id=session.job.id,
            scorecard=session.scorecard.model_copy(deep=True),
            close_reason=session.close_reason,
            transcript=[turn.model_copy() for turn in self._transcript],
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
