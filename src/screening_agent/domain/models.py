"""Data models for the job, the screening scorecard and saved conversations."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, Field


def utc_now() -> datetime:
    return datetime.now(UTC)


class Language(StrEnum):
    """Language of a conversation. Voice calls can switch it (see ADR-0009)."""

    EN = "en"
    ES = "es"


class Stage(StrEnum):
    """Stage of the screening state machine."""

    CONSENT = "consent"
    MUST_HAVES = "must_haves"
    EXPERIENCE = "experience"
    SCHEDULING = "scheduling"
    BOOKED = "booked"
    NOT_ELIGIBLE = "not_eligible"
    ESCALATED = "escalated"
    ENDED = "ended"


class ConversationStatus(StrEnum):
    IN_PROGRESS = "in_progress"
    BOOKED = "booked"
    NOT_ELIGIBLE = "not_eligible"
    ESCALATED = "escalated"
    # The candidate declined the screening or withdrew.
    ENDED = "ended"
    # The call dropped before the screening closed.
    ABANDONED = "abandoned"


class Requirement(StrEnum):
    """A must-have: a failed, confirmed must-have ends the application."""

    WORK_PERMIT = "work_permit"
    SHIFTS = "shifts"
    START_DATE = "start_date"
    COMMUTE = "commute"


class Shift(StrEnum):
    MORNING = "morning"
    AFTERNOON = "afternoon"
    NIGHT = "night"


class Specificity(StrEnum):
    """Specific: a real example with the candidate's own actions. Vague: general claims."""

    SPECIFIC = "specific"
    VAGUE = "vague"


class Recommendation(StrEnum):
    """The code's recommendation for the recruiter. A person makes the decision."""

    ADVANCE = "advance"
    REVIEW = "review"
    REJECT = "reject"
    INCOMPLETE = "incomplete"


class ExperienceQuestion(BaseModel):
    id: str
    text: str
    looks_for: str = Field(description="What a strong answer shows. Used to score it.")


class JobPosting(BaseModel):
    """The role that the screening is for. Data, not code: see data/job.json."""

    id: str
    company: str
    title: str
    site: str
    summary: str
    required_shifts: list[Shift]
    max_start_days: int = Field(ge=0, description="The latest start, in days from today.")
    questions: list[ExperienceQuestion]

    def question(self, question_id: str) -> ExperienceQuestion | None:
        return next((q for q in self.questions if q.id == question_id), None)


class MustHaveCheck(BaseModel):
    requirement: Requirement
    answer: str
    passed: bool
    # A failed answer must be read back and confirmed before it closes the call,
    # because speech recognition can mishear a "yes" or a date.
    confirmed: bool = True


class ExperienceAnswer(BaseModel):
    question_id: str
    summary: str
    evidence_quote: str
    specificity: Specificity
    score: int = Field(ge=1, le=4)
    follow_up_asked: bool = False


class InterviewSlot(BaseModel):
    id: str
    starts_at: datetime


class Scorecard(BaseModel):
    """What the recruiter gets. Gemini proposes values; the domain validates them."""

    candidate_name: str | None = None
    must_haves: dict[Requirement, MustHaveCheck] = Field(default_factory=dict)
    answers: dict[str, ExperienceAnswer] = Field(default_factory=dict)
    open_questions: list[str] = Field(default_factory=list)
    interview: InterviewSlot | None = None
    booking_id: str | None = None
    recommendation: Recommendation = Recommendation.INCOMPLETE


class FrustrationAction(StrEnum):
    """What the agent must do after a frustration reading."""

    NONE = "none"
    ACKNOWLEDGE = "acknowledge"
    OFFER_HUMAN = "offer_human"


class SentimentDraft(BaseModel):
    """Fields that the sentiment classifier fills (structured output schema)."""

    frustration: float = Field(
        ge=0.0, le=1.0, description="0.0 calm, 0.6 clearly frustrated, 1.0 furious."
    )
    reason: str = Field(description="Why, in at most 12 words.")


class SentimentReading(SentimentDraft):
    """A frustration reading for one candidate turn, with the action it caused."""

    action: FrustrationAction = FrustrationAction.NONE


class Turn(BaseModel):
    role: Literal["user", "assistant"]
    text: str
    at: datetime = Field(default_factory=utc_now)
    sentiment: SentimentReading | None = None


class ValidationEvent(BaseModel):
    field: str
    value: Any
    error: str
    at: datetime = Field(default_factory=utc_now)


class ToolCallEvent(BaseModel):
    name: str
    arguments: dict[str, Any]
    result: dict[str, Any]
    at: datetime = Field(default_factory=utc_now)


class Sentiment(StrEnum):
    POSITIVE = "positive"
    NEUTRAL = "neutral"
    FRUSTRATED = "frustrated"
    ANGRY = "angry"


class SummaryDraft(BaseModel):
    """Summary fields that the LLM fills (structured output schema)."""

    headline: str = Field(description="One sentence: who the candidate is and how the call ended.")
    key_points: list[str] = Field(description="Three to six short facts from the conversation.")
    candidate_sentiment: Sentiment
    next_action: str = Field(description="The next step the recruiting team must take.")


class Summary(SummaryDraft):
    generated_by: str = Field(description="Model ID, or 'fallback' when the LLM was not used.")


class ConversationRecord(BaseModel):
    """Everything saved for one conversation."""

    id: str
    channel: Literal["text", "voice"]
    language: Language = Language.EN
    job_id: str
    status: ConversationStatus
    stage: Stage
    started_at: datetime
    ended_at: datetime | None = None
    scorecard: Scorecard
    close_reason: str | None = None
    transcript: list[Turn] = Field(default_factory=list)
    tool_calls: list[ToolCallEvent] = Field(default_factory=list)
    validation_events: list[ValidationEvent] = Field(default_factory=list)
    summary: Summary | None = None
