"""Data models for orders, tickets and saved conversations."""

from __future__ import annotations

from datetime import UTC, date, datetime
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, Field


def utc_now() -> datetime:
    return datetime.now(UTC)


class Language(StrEnum):
    """Language of one conversation. Fixed at the start (see ADR-0008)."""

    EN = "en"
    ES = "es"


class Category(StrEnum):
    SHIPPING = "shipping"
    BILLING = "billing"
    PRODUCT_DEFECT = "product_defect"
    RETURN_REFUND = "return_refund"
    ACCOUNT = "account"
    OTHER = "other"


class Urgency(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class OrderStatus(StrEnum):
    PROCESSING = "processing"
    SHIPPED = "shipped"
    DELIVERED = "delivered"
    CANCELLED = "cancelled"
    RETURNED = "returned"


class Stage(StrEnum):
    """Stage of the ticket state machine."""

    COLLECTING = "collecting"
    CONFIRMING = "confirming"
    COMPLETED = "completed"
    ESCALATED = "escalated"
    ENDED = "ended"


class ConversationStatus(StrEnum):
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    ESCALATED = "escalated"
    ABANDONED = "abandoned"


class Order(BaseModel):
    order_number: str
    customer_name: str
    email: str
    status: OrderStatus
    items: list[str]
    total: float
    placed_on: date
    delivered_on: date | None = None


REQUIRED_FIELDS: tuple[str, ...] = ("order_number", "category", "description", "urgency")


class Ticket(BaseModel):
    """Support ticket. Fields stay empty until the customer gives a valid value."""

    ticket_id: str | None = None
    order_number: str | None = None
    customer_name: str | None = None
    category: Category | None = None
    description: str | None = None
    urgency: Urgency | None = None

    def missing_fields(self) -> list[str]:
        return [name for name in REQUIRED_FIELDS if getattr(self, name) is None]


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
    """A frustration reading for one customer turn, with the action it caused."""

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

    headline: str = Field(description="One sentence that states the customer's problem.")
    key_points: list[str] = Field(description="Three to six short facts from the conversation.")
    customer_sentiment: Sentiment
    next_action: str = Field(description="The next step the support team must take.")


class Summary(SummaryDraft):
    generated_by: str = Field(description="Model ID, or 'fallback' when the LLM was not used.")


class ConversationRecord(BaseModel):
    """Everything saved for one conversation."""

    id: str
    channel: Literal["text", "voice"]
    language: Language = Language.EN
    status: ConversationStatus
    stage: Stage
    started_at: datetime
    ended_at: datetime | None = None
    ticket: Ticket
    order: Order | None = None
    escalation_reason: str | None = None
    transcript: list[Turn] = Field(default_factory=list)
    tool_calls: list[ToolCallEvent] = Field(default_factory=list)
    validation_events: list[ValidationEvent] = Field(default_factory=list)
    summary: Summary | None = None
