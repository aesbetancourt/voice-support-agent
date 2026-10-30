"""Conversation summary rules: the LLM input, and a rule-based fallback.

The LLM call itself is an outbound adapter (``adapters.outbound.gemini``).
"""

from __future__ import annotations

import json

from loguru import logger

from support_agent.application.ports import Summarizer
from support_agent.domain.frustration import sentiment_from_peak
from support_agent.domain.models import ConversationRecord, Stage, Summary, Turn

SUMMARY_INSTRUCTION = """\
You summarize customer support conversations for the support team.
Use only facts from the transcript and the ticket. Do not invent details.
Write short, plain sentences.
Always write the summary in English, even if the conversation is in another language,
because the support team works in English.
"""


class SafeSummarizer:
    """Uses the primary summarizer, and the fallback summary if it fails."""

    def __init__(self, primary: Summarizer | None):
        self._primary = primary

    async def summarize(self, record: ConversationRecord) -> Summary:
        if self._primary is not None:
            try:
                return await self._primary.summarize(record)
            except Exception as exc:  # noqa: BLE001 - any failure must fall back
                logger.warning(f"LLM summary failed, using fallback: {exc}")
        return fallback_summary(record)


def fallback_summary(record: ConversationRecord) -> Summary:
    ticket = record.ticket
    key_points = [
        f"{label}: {value}"
        for label, value in (
            ("Order", ticket.order_number),
            ("Category", ticket.category),
            ("Urgency", ticket.urgency),
            ("Description", ticket.description),
            ("Ticket", ticket.ticket_id),
        )
        if value
    ]
    if record.escalation_reason:
        key_points.append(f"Escalated: {record.escalation_reason}")

    match record.stage:
        case Stage.COMPLETED:
            headline = f"Ticket {ticket.ticket_id} opened for order {ticket.order_number}."
            next_action = "Handle the ticket in the support queue."
        case Stage.ESCALATED:
            headline = "Conversation escalated to a human agent."
            next_action = "A human agent must contact the customer."
        case _:
            headline = "Conversation ended before a ticket was opened."
            next_action = "No action, unless the customer contacts support again."

    return Summary(
        headline=headline,
        key_points=key_points or ["No ticket details were collected."],
        customer_sentiment=sentiment_from_peak(peak_frustration(record)),
        next_action=next_action,
        generated_by="fallback",
    )


def peak_frustration(record: ConversationRecord) -> float | None:
    readings = [turn.sentiment.frustration for turn in record.transcript if turn.sentiment]
    return max(readings) if readings else None


def summary_input(record: ConversationRecord) -> str:
    """The text an LLM summarizer receives for one conversation."""
    transcript = "\n".join(_transcript_line(turn) for turn in record.transcript)
    ticket = json.dumps(record.ticket.model_dump(mode="json"), indent=2)
    return (
        f"Final stage: {record.stage.value}\n"
        f"Escalation reason: {record.escalation_reason or 'none'}\n\n"
        f"Ticket:\n{ticket}\n\nTranscript:\n{transcript}"
    )


def _transcript_line(turn: Turn) -> str:
    if turn.sentiment is None:
        return f"{turn.role}: {turn.text}"
    return f"{turn.role} (frustration {turn.sentiment.frustration:.1f}): {turn.text}"
