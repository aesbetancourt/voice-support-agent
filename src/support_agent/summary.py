"""Conversation summary: Gemini structured output, with a rule-based fallback."""

from __future__ import annotations

import json
from typing import Protocol

from google import genai
from google.genai import types
from loguru import logger

from support_agent.models import ConversationRecord, Sentiment, Stage, Summary, SummaryDraft

_INSTRUCTION = """\
You summarize customer support conversations for the support team.
Use only facts from the transcript and the ticket. Do not invent details.
Write short, plain sentences.
"""


class Summarizer(Protocol):
    async def summarize(self, record: ConversationRecord) -> Summary: ...


class GeminiSummarizer:
    def __init__(self, client: genai.Client, model: str):
        self._client = client
        self._model = model

    async def summarize(self, record: ConversationRecord) -> Summary:
        response = await self._client.aio.models.generate_content(
            model=self._model,
            contents=_summary_input(record),
            config=types.GenerateContentConfig(
                system_instruction=_INSTRUCTION,
                response_mime_type="application/json",
                response_schema=SummaryDraft,
                temperature=0.2,
            ),
        )
        draft = response.parsed
        if not isinstance(draft, SummaryDraft):
            draft = SummaryDraft.model_validate_json(response.text or "")
        return Summary(**draft.model_dump(), generated_by=self._model)


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
        customer_sentiment=Sentiment.NEUTRAL,
        next_action=next_action,
        generated_by="fallback",
    )


def _summary_input(record: ConversationRecord) -> str:
    transcript = "\n".join(f"{turn.role}: {turn.text}" for turn in record.transcript)
    ticket = json.dumps(record.ticket.model_dump(mode="json"), indent=2)
    return (
        f"Final stage: {record.stage.value}\n"
        f"Escalation reason: {record.escalation_reason or 'none'}\n\n"
        f"Ticket:\n{ticket}\n\nTranscript:\n{transcript}"
    )
