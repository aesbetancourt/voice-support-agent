"""Conversation summary rules: the LLM input, and a rule-based fallback.

The LLM call itself is an outbound adapter (``adapters.outbound.gemini``).
"""

from __future__ import annotations

import json

from loguru import logger

from screening_agent.application.ports import Summarizer
from screening_agent.domain.frustration import sentiment_from_peak
from screening_agent.domain.models import ConversationRecord, Stage, Summary, Turn

SUMMARY_INSTRUCTION = """\
You summarize candidate screening calls for the recruiting team.
Use only facts from the transcript and the scorecard. Do not invent details.
The recommendation in the scorecard comes from fixed rules: report it, do not change it.
Do not mention age, nationality, health, family or other personal topics, even if the
candidate mentioned them.
Write short, plain sentences.
Always write the summary in English, even if the call is in another language,
because the recruiting team works in English.
"""


class SafeSummarizer:
    """Uses the primary summarizer, and the fallback summary if it fails."""

    def __init__(self, primary: Summarizer | None) -> None:
        self._primary = primary

    async def summarize(self, record: ConversationRecord) -> Summary:
        if self._primary is not None:
            try:
                return await self._primary.summarize(record)
            except Exception as exc:  # noqa: BLE001 - any failure must fall back
                logger.warning(f"LLM summary failed, using fallback: {exc}")
        return fallback_summary(record)


def fallback_summary(record: ConversationRecord) -> Summary:
    card = record.scorecard
    name = card.candidate_name or "The candidate"
    key_points = [
        f"{check.requirement.value}: {check.answer} ({'meets' if check.passed else 'fails'})"
        for check in card.must_haves.values()
    ]
    key_points += [
        f"{answer.question_id}: score {answer.score}/4, {answer.specificity.value}"
        for answer in card.answers.values()
    ]
    key_points += [f"Open question: {question}" for question in card.open_questions]
    key_points.append(f"Recommendation: {card.recommendation.value}")

    match record.stage:
        case Stage.BOOKED:
            when = card.interview.starts_at.isoformat(" ", "minutes") if card.interview else "?"
            headline = f"{name} passed the screening and booked an interview for {when}."
            next_action = "Hold the interview, and answer the open questions first."
        case Stage.NOT_ELIGIBLE:
            headline = f"{name} does not meet a must-have requirement."
            next_action = "No action, unless the candidate applies again."
        case Stage.ESCALATED:
            headline = f"{name} needs a call back from a recruiter."
            next_action = f"Call the candidate back. Reason: {record.close_reason or 'unknown'}."
        case _:
            headline = f"{name} ended the screening before it was complete."
            next_action = "No action, unless the candidate contacts the team again."

    return Summary(
        headline=headline,
        key_points=key_points,
        candidate_sentiment=sentiment_from_peak(peak_frustration(record)),
        next_action=next_action,
        generated_by="fallback",
    )


def peak_frustration(record: ConversationRecord) -> float | None:
    readings = [turn.sentiment.frustration for turn in record.transcript if turn.sentiment]
    return max(readings) if readings else None


def summary_input(record: ConversationRecord) -> str:
    """The text an LLM summarizer receives for one conversation."""
    transcript = "\n".join(_transcript_line(turn) for turn in record.transcript)
    scorecard = json.dumps(record.scorecard.model_dump(mode="json"), indent=2, ensure_ascii=False)
    return (
        f"Final stage: {record.stage.value}\n"
        f"Close reason: {record.close_reason or 'none'}\n\n"
        f"Scorecard:\n{scorecard}\n\nTranscript:\n{transcript}"
    )


def _transcript_line(turn: Turn) -> str:
    role = "candidate" if turn.role == "user" else "agent"
    if turn.sentiment is None:
        return f"{role}: {turn.text}"
    return f"{role} (frustration {turn.sentiment.frustration:.1f}): {turn.text}"
