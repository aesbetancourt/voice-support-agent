"""Tools the LLM can call, shared by the text and voice channels.

Each channel converts ``TOOL_SPECS`` to its own SDK format and sends every call
to ``ToolExecutor``. When a call closes the conversation, the executor returns a
fixed closing message. The channel speaks or prints it, so the booking time, the
reference and the closing words never depend on the LLM.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from loguru import logger

from screening_agent.application.messages import closing_message
from screening_agent.application.ports import InterviewCalendar, JobKnowledge, KnowledgeHit
from screening_agent.domain.models import Language, Requirement, Specificity
from screening_agent.domain.state import ScreeningSession, SessionClosedError

# Three pieces cover a question that touches two topics ("pay and shifts?").
SEARCH_LIMIT = 3
# A search takes about 0.5 s. After 4 s the candidate hears too long a silence.
SEARCH_TIMEOUT_SECS = 4.0
# Enough choice for most candidates; Ava offers two or three at a time.
SLOT_LIMIT = 6


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    properties: dict[str, Any] = field(default_factory=dict)
    required: list[str] = field(default_factory=list)

    def json_schema(self) -> dict[str, Any]:
        return {"type": "object", "properties": self.properties, "required": self.required}


TOOL_SPECS: list[ToolSpec] = [
    ToolSpec(
        name="start_screening",
        description=(
            "Call after the candidate says their name and answers if they agree to the "
            "screening call. Call it also when they do not agree."
        ),
        properties={
            "candidate_name": {"type": "string", "description": "The name the candidate gave."},
            "consent": {
                "type": "boolean",
                "description": "True if the candidate agrees to continue with the screening.",
            },
        },
        required=["candidate_name", "consent"],
    ),
    ToolSpec(
        name="record_must_have",
        description=(
            "Save the candidate's answer to one requirement, as soon as they give it. "
            "The system decides if it meets the requirement."
        ),
        properties={
            "requirement": {"type": "string", "enum": [r.value for r in Requirement]},
            "answer": {
                "type": "string",
                "description": (
                    "work_permit and commute: 'yes' or 'no'. shifts: exactly the shifts the "
                    "candidate named, from morning, afternoon and night; 'any' only if they "
                    "said any shift. start_date: the earliest start date as YYYY-MM-DD, or "
                    "'immediately'."
                ),
            },
            "confirmed": {
                "type": "boolean",
                "description": (
                    "True only when the system asked you to confirm this answer and the "
                    "candidate confirmed it."
                ),
            },
        },
        required=["requirement", "answer"],
    ),
    ToolSpec(
        name="record_answer",
        description=(
            "Save your assessment of the candidate's answer to one experience question. "
            "The candidate never hears it."
        ),
        properties={
            "question_id": {"type": "string", "description": "The id of the question."},
            "summary": {
                "type": "string",
                "description": "What the candidate said, in one or two sentences in English.",
            },
            "evidence_quote": {
                "type": "string",
                "description": (
                    "Three to fifteen words copied exactly from the candidate's answer, in "
                    "their language, that support your score."
                ),
            },
            "specificity": {
                "type": "string",
                "enum": [s.value for s in Specificity],
                "description": (
                    "specific: a real situation with the candidate's own actions. "
                    "vague: general claims ('I am a team player') without a real example."
                ),
            },
            "score": {
                "type": "integer",
                "description": (
                    "1: no relevant example. 2: weak or unclear example. 3: good example. "
                    "4: strong example with clear actions and result."
                ),
            },
        },
        required=["question_id", "summary", "evidence_quote", "specificity", "score"],
    ),
    ToolSpec(
        name="search_job_info",
        description=(
            "Find information about the job (pay, shifts, location, contract, tasks, "
            "training, benefits, next steps) to answer a candidate's question."
        ),
        properties={
            "question": {
                "type": "string",
                "description": "The candidate's question, in English, as a full sentence.",
            }
        },
        required=["question"],
    ),
    ToolSpec(
        name="get_interview_slots",
        description="Get the free times for the video interview with the warehouse manager.",
    ),
    ToolSpec(
        name="book_interview",
        description="Book the interview time that the candidate chose.",
        properties={"slot_id": {"type": "string", "description": "A slot_id from the slots."}},
        required=["slot_id"],
    ),
    ToolSpec(
        name="escalate_to_human",
        description=(
            "Pass the call to a recruiter, who calls back. Use when the candidate asks for a "
            "person, is very upset, no interview time fits, or the request is out of scope."
        ),
        properties={"reason": {"type": "string", "description": "Short reason, in English."}},
        required=["reason"],
    ),
    ToolSpec(
        name="end_conversation",
        description="End the call when the candidate withdraws or wants to stop.",
        properties={"reason": {"type": "string", "description": "Short reason, in English."}},
        required=["reason"],
    ),
]


@dataclass(frozen=True)
class ToolOutcome:
    payload: dict[str, Any]
    # Set only on the call that closed the conversation, so the channel says it once.
    closing_message: str | None = None


ToolCallListener = Callable[[str, dict[str, Any], dict[str, Any]], None]


class ToolExecutor:
    def __init__(
        self,
        session: ScreeningSession,
        knowledge: JobKnowledge | None,
        calendar: InterviewCalendar,
        on_call: ToolCallListener | None = None,
        language: Language = Language.EN,
    ) -> None:
        self.session = session
        self.language = language
        self._knowledge = knowledge
        self._calendar = calendar
        self._on_call = on_call

    async def execute(self, name: str, arguments: dict[str, Any] | None) -> ToolOutcome:
        args = dict(arguments or {})
        was_open = not self.session.is_closed
        try:
            payload = await self._dispatch(name, args)
        except SessionClosedError as exc:
            payload = {"ok": False, "error": str(exc)}
        except KeyError as exc:
            payload = {"ok": False, "error": f"Missing argument {exc} for {name}."}
        except (TypeError, ValueError) as exc:
            payload = {"ok": False, "error": f"Invalid arguments for {name}: {exc}"}

        if self._on_call is not None:
            self._on_call(name, args, payload)
        closing = None
        if was_open and self.session.is_closed:
            closing = closing_message(self.language, self.session)
        return ToolOutcome(payload=payload, closing_message=closing)

    async def _dispatch(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        session = self.session
        match name:
            case "start_screening":
                return session.start_screening(str(args["candidate_name"]), args["consent"])
            case "record_must_have":
                return session.record_must_have(
                    str(args["requirement"]), args["answer"], args.get("confirmed", False)
                )
            case "record_answer":
                return session.record_answer(
                    str(args["question_id"]),
                    str(args["summary"]),
                    str(args["evidence_quote"]),
                    str(args["specificity"]),
                    args["score"],
                )
            case "search_job_info":
                return await self._search(str(args["question"]))
            case "get_interview_slots":
                return session.offer_slots(self._calendar.free_slots(SLOT_LIMIT))
            case "book_interview":
                result = session.book_interview(str(args["slot_id"]))
                if result["ok"]:
                    self._calendar.book(str(args["slot_id"]))
                return result
            case "escalate_to_human":
                return session.escalate(str(args.get("reason", "")))
            case "end_conversation":
                return session.end(str(args.get("reason", "")))
            case _:
                return {"ok": False, "error": f"Unknown tool '{name}'."}

    async def _search(self, question: str) -> dict[str, Any]:
        if self.session.is_closed:
            raise SessionClosedError(f"The conversation is already {self.session.stage.value}.")
        hits = await self._find(question)
        if not hits:
            # The recruiter answers it later; the question goes on the scorecard.
            self.session.add_open_question(question)
            return {
                "ok": True,
                "found": False,
                "instruction": (
                    "The job information has no answer. Say that you do not have that detail "
                    "and that the recruiter will answer it. Do not guess. Then continue."
                ),
            }
        return {
            "ok": True,
            "found": True,
            "information": [hit.text for hit in hits],
            "instruction": (
                "Answer only from this information, in one or two short sentences. If it does "
                "not answer the question, say that the recruiter will answer it. "
                "Then continue the screening where you stopped."
            ),
        }

    async def _find(self, question: str) -> list[KnowledgeHit]:
        if self._knowledge is None:
            return []
        try:
            return await asyncio.wait_for(
                self._knowledge.search(question, SEARCH_LIMIT), SEARCH_TIMEOUT_SECS
            )
        except Exception as exc:  # noqa: BLE001 - a failed search must not break the call
            logger.warning(f"Job search failed, the recruiter answers instead: {exc!r}")
            return []
