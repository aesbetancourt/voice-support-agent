"""Screening state machine.

Gemini proposes values through tool calls. This module decides if they are valid,
what the candidate's answers mean for the application, and when the call closes.
Gemini cannot skip these rules.

    CONSENT --consent--> MUST_HAVES --all pass--> EXPERIENCE --all answered--> SCHEDULING
    SCHEDULING --book an offered slot--> BOOKED
    MUST_HAVES --a failed answer, confirmed by the candidate--> NOT_ELIGIBLE
    CONSENT --no consent--> ENDED;  any open stage --candidate withdraws--> ENDED
    any open stage --asks for a person, or too many invalid values--> ESCALATED
"""

from __future__ import annotations

import secrets
from collections import Counter
from collections.abc import Callable
from datetime import date, timedelta
from typing import Any

from screening_agent.domain.models import (
    ExperienceAnswer,
    InterviewSlot,
    JobPosting,
    MustHaveCheck,
    Recommendation,
    Requirement,
    Scorecard,
    Shift,
    Specificity,
    Stage,
    ValidationEvent,
)
from screening_agent.domain.validators import (
    parse_requirement,
    parse_score,
    parse_shifts,
    parse_specificity,
    parse_start_date,
    parse_yes_no,
    quote_in_words,
    validate_name,
    validate_summary,
)

_CLOSED_STAGES = {Stage.BOOKED, Stage.NOT_ELIGIBLE, Stage.ESCALATED, Stage.ENDED}
# One follow-up gives a vague candidate a second chance; more feels like an interrogation.
MAX_FOLLOW_UPS = 1
# A vague answer, even after a follow-up, shows no real example: it cannot score high.
VAGUE_SCORE_CAP = 2
# Advance when the answers are good on average, and none is weak.
ADVANCE_MIN_AVERAGE = 3.0
ADVANCE_MIN_SCORE = 2

# Returns everything the candidate said so far. The evidence quotes are checked against it.
CandidateWords = Callable[[], list[str]]


def _is_iso_date(text: str) -> bool:
    try:
        date.fromisoformat(text.strip())
    except ValueError:
        return False
    return True


class SessionClosedError(RuntimeError):
    """Raised when a tool is called after the conversation has closed."""


def recommend(scorecard: Scorecard, job: JobPosting) -> Recommendation:
    """The code's recommendation for the recruiter. Gemini never decides it."""
    checks = scorecard.must_haves.values()
    if any(not check.passed and check.confirmed for check in checks):
        return Recommendation.REJECT
    all_passed = len(scorecard.must_haves) == len(Requirement) and all(c.passed for c in checks)
    if not all_passed or len(scorecard.answers) < len(job.questions):
        return Recommendation.INCOMPLETE
    scores = [answer.score for answer in scorecard.answers.values()]
    if sum(scores) / len(scores) >= ADVANCE_MIN_AVERAGE and min(scores) >= ADVANCE_MIN_SCORE:
        return Recommendation.ADVANCE
    return Recommendation.REVIEW


class ScreeningSession:
    def __init__(
        self,
        job: JobPosting,
        today: date,
        candidate_words: CandidateWords,
        max_attempts: int = 3,
    ) -> None:
        self.job = job
        self.today = today
        self._candidate_words = candidate_words
        self.max_attempts = max_attempts
        self.stage = Stage.CONSENT
        self.scorecard = Scorecard()
        self.close_reason: str | None = None
        # The requirement that a confirmed failure closed the application on.
        self.failed_requirement: Requirement | None = None
        self.validation_events: list[ValidationEvent] = []
        self._follow_ups: Counter[str] = Counter()
        self._offered: dict[str, InterviewSlot] = {}
        self._failures: Counter[str] = Counter()

    @property
    def is_closed(self) -> bool:
        return self.stage in _CLOSED_STAGES

    @property
    def latest_start(self) -> date:
        return self.today + timedelta(days=self.job.max_start_days)

    # --- Consent -----------------------------------------------------------------------

    def start_screening(self, candidate_name: str, consent: object) -> dict[str, Any]:
        self._ensure_open()
        if error := self._wrong_stage(Stage.CONSENT):
            return error
        try:
            agreed = parse_yes_no(consent)
            name = validate_name(candidate_name)
        except ValueError as exc:
            return self._reject("consent", {"name": candidate_name, "consent": consent}, str(exc))
        self.scorecard.candidate_name = name
        if not agreed:
            self.stage = Stage.ENDED
            self.close_reason = "The candidate did not agree to the screening."
            return {"ok": True, "ended": True, **self._progress()}
        self.stage = Stage.MUST_HAVES
        return {
            "ok": True,
            "instruction": "Ask about the requirements, one at a time.",
            **self._progress(),
        }

    # --- Must-haves ----------------------------------------------------------------------

    def record_must_have(
        self, requirement: str, answer: object, confirmed: object = False
    ) -> dict[str, Any]:
        self._ensure_open()
        if error := self._wrong_stage(Stage.MUST_HAVES):
            return error
        try:
            req = parse_requirement(requirement)
            canonical, passed = self._evaluate(req, answer)
            is_confirmed = parse_yes_no(confirmed) if confirmed not in (None, "") else False
        except ValueError as exc:
            return self._reject(str(requirement), answer, str(exc))

        if passed:
            self.scorecard.must_haves[req] = MustHaveCheck(
                requirement=req, answer=canonical, passed=True
            )
            return self._after_must_have()

        previous = self.scorecard.must_haves.get(req)
        pending_same = previous is not None and not previous.passed and not previous.confirmed
        if is_confirmed and pending_same:
            self.scorecard.must_haves[req] = MustHaveCheck(
                requirement=req, answer=canonical, passed=False, confirmed=True
            )
            self.stage = Stage.NOT_ELIGIBLE
            self.failed_requirement = req
            self.close_reason = f"Does not meet the requirement '{req.value}': {canonical}."
            return {"ok": True, "not_eligible": True, **self._progress()}

        self.scorecard.must_haves[req] = MustHaveCheck(
            requirement=req, answer=canonical, passed=False, confirmed=False
        )
        return {
            "ok": True,
            "meets_requirement": False,
            "instruction": (
                "This answer does not meet the requirement. Repeat what you understood in "
                "neutral words and ask the candidate if it is correct. Do not say that it "
                "fails. If they confirm, call record_must_have again with the same answer and "
                "confirmed=true. If they correct it, record the corrected answer."
            ),
            **self._progress(),
        }

    def _evaluate(self, req: Requirement, answer: object) -> tuple[str, bool]:
        match req:
            case Requirement.WORK_PERMIT | Requirement.COMMUTE:
                yes = parse_yes_no(answer)
                return ("yes" if yes else "no"), yes
            case Requirement.SHIFTS:
                shifts = parse_shifts(answer)
                # "Any" stays "any": the recruiter never sees a shift the candidate did not name.
                named = ", ".join(s.value for s in Shift if s in shifts) or "none"
                canonical = "any" if shifts == set(Shift) else named
                return canonical, set(self.job.required_shifts) <= shifts
            case Requirement.START_DATE:
                start = parse_start_date(str(answer), self.today)
                # Keep the candidate's "immediately" visible next to the date it stands for.
                said_now = start == self.today and not _is_iso_date(str(answer))
                canonical = f"immediately ({start.isoformat()})" if said_now else start.isoformat()
                return canonical, start <= self.latest_start

    def _after_must_have(self) -> dict[str, Any]:
        checks = self.scorecard.must_haves
        if len(checks) == len(Requirement) and all(c.passed for c in checks.values()):
            self.stage = Stage.EXPERIENCE
            return {
                "ok": True,
                "instruction": "All requirements are met. Do not say so. Ask the next question.",
                **self._progress(),
            }
        return {"ok": True, **self._progress()}

    # --- Experience questions ------------------------------------------------------------

    def record_answer(
        self,
        question_id: str,
        summary: str,
        evidence_quote: str,
        specificity: str,
        score: object,
    ) -> dict[str, Any]:
        self._ensure_open()
        if error := self._wrong_stage(Stage.EXPERIENCE):
            return error
        question = self.job.question(str(question_id))
        if question is None:
            ids = ", ".join(q.id for q in self.job.questions)
            return {"ok": False, "error": f"Unknown question '{question_id}'. Use one of: {ids}."}
        if question.id in self.scorecard.answers:
            return {"ok": False, "error": f"Question '{question.id}' is already answered."}
        try:
            text = validate_summary(summary)
            spec = parse_specificity(specificity)
            points = parse_score(score)
        except ValueError as exc:
            return {"ok": False, "error": str(exc), **self._progress()}
        # A wrong quote is Gemini's error, not the candidate's: it never escalates the call.
        if not quote_in_words(evidence_quote, self._candidate_words()):
            return {
                "ok": False,
                "error": (
                    "The evidence_quote is not in the candidate's words. Copy three or more "
                    "words exactly as the candidate said them, in their language."
                ),
            }

        if spec is Specificity.VAGUE and self._follow_ups[question.id] < MAX_FOLLOW_UPS:
            self._follow_ups[question.id] += 1
            return {
                "ok": True,
                "needs_follow_up": True,
                "instruction": (
                    "The answer is vague. Ask one short follow-up for a real example: what "
                    "the candidate did personally, and what happened. Then call record_answer "
                    "again for this question."
                ),
                **self._progress(),
            }

        if spec is Specificity.VAGUE:
            points = min(points, VAGUE_SCORE_CAP)
        self.scorecard.answers[question.id] = ExperienceAnswer(
            question_id=question.id,
            summary=text,
            evidence_quote=" ".join(evidence_quote.split()),
            specificity=spec,
            score=points,
            follow_up_asked=self._follow_ups[question.id] > 0,
        )
        self._refresh()
        if len(self.scorecard.answers) == len(self.job.questions):
            self.stage = Stage.SCHEDULING
            return {
                "ok": True,
                "instruction": "Thank the candidate. Call get_interview_slots and offer times.",
                **self._progress(),
            }
        return {"ok": True, **self._progress()}

    # --- Scheduling ----------------------------------------------------------------------

    def offer_slots(self, slots: list[InterviewSlot]) -> dict[str, Any]:
        self._ensure_open()
        if error := self._wrong_stage(Stage.SCHEDULING):
            return error
        self._offered.update({slot.id: slot for slot in slots})
        if not slots:
            return {
                "ok": False,
                "error": "No free slots. Call escalate_to_human so a recruiter calls back.",
            }
        return {
            "ok": True,
            "slots": [
                {"slot_id": slot.id, "starts_at": slot.starts_at.strftime("%A %Y-%m-%d %H:%M")}
                for slot in slots
            ],
            "instruction": "Offer two or three of these times. Book the one the candidate picks.",
        }

    def book_interview(self, slot_id: str) -> dict[str, Any]:
        self._ensure_open()
        if error := self._wrong_stage(Stage.SCHEDULING):
            return error
        slot = self._offered.get(str(slot_id))
        if slot is None:
            return {
                "ok": False,
                "error": (
                    f"Slot '{slot_id}' was not offered. Use a slot_id from get_interview_slots."
                ),
            }
        self.scorecard.interview = slot
        self.scorecard.booking_id = f"INT-{secrets.randbelow(10**6):06d}"
        self.stage = Stage.BOOKED
        self.close_reason = "Interview booked."
        return {"ok": True, "booking_id": self.scorecard.booking_id, **self._progress()}

    # --- Any stage -----------------------------------------------------------------------

    def add_open_question(self, question: str) -> None:
        """A candidate question with no answer in the job information, for the recruiter."""
        text = " ".join(question.split())
        if text and text not in self.scorecard.open_questions:
            self.scorecard.open_questions.append(text)

    def escalate(self, reason: str) -> dict[str, Any]:
        self._ensure_open()
        self.stage = Stage.ESCALATED
        self.close_reason = reason.strip() or "unspecified"
        return {"ok": True, "escalated": True, "reason": self.close_reason}

    def end(self, reason: str) -> dict[str, Any]:
        self._ensure_open()
        self.stage = Stage.ENDED
        self.close_reason = reason.strip() or "The candidate ended the call."
        return {"ok": True, "ended": True, "reason": self.close_reason}

    # --- Internals -----------------------------------------------------------------------

    def _ensure_open(self) -> None:
        if self.is_closed:
            raise SessionClosedError(f"The conversation is already {self.stage.value}.")

    def _wrong_stage(self, expected: Stage) -> dict[str, Any] | None:
        if self.stage is expected:
            return None
        return {
            "ok": False,
            "error": f"Not possible in stage '{self.stage.value}'. {self._next_step()}",
            **self._progress(),
        }

    def _reject(self, field: str, value: Any, error: str) -> dict[str, Any]:
        self.validation_events.append(ValidationEvent(field=field, value=value, error=error))
        self._failures[field] += 1
        if self._failures[field] >= self.max_attempts:
            self.stage = Stage.ESCALATED
            self.close_reason = f"Could not get a valid {field} after {self.max_attempts} attempts."
        return {"ok": False, "rejected": {field: error}, **self._progress()}

    def _refresh(self) -> None:
        self.scorecard.recommendation = recommend(self.scorecard, self.job)

    def _next_step(self) -> str:
        match self.stage:
            case Stage.CONSENT:
                return "Get the candidate's name and consent, then call start_screening."
            case Stage.MUST_HAVES:
                return f"Ask about the open requirements: {', '.join(self._open_requirements())}."
            case Stage.EXPERIENCE:
                return "Ask the next experience question."
            case Stage.SCHEDULING:
                return "Call get_interview_slots and book a time."
            case _:
                return "The conversation is closed."

    def _open_requirements(self) -> list[str]:
        checks = self.scorecard.must_haves
        return [r.value for r in Requirement if r not in checks or not checks[r].passed]

    def _progress(self) -> dict[str, Any]:
        self._refresh()
        progress: dict[str, Any] = {"stage": self.stage.value}
        if self.stage is Stage.MUST_HAVES:
            progress["open_requirements"] = self._open_requirements()
        if self.stage is Stage.EXPERIENCE:
            next_q = next(q for q in self.job.questions if q.id not in self.scorecard.answers)
            progress["next_question"] = {"id": next_q.id, "text": next_q.text}
        if self.stage is Stage.ESCALATED:
            progress["escalation_reason"] = self.close_reason
        return progress
