from datetime import datetime

import pytest

from screening_agent.domain.models import InterviewSlot, Recommendation, Requirement, Stage
from screening_agent.domain.state import SessionClosedError

TEAMWORK = (
    "Last Black Friday at the supermarket we had two trucks late, so I organized "
    "the team in pairs and we unloaded everything before opening."
)
SAFETY = (
    "I saw a pallet leaning near the exit, so I stopped the forklift and told my "
    "supervisor, and we restacked it before anyone got hurt."
)


def pass_must_haves(session):
    session.start_screening("Lucia Gomez", True)
    session.record_must_have("work_permit", "yes")
    session.record_must_have("shifts", "morning and afternoon")
    session.record_must_have("start_date", "2026-11-20")
    return session.record_must_have("commute", True)


def answer(session, words, question_id, text, specificity="specific", score=4):
    words.lines.append(text)
    quote = " ".join(text.split()[2:10])
    return session.record_answer(question_id, f"Summary of the {question_id} answer.", quote,
                                 specificity, score)  # fmt: skip


def to_scheduling(session, words):
    pass_must_haves(session)
    answer(session, words, "teamwork", TEAMWORK)
    return answer(session, words, "safety", SAFETY)


SLOTS = [
    InterviewSlot(id="S1", starts_at=datetime(2026, 11, 11, 10, 0)),
    InterviewSlot(id="S2", starts_at=datetime(2026, 11, 11, 16, 0)),
]


# --- Consent ---------------------------------------------------------------------------


def test_screening_starts_in_consent_and_needs_it(session):
    assert session.stage is Stage.CONSENT
    result = session.record_must_have("work_permit", "yes")
    assert result["ok"] is False
    assert "start_screening" in result["error"]


def test_no_consent_ends_the_call(session):
    result = session.start_screening("Lucia", "no")
    assert result["ended"] is True
    assert session.stage is Stage.ENDED
    assert session.is_closed


def test_consent_moves_to_must_haves(session):
    result = session.start_screening("Lucia Gomez", "sí")
    assert result["ok"] is True
    assert session.stage is Stage.MUST_HAVES
    assert session.scorecard.candidate_name == "Lucia Gomez"
    assert result["open_requirements"] == [r.value for r in Requirement]


# --- Must-haves ------------------------------------------------------------------------


def test_all_must_haves_pass_moves_to_experience(session):
    result = pass_must_haves(session)
    assert session.stage is Stage.EXPERIENCE
    assert result["next_question"]["id"] == "teamwork"
    assert all(check.passed for check in session.scorecard.must_haves.values())


def test_a_failed_must_have_needs_confirmation_first(session):
    session.start_screening("Lucia", True)
    result = session.record_must_have("work_permit", "no")
    assert result["meets_requirement"] is False
    assert "confirm" in result["instruction"]
    assert session.stage is Stage.MUST_HAVES


def test_a_confirmed_failure_closes_as_not_eligible(session):
    session.start_screening("Lucia", True)
    session.record_must_have("work_permit", "no")
    result = session.record_must_have("work_permit", "no", confirmed=True)
    assert result["not_eligible"] is True
    assert session.stage is Stage.NOT_ELIGIBLE
    assert session.failed_requirement is Requirement.WORK_PERMIT
    assert session.scorecard.recommendation is Recommendation.REJECT


def test_confirmed_true_on_the_first_failure_still_asks_to_confirm(session):
    # Gemini cannot skip the read-back by sending confirmed=true at once.
    session.start_screening("Lucia", True)
    result = session.record_must_have("commute", "no", confirmed=True)
    assert result["meets_requirement"] is False
    assert session.stage is Stage.MUST_HAVES


def test_a_correction_after_a_misheard_failure_passes(session):
    session.start_screening("Lucia", True)
    session.record_must_have("work_permit", "no")
    session.record_must_have("work_permit", "yes")
    assert session.scorecard.must_haves[Requirement.WORK_PERMIT].passed


def test_missing_a_required_shift_fails(session):
    session.start_screening("Lucia", True)
    result = session.record_must_have("shifts", "only mornings")
    assert result["meets_requirement"] is False
    assert session.scorecard.must_haves[Requirement.SHIFTS].answer == "morning"


def test_any_shift_is_kept_as_any(session):
    session.start_screening("Lucia", True)
    session.record_must_have("shifts", "any shift you need")
    check = session.scorecard.must_haves[Requirement.SHIFTS]
    assert (check.answer, check.passed) == ("any", True)


def test_start_date_after_the_limit_fails(session):
    session.start_screening("Lucia", True)
    assert session.record_must_have("start_date", "2026-12-10")["ok"] is True
    assert session.scorecard.must_haves[Requirement.START_DATE].passed
    result = session.record_must_have("start_date", "2027-01-15")
    assert result["meets_requirement"] is False


def test_invalid_values_escalate_after_max_attempts(session):
    session.start_screening("Lucia", True)
    for _ in range(3):
        result = session.record_must_have("start_date", "sometime soon")
    assert result["ok"] is False
    assert session.stage is Stage.ESCALATED
    assert "start_date" in session.close_reason


# --- Experience questions --------------------------------------------------------------


def test_answers_need_the_experience_stage(session, words):
    session.start_screening("Lucia", True)
    result = answer(session, words, "teamwork", TEAMWORK)
    assert result["ok"] is False


def test_a_specific_answer_is_recorded(session, words):
    pass_must_haves(session)
    result = answer(session, words, "teamwork", TEAMWORK)
    assert result["ok"] is True
    assert result["next_question"]["id"] == "safety"
    assert session.scorecard.answers["teamwork"].score == 4


def test_a_vague_answer_gets_one_follow_up(session, words):
    pass_must_haves(session)
    vague = "I am a very good team player and I always work hard with everyone."
    result = answer(session, words, "teamwork", vague, specificity="vague", score=2)
    assert result["needs_follow_up"] is True
    assert "teamwork" not in session.scorecard.answers

    result = answer(session, words, "teamwork", TEAMWORK)
    saved = session.scorecard.answers["teamwork"]
    assert result["ok"] is True
    assert saved.follow_up_asked is True
    assert saved.score == 4


def test_still_vague_after_the_follow_up_is_capped(session, words):
    pass_must_haves(session)
    vague = "I am a very good team player and I always work hard with everyone."
    answer(session, words, "teamwork", vague, specificity="vague", score=4)
    answer(session, words, "teamwork", vague, specificity="vague", score=4)
    assert session.scorecard.answers["teamwork"].score == 2


def test_an_invented_quote_is_rejected(session, words):
    pass_must_haves(session)
    words.lines.append(TEAMWORK)
    result = session.record_answer(
        "teamwork", "Led the team.", "I led a team of twenty people", "specific", 4
    )
    assert result["ok"] is False
    assert "evidence_quote" in result["error"]
    assert session.stage is Stage.EXPERIENCE


def test_a_quote_with_small_transcription_changes_is_accepted(session, words):
    pass_must_haves(session)
    words.lines.append("so I organised the team in pairs and we unloaded everything")
    result = session.record_answer(
        "teamwork", "Organized the team in pairs.", "I organized the team in pairs", "specific", 3
    )
    assert result["ok"] is True


def test_unknown_question_is_rejected(session, words):
    pass_must_haves(session)
    words.lines.append(TEAMWORK)
    result = session.record_answer("salary", "x" * 20, "we had two trucks late", "specific", 3)
    assert "Unknown question" in result["error"]


def test_all_answers_move_to_scheduling(session, words):
    result = to_scheduling(session, words)
    assert session.stage is Stage.SCHEDULING
    assert "get_interview_slots" in result["instruction"]
    assert session.scorecard.recommendation is Recommendation.ADVANCE


def test_low_scores_ask_for_a_review(session, words):
    pass_must_haves(session)
    answer(session, words, "teamwork", TEAMWORK, score=2)
    answer(session, words, "safety", SAFETY, score=3)
    assert session.scorecard.recommendation is Recommendation.REVIEW


# --- Scheduling ------------------------------------------------------------------------


def test_only_an_offered_slot_can_be_booked(session, words):
    to_scheduling(session, words)
    assert session.book_interview("S1")["ok"] is False
    session.offer_slots(SLOTS)
    result = session.book_interview("S2")
    assert result["ok"] is True
    assert session.stage is Stage.BOOKED
    assert session.scorecard.interview.id == "S2"
    assert session.scorecard.booking_id.startswith("INT-")


def test_no_free_slots_asks_for_a_recruiter(session, words):
    to_scheduling(session, words)
    result = session.offer_slots([])
    assert "escalate_to_human" in result["error"]


# --- Any stage -------------------------------------------------------------------------


def test_open_questions_are_kept_once(session):
    session.add_open_question("Can I bring  my dog?")
    session.add_open_question("Can I bring my dog?")
    assert session.scorecard.open_questions == ["Can I bring my dog?"]


def test_escalate_and_end_close_the_call(session):
    session.escalate("Candidate asked for a person.")
    assert session.stage is Stage.ESCALATED
    with pytest.raises(SessionClosedError):
        session.end("bye")
