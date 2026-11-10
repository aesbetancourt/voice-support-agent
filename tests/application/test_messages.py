"""Candidate-facing messages and prompts in every supported language."""

from datetime import datetime

import pytest
from conftest import TODAY

from screening_agent.application.messages import (
    LANGUAGE_NAMES,
    closing_message,
    closing_stages,
    retry_message,
    spoken_time,
    technical_error_message,
)
from screening_agent.application.prompts import greeting_trigger, system_prompt
from screening_agent.application.tools import ToolExecutor
from screening_agent.domain.models import InterviewSlot, Language, Requirement, Stage

CLOSING_STAGES = {Stage.BOOKED, Stage.NOT_ELIGIBLE, Stage.ESCALATED, Stage.ENDED}


@pytest.mark.parametrize("language", list(Language))
def test_every_language_has_every_closing_message(language):
    assert closing_stages(language) == CLOSING_STAGES
    assert language in LANGUAGE_NAMES
    assert retry_message(language)
    assert technical_error_message(language)


@pytest.mark.parametrize(
    ("language", "expected"),
    [
        (Language.EN, "Thursday 12 November at 16:00"),
        (Language.ES, "el jueves 12 de noviembre a las 16:00"),
    ],
)
def test_spoken_time(language, expected):
    assert spoken_time(language, datetime(2026, 11, 12, 16, 0)) == expected


@pytest.mark.parametrize("language", list(Language))
def test_booked_message_has_the_time_and_reference(session, language):
    session.stage = Stage.BOOKED
    session.scorecard.interview = InterviewSlot(id="S1", starts_at=datetime(2026, 11, 12, 10, 0))
    session.scorecard.booking_id = "INT-123456"
    message = closing_message(language, session)
    assert "12" in message
    assert "10:00" in message
    assert "INT-123456" in message


@pytest.mark.parametrize("requirement", list(Requirement))
@pytest.mark.parametrize("language", list(Language))
def test_not_eligible_message_names_every_requirement(session, language, requirement):
    session.stage = Stage.NOT_ELIGIBLE
    session.failed_requirement = requirement
    message = closing_message(language, session)
    assert "{" not in message
    if requirement is Requirement.SHIFTS:
        assert ("morning and afternoon" in message) or ("mañana y tarde" in message)
    if requirement is Requirement.START_DATE:
        assert "30" in message


def test_open_stages_have_no_closing_message(session):
    assert closing_message(Language.ES, session) is None


def test_prompts_name_the_session_language(job):
    assert "speak only Spanish" in system_prompt("voice", job, TODAY, Language.ES)
    assert "speak only English" in system_prompt("text", job, TODAY, Language.EN)
    assert greeting_trigger(job, Language.ES).endswith("Speak Spanish.")


def test_prompt_has_the_job_the_date_and_the_questions(job):
    prompt = system_prompt("text", job, TODAY)
    assert "Harbor Logistics" in prompt
    assert "Tuesday 2026-11-10" in prompt
    assert all(q.text in prompt for q in job.questions)
    assert "morning and afternoon" in prompt


def test_prompt_forbids_leaks_and_protected_topics(job):
    prompt = system_prompt("voice", job, TODAY)
    assert "Never say a score" in prompt
    assert "Never ask about age, nationality" in prompt


def test_spanish_prompt_uses_the_same_register_as_the_fixed_messages(job, session):
    assert '"usted", never "tú"' in system_prompt("voice", job, TODAY, Language.ES)
    session.stage = Stage.ENDED
    assert "Gracias por su tiempo" in closing_message(Language.ES, session)
    assert "usted" not in system_prompt("voice", job, TODAY, Language.EN)


async def test_executor_closes_in_the_session_language(session, knowledge, calendar):
    executor = ToolExecutor(session, knowledge, calendar, language=Language.ES)
    outcome = await executor.execute("escalate_to_human", {"reason": "pide una persona"})
    assert outcome.closing_message.startswith("Paso su candidatura")


def test_voice_prompt_can_switch_language_and_keeps_every_register_rule(job):
    prompt = system_prompt("voice", job, TODAY, Language.EN, switchable=True)
    assert "start in English" in prompt
    assert "speak only" not in prompt.split("A developer note")[0]
    assert '"usted", never "tú"' in prompt
