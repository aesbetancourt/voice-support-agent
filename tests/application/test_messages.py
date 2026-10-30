"""Customer-facing messages and prompts in every supported language."""

import pytest

from support_agent.application.messages import LANGUAGE_NAMES, closing_message, closing_messages
from support_agent.application.prompts import greeting_trigger, system_prompt
from support_agent.application.tools import ToolExecutor
from support_agent.domain.models import Language, Stage

CLOSING_STAGES = {Stage.COMPLETED, Stage.ESCALATED, Stage.ENDED}


@pytest.mark.parametrize("language", list(Language))
def test_every_language_has_every_closing_message(language):
    assert set(closing_messages(language)) == CLOSING_STAGES
    assert language in LANGUAGE_NAMES


@pytest.mark.parametrize("language", list(Language))
def test_completed_message_contains_the_ticket_id(language):
    assert "TCK-123456" in closing_message(language, Stage.COMPLETED, "TCK-123456")


def test_open_stages_have_no_closing_message():
    assert closing_message(Language.ES, Stage.COLLECTING, None) is None


def test_prompts_name_the_session_language():
    assert "speak only Spanish" in system_prompt("voice", Language.ES)
    assert "speak only English" in system_prompt("text", Language.EN)
    assert greeting_trigger(Language.ES).endswith("Speak Spanish.")


def test_spanish_prompt_uses_the_same_register_as_the_fixed_messages():
    assert '"usted", never "tú"' in system_prompt("voice", Language.ES)
    assert "usted" in closing_message(Language.ES, Stage.COMPLETED, "TCK-1")
    assert "usted" not in system_prompt("voice", Language.EN)


def test_executor_closes_in_the_session_language(session):
    executor = ToolExecutor(session, language=Language.ES)
    outcome = executor.execute("escalate_to_human", {"reason": "pide una persona"})
    assert outcome.closing_message.startswith("Ahora le transfiero con un agente humano")


def test_voice_prompt_can_switch_language_and_keeps_every_register_rule():
    prompt = system_prompt("voice", Language.EN, switchable=True)
    assert "start in English" in prompt
    assert "speak only" not in prompt.split("A developer note")[0]
    assert '"usted", never "tú"' in prompt
