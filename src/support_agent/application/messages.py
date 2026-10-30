"""Fixed customer-facing messages, one set per language.

Code says these messages itself (see ADR-0001), so the LLM cannot change them.
They must be translated here, not by the LLM. Add a language by adding one entry
to each dictionary; ``tests/application/test_messages.py`` checks that none is missing.
"""

from __future__ import annotations

from support_agent.domain.models import Language, Stage

LANGUAGE_NAMES = {
    Language.EN: "English",
    Language.ES: "Spanish",
}

_CLOSING_MESSAGES = {
    Language.EN: {
        Stage.COMPLETED: (
            "Your ticket is submitted. The reference is {ticket_id}. "
            "Our team will contact you soon. Thank you for contacting Acme Store. Goodbye!"
        ),
        Stage.ESCALATED: (
            "I am transferring you to a human agent now. "
            "They will see everything we discussed. Please hold."
        ),
        Stage.ENDED: "Thank you for contacting Acme Store. Goodbye!",
    },
    Language.ES: {
        Stage.COMPLETED: (
            "Su ticket está registrado. La referencia es {ticket_id}. "
            "Nuestro equipo se comunicará con usted pronto. "
            "Gracias por contactar a Acme Store. ¡Hasta luego!"
        ),
        Stage.ESCALATED: (
            "Ahora le transfiero con un agente humano. "
            "Verá todo lo que hablamos. Por favor, espere en línea."
        ),
        Stage.ENDED: "Gracias por contactar a Acme Store. ¡Hasta luego!",
    },
}


_RETRY_MESSAGES = {
    Language.EN: "Sorry, I did not catch that. Could you say it again?",
    Language.ES: "Disculpe, no le entendí. ¿Podría repetirlo?",
}

_TECHNICAL_ERROR_MESSAGES = {
    Language.EN: "Sorry, I have a technical problem and must stop. Your conversation is saved.",
    Language.ES: (
        "Disculpe, tengo un problema técnico y debo terminar. Su conversación está guardada."
    ),
}


def retry_message(language: Language) -> str:
    return _RETRY_MESSAGES[language]


def technical_error_message(language: Language) -> str:
    return _TECHNICAL_ERROR_MESSAGES[language]


def closing_message(language: Language, stage: Stage, ticket_id: str | None) -> str | None:
    template = _CLOSING_MESSAGES[language].get(stage)
    return template.format(ticket_id=ticket_id or "") if template else None


def closing_messages(language: Language) -> dict[Stage, str]:
    """All closing templates for one language."""
    return dict(_CLOSING_MESSAGES[language])
