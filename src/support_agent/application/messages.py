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
            "Our team will contact you soon. Thank you for contacting Orbio Store. Goodbye!"
        ),
        Stage.ESCALATED: (
            "I am transferring you to a human agent now. "
            "They will see everything we discussed. Please hold."
        ),
        Stage.ENDED: "Thank you for contacting Orbio Store. Goodbye!",
    },
    Language.ES: {
        Stage.COMPLETED: (
            "Su ticket está registrado. La referencia es {ticket_id}. "
            "Nuestro equipo se comunicará con usted pronto. "
            "Gracias por contactar a Orbio Store. ¡Hasta luego!"
        ),
        Stage.ESCALATED: (
            "Ahora le transfiero con un agente humano. "
            "Verá todo lo que hablamos. Por favor, espere en línea."
        ),
        Stage.ENDED: "Gracias por contactar a Orbio Store. ¡Hasta luego!",
    },
}


def closing_message(language: Language, stage: Stage, ticket_id: str | None) -> str | None:
    template = _CLOSING_MESSAGES[language].get(stage)
    return template.format(ticket_id=ticket_id or "") if template else None


def closing_messages(language: Language) -> dict[Stage, str]:
    """All closing templates for one language (used by tests)."""
    return dict(_CLOSING_MESSAGES[language])
