"""Inbound adapter: Gemini plays a customer persona and drives the text agent.

Use it to test the agent on difficult behavior without manual calls.
Each run is saved like a real conversation.
"""

from __future__ import annotations

from dataclasses import dataclass

from google import genai
from google.genai import types

from support_agent.application.messages import LANGUAGE_NAMES
from support_agent.bootstrap import (
    make_sentiment_analyzer,
    make_summarizer,
    make_text_agent,
    new_conversation,
)
from support_agent.config import Settings
from support_agent.domain.models import ConversationRecord, ConversationStatus, Language

_CUSTOMER_RULES = """\
You play a customer who contacts Orbio Store support by chat.
Stay in character. Write only the customer's next message, one or two sentences.
Do not invent a new order number unless your persona says so.
When the agent says goodbye or transfers you, write exactly: [END]
"""


@dataclass(frozen=True)
class Persona:
    name: str
    brief: str
    expected: ConversationStatus


PERSONAS: list[Persona] = [
    Persona(
        "cooperative",
        "Your order ORD-104233 (wireless headphones) arrived with a cracked left ear cup. "
        "You are calm and answer every question directly.",
        ConversationStatus.COMPLETED,
    ),
    Persona(
        "typo_order_number",
        "Your package for order ORD-208871 has not moved for six days. You first give the "
        "wrong number ORD-208817. When told it is not found, give the right one.",
        ConversationStatus.COMPLETED,
    ),
    Persona(
        "corrects_details",
        "You were charged twice for order ORD-310552. You first say urgency is low, but "
        "during the read-back you change it to high because rent is due.",
        ConversationStatus.COMPLETED,
    ),
    Persona(
        "rambler",
        "You talk about your weekend and your dog before you get to the point: "
        "the smart watch from order ORD-412907 does not turn on. Give the order number "
        "only when asked twice.",
        ConversationStatus.COMPLETED,
    ),
    Persona(
        "wants_human",
        "You are angry about a late refund for order ORD-523316. After the agent's first "
        "question, you demand to talk to a real person and refuse to answer anything else.",
        ConversationStatus.ESCALATED,
    ),
    Persona(
        "growing_frustration",
        "Your monitor from order ORD-208871 is six days late and this is your third contact. "
        "You answer questions, but each answer is more impatient and angry than the last. "
        "You never ask for a human yourself. If the agent offers a human agent, accept it.",
        ConversationStatus.ESCALATED,
    ),
    Persona(
        "never_valid_order",
        "You want help with a broken speaker but you do not have a valid order number. "
        "Each time you are asked, give a different made-up number like 12345 or ORD-99.",
        ConversationStatus.ESCALATED,
    ),
]


@dataclass(frozen=True)
class SimulationResult:
    persona: Persona
    record: ConversationRecord
    turns: int

    @property
    def passed(self) -> bool:
        return self.record.status is self.persona.expected


class SimulatedCustomer:
    def __init__(
        self, client: genai.Client, model: str, persona: Persona, language: Language = Language.EN
    ):
        self._client = client
        self._model = model
        language_rule = f"Write only in {LANGUAGE_NAMES[language]}."
        self._config = types.GenerateContentConfig(
            system_instruction=f"{_CUSTOMER_RULES}\n{language_rule}\nPersona: {persona.brief}",
            temperature=0.8,
        )
        self._history: list[types.Content] = []

    async def reply(self, agent_text: str) -> str:
        # Roles are mirrored: the agent is the "user" from the customer's point of view.
        self._history.append(
            types.Content(role="user", parts=[types.Part.from_text(text=agent_text)])
        )
        response = await self._client.aio.models.generate_content(
            model=self._model, contents=self._history, config=self._config
        )
        text = (response.text or "").strip()
        self._history.append(types.Content(role="model", parts=[types.Part.from_text(text=text)]))
        return text


async def run_persona(
    settings: Settings, client: genai.Client, persona: Persona, max_turns: int
) -> SimulationResult:
    conversation = new_conversation(
        settings,
        "text",
        make_summarizer(settings, client),
        make_sentiment_analyzer(settings, client),
    )
    agent = make_text_agent(settings, client, conversation)
    customer = SimulatedCustomer(client, settings.model, persona, settings.default_language)

    turns = 0
    reply = await agent.greet()
    try:
        while not reply.ended and turns < max_turns:
            customer_text = await customer.reply(reply.text)
            if not customer_text or "[END]" in customer_text:
                break
            reply = await agent.respond(customer_text)
            turns += 1
    finally:
        record = await conversation.recorder.finish()
    return SimulationResult(persona=persona, record=record, turns=turns)
