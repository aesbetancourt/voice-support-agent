"""Tools the LLM can call, shared by the text and voice channels.

Each channel converts ``TOOL_SPECS`` to its own SDK format and sends every call
to ``ToolExecutor``. When a call closes the conversation, the executor returns a
fixed closing message. The channel speaks or prints it, so the ticket ID and
the closing words never depend on the LLM.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from support_agent.application.messages import closing_message
from support_agent.domain.models import Category, Language, Urgency
from support_agent.domain.state import SessionClosedError, TicketSession


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
        name="lookup_order",
        description=(
            "Find the customer's order and attach it to the ticket. Call it as soon as the "
            "customer gives an order number, also when they correct it. Pass the number "
            "exactly as the customer said it; the system normalizes it."
        ),
        properties={
            "order_number": {
                "type": "string",
                "description": "Order number as said by the customer, e.g. 'ORD-104233'.",
            }
        },
        required=["order_number"],
    ),
    ToolSpec(
        name="update_ticket",
        description=(
            "Save one or more ticket fields. Send only the fields you learned or the "
            "customer corrected. The result lists rejected values and missing fields."
        ),
        properties={
            "category": {
                "type": "string",
                "enum": [c.value for c in Category],
                "description": "Problem category.",
            },
            "description": {
                "type": "string",
                "description": "The problem in the customer's words, one to three sentences.",
            },
            "urgency": {
                "type": "string",
                "enum": [u.value for u in Urgency],
                "description": (
                    "high: customer is blocked, money is at risk, or a safety issue. "
                    "medium: a real problem with a workaround. low: a question or minor issue."
                ),
            },
        },
    ),
    ToolSpec(
        name="review_ticket",
        description=(
            "Call when no fields are missing. Returns the ticket to read back to the customer "
            "for confirmation."
        ),
    ),
    ToolSpec(
        name="finalize_ticket",
        description=(
            "Submit the ticket. Call only after review_ticket and after the customer clearly "
            "confirms that the read-back is correct."
        ),
    ),
    ToolSpec(
        name="escalate_to_human",
        description=(
            "Transfer to a human agent. Use when the customer asks for a person, is very "
            "upset, reports a safety issue, or the request is outside support scope."
        ),
        properties={"reason": {"type": "string", "description": "Short reason for the transfer."}},
        required=["reason"],
    ),
    ToolSpec(
        name="end_conversation",
        description=(
            "End the conversation without a ticket, when the customer wants to stop or "
            "does not need help."
        ),
        properties={"reason": {"type": "string", "description": "Short reason."}},
        required=["reason"],
    ),
]


@dataclass(frozen=True)
class ToolOutcome:
    payload: dict[str, Any]
    closing_message: str | None = None

    @property
    def ends_conversation(self) -> bool:
        return self.closing_message is not None


ToolCallListener = Callable[[str, dict[str, Any], dict[str, Any]], None]


class ToolExecutor:
    def __init__(
        self,
        session: TicketSession,
        on_call: ToolCallListener | None = None,
        language: Language = Language.EN,
    ):
        self.session = session
        self.language = language
        self._on_call = on_call

    def execute(self, name: str, arguments: dict[str, Any] | None) -> ToolOutcome:
        args = dict(arguments or {})
        try:
            payload = self._dispatch(name, args)
        except SessionClosedError as exc:
            payload = {"ok": False, "error": str(exc)}
        except KeyError as exc:
            payload = {"ok": False, "error": f"Missing argument {exc} for {name}."}
        except (TypeError, ValueError) as exc:
            payload = {"ok": False, "error": f"Invalid arguments for {name}: {exc}"}

        if self._on_call is not None:
            self._on_call(name, args, payload)
        return ToolOutcome(payload=payload, closing_message=self._closing_message())

    def _dispatch(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        session = self.session
        match name:
            case "lookup_order":
                return session.lookup_order(str(args["order_number"]))
            case "update_ticket":
                return session.update_ticket(args)
            case "review_ticket":
                return session.review_ticket()
            case "finalize_ticket":
                return session.finalize_ticket()
            case "escalate_to_human":
                return session.escalate(str(args.get("reason", "")))
            case "end_conversation":
                return session.end(str(args.get("reason", "")))
            case _:
                return {"ok": False, "error": f"Unknown tool '{name}'."}

    def _closing_message(self) -> str | None:
        session = self.session
        return closing_message(self.language, session.stage, session.ticket.ticket_id)
