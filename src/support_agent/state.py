"""Ticket state machine.

The LLM proposes values through tool calls. This module decides if they are
valid and when the ticket is complete. The LLM cannot skip these rules.

    COLLECTING --review (all fields valid)--> CONFIRMING --finalize--> COMPLETED
    CONFIRMING --update (customer correction)--> COLLECTING
    COLLECTING/CONFIRMING --escalate or too many failures--> ESCALATED
    COLLECTING/CONFIRMING --end (customer leaves)--> ENDED
"""

from __future__ import annotations

import secrets
from collections import Counter
from typing import Any

from support_agent.models import (
    Category,
    Order,
    OrderStatus,
    Stage,
    Ticket,
    ValidationEvent,
)
from support_agent.orders import OrderRepository
from support_agent.validators import (
    normalize_order_number,
    parse_category,
    parse_urgency,
    validate_description,
)

_CLOSED_STAGES = {Stage.COMPLETED, Stage.ESCALATED, Stage.ENDED}
_NEEDS_DELIVERY = {Category.PRODUCT_DEFECT, Category.RETURN_REFUND}
_FIELD_PARSERS = {
    "category": parse_category,
    "description": validate_description,
    "urgency": parse_urgency,
}


class SessionClosedError(RuntimeError):
    """Raised when a tool is called after the conversation has closed."""


class TicketSession:
    def __init__(self, orders: OrderRepository, max_attempts: int = 3):
        self._orders = orders
        self.max_attempts = max_attempts
        self.stage = Stage.COLLECTING
        self.ticket = Ticket()
        self.order: Order | None = None
        self.escalation_reason: str | None = None
        self.validation_events: list[ValidationEvent] = []
        self._failures: Counter[str] = Counter()

    @property
    def is_closed(self) -> bool:
        return self.stage in _CLOSED_STAGES

    def lookup_order(self, order_number: str) -> dict[str, Any]:
        self._ensure_open()
        try:
            canonical = normalize_order_number(order_number)
        except ValueError as exc:
            return self._reject("order_number", order_number, str(exc))

        order = self._orders.get(canonical)
        if order is None:
            return self._reject(
                "order_number",
                canonical,
                f"No order {canonical} exists. Ask the customer to check the number.",
            )

        self._reopen_if_confirming()
        self.order = order
        self.ticket.order_number = order.order_number
        self.ticket.customer_name = order.customer_name
        return {
            "ok": True,
            "order": order.model_dump(mode="json", exclude={"email"}),
            **self._progress(),
        }

    def update_ticket(self, fields: dict[str, Any]) -> dict[str, Any]:
        self._ensure_open()
        accepted: dict[str, str] = {}
        rejected: dict[str, str] = {}
        for name, raw in fields.items():
            if raw is None or raw == "":
                continue
            parser = _FIELD_PARSERS.get(name)
            if parser is None:
                rejected[name] = (
                    "Use lookup_order for the order number."
                    if name == "order_number"
                    else f"Unknown field '{name}'."
                )
                continue
            try:
                value = parser(str(raw))
            except ValueError as exc:
                rejected[name] = str(exc)
                self._record_failure(name, raw, str(exc))
                continue
            setattr(self.ticket, name, value)
            accepted[name] = str(value)

        if accepted:
            self._reopen_if_confirming()
        result: dict[str, Any] = {"ok": not rejected, "accepted": accepted}
        if rejected:
            result["rejected"] = rejected
        if warning := self._category_warning():
            result["warning"] = warning
        result.update(self._progress())
        return result

    def review_ticket(self) -> dict[str, Any]:
        self._ensure_open()
        missing = self.ticket.missing_fields()
        if missing:
            return {
                "ok": False,
                "error": "The ticket is not complete. Collect the missing fields first.",
                **self._progress(),
            }
        self.stage = Stage.CONFIRMING
        return {
            "ok": True,
            "instruction": "Read this ticket back to the customer and ask them to confirm it.",
            "ticket": self.ticket.model_dump(mode="json", exclude={"ticket_id"}),
            **self._progress(),
        }

    def finalize_ticket(self) -> dict[str, Any]:
        self._ensure_open()
        if self.stage is not Stage.CONFIRMING:
            return {
                "ok": False,
                "error": "Call review_ticket and get the customer's confirmation first.",
                **self._progress(),
            }
        self.ticket.ticket_id = f"TCK-{secrets.randbelow(10**6):06d}"
        self.stage = Stage.COMPLETED
        return {"ok": True, "ticket_id": self.ticket.ticket_id, **self._progress()}

    def escalate(self, reason: str) -> dict[str, Any]:
        self._ensure_open()
        self.stage = Stage.ESCALATED
        self.escalation_reason = reason.strip() or "unspecified"
        return {"ok": True, "escalated": True, "reason": self.escalation_reason}

    def end(self, reason: str) -> dict[str, Any]:
        self._ensure_open()
        self.stage = Stage.ENDED
        return {"ok": True, "ended": True, "reason": reason}

    def _ensure_open(self) -> None:
        if self.is_closed:
            raise SessionClosedError(f"The conversation is already {self.stage.value}.")

    def _reopen_if_confirming(self) -> None:
        # A change after the read-back invalidates the customer's confirmation.
        if self.stage is Stage.CONFIRMING:
            self.stage = Stage.COLLECTING

    def _reject(self, field: str, value: Any, error: str) -> dict[str, Any]:
        self._record_failure(field, value, error)
        return {"ok": False, "rejected": {field: error}, **self._progress()}

    def _record_failure(self, field: str, value: Any, error: str) -> None:
        self.validation_events.append(ValidationEvent(field=field, value=value, error=error))
        self._failures[field] += 1
        if self._failures[field] >= self.max_attempts and not self.is_closed:
            self.stage = Stage.ESCALATED
            self.escalation_reason = (
                f"Could not get a valid {field} after {self.max_attempts} attempts."
            )

    def _category_warning(self) -> str | None:
        if (
            self.order is not None
            and self.ticket.category in _NEEDS_DELIVERY
            and self.order.status is not OrderStatus.DELIVERED
        ):
            return (
                f"Order {self.order.order_number} is '{self.order.status.value}', not delivered. "
                "Confirm the category with the customer."
            )
        return None

    def _progress(self) -> dict[str, Any]:
        progress: dict[str, Any] = {
            "stage": self.stage.value,
            "missing_fields": self.ticket.missing_fields(),
        }
        if self.stage is Stage.ESCALATED:
            progress["escalation_reason"] = self.escalation_reason
        return progress
