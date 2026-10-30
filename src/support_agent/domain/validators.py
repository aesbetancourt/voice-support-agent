"""Validation and normalization of customer-provided values.

Values come from speech transcripts or chat, so each parser accepts the
common spoken or informal forms and returns the canonical value.
Each parser raises ``ValueError`` with a message the agent can say to the customer.
"""

from __future__ import annotations

import re

from support_agent.domain.models import Category, Urgency

ORDER_NUMBER_RE = re.compile(r"^ORD(\d{6})$")
ORDER_NUMBER_HINT = "Order numbers look like ORD-123456: the letters ORD and six digits."

_DIGIT_WORDS = {
    "zero": "0",
    "oh": "0",
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
}
_ORDER_FILLER_WORDS = {
    "order",
    "number",
    "no",
    "num",
    "is",
    "my",
    "it",
    "its",
    "the",
    "dash",
    "hyphen",
}
_REPEAT_WORDS = {"double": 2, "triple": 3}

DESCRIPTION_MIN_LENGTH = 10
DESCRIPTION_MAX_LENGTH = 1000

_CATEGORY_ALIASES = {
    "delivery": Category.SHIPPING,
    "shipment": Category.SHIPPING,
    "payment": Category.BILLING,
    "charge": Category.BILLING,
    "invoice": Category.BILLING,
    "defect": Category.PRODUCT_DEFECT,
    "defective": Category.PRODUCT_DEFECT,
    "damaged": Category.PRODUCT_DEFECT,
    "broken": Category.PRODUCT_DEFECT,
    "return": Category.RETURN_REFUND,
    "refund": Category.RETURN_REFUND,
    "login": Category.ACCOUNT,
}
_URGENCY_ALIASES = {
    "urgent": Urgency.HIGH,
    "critical": Urgency.HIGH,
    "asap": Urgency.HIGH,
    "normal": Urgency.MEDIUM,
    "moderate": Urgency.MEDIUM,
    "minor": Urgency.LOW,
}


def normalize_order_number(raw: str) -> str:
    """Return the canonical ``ORD-123456`` form of a spoken or typed order number."""
    tokens = re.findall(r"[a-z0-9]+", raw.lower().replace("'", ""))
    parts: list[str] = []
    repeat = 1
    for token in tokens:
        if token in _REPEAT_WORDS:
            repeat = _REPEAT_WORDS[token]
            continue
        if token in _ORDER_FILLER_WORDS:
            continue
        parts.append(_DIGIT_WORDS.get(token, token) * repeat)
        repeat = 1

    compact = "".join(parts).upper()
    # Speech-to-text often writes the "ORD" prefix as "OR D" or "O R D", which the join
    # fixes. A bare six-digit number is accepted too, because customers often omit the prefix.
    if compact.isdigit():
        compact = f"ORD{compact}"
    match = ORDER_NUMBER_RE.match(compact)
    if not match:
        raise ValueError(f"'{raw}' is not a valid order number. {ORDER_NUMBER_HINT}")
    return f"ORD-{match.group(1)}"


def _normalize_key(raw: str) -> str:
    return re.sub(r"[\s\-/]+", "_", raw.strip().lower())


def parse_category(raw: str) -> Category:
    key = _normalize_key(raw)
    try:
        return Category(key)
    except ValueError:
        pass
    if key in _CATEGORY_ALIASES:
        return _CATEGORY_ALIASES[key]
    allowed = ", ".join(c.value for c in Category)
    raise ValueError(f"'{raw}' is not a known category. Allowed: {allowed}.")


def parse_urgency(raw: str) -> Urgency:
    key = _normalize_key(raw)
    try:
        return Urgency(key)
    except ValueError:
        pass
    if key in _URGENCY_ALIASES:
        return _URGENCY_ALIASES[key]
    allowed = ", ".join(u.value for u in Urgency)
    raise ValueError(f"'{raw}' is not a known urgency level. Allowed: {allowed}.")


def validate_description(raw: str) -> str:
    text = " ".join(raw.split())
    if len(text) < DESCRIPTION_MIN_LENGTH:
        raise ValueError(
            "The description is too short. Ask the customer what happened, in a full sentence."
        )
    if len(text) > DESCRIPTION_MAX_LENGTH:
        text = text[: DESCRIPTION_MAX_LENGTH - 1].rstrip() + "…"
    return text
