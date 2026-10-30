"""Validation and normalization of customer-provided values.

Values come from speech transcripts or chat, so each parser accepts the
common spoken or informal forms and returns the canonical value.
Each parser raises ``ValueError`` with a message the agent can say to the customer.
"""

from __future__ import annotations

import re
import unicodedata

from support_agent.domain.models import Category, Language, Urgency

ORDER_NUMBER_RE = re.compile(r"^ORD(\d{6})$")
ORDER_NUMBER_HINT = "Order numbers look like ORD-123456: the letters ORD and six digits."

# Spoken forms in every supported language. They are merged, not chosen by the session
# language, because customers mix languages and STT can write either form.
_DIGIT_WORDS = {
    # English
    "zero": "0", "oh": "0", "one": "1", "two": "2", "three": "3", "four": "4",
    "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
    # Spanish (accents are removed before lookup)
    "cero": "0", "uno": "1", "una": "1", "dos": "2", "tres": "3", "cuatro": "4",
    "cinco": "5", "seis": "6", "siete": "7", "ocho": "8", "nueve": "9",
}  # fmt: skip
_LETTER_WORDS = {"erre": "r"}  # Spanish name of the letter R, as in "o erre de".
_ORDER_FILLER_WORDS = {
    # English
    "order", "number", "no", "num", "is", "my", "it", "its", "the", "dash", "hyphen",
    # Spanish
    "orden", "pedido", "numero", "es", "mi", "el", "la", "de", "del", "guion",
}  # fmt: skip
_REPEAT_WORDS = {"double": 2, "triple": 3, "doble": 2}

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
    # Spanish
    "envio": Category.SHIPPING,
    "entrega": Category.SHIPPING,
    "retraso": Category.SHIPPING,
    "pago": Category.BILLING,
    "cobro": Category.BILLING,
    "cargo": Category.BILLING,
    "factura": Category.BILLING,
    "defecto": Category.PRODUCT_DEFECT,
    "defectuoso": Category.PRODUCT_DEFECT,
    "roto": Category.PRODUCT_DEFECT,
    "danado": Category.PRODUCT_DEFECT,
    "averiado": Category.PRODUCT_DEFECT,
    "devolucion": Category.RETURN_REFUND,
    "reembolso": Category.RETURN_REFUND,
    "cuenta": Category.ACCOUNT,
    "acceso": Category.ACCOUNT,
    "otro": Category.OTHER,
}
_URGENCY_ALIASES = {
    "urgent": Urgency.HIGH,
    "critical": Urgency.HIGH,
    "asap": Urgency.HIGH,
    "normal": Urgency.MEDIUM,
    "moderate": Urgency.MEDIUM,
    "minor": Urgency.LOW,
    # Spanish
    "urgente": Urgency.HIGH,
    "alta": Urgency.HIGH,
    "critica": Urgency.HIGH,
    "media": Urgency.MEDIUM,
    "baja": Urgency.LOW,
}
_LANGUAGE_ALIASES = {
    "english": Language.EN,
    "ingles": Language.EN,
    "spanish": Language.ES,
    "espanol": Language.ES,
}


def _fold(raw: str) -> str:
    """Lowercase and remove accents, so "Número" and "numero" match."""
    decomposed = unicodedata.normalize("NFKD", raw.lower())
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch))


def normalize_order_number(raw: str) -> str:
    """Return the canonical ``ORD-123456`` form of a spoken or typed order number."""
    tokens = re.findall(r"[a-z0-9]+", _fold(raw).replace("'", ""))
    parts: list[str] = []
    repeat = 1
    for token in tokens:
        if token in _REPEAT_WORDS:
            repeat = _REPEAT_WORDS[token]
            continue
        if token in _ORDER_FILLER_WORDS:
            continue
        token = _LETTER_WORDS.get(token, token)
        parts.append(_DIGIT_WORDS.get(token, token) * repeat)
        repeat = 1

    compact = "".join(parts).upper()
    # Speech-to-text often writes the "ORD" prefix as "OR D" or "O R D", which the join
    # fixes. A bare six-digit number is accepted too, because customers often omit the prefix.
    if compact.isdigit():
        compact = f"ORD{compact}"
    # Spanish "o erre de" loses its "de" as a filler word, which leaves "OR" + digits.
    if re.fullmatch(r"OR\d{6}", compact):
        compact = f"ORD{compact[2:]}"
    match = ORDER_NUMBER_RE.match(compact)
    if not match:
        raise ValueError(f"'{raw}' is not a valid order number. {ORDER_NUMBER_HINT}")
    return f"ORD-{match.group(1)}"


def _normalize_key(raw: str) -> str:
    return re.sub(r"[\s\-/]+", "_", _fold(raw.strip()))


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


def parse_language(raw: str) -> Language:
    """Accept "es", "es-US", "Spanish" or "español", and return the ``Language``."""
    key = _fold(raw.strip())
    try:
        return Language(key.split("-")[0].split("_")[0])
    except ValueError:
        pass
    if key in _LANGUAGE_ALIASES:
        return _LANGUAGE_ALIASES[key]
    allowed = ", ".join(lang.value for lang in Language)
    raise ValueError(f"'{raw}' is not a supported language. Supported: {allowed}.")


# Short replies ("ok", "sí", "no") give unreliable language detection.
MIN_WORDS_TO_SWITCH_LANGUAGE = 3


def language_switch(current: Language, detected: str | None, transcript: str) -> Language | None:
    """Return the new language if speech recognition detected a change, else None."""
    if not detected or len(transcript.split()) < MIN_WORDS_TO_SWITCH_LANGUAGE:
        return None
    try:
        language = parse_language(detected)
    except ValueError:
        return None
    return language if language is not current else None
