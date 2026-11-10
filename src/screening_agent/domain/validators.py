"""Validation and normalization of values that Gemini proposes from the candidate's words.

Values come from speech transcripts or chat, so each parser accepts the common
spoken or informal forms in every supported language and returns the canonical value.
Each parser raises ``ValueError`` with a message that tells the agent what to do.
"""

from __future__ import annotations

import re
import unicodedata
from datetime import date
from difflib import SequenceMatcher

from screening_agent.domain.models import Language, Requirement, Shift, Specificity

_YES = {"yes", "y", "true", "si", "yeah", "yep", "sure", "claro", "1"}
_NO = {"no", "n", "false", "nope", "0"}

_SHIFT_ALIASES = {
    "morning": Shift.MORNING,
    "mornings": Shift.MORNING,
    "early": Shift.MORNING,
    "manana": Shift.MORNING,
    "mananas": Shift.MORNING,
    "afternoon": Shift.AFTERNOON,
    "afternoons": Shift.AFTERNOON,
    "evening": Shift.AFTERNOON,
    "tarde": Shift.AFTERNOON,
    "tardes": Shift.AFTERNOON,
    "night": Shift.NIGHT,
    "nights": Shift.NIGHT,
    "noche": Shift.NIGHT,
    "noches": Shift.NIGHT,
}
_ALL_SHIFTS_WORDS = {"any", "all", "every", "anything", "cualquiera", "todos", "todas"}
_NO_SHIFT_WORDS = {"none", "ninguno", "ninguna"}
_NOW_WORDS = {"today", "now", "immediately", "asap", "hoy", "ya", "inmediatamente"}

_LANGUAGE_ALIASES = {
    "english": Language.EN,
    "ingles": Language.EN,
    "spanish": Language.ES,
    "espanol": Language.ES,
}

# Shorter is not a usable summary; longer is not a summary.
SUMMARY_MIN_LENGTH = 10
SUMMARY_MAX_LENGTH = 600
# A quote of one or two words proves nothing about the answer.
QUOTE_MIN_WORDS = 3
# Speech recognition and Gemini can change a few characters (accents, "gonna"), not the words.
QUOTE_MATCH_RATIO = 0.85
NAME_MIN_LENGTH = 2
NAME_MAX_LENGTH = 80


def fold(raw: str) -> str:
    """Lowercase and remove accents, so "Mañana" and "manana" match."""
    decomposed = unicodedata.normalize("NFKD", raw.lower())
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch))


def _words(raw: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", fold(raw))


def parse_requirement(raw: str) -> Requirement:
    key = re.sub(r"[\s\-]+", "_", fold(raw.strip()))
    try:
        return Requirement(key)
    except ValueError:
        allowed = ", ".join(r.value for r in Requirement)
        raise ValueError(f"'{raw}' is not a known requirement. Allowed: {allowed}.") from None


def parse_yes_no(raw: object) -> bool:
    if isinstance(raw, bool):
        return raw
    words = _words(str(raw))
    if words and words[0] in _YES:
        return True
    if words and words[0] in _NO:
        return False
    raise ValueError(f"'{raw}' is not a yes or no answer. Ask the candidate a yes or no question.")


def parse_shifts(raw: object) -> set[Shift]:
    """Accept "morning and afternoon", "mañanas y tardes", "any shift" or a list."""
    text = " ".join(str(item) for item in raw) if isinstance(raw, list) else str(raw)
    words = _words(text)
    if _ALL_SHIFTS_WORDS & set(words):
        return set(Shift)
    shifts = {_SHIFT_ALIASES[word] for word in words if word in _SHIFT_ALIASES}
    if not shifts and not (_NO_SHIFT_WORDS & set(words)):
        allowed = ", ".join(s.value for s in Shift)
        raise ValueError(f"'{raw}' names no shift. Use one or more of: {allowed}.")
    return shifts


def parse_start_date(raw: str, today: date) -> date:
    """Accept an ISO date (YYYY-MM-DD) or a word for "now". Gemini converts other dates."""
    text = str(raw).strip()
    if set(_words(text)) & _NOW_WORDS:
        return today
    try:
        start = date.fromisoformat(text)
    except ValueError:
        raise ValueError(
            f"'{raw}' is not a date. Send the earliest start date as YYYY-MM-DD. "
            f"Today is {today.isoformat()}."
        ) from None
    if start < today:
        raise ValueError(
            f"{start.isoformat()} is in the past. Today is {today.isoformat()}. "
            "Ask when the candidate can start."
        )
    return start


def parse_specificity(raw: str) -> Specificity:
    try:
        return Specificity(fold(raw.strip()))
    except ValueError:
        raise ValueError(f"'{raw}' is not a specificity. Use 'specific' or 'vague'.") from None


def parse_score(raw: object) -> int:
    try:
        score = int(str(raw).strip())
    except ValueError:
        raise ValueError(f"'{raw}' is not a score. Use a whole number from 1 to 4.") from None
    if not 1 <= score <= 4:
        raise ValueError(f"{score} is out of range. Use a whole number from 1 to 4.")
    return score


def validate_name(raw: str) -> str:
    name = " ".join(str(raw).split())
    if not NAME_MIN_LENGTH <= len(name) <= NAME_MAX_LENGTH:
        raise ValueError("That is not a usable name. Ask the candidate for their name.")
    return name


def validate_summary(raw: str) -> str:
    text = " ".join(str(raw).split())
    if len(text) < SUMMARY_MIN_LENGTH:
        raise ValueError("The summary is too short. Describe the answer in one or two sentences.")
    if len(text) > SUMMARY_MAX_LENGTH:
        text = text[: SUMMARY_MAX_LENGTH - 1].rstrip() + "…"
    return text


def quote_in_words(quote: str, candidate_texts: list[str]) -> bool:
    """True if the quote appears, almost word for word, in what the candidate said.

    This stops an invented quote: every score in the scorecard must point to real words.
    """
    quote_words = _words(quote)
    if len(quote_words) < QUOTE_MIN_WORDS:
        return False
    needle = " ".join(quote_words)
    for text in candidate_texts:
        words = _words(text)
        haystack = " ".join(words)
        if needle in haystack:
            return True
        # Compare with every window of the same length, to allow small changes.
        size = len(quote_words)
        for start in range(max(1, len(words) - size + 1)):
            window = " ".join(words[start : start + size])
            if SequenceMatcher(None, needle, window).ratio() >= QUOTE_MATCH_RATIO:
                return True
    return False


def parse_language(raw: str) -> Language:
    """Accept "es", "es-US", "Spanish" or "español", and return the ``Language``."""
    key = fold(raw.strip())
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
