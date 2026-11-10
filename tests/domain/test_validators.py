from datetime import date

import pytest

from screening_agent.domain.models import Language, Requirement, Shift, Specificity
from screening_agent.domain.validators import (
    language_switch,
    parse_language,
    parse_requirement,
    parse_score,
    parse_shifts,
    parse_specificity,
    parse_start_date,
    parse_yes_no,
    quote_in_words,
    validate_name,
    validate_summary,
)

TODAY = date(2026, 11, 10)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [(True, True), (False, False), ("yes", True), ("Sí", True), ("si, claro", True),
     ("no", False), ("No, I don't", False), ("true", True)],
)  # fmt: skip
def test_yes_no(raw, expected):
    assert parse_yes_no(raw) is expected


def test_yes_no_rejects_other_answers():
    with pytest.raises(ValueError, match="yes or no"):
        parse_yes_no("maybe later")


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("morning and afternoon", {Shift.MORNING, Shift.AFTERNOON}),
        ("Mañanas y tardes", {Shift.MORNING, Shift.AFTERNOON}),
        (["morning", "night"], {Shift.MORNING, Shift.NIGHT}),
        ("any shift", set(Shift)),
        ("cualquiera", set(Shift)),
        ("none", set()),
    ],
)
def test_shifts(raw, expected):
    assert parse_shifts(raw) == expected


def test_shifts_rejects_text_without_a_shift():
    with pytest.raises(ValueError, match="names no shift"):
        parse_shifts("whenever")


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("2026-11-20", date(2026, 11, 20)), ("immediately", TODAY), ("ya", TODAY), ("hoy", TODAY)],
)
def test_start_date(raw, expected):
    assert parse_start_date(raw, TODAY) == expected


def test_start_date_rejects_past_and_free_text():
    with pytest.raises(ValueError, match="in the past"):
        parse_start_date("2026-11-01", TODAY)
    with pytest.raises(ValueError, match="YYYY-MM-DD"):
        parse_start_date("next month", TODAY)


def test_requirement_and_specificity():
    assert parse_requirement("Work permit") is Requirement.WORK_PERMIT
    assert parse_specificity("VAGUE") is Specificity.VAGUE
    with pytest.raises(ValueError, match="Allowed"):
        parse_requirement("salary")


@pytest.mark.parametrize("raw", ["0", "5", "three", ""])
def test_score_rejects_out_of_range(raw):
    with pytest.raises(ValueError):
        parse_score(raw)


def test_score_accepts_text_numbers():
    assert parse_score("3") == 3


def test_name_and_summary():
    assert validate_name("  Lucía   Gómez ") == "Lucía Gómez"
    with pytest.raises(ValueError):
        validate_name("x")
    with pytest.raises(ValueError, match="too short"):
        validate_summary("ok")


def test_quote_must_come_from_the_candidate():
    said = ["Bueno, en el súper organicé al equipo por parejas y descargamos todo a tiempo."]
    assert quote_in_words("organicé al equipo por parejas", said)
    assert quote_in_words("organice al equipo por parejas", said)  # STT dropped the accent
    assert not quote_in_words("lideré un equipo de veinte personas", said)
    assert not quote_in_words("equipo", said)  # too short to prove anything


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("en", Language.EN),
        ("es", Language.ES),
        ("es-US", Language.ES),
        ("en_US", Language.EN),
        ("Spanish", Language.ES),
        ("Español", Language.ES),
    ],
)
def test_parse_language(raw, expected):
    assert parse_language(raw) is expected


def test_parse_language_rejects_unsupported():
    with pytest.raises(ValueError, match="Supported: en, es"):
        parse_language("fr")


@pytest.mark.parametrize(
    ("current", "detected", "transcript", "expected"),
    [
        (Language.EN, "es-US", "Hola, tengo permiso de trabajo", Language.ES),
        (Language.ES, "en-US", "I have a work permit", Language.EN),
        (Language.EN, "en-US", "I have a work permit", None),
        (Language.EN, "es-US", "Sí, gracias", None),
        (Language.EN, "fr-FR", "Bonjour, j'ai un permis de travail", None),
        (Language.EN, "", "Hola, tengo permiso de trabajo", None),
        (Language.EN, None, "Hola, tengo permiso de trabajo", None),
    ],
)
def test_language_switch(current, detected, transcript, expected):
    assert language_switch(current, detected, transcript) is expected
