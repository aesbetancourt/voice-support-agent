import pytest

from support_agent.domain.models import Category, Urgency
from support_agent.domain.validators import (
    normalize_order_number,
    parse_category,
    parse_urgency,
    validate_description,
)


@pytest.mark.parametrize(
    "raw",
    [
        "ORD-104233",
        "ord104233",
        "ORD 104 233",
        "O R D 1 0 4 2 3 3",
        "or d 104233",
        "104233",
        "my order number is ORD-104233",
        "o r d one zero four two three three",
        "ORD one oh four two double three",
    ],
)
def test_order_number_spoken_and_typed_forms(raw):
    assert normalize_order_number(raw) == "ORD-104233"


@pytest.mark.parametrize("raw", ["", "12345", "ORD-99", "ORD-1042331", "ABC-104233", "hello"])
def test_order_number_rejects_invalid_values(raw):
    with pytest.raises(ValueError, match="ORD-123456"):
        normalize_order_number(raw)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("shipping", Category.SHIPPING),
        ("Product Defect", Category.PRODUCT_DEFECT),
        ("return-refund", Category.RETURN_REFUND),
        ("refund", Category.RETURN_REFUND),
        ("damaged", Category.PRODUCT_DEFECT),
    ],
)
def test_category_accepts_values_and_aliases(raw, expected):
    assert parse_category(raw) is expected


def test_category_rejects_unknown_value():
    with pytest.raises(ValueError, match="Allowed: shipping"):
        parse_category("weather")


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("HIGH", Urgency.HIGH), ("urgent", Urgency.HIGH), ("normal", Urgency.MEDIUM)],
)
def test_urgency_accepts_values_and_aliases(raw, expected):
    assert parse_urgency(raw) is expected


def test_urgency_rejects_unknown_value():
    with pytest.raises(ValueError):
        parse_urgency("whenever")


def test_description_is_cleaned_and_length_checked():
    assert validate_description("  The   screen is   cracked  ") == "The screen is cracked"
    with pytest.raises(ValueError, match="too short"):
        validate_description("broken")
    assert len(validate_description("x" * 5000)) == 1000
