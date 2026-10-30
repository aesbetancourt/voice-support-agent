import pytest

from support_agent.domain.models import Stage
from support_agent.domain.state import SessionClosedError, TicketSession


def fill_ticket(session: TicketSession) -> None:
    session.lookup_order("ORD-104233")
    session.update_ticket(
        {
            "category": "product_defect",
            "description": "The left ear cup arrived cracked.",
            "urgency": "medium",
        }
    )


def test_happy_path_reaches_completed(session):
    result = session.lookup_order("o r d one zero four two three three")
    assert result["ok"]
    assert result["order"]["status"] == "delivered"
    assert "email" not in result["order"]
    assert session.ticket.customer_name == "Maria Lopez"

    result = session.update_ticket(
        {"category": "product_defect", "description": "Left ear cup is cracked.", "urgency": "low"}
    )
    assert result["ok"]
    assert result["missing_fields"] == []

    assert session.review_ticket()["ok"]
    assert session.stage is Stage.CONFIRMING

    result = session.finalize_ticket()
    assert result["ok"]
    assert result["ticket_id"].startswith("TCK-")
    assert session.stage is Stage.COMPLETED


def test_unknown_order_is_rejected_and_logged(session):
    result = session.lookup_order("ORD-999999")
    assert not result["ok"]
    assert "No order ORD-999999" in result["rejected"]["order_number"]
    assert session.ticket.order_number is None
    assert session.validation_events[0].field == "order_number"


def test_partial_update_keeps_valid_fields(session):
    result = session.update_ticket({"category": "weather", "urgency": "high"})
    assert not result["ok"]
    assert result["accepted"] == {"urgency": "high"}
    assert "category" in result["rejected"]
    assert set(result["missing_fields"]) == {"order_number", "category", "description"}


def test_order_number_through_update_ticket_is_redirected(session):
    result = session.update_ticket({"order_number": "ORD-104233"})
    assert result["rejected"]["order_number"] == "Use lookup_order for the order number."


def test_review_requires_all_fields(session):
    session.lookup_order("ORD-104233")
    result = session.review_ticket()
    assert not result["ok"]
    assert session.stage is Stage.COLLECTING


def test_finalize_requires_review_first(session):
    fill_ticket(session)
    result = session.finalize_ticket()
    assert not result["ok"]
    assert session.stage is Stage.COLLECTING


def test_correction_after_review_requires_new_review(session):
    fill_ticket(session)
    session.review_ticket()
    session.update_ticket({"urgency": "high"})
    assert session.stage is Stage.COLLECTING
    assert not session.finalize_ticket()["ok"]
    session.review_ticket()
    assert session.finalize_ticket()["ok"]


def test_repeated_failures_escalate(session):
    for attempt in ("1", "2", "3"):
        session.lookup_order(f"ORD-99{attempt}")
    assert session.stage is Stage.ESCALATED
    assert "order_number" in session.escalation_reason


def test_category_warning_for_undelivered_order(session):
    session.lookup_order("ORD-208871")  # shipped, not delivered
    result = session.update_ticket({"category": "product_defect"})
    assert "not delivered" in result["warning"]


def test_closed_session_rejects_tool_calls(session):
    session.escalate("customer asked for a human")
    with pytest.raises(SessionClosedError):
        session.update_ticket({"urgency": "high"})
