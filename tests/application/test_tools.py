from support_agent.application.tools import TOOL_SPECS, ToolExecutor


def test_tool_specs_have_valid_schemas():
    names = [spec.name for spec in TOOL_SPECS]
    assert len(names) == len(set(names))
    for spec in TOOL_SPECS:
        schema = spec.json_schema()
        assert schema["type"] == "object"
        assert set(spec.required) <= set(schema["properties"])


def test_executor_records_calls_and_returns_payload(session):
    calls = []
    executor = ToolExecutor(session, on_call=lambda *args: calls.append(args))
    outcome = executor.execute("lookup_order", {"order_number": "ORD-104233"})
    assert outcome.payload["ok"]
    assert outcome.closing_message is None
    assert calls[0][0] == "lookup_order"


def test_unknown_tool_and_missing_arguments_are_errors(session):
    executor = ToolExecutor(session)
    assert "Unknown tool" in executor.execute("delete_everything", {}).payload["error"]
    assert not executor.execute("lookup_order", {}).payload["ok"]


def test_finalize_returns_fixed_closing_message(session):
    executor = ToolExecutor(session)
    executor.execute("lookup_order", {"order_number": "ORD-104233"})
    executor.execute(
        "update_ticket",
        {"category": "shipping", "description": "Box arrived open and empty.", "urgency": "high"},
    )
    executor.execute("review_ticket", {})
    outcome = executor.execute("finalize_ticket", {})
    assert session.ticket.ticket_id in outcome.closing_message


def test_auto_escalation_ends_conversation(session):
    executor = ToolExecutor(session)
    outcomes = [executor.execute("lookup_order", {"order_number": "nope"}) for _ in range(3)]
    assert outcomes[1].closing_message is None
    assert "human agent" in outcomes[2].closing_message


def test_calls_after_close_are_rejected(session):
    executor = ToolExecutor(session)
    executor.execute("end_conversation", {"reason": "customer left"})
    outcome = executor.execute("update_ticket", {"urgency": "high"})
    assert outcome.payload["ok"] is False
    assert "already ended" in outcome.payload["error"]


def test_only_the_closing_call_returns_the_closing_message(session):
    # A second call after the close (for example in the same batch) must not repeat it.
    executor = ToolExecutor(session)
    first = executor.execute("escalate_to_human", {"reason": "customer asked"})
    second = executor.execute("end_conversation", {"reason": "duplicate"})
    assert "human agent" in first.closing_message
    assert second.closing_message is None
    assert second.payload["ok"] is False
