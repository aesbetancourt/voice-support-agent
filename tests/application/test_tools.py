from conftest import FakeKnowledge

from screening_agent.application.tools import TOOL_SPECS, ToolExecutor
from screening_agent.domain.models import Stage

TEAMWORK = "At the supermarket two trucks came late, so I split the team in pairs and we finished."
SAFETY = "I saw a leaning pallet near the exit, stopped work and told my supervisor right away."


async def pass_screening(executor, words):
    await executor.execute("start_screening", {"candidate_name": "Lucia", "consent": True})
    for requirement, answer in [
        ("work_permit", "yes"),
        ("shifts", "morning, afternoon"),
        ("start_date", "immediately"),
        ("commute", "yes"),
    ]:
        await executor.execute("record_must_have", {"requirement": requirement, "answer": answer})
    for question_id, text in [("teamwork", TEAMWORK), ("safety", SAFETY)]:
        words.lines.append(text)
        await executor.execute(
            "record_answer",
            {
                "question_id": question_id,
                "summary": f"A real {question_id} example.",
                "evidence_quote": " ".join(text.split()[3:9]),
                "specificity": "specific",
                "score": 3,
            },
        )


def test_tool_specs_have_valid_schemas():
    names = [spec.name for spec in TOOL_SPECS]
    assert len(names) == len(set(names))
    for spec in TOOL_SPECS:
        schema = spec.json_schema()
        assert schema["type"] == "object"
        assert set(spec.required) <= set(schema["properties"])


async def test_executor_records_calls_and_returns_payload(session, knowledge, calendar):
    calls = []
    executor = ToolExecutor(session, knowledge, calendar, on_call=lambda *args: calls.append(args))
    outcome = await executor.execute("start_screening", {"candidate_name": "Lucia", "consent": 1})
    assert outcome.payload["ok"]
    assert outcome.closing_message is None
    assert calls[0][0] == "start_screening"


async def test_unknown_tool_and_missing_arguments_are_errors(executor):
    assert "Unknown tool" in (await executor.execute("hire_me", {})).payload["error"]
    assert "Missing argument" in (await executor.execute("start_screening", {})).payload["error"]


async def test_full_screening_books_and_says_the_fixed_time(executor, session, words, calendar):
    await pass_screening(executor, words)
    assert session.stage is Stage.SCHEDULING
    slots = (await executor.execute("get_interview_slots", {})).payload["slots"]
    outcome = await executor.execute("book_interview", {"slot_id": slots[0]["slot_id"]})
    assert session.stage is Stage.BOOKED
    assert "Wednesday 11 November at 10:00" in outcome.closing_message
    assert session.scorecard.booking_id in outcome.closing_message
    # The booked time is not offered again.
    assert calendar.free_slots(1)[0].id != slots[0]["slot_id"]


async def test_job_question_returns_information(executor):
    outcome = await executor.execute("search_job_info", {"question": "How much is the pay?"})
    assert outcome.payload["found"] is True
    assert "1,580 euros" in outcome.payload["information"][0]


async def test_unknown_job_question_goes_to_the_recruiter(executor, session):
    outcome = await executor.execute("search_job_info", {"question": "Can I bring my dog?"})
    assert outcome.payload["found"] is False
    assert "recruiter" in outcome.payload["instruction"]
    assert session.scorecard.open_questions == ["Can I bring my dog?"]


async def test_a_failed_search_does_not_break_the_call(session, calendar):
    executor = ToolExecutor(session, FakeKnowledge(broken=True), calendar)
    outcome = await executor.execute("search_job_info", {"question": "What is the pay?"})
    assert outcome.payload["found"] is False
    assert session.scorecard.open_questions == ["What is the pay?"]


async def test_confirmed_failure_closes_with_the_reason(executor):
    await executor.execute("start_screening", {"candidate_name": "Samir", "consent": True})
    first = await executor.execute(
        "record_must_have", {"requirement": "work_permit", "answer": "no"}
    )
    assert first.closing_message is None
    outcome = await executor.execute(
        "record_must_have", {"requirement": "work_permit", "answer": "no", "confirmed": True}
    )
    assert "a valid permit to work in Spain" in outcome.closing_message


async def test_auto_escalation_ends_conversation(executor):
    await executor.execute("start_screening", {"candidate_name": "Lucia", "consent": True})
    outcomes = [
        await executor.execute("record_must_have", {"requirement": "start_date", "answer": "soon"})
        for _ in range(3)
    ]
    assert outcomes[1].closing_message is None
    assert "recruiter" in outcomes[2].closing_message


async def test_calls_after_close_are_rejected(executor):
    await executor.execute("end_conversation", {"reason": "candidate withdrew"})
    outcome = await executor.execute("search_job_info", {"question": "pay?"})
    assert outcome.payload["ok"] is False
    assert "already ended" in outcome.payload["error"]


async def test_only_the_closing_call_returns_the_closing_message(executor):
    # A second call after the close (for example in the same batch) must not repeat it.
    first = await executor.execute("escalate_to_human", {"reason": "asked for a person"})
    second = await executor.execute("end_conversation", {"reason": "duplicate"})
    assert "recruiter" in first.closing_message
    assert second.closing_message is None
    assert second.payload["ok"] is False
