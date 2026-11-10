"""GeminiChatModel against a fake google-genai client (no network)."""

from types import SimpleNamespace

import pytest
from google.genai import types

from screening_agent.adapters.outbound.gemini import (
    GeminiChatModel,
    GeminiSentimentAnalyzer,
    ThoughtTagStripper,
)
from screening_agent.application.ports import FunctionCall
from screening_agent.domain.models import Turn


class FakeModels:
    def __init__(self, responses):
        self._responses = list(responses)
        self.requests = []

    async def generate_content(self, model, contents, config):
        self.requests.append(contents if isinstance(contents, str) else list(contents))
        return self._responses.pop(0)


def model_response(*parts):
    return types.GenerateContentResponse(
        candidates=[types.Candidate(content=types.Content(role="model", parts=list(parts)))]
    )


async def test_parses_text_and_calls_and_keeps_history():
    fake = FakeModels(
        [
            model_response(
                types.Part(text="internal reasoning", thought=True),
                types.Part(
                    function_call=types.FunctionCall(
                        id="call-1", name="lookup_order", args={"order_number": "ORD-104233"}
                    )
                ),
            ),
            model_response(types.Part(text="Thanks, Maria. What happened?")),
        ]
    )
    client = SimpleNamespace(aio=SimpleNamespace(models=fake))
    chat = GeminiChatModel(client, "gemini-test", "system prompt")

    reply = await chat.send_text("My order is ORD-104233")
    assert reply.text == ""
    assert reply.function_calls == [
        FunctionCall("lookup_order", {"order_number": "ORD-104233"}, "call-1")
    ]

    reply = await chat.send_tool_results([(reply.function_calls[0], {"ok": True})])
    assert reply.text == "Thanks, Maria. What happened?"

    # Second request: user text, model turn (with thought part kept), function response.
    user, model_turn, tool_turn = fake.requests[1]
    assert model_turn.parts[0].thought
    response = tool_turn.parts[0].function_response
    assert (response.id, response.name, response.response) == (
        "call-1",
        "lookup_order",
        {"ok": True},
    )


async def test_empty_candidate_returns_empty_reply():
    fake = FakeModels([types.GenerateContentResponse(candidates=[])])
    chat = GeminiChatModel(SimpleNamespace(aio=SimpleNamespace(models=fake)), "m", "p")
    reply = await chat.send_text("hello")
    assert reply.text == ""
    assert reply.function_calls == []


async def test_instruction_is_sent_before_the_candidate_text():
    fake = FakeModels([model_response(types.Part(text="I understand."))])
    chat = GeminiChatModel(SimpleNamespace(aio=SimpleNamespace(models=fake)), "m", "p")
    await chat.send_text("When can I start?", instruction="Acknowledge the frustration.")
    note, candidate = fake.requests[0][0].parts
    assert note.text.startswith("[Guidance for the agent, not said by the candidate:")
    assert candidate.text == "When can I start?"


async def test_sentiment_analyzer_parses_structured_output():
    response = types.GenerateContentResponse(
        candidates=[
            types.Candidate(
                content=types.Content(
                    role="model",
                    parts=[types.Part(text='{"frustration": 0.8, "reason": "third call"}')],
                )
            )
        ]
    )
    fake = FakeModels([response])
    analyzer = GeminiSentimentAnalyzer(SimpleNamespace(aio=SimpleNamespace(models=fake)), "lite")
    recent = [Turn(role="assistant", text="How can I help?")]
    draft = await analyzer.analyze("This is my third call!", recent)
    assert (draft.frustration, draft.reason) == (0.8, "third call")
    assert "assistant: How can I help?" in fake.requests[0]


@pytest.mark.parametrize(
    ("chunks", "expected"),
    [
        (["Hello there."], "Hello there."),
        # Seen in a live Spanish call: an unclosed tag before the real reply.
        (
            ["<thought>\nEntiendo. ¿Pudo encontrar su número de orden?"],
            "\nEntiendo. ¿Pudo encontrar su número de orden?",
        ),
        (["<thought>The user is upset.</thought>I understand."], "I understand."),
        (["<tho", "ught>plan</tho", "ught>Sure, ", "one moment."], "Sure, one moment."),
        (["Price < 50 dollars."], "Price < 50 dollars."),
        (["Ends with <"], "Ends with <"),
        (["<THOUGHT>x</THOUGHT>Ok."], "Ok."),
    ],
)
def test_thought_tags_are_removed_from_streamed_text(chunks, expected):
    stripper = ThoughtTagStripper()
    text = "".join(stripper.feed(chunk) for chunk in chunks) + stripper.flush()
    assert text == expected


def test_thought_text_before_a_tag_is_sent_at_once():
    stripper = ThoughtTagStripper()
    assert stripper.feed("Sure. <thou") == "Sure. "


async def test_open_tool_calls_are_answered_before_new_candidate_text():
    # The agent loop can stop at MAX_TOOL_ROUNDS with calls still open.
    open_call = types.Part(
        function_call=types.FunctionCall(id="call-9", name="review_ticket", args={})
    )
    fake = FakeModels([model_response(open_call), model_response(types.Part(text="Sorry."))])
    chat = GeminiChatModel(SimpleNamespace(aio=SimpleNamespace(models=fake)), "m", "prompt")
    await chat.send_text("hello")
    await chat.send_text("are you there?")
    last_user_turn = fake.requests[1][-1]
    response, text = last_user_turn.parts
    assert response.function_response.id == "call-9"
    assert response.function_response.response["ok"] is False
    assert text.text == "are you there?"
