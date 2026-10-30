"""GeminiChatModel against a fake google-genai client (no network)."""

from types import SimpleNamespace

from google.genai import types

from support_agent.text_agent import FunctionCall, GeminiChatModel


class FakeModels:
    def __init__(self, responses):
        self._responses = list(responses)
        self.requests = []

    async def generate_content(self, model, contents, config):
        self.requests.append(list(contents))
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
