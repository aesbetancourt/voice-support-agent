"""Text channel: the agent loop over Gemini with manual function calling."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from google import genai
from google.genai import types

from support_agent.prompts import GREETING_TRIGGER
from support_agent.recorder import ConversationRecorder
from support_agent.tools import TOOL_SPECS, ToolExecutor

# Stops a model that keeps calling tools without ever answering the customer.
MAX_TOOL_ROUNDS = 6
FALLBACK_REPLY = "Sorry, I did not catch that. Could you say it again?"


@dataclass(frozen=True)
class FunctionCall:
    name: str
    arguments: dict[str, Any]
    call_id: str | None = None


@dataclass(frozen=True)
class ModelReply:
    text: str
    function_calls: list[FunctionCall] = field(default_factory=list)


@dataclass(frozen=True)
class AgentReply:
    text: str
    ended: bool = False


class ChatModel(Protocol):
    """A chat model that keeps its own conversation history."""

    async def send_text(self, text: str) -> ModelReply: ...

    async def send_tool_results(
        self, results: list[tuple[FunctionCall, dict[str, Any]]]
    ) -> ModelReply: ...


class GeminiChatModel:
    def __init__(self, client: genai.Client, model: str, system_prompt: str):
        self._client = client
        self._model = model
        self._history: list[types.Content] = []
        self._config = types.GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=0.4,
            tools=[
                types.Tool(
                    function_declarations=[
                        types.FunctionDeclaration(
                            name=spec.name,
                            description=spec.description,
                            parameters_json_schema=spec.json_schema(),
                        )
                        for spec in TOOL_SPECS
                    ]
                )
            ],
            # The agent loop runs the tools itself, so it can enforce the state machine.
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        )

    async def send_text(self, text: str) -> ModelReply:
        self._history.append(types.Content(role="user", parts=[types.Part.from_text(text=text)]))
        return await self._generate()

    async def send_tool_results(
        self, results: list[tuple[FunctionCall, dict[str, Any]]]
    ) -> ModelReply:
        parts = [
            types.Part(
                function_response=types.FunctionResponse(
                    id=call.call_id, name=call.name, response=payload
                )
            )
            for call, payload in results
        ]
        self._history.append(types.Content(role="user", parts=parts))
        return await self._generate()

    async def _generate(self) -> ModelReply:
        response = await self._client.aio.models.generate_content(
            model=self._model, contents=self._history, config=self._config
        )
        content = response.candidates[0].content if response.candidates else None
        if content is None:
            return ModelReply(text="")
        # Keep the full model turn: Gemini needs its thought signatures in later requests.
        self._history.append(content)
        parts = content.parts or []
        text = "".join(p.text for p in parts if p.text and not p.thought)
        calls = [
            FunctionCall(
                name=p.function_call.name or "",
                arguments=dict(p.function_call.args or {}),
                call_id=p.function_call.id,
            )
            for p in parts
            if p.function_call
        ]
        return ModelReply(text=text.strip(), function_calls=calls)


class TextAgent:
    def __init__(self, model: ChatModel, executor: ToolExecutor, recorder: ConversationRecorder):
        self._model = model
        self._executor = executor
        self._recorder = recorder

    async def greet(self) -> AgentReply:
        # The greeting trigger is an instruction, not customer speech, so it is not recorded.
        return await self._run(await self._model.send_text(GREETING_TRIGGER))

    async def respond(self, user_text: str) -> AgentReply:
        self._recorder.add_turn("user", user_text)
        return await self._run(await self._model.send_text(user_text))

    async def _run(self, reply: ModelReply) -> AgentReply:
        for _ in range(MAX_TOOL_ROUNDS):
            if not reply.function_calls:
                break
            results: list[tuple[FunctionCall, dict[str, Any]]] = []
            for call in reply.function_calls:
                outcome = self._executor.execute(call.name, call.arguments)
                if outcome.closing_message is not None:
                    return self._say(outcome.closing_message, ended=True)
                results.append((call, outcome.payload))
            reply = await self._model.send_tool_results(results)
        return self._say(reply.text or FALLBACK_REPLY)

    def _say(self, text: str, ended: bool = False) -> AgentReply:
        self._recorder.add_turn("assistant", text)
        return AgentReply(text=text, ended=ended)
