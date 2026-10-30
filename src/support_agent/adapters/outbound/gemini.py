"""Gemini on Vertex AI adapters for the ``ChatModel`` and ``Summarizer`` ports."""

from __future__ import annotations

from typing import Any

from google import genai
from google.genai import types

from support_agent.application.ports import FunctionCall, ModelReply
from support_agent.application.summary import SUMMARY_INSTRUCTION, summary_input
from support_agent.application.tools import TOOL_SPECS
from support_agent.config import Settings
from support_agent.domain.models import ConversationRecord, Summary, SummaryDraft

# Retry rate limits and server errors with exponential backoff.
_RETRY = types.HttpRetryOptions(attempts=3, initial_delay=1.0, max_delay=8.0)
_TIMEOUT_MS = 30_000


def make_client(settings: Settings) -> genai.Client:
    """Create a Vertex AI client. Auth uses Application Default Credentials."""
    return genai.Client(
        vertexai=True,
        project=settings.require_project(),
        location=settings.location,
        http_options=types.HttpOptions(timeout=_TIMEOUT_MS, retry_options=_RETRY),
    )


class GeminiChatModel:
    """``ChatModel`` port with manual function calling."""

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


class GeminiSummarizer:
    """``Summarizer`` port with Gemini structured output."""

    def __init__(self, client: genai.Client, model: str):
        self._client = client
        self._model = model

    async def summarize(self, record: ConversationRecord) -> Summary:
        response = await self._client.aio.models.generate_content(
            model=self._model,
            contents=summary_input(record),
            config=types.GenerateContentConfig(
                system_instruction=SUMMARY_INSTRUCTION,
                response_mime_type="application/json",
                response_schema=SummaryDraft,
                temperature=0.2,
            ),
        )
        draft = response.parsed
        if not isinstance(draft, SummaryDraft):
            draft = SummaryDraft.model_validate_json(response.text or "")
        return Summary(**draft.model_dump(), generated_by=self._model)
