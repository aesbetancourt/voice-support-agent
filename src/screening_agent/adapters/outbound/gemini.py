"""Gemini on Vertex AI adapters for the ChatModel, Summarizer and SentimentAnalyzer ports."""

from __future__ import annotations

import re
from typing import Any

from google import genai
from google.genai import types

from screening_agent.adapters.constants import THOUGHT_CLOSE_TAG, THOUGHT_OPEN_TAG
from screening_agent.application.ports import FunctionCall, ModelReply
from screening_agent.application.prompts import agent_note
from screening_agent.application.sentiment import SENTIMENT_INSTRUCTION
from screening_agent.application.summary import SUMMARY_INSTRUCTION, summary_input
from screening_agent.application.tools import TOOL_SPECS
from screening_agent.config import Settings
from screening_agent.domain.models import (
    ConversationRecord,
    SentimentDraft,
    Summary,
    SummaryDraft,
    Turn,
)

_OPEN_TAG = THOUGHT_OPEN_TAG
_CLOSE_TAG = THOUGHT_CLOSE_TAG


class ThoughtTagStripper:
    """Remove ``<thought>`` tags that Gemini sometimes writes as plain reply text.

    These are not real thought parts (those are filtered by ``part.thought``), so they
    would reach the customer. A closed ``<thought>…</thought>`` block is dropped. An
    unclosed tag is dropped, and the text after it is kept: in live calls it was the reply.
    Feed streamed chunks in order, then call ``flush()`` at the end of the response.
    """

    def __init__(self) -> None:
        self._buffer = ""
        self._in_block = False

    def feed(self, chunk: str) -> str:
        self._buffer += chunk
        out = []
        while True:
            lower = self._buffer.lower()
            if self._in_block:
                end = lower.find(_CLOSE_TAG)
                if end < 0:
                    return "".join(out)  # Hold the text until the block closes or ends.
                self._buffer = self._buffer[end + len(_CLOSE_TAG) :]
                self._in_block = False
                continue
            start = lower.find(_OPEN_TAG)
            if start >= 0:
                out.append(self._buffer[:start])
                self._buffer = self._buffer[start + len(_OPEN_TAG) :]
                self._in_block = True
                continue
            self._buffer = re.sub(re.escape(_CLOSE_TAG), "", self._buffer, flags=re.IGNORECASE)
            keep = _partial_tag_length(self._buffer)
            out.append(self._buffer[: len(self._buffer) - keep])
            self._buffer = self._buffer[len(self._buffer) - keep :]
            return "".join(out)

    def flush(self) -> str:
        """Return the held text. The text after an unclosed tag counts as the reply."""
        rest, self._buffer, self._in_block = self._buffer, "", False
        return rest


def _partial_tag_length(text: str) -> int:
    """Length of a trailing piece that can still become a tag, such as ``<tho``."""
    lower = text.lower()
    for size in range(min(len(lower), len(_CLOSE_TAG) - 1), 0, -1):
        tail = lower[-size:]
        if _OPEN_TAG.startswith(tail) or _CLOSE_TAG.startswith(tail):
            return size
    return 0


def strip_thought_tags(text: str) -> str:
    stripper = ThoughtTagStripper()
    return stripper.feed(text) + stripper.flush()


def make_client(settings: Settings, location: str | None = None) -> genai.Client:
    """Create a Vertex AI client. Auth uses Application Default Credentials."""
    gemini = settings.gemini
    retry = types.HttpRetryOptions(
        attempts=gemini.retry_attempts,
        initial_delay=gemini.retry_initial_delay_secs,
        max_delay=gemini.retry_max_delay_secs,
    )
    return genai.Client(
        vertexai=True,
        project=settings.require_project(),
        location=location or gemini.location,
        http_options=types.HttpOptions(timeout=gemini.timeout_ms, retry_options=retry),
    )


class GeminiChatModel:
    """``ChatModel`` port with manual function calling."""

    def __init__(
        self, client: genai.Client, model: str, system_prompt: str, temperature: float
    ) -> None:
        self._client = client
        self._model = model
        self._history: list[types.Content] = []
        self._config = types.GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=temperature,
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

    async def send_text(self, text: str, instruction: str | None = None) -> ModelReply:
        parts = self._close_open_calls()
        if instruction:
            parts.append(types.Part.from_text(text=agent_note(instruction)))
        parts.append(types.Part.from_text(text=text))
        self._history.append(types.Content(role="user", parts=parts))
        return await self._generate()

    def _close_open_calls(self) -> list[types.Part]:
        """Responses for tool calls left open when the agent loop hit MAX_TOOL_ROUNDS.

        Gemini rejects a request whose last model turn has calls without responses.
        """
        last = self._history[-1] if self._history else None
        if last is None or last.role != "model":
            return []
        error = {"ok": False, "error": "Not run: too many tool calls in one turn."}
        return [
            types.Part(
                function_response=types.FunctionResponse(
                    id=part.function_call.id, name=part.function_call.name, response=error
                )
            )
            for part in last.parts or []
            if part.function_call
        ]

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
        text = strip_thought_tags("".join(p.text for p in parts if p.text and not p.thought))
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

    def __init__(self, client: genai.Client, model: str, temperature: float) -> None:
        self._client = client
        self._model = model
        self._temperature = temperature

    async def summarize(self, record: ConversationRecord) -> Summary:
        response = await self._client.aio.models.generate_content(
            model=self._model,
            contents=summary_input(record),
            config=types.GenerateContentConfig(
                system_instruction=SUMMARY_INSTRUCTION,
                response_mime_type="application/json",
                response_schema=SummaryDraft,
                temperature=self._temperature,
            ),
        )
        draft = response.parsed
        if not isinstance(draft, SummaryDraft):
            draft = SummaryDraft.model_validate_json(response.text or "")
        return Summary(**draft.model_dump(), generated_by=self._model)


class GeminiSentimentAnalyzer:
    """``SentimentAnalyzer`` port with a small, fast Gemini model and structured output."""

    def __init__(self, client: genai.Client, model: str, temperature: float) -> None:
        self._client = client
        self._model = model
        self._temperature = temperature

    async def analyze(self, candidate_text: str, recent_turns: list[Turn]) -> SentimentDraft:
        context = "\n".join(f"{turn.role}: {turn.text}" for turn in recent_turns) or "(none)"
        prompt = f"Recent conversation:\n{context}\n\nLatest candidate message:\n{candidate_text}"
        response = await self._client.aio.models.generate_content(
            model=self._model,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=SENTIMENT_INSTRUCTION,
                response_mime_type="application/json",
                response_schema=SentimentDraft,
                temperature=self._temperature,
            ),
        )
        draft = response.parsed
        if not isinstance(draft, SentimentDraft):
            draft = SentimentDraft.model_validate_json(response.text or "")
        return draft
