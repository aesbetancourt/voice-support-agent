"""The agent loop for turn-based channels (text chat and simulations).

It talks to any ``ChatModel`` port and runs every tool call through the
``ToolExecutor``. The voice channel uses Pipecat's own loop instead,
with the same executor.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

from screening_agent.application.messages import retry_message
from screening_agent.application.ports import ChatModel, FunctionCall, ModelReply
from screening_agent.application.prompts import greeting_trigger
from screening_agent.application.recorder import ConversationRecorder
from screening_agent.application.sentiment import SentimentMonitor
from screening_agent.application.tools import ToolExecutor

# Stops a model that keeps calling tools without ever answering the customer.
MAX_TOOL_ROUNDS = 6


@dataclass(frozen=True)
class AgentReply:
    text: str
    ended: bool = False


class TextAgent:
    def __init__(
        self,
        model: ChatModel,
        executor: ToolExecutor,
        recorder: ConversationRecorder,
        sentiment: SentimentMonitor | None = None,
    ) -> None:
        self._model = model
        self._executor = executor
        self._recorder = recorder
        self._sentiment = sentiment
        self._reading: asyncio.Task[str | None] | None = None

    async def greet(self) -> AgentReply:
        # The greeting trigger is an instruction, not candidate speech, so it is not recorded.
        trigger = greeting_trigger(self._executor.session.job, self._recorder.language)
        return await self._run(await self._model.send_text(trigger))

    async def respond(self, user_text: str) -> AgentReply:
        # The previous turn's reading had the candidate's whole typing time to finish.
        guidance = await self.wait_for_reading()
        turn_index = self._recorder.add_turn("user", user_text)
        # This turn's reading runs in the background, like in the voice channel,
        # so it never delays the reply. Its guidance applies from the next turn.
        if self._sentiment is not None:
            self._reading = asyncio.create_task(self._sentiment.observe(turn_index, user_text))
        return await self._run(await self._model.send_text(user_text, instruction=guidance))

    async def wait_for_reading(self) -> str | None:
        """Wait for the last frustration reading; call it before the record is finished."""
        reading, self._reading = self._reading, None
        return await reading if reading is not None else None

    async def _run(self, reply: ModelReply) -> AgentReply:
        for _ in range(MAX_TOOL_ROUNDS):
            if not reply.function_calls:
                break
            results: list[tuple[FunctionCall, dict[str, Any]]] = []
            for call in reply.function_calls:
                outcome = await self._executor.execute(call.name, call.arguments)
                if outcome.closing_message is not None:
                    return self._say(outcome.closing_message, ended=True)
                results.append((call, outcome.payload))
            reply = await self._model.send_tool_results(results)
        return self._say(reply.text or retry_message(self._executor.language))

    def _say(self, text: str, ended: bool = False) -> AgentReply:
        self._recorder.add_turn("assistant", text)
        return AgentReply(text=text, ended=ended)
