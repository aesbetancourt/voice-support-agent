"""The agent loop for turn-based channels (text chat and simulations).

It talks to any ``ChatModel`` port and runs every tool call through the
``ToolExecutor``. The voice channel uses Pipecat's own loop instead,
with the same executor.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

from support_agent.application.ports import ChatModel, FunctionCall, ModelReply
from support_agent.application.prompts import greeting_trigger
from support_agent.application.recorder import ConversationRecorder
from support_agent.application.sentiment import SentimentMonitor
from support_agent.application.tools import ToolExecutor

# Stops a model that keeps calling tools without ever answering the customer.
MAX_TOOL_ROUNDS = 6
FALLBACK_REPLY = "Sorry, I did not catch that. Could you say it again?"


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
    ):
        self._model = model
        self._executor = executor
        self._recorder = recorder
        self._sentiment = sentiment
        self._pending_guidance: str | None = None

    async def greet(self) -> AgentReply:
        # The greeting trigger is an instruction, not customer speech, so it is not recorded.
        trigger = greeting_trigger(self._recorder.language)
        return await self._run(await self._model.send_text(trigger))

    async def respond(self, user_text: str) -> AgentReply:
        turn_index = self._recorder.add_turn("user", user_text)
        # The sentiment reading runs in parallel with the reply, like in the voice channel,
        # so its guidance applies from the next turn.
        sentiment = (
            asyncio.create_task(self._sentiment.observe(turn_index, user_text))
            if self._sentiment
            else None
        )
        guidance, self._pending_guidance = self._pending_guidance, None
        try:
            reply = await self._run(await self._model.send_text(user_text, instruction=guidance))
        finally:
            if sentiment is not None:
                self._pending_guidance = await sentiment
        return reply

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
