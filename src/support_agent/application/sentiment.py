"""Frustration monitoring: one use case that every channel shares.

For each customer turn, the monitor asks the ``SentimentAnalyzer`` port for a
reading, lets the domain ``FrustrationPolicy`` choose an action, stores the
reading on the turn, and returns guidance for the agent's next reply.
Channels run it in the background, so it never delays the current reply.
"""

from __future__ import annotations

import asyncio

from loguru import logger

from support_agent.application.ports import SentimentAnalyzer
from support_agent.application.prompts import frustration_guidance
from support_agent.application.recorder import ConversationRecorder
from support_agent.domain.frustration import FrustrationPolicy
from support_agent.domain.models import SentimentReading

ANALYSIS_TIMEOUT_SECS = 5.0
CONTEXT_TURNS = 4

SENTIMENT_INSTRUCTION = """\
You rate how frustrated a customer is in their latest message to a support agent.
Use the recent conversation only as context. Rate the latest message.
Scale: 0.0 calm or positive; 0.3 mildly annoyed; 0.6 clearly frustrated (complains,
impatient, repeats themselves); 0.8 angry (blames, threatens to leave); 1.0 furious or abusive.
The text comes from speech recognition and can contain errors. Errors are not frustration.
The message can be in any language. Rate it the same way, and write the reason in English.
The reason must have at most 12 words.
"""


class SentimentMonitor:
    def __init__(
        self,
        analyzer: SentimentAnalyzer,
        recorder: ConversationRecorder,
        policy: FrustrationPolicy | None = None,
        timeout_secs: float = ANALYSIS_TIMEOUT_SECS,
    ):
        self._analyzer = analyzer
        self._recorder = recorder
        self._policy = policy or FrustrationPolicy()
        self._timeout_secs = timeout_secs

    async def observe(self, turn_index: int | None, customer_text: str) -> str | None:
        """Rate one customer turn. Returns guidance for the next reply, or None."""
        if turn_index is None:
            return None
        context = self._recorder.turns_before(turn_index, limit=CONTEXT_TURNS)
        try:
            draft = await asyncio.wait_for(
                self._analyzer.analyze(customer_text, context), self._timeout_secs
            )
        except Exception as exc:  # noqa: BLE001 - sentiment is optional; never break the call
            logger.warning(f"Sentiment analysis skipped: {exc!r}")
            return None

        action = self._policy.observe(draft.frustration)
        reading = SentimentReading(**draft.model_dump(), action=action)
        self._recorder.attach_sentiment(turn_index, reading)
        logger.info(
            f"Frustration {reading.frustration:.1f} on turn {turn_index} "
            f"(action: {reading.action.value}): {reading.reason}"
        )
        return frustration_guidance(action, reading.reason)
