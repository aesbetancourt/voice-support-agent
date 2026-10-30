"""Frustration policy: turns frustration readings into agent actions.

A classifier measures frustration. This policy decides what to do about it,
so the reaction is deterministic and testable (see ADR-0001 and ADR-0007).

    first high reading            → ACKNOWLEDGE
    high readings in a row (2)    → OFFER_HUMAN, once per conversation
    reading below the threshold   → streak resets
"""

from __future__ import annotations

from support_agent.domain.models import FrustrationAction, Sentiment

HIGH_FRUSTRATION = 0.7
OFFER_HUMAN_AFTER = 2


class FrustrationPolicy:
    def __init__(self, high: float = HIGH_FRUSTRATION, offer_after: int = OFFER_HUMAN_AFTER):
        self.high = high
        self.offer_after = offer_after
        self._streak = 0
        self._offered_human = False

    def observe(self, frustration: float) -> FrustrationAction:
        if frustration < self.high:
            self._streak = 0
            return FrustrationAction.NONE
        self._streak += 1
        if self._streak >= self.offer_after and not self._offered_human:
            self._offered_human = True
            return FrustrationAction.OFFER_HUMAN
        if self._streak == 1:
            return FrustrationAction.ACKNOWLEDGE
        # Still frustrated, and a human was already offered: do not repeat the offer.
        return FrustrationAction.NONE


def sentiment_from_peak(peak_frustration: float | None) -> Sentiment:
    """Overall sentiment for a conversation, from its highest frustration reading."""
    if peak_frustration is None:
        return Sentiment.NEUTRAL
    if peak_frustration >= 0.85:
        return Sentiment.ANGRY
    if peak_frustration >= 0.5:
        return Sentiment.FRUSTRATED
    return Sentiment.NEUTRAL
