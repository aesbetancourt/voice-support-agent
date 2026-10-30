import pytest

from support_agent.domain.frustration import FrustrationPolicy, sentiment_from_peak
from support_agent.domain.models import FrustrationAction, Sentiment

NONE, ACK, OFFER = (
    FrustrationAction.NONE,
    FrustrationAction.ACKNOWLEDGE,
    FrustrationAction.OFFER_HUMAN,
)


def observe_all(scores: list[float]) -> list[FrustrationAction]:
    policy = FrustrationPolicy()
    return [policy.observe(score) for score in scores]


@pytest.mark.parametrize(
    ("scores", "expected"),
    [
        ([0.1, 0.3, 0.5], [NONE, NONE, NONE]),
        ([0.8], [ACK]),
        ([0.8, 0.9], [ACK, OFFER]),
        ([0.8, 0.2, 0.8], [ACK, NONE, ACK]),
        # The human is offered once per conversation.
        ([0.8, 0.9, 0.9, 0.2, 0.8, 0.9], [ACK, OFFER, NONE, NONE, ACK, NONE]),
        ([0.7], [ACK]),
    ],
)
def test_policy_actions(scores, expected):
    assert observe_all(scores) == expected


@pytest.mark.parametrize(
    ("peak", "expected"),
    [
        (None, Sentiment.NEUTRAL),
        (0.2, Sentiment.NEUTRAL),
        (0.5, Sentiment.NEUTRAL),
        (0.6, Sentiment.FRUSTRATED),
        (0.8, Sentiment.ANGRY),
    ],
)
def test_sentiment_from_peak(peak, expected):
    assert sentiment_from_peak(peak) is expected
