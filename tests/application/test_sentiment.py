"""SentimentMonitor with fake analyzers (no network)."""

import asyncio

import pytest

from screening_agent.adapters.outbound.json_store import JsonConversationStore
from screening_agent.application.recorder import ConversationRecorder
from screening_agent.application.sentiment import SentimentMonitor
from screening_agent.application.summary import SafeSummarizer
from screening_agent.domain.models import FrustrationAction, SentimentDraft


class FixedAnalyzer:
    def __init__(self, frustration: float):
        self.frustration = frustration
        self.contexts = []

    async def analyze(self, candidate_text, recent_turns):
        self.contexts.append([turn.text for turn in recent_turns])
        return SentimentDraft(frustration=self.frustration, reason="waited a week")


class BrokenAnalyzer:
    async def analyze(self, candidate_text, recent_turns):
        raise RuntimeError("Vertex AI is down")


class SlowAnalyzer:
    async def analyze(self, candidate_text, recent_turns):
        await asyncio.sleep(1)
        return SentimentDraft(frustration=1.0, reason="late")


@pytest.fixture
def recorder(session, tmp_path):
    return ConversationRecorder(
        "text", session, JsonConversationStore(tmp_path), SafeSummarizer(None)
    )


async def test_high_frustration_is_stored_and_returns_guidance(recorder):
    analyzer = FixedAnalyzer(0.9)
    recorder.add_turn("assistant", "Hi, how can I help?")
    index = recorder.add_turn("user", "I have waited a week for this call!")

    guidance = await SentimentMonitor(analyzer, recorder).observe(index, "I have waited a week!")

    assert "acknowledge" in guidance
    assert "waited a week" in guidance
    assert analyzer.contexts == [["Hi, how can I help?"]]
    reading = recorder.build_record().transcript[index].sentiment
    assert (reading.frustration, reading.action) == (0.9, FrustrationAction.ACKNOWLEDGE)


async def test_calm_turn_returns_no_guidance(recorder):
    index = recorder.add_turn("user", "Hello, I have a question.")
    assert await SentimentMonitor(FixedAnalyzer(0.1), recorder).observe(index, "Hello") is None
    assert recorder.build_record().transcript[index].sentiment.frustration == 0.1


@pytest.mark.parametrize("analyzer", [BrokenAnalyzer(), SlowAnalyzer()])
async def test_failures_and_timeouts_are_skipped(recorder, analyzer):
    index = recorder.add_turn("user", "When can I start?")
    monitor = SentimentMonitor(analyzer, recorder, timeout_secs=0.05)
    assert await monitor.observe(index, "When can I start?") is None
    assert recorder.build_record().transcript[index].sentiment is None


async def test_skipped_turn_is_not_analyzed(recorder):
    analyzer = FixedAnalyzer(0.9)
    assert await SentimentMonitor(analyzer, recorder).observe(None, "") is None
    assert analyzer.contexts == []
