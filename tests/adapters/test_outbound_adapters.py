"""The job search ranking and the mock calendar (no network)."""

from datetime import date

from screening_agent.adapters.outbound.embeddings_knowledge import (
    GeminiJobKnowledge,
    KnowledgeChunk,
    cosine,
    load_chunks,
    rank,
)
from screening_agent.adapters.outbound.mock_calendar import MockInterviewCalendar

PAY = KnowledgeChunk("pay", "Pay: 1,580 euros gross per month.")
SHIFTS = KnowledgeChunk("shifts", "Shifts rotate every week.")


def test_cosine():
    assert cosine([1.0, 0.0], [1.0, 0.0]) == 1.0
    assert cosine([1.0, 0.0], [0.0, 1.0]) == 0.0
    assert cosine([0.0, 0.0], [1.0, 0.0]) == 0.0


def test_rank_returns_the_best_matches_above_the_threshold():
    index = [(PAY, [1.0, 0.0]), (SHIFTS, [0.6, 0.8])]
    hits = rank([1.0, 0.0], index, limit=3, min_score=0.5)
    assert [h.id for h in hits] == ["pay", "shifts"]
    assert [h.id for h in rank([1.0, 0.0], index, limit=3, min_score=0.7)] == ["pay"]
    assert rank([0.0, 1.0], [(PAY, [1.0, 0.0])], limit=3, min_score=0.5) == []


def test_the_knowledge_file_loads():
    from conftest import DATA

    chunks = load_chunks(DATA / "job_knowledge.json")
    assert {"pay", "shifts", "location", "next_steps"} <= {c.id for c in chunks}


class FakeEmbeddings:
    """Embeds by keyword, and counts the calls."""

    def __init__(self):
        self.calls = 0

    async def embed_content(self, model, contents, config):
        from google.genai import types

        self.calls += 1
        vector = [1.0, 0.0] if "pay" in contents.lower() else [0.0, 1.0]
        return types.EmbedContentResponse(embeddings=[types.ContentEmbedding(values=vector)])


class FakeClient:
    def __init__(self):
        self.models = FakeEmbeddings()
        self.aio = self


async def test_search_builds_the_index_once():
    client = FakeClient()
    knowledge = GeminiJobKnowledge(client, "embedding-model", [PAY, SHIFTS], min_score=0.5)
    await knowledge.warm_up()
    hits = await knowledge.search("What is the pay?", 3)
    await knowledge.search("And the pay on Saturdays?", 3)
    assert [h.id for h in hits] == ["pay"]
    assert client.models.calls == 2 + 2  # Two chunks once, then one call per question.


def test_calendar_offers_working_days_only_and_hides_booked_slots():
    calendar = MockInterviewCalendar(today=lambda: date(2026, 11, 13))  # A Friday.
    slots = calendar.free_slots(10)
    assert {slot.starts_at.weekday() for slot in slots} <= {0, 1, 2, 3, 4}
    assert slots[0].starts_at.date() == date(2026, 11, 16)  # The next Monday.
    calendar.book(slots[0].id)
    assert calendar.free_slots(1)[0].id == slots[1].id
