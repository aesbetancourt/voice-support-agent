from datetime import date
from pathlib import Path

import pytest

from screening_agent.domain.models import JobPosting
from screening_agent.domain.state import ScreeningSession

DATA = Path(__file__).resolve().parents[1] / "data"
TODAY = date(2026, 11, 10)


@pytest.fixture
def job() -> JobPosting:
    return JobPosting.model_validate_json((DATA / "job.json").read_text(encoding="utf-8"))


class CandidateWords:
    """What the candidate said so far. Tests add lines to check the evidence quotes."""

    def __init__(self):
        self.lines: list[str] = []

    def __call__(self) -> list[str]:
        return list(self.lines)


@pytest.fixture
def words() -> CandidateWords:
    return CandidateWords()


@pytest.fixture
def session(job: JobPosting, words: CandidateWords) -> ScreeningSession:
    return ScreeningSession(job, TODAY, words, max_attempts=3)


class FakeKnowledge:
    """Answers from a fixed table. ``broken`` simulates a failed search."""

    def __init__(self, answers: dict[str, str] | None = None, broken: bool = False):
        self.answers = answers or {"pay": "Pay: 1,580 euros gross per month."}
        self.broken = broken
        self.questions: list[str] = []

    async def search(self, question, limit):
        from screening_agent.application.ports import KnowledgeHit

        self.questions.append(question)
        if self.broken:
            raise RuntimeError("Vertex AI is down")
        return [
            KnowledgeHit(id=key, text=text, score=0.9)
            for key, text in self.answers.items()
            if key in question.lower()
        ][:limit]


@pytest.fixture
def knowledge() -> FakeKnowledge:
    return FakeKnowledge()


@pytest.fixture
def calendar():
    from screening_agent.adapters.outbound.mock_calendar import MockInterviewCalendar

    return MockInterviewCalendar(today=lambda: TODAY)


@pytest.fixture
def executor(session, knowledge, calendar):
    from screening_agent.application.tools import ToolExecutor

    return ToolExecutor(session, knowledge, calendar)
