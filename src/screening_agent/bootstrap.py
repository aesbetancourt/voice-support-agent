"""Composition root: the only module that connects adapters to ports.

Inbound adapters (CLI, voice, simulation) call these factories, so they never build
outbound adapters themselves. Every adapter gets its settings from here.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from functools import cache
from pathlib import Path
from typing import Literal

from google import genai

from screening_agent.adapters.outbound.embeddings_knowledge import (
    GeminiJobKnowledge,
    load_chunks,
)
from screening_agent.adapters.outbound.gemini import (
    GeminiChatModel,
    GeminiSentimentAnalyzer,
    GeminiSummarizer,
    make_client,
)
from screening_agent.adapters.outbound.json_job import load_job
from screening_agent.adapters.outbound.json_store import JsonConversationStore
from screening_agent.adapters.outbound.mock_calendar import MockInterviewCalendar
from screening_agent.application.agent import TextAgent
from screening_agent.application.ports import SentimentAnalyzer, Summarizer
from screening_agent.application.prompts import system_prompt
from screening_agent.application.recorder import ConversationRecorder
from screening_agent.application.sentiment import SentimentMonitor
from screening_agent.application.summary import SafeSummarizer
from screening_agent.application.tools import ToolExecutor
from screening_agent.config import CalendarSettings, EmbeddingSettings, Settings
from screening_agent.domain.models import JobPosting, Language
from screening_agent.domain.state import ScreeningSession


@dataclass(frozen=True)
class Conversation:
    session: ScreeningSession
    recorder: ConversationRecorder
    executor: ToolExecutor
    sentiment: SentimentMonitor | None


def make_summarizer(settings: Settings, client: genai.Client) -> Summarizer:
    gemini = settings.gemini
    return SafeSummarizer(
        GeminiSummarizer(client, gemini.summary_model, gemini.summary_temperature)
    )


def make_sentiment_analyzer(settings: Settings, client: genai.Client) -> SentimentAnalyzer:
    gemini = settings.gemini
    return GeminiSentimentAnalyzer(client, gemini.sentiment_model, gemini.sentiment_temperature)


@cache
def _job_posting(path: Path) -> JobPosting:
    return load_job(path)


def job_posting(settings: Settings) -> JobPosting:
    return _job_posting(settings.app.job_path)


# One index and one calendar per process: every conversation shares them, so the embeddings
# are built once and a booked time is not offered again. The keys are the settings they use.
_knowledge: dict[tuple[EmbeddingSettings, Path], GeminiJobKnowledge] = {}


def job_knowledge(settings: Settings) -> GeminiJobKnowledge:
    embeddings, path = settings.embeddings, settings.app.knowledge_path
    key = (embeddings, path)
    if key not in _knowledge:
        client = make_client(settings, location=embeddings.location)
        _knowledge[key] = GeminiJobKnowledge(
            client, embeddings.model, load_chunks(path), embeddings.min_score
        )
    return _knowledge[key]


@cache
def _interview_calendar(calendar: CalendarSettings) -> MockInterviewCalendar:
    return MockInterviewCalendar(calendar.interview_times, calendar.working_days_ahead)


def interview_calendar(settings: Settings) -> MockInterviewCalendar:
    return _interview_calendar(settings.calendar)


def new_conversation(
    settings: Settings,
    channel: Literal["text", "voice"],
    summarizer: Summarizer,
    sentiment_analyzer: SentimentAnalyzer | None = None,
    language: Language | None = None,
) -> Conversation:
    app = settings.app
    language = language or app.default_language
    recorder: ConversationRecorder | None = None

    def candidate_words() -> list[str]:
        return recorder.candidate_texts() if recorder is not None else []

    session = ScreeningSession(
        job_posting(settings), date.today(), candidate_words, app.max_field_attempts
    )
    recorder = ConversationRecorder(
        channel, session, JsonConversationStore(app.conversations_dir), summarizer, language
    )
    executor = ToolExecutor(
        session,
        job_knowledge(settings),
        interview_calendar(settings),
        on_call=recorder.add_tool_call,
        language=language,
    )
    sentiment = SentimentMonitor(sentiment_analyzer, recorder) if sentiment_analyzer else None
    return Conversation(session=session, recorder=recorder, executor=executor, sentiment=sentiment)


def make_text_agent(
    settings: Settings, client: genai.Client, conversation: Conversation
) -> TextAgent:
    session = conversation.session
    prompt = system_prompt("text", session.job, session.today, conversation.recorder.language)
    model = GeminiChatModel(client, settings.gemini.model, prompt, settings.gemini.temperature)
    return TextAgent(model, conversation.executor, conversation.recorder, conversation.sentiment)
