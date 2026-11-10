"""Composition root: the only module that connects adapters to ports.

Inbound adapters (CLI, voice, simulation) call these factories,
so they never build outbound adapters themselves.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from functools import cache
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
from screening_agent.config import Settings
from screening_agent.domain.models import JobPosting, Language
from screening_agent.domain.state import ScreeningSession


@dataclass(frozen=True)
class Conversation:
    session: ScreeningSession
    recorder: ConversationRecorder
    executor: ToolExecutor
    sentiment: SentimentMonitor | None


def make_summarizer(settings: Settings, client: genai.Client) -> Summarizer:
    return SafeSummarizer(GeminiSummarizer(client, settings.summary_model))


def make_sentiment_analyzer(settings: Settings, client: genai.Client) -> SentimentAnalyzer:
    return GeminiSentimentAnalyzer(client, settings.sentiment_model)


@cache
def job_posting(settings: Settings) -> JobPosting:
    return load_job(settings.job_path)


# One index and one calendar per process: every conversation shares them, so the
# embeddings are built once and a booked time is not offered again.
@cache
def job_knowledge(settings: Settings) -> GeminiJobKnowledge:
    client = make_client(settings, location=settings.embedding_location)
    return GeminiJobKnowledge(
        client, settings.embedding_model, load_chunks(settings.knowledge_path)
    )


@cache
def interview_calendar(settings: Settings) -> MockInterviewCalendar:
    return MockInterviewCalendar()


def new_conversation(
    settings: Settings,
    channel: Literal["text", "voice"],
    summarizer: Summarizer,
    sentiment_analyzer: SentimentAnalyzer | None = None,
    language: Language | None = None,
) -> Conversation:
    language = language or settings.default_language
    recorder: ConversationRecorder | None = None

    def candidate_words() -> list[str]:
        return recorder.candidate_texts() if recorder is not None else []

    session = ScreeningSession(
        job_posting(settings), date.today(), candidate_words, settings.max_field_attempts
    )
    recorder = ConversationRecorder(
        channel, session, JsonConversationStore(settings.conversations_dir), summarizer, language
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
    model = GeminiChatModel(client, settings.model, prompt)
    return TextAgent(model, conversation.executor, conversation.recorder, conversation.sentiment)
