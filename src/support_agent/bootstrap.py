"""Composition root: the only module that connects adapters to ports.

Inbound adapters (CLI, voice, simulation) call these factories,
so they never build outbound adapters themselves.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from google import genai

from support_agent.adapters.outbound.gemini import (
    GeminiChatModel,
    GeminiSentimentAnalyzer,
    GeminiSummarizer,
)
from support_agent.adapters.outbound.json_orders import JsonOrderRepository
from support_agent.adapters.outbound.json_store import JsonConversationStore
from support_agent.application.agent import TextAgent
from support_agent.application.ports import SentimentAnalyzer, Summarizer
from support_agent.application.prompts import system_prompt
from support_agent.application.recorder import ConversationRecorder
from support_agent.application.sentiment import SentimentMonitor
from support_agent.application.summary import SafeSummarizer
from support_agent.application.tools import ToolExecutor
from support_agent.config import Settings
from support_agent.domain.state import TicketSession


@dataclass(frozen=True)
class Conversation:
    session: TicketSession
    recorder: ConversationRecorder
    executor: ToolExecutor
    sentiment: SentimentMonitor | None


def make_summarizer(settings: Settings, client: genai.Client) -> Summarizer:
    return SafeSummarizer(GeminiSummarizer(client, settings.summary_model))


def make_sentiment_analyzer(settings: Settings, client: genai.Client) -> SentimentAnalyzer:
    return GeminiSentimentAnalyzer(client, settings.sentiment_model)


def new_conversation(
    settings: Settings,
    channel: Literal["text", "voice"],
    summarizer: Summarizer,
    sentiment_analyzer: SentimentAnalyzer | None = None,
) -> Conversation:
    orders = JsonOrderRepository.from_file(settings.orders_path)
    session = TicketSession(orders.get, max_attempts=settings.max_field_attempts)
    recorder = ConversationRecorder(
        channel, session, JsonConversationStore(settings.conversations_dir), summarizer
    )
    executor = ToolExecutor(session, on_call=recorder.add_tool_call)
    sentiment = SentimentMonitor(sentiment_analyzer, recorder) if sentiment_analyzer else None
    return Conversation(session=session, recorder=recorder, executor=executor, sentiment=sentiment)


def make_text_agent(
    settings: Settings, client: genai.Client, conversation: Conversation
) -> TextAgent:
    model = GeminiChatModel(client, settings.model, system_prompt("text"))
    return TextAgent(model, conversation.executor, conversation.recorder, conversation.sentiment)
