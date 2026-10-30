"""Builds the per-conversation objects that both channels share."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from support_agent.config import Settings
from support_agent.orders import OrderRepository
from support_agent.recorder import ConversationRecorder
from support_agent.state import TicketSession
from support_agent.storage import ConversationStore
from support_agent.summary import Summarizer
from support_agent.tools import ToolExecutor


@dataclass(frozen=True)
class Conversation:
    session: TicketSession
    recorder: ConversationRecorder
    executor: ToolExecutor


def new_conversation(
    settings: Settings, channel: Literal["text", "voice"], summarizer: Summarizer
) -> Conversation:
    session = TicketSession(
        OrderRepository.from_json(settings.orders_path),
        max_attempts=settings.max_field_attempts,
    )
    recorder = ConversationRecorder(
        channel, session, ConversationStore(settings.conversations_dir), summarizer
    )
    executor = ToolExecutor(session, on_call=recorder.add_tool_call)
    return Conversation(session=session, recorder=recorder, executor=executor)
