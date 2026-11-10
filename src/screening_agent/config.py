"""Settings read from environment variables (and an optional ``.env`` file)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

from screening_agent.domain.models import Language
from screening_agent.domain.validators import parse_language

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Low enough for consistent tool calls, high enough for natural replies. Voice and text share it.
AGENT_TEMPERATURE = 0.4


class ConfigError(RuntimeError):
    """Raised when a required setting is missing."""


@dataclass(frozen=True)
class Settings:
    project_id: str | None
    location: str
    model: str
    summary_model: str
    sentiment_model: str
    judge_model: str
    default_language: Language
    embedding_model: str
    embedding_location: str
    stt_phrase_hints: bool
    job_path: Path
    knowledge_path: Path
    conversations_dir: Path
    max_field_attempts: int

    @classmethod
    def from_env(cls) -> Settings:
        load_dotenv(PROJECT_ROOT / ".env")
        model = os.getenv("GEMINI_MODEL", "gemini-3.5-flash")
        return cls(
            project_id=os.getenv("GOOGLE_CLOUD_PROJECT") or None,
            location=os.getenv("GOOGLE_CLOUD_LOCATION", "global"),
            model=model,
            summary_model=os.getenv("GEMINI_SUMMARY_MODEL", model),
            sentiment_model=os.getenv("GEMINI_SENTIMENT_MODEL", "gemini-3.5-flash-lite"),
            # The judge runs offline, so a slower model is fine.
            judge_model=os.getenv("GEMINI_JUDGE_MODEL", model),
            default_language=parse_language(os.getenv("DEFAULT_LANGUAGE", "en")),
            embedding_model=os.getenv("GEMINI_EMBEDDING_MODEL", "gemini-embedding-001"),
            # Embedding models are served from regional locations, not from "global".
            embedding_location=os.getenv("EMBEDDING_LOCATION", "us-central1"),
            stt_phrase_hints=os.getenv("GOOGLE_STT_PHRASE_HINTS", "true").lower() != "false",
            job_path=Path(os.getenv("JOB_PATH", PROJECT_ROOT / "data" / "job.json")),
            knowledge_path=Path(
                os.getenv("KNOWLEDGE_PATH", PROJECT_ROOT / "data" / "job_knowledge.json")
            ),
            conversations_dir=Path(os.getenv("CONVERSATIONS_DIR", PROJECT_ROOT / "conversations")),
            max_field_attempts=int(os.getenv("MAX_FIELD_ATTEMPTS", "3")),
        )

    def require_project(self) -> str:
        if not self.project_id:
            raise ConfigError(
                "GOOGLE_CLOUD_PROJECT is not set. Copy .env.example to .env and set it."
            )
        return self.project_id
