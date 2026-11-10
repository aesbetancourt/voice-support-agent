"""Settings: defaults from ``config/settings.toml``, overridden by environment variables.

The TOML file holds every default for the app and the adapters. ``.env`` holds secrets and
per-machine choices. ``ENV_OVERRIDES`` lists each environment variable and the setting it sets.
"""

from __future__ import annotations

import os
import tomllib
from datetime import time
from enum import StrEnum
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, ValidationError, field_validator

from screening_agent.domain.models import Language
from screening_agent.domain.validators import parse_language

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "config" / "settings.toml"
ENV_PATH = PROJECT_ROOT / ".env"

# Environment variable → path of the setting in the TOML file.
ENV_OVERRIDES: dict[str, tuple[str, ...]] = {
    "DEFAULT_LANGUAGE": ("app", "default_language"),
    "MAX_FIELD_ATTEMPTS": ("app", "max_field_attempts"),
    "JOB_PATH": ("app", "job_path"),
    "KNOWLEDGE_PATH": ("app", "knowledge_path"),
    "CONVERSATIONS_DIR": ("app", "conversations_dir"),
    "GOOGLE_CLOUD_PROJECT": ("google", "project"),
    "GOOGLE_CLOUD_LOCATION": ("gemini", "location"),
    "GEMINI_MODEL": ("gemini", "model"),
    "GEMINI_SUMMARY_MODEL": ("gemini", "summary_model"),
    "GEMINI_SENTIMENT_MODEL": ("gemini", "sentiment_model"),
    "GEMINI_JUDGE_MODEL": ("gemini", "judge_model"),
    "GEMINI_EMBEDDING_MODEL": ("embeddings", "model"),
    "EMBEDDING_LOCATION": ("embeddings", "location"),
    "GOOGLE_STT_PHRASE_HINTS": ("speech", "phrase_hints"),
    "STT_PROVIDER": ("stt", "provider"),
    "DEEPGRAM_API_KEY": ("stt", "deepgram", "api_key"),
    "DEEPGRAM_MODEL": ("stt", "deepgram", "model"),
    "TTS_PROVIDER": ("tts", "provider"),
    "ELEVENLABS_API_KEY": ("tts", "elevenlabs", "api_key"),
    "ELEVENLABS_VOICE_ID": ("tts", "elevenlabs", "voice_id"),
    "ELEVENLABS_MODEL": ("tts", "elevenlabs", "model"),
}


class ConfigError(RuntimeError):
    """Raised when a setting is missing or invalid."""


class SttProvider(StrEnum):
    """The speech-to-text service of the voice channel."""

    GOOGLE = "google"
    DEEPGRAM = "deepgram"


class TtsProvider(StrEnum):
    """The text-to-speech service of the voice channel."""

    GOOGLE = "google"
    ELEVENLABS = "elevenlabs"


class _Section(BaseModel):
    # Frozen, so a section is hashable and can key a process-wide cache.
    model_config = ConfigDict(frozen=True, extra="forbid")


class AppSettings(_Section):
    default_language: Language
    max_field_attempts: int
    job_path: Path
    knowledge_path: Path
    conversations_dir: Path

    @field_validator("default_language", mode="before")
    @classmethod
    def _parse_language(cls, raw: object) -> Language:
        return parse_language(str(raw))

    @field_validator("job_path", "knowledge_path", "conversations_dir")
    @classmethod
    def _from_project_root(cls, path: Path) -> Path:
        return path if path.is_absolute() else PROJECT_ROOT / path


class GoogleSettings(_Section):
    project: str


class GeminiSettings(_Section):
    location: str
    model: str
    summary_model: str
    sentiment_model: str
    judge_model: str
    temperature: float
    summary_temperature: float
    sentiment_temperature: float
    judge_temperature: float
    candidate_temperature: float
    timeout_ms: int
    retry_attempts: int
    retry_initial_delay_secs: float
    retry_max_delay_secs: float


class EmbeddingSettings(_Section):
    model: str
    location: str
    min_score: float


class CalendarSettings(_Section):
    interview_times: tuple[time, ...]
    working_days_ahead: int


class CliSettings(_Section):
    max_input_chars: int
    simulate_max_turns: int


class SpeechSettings(_Section):
    locales: dict[Language, str]
    phrase_hints: bool
    terms: dict[Language, tuple[str, ...]]


class GoogleSttSettings(_Section):
    model: str
    location: str
    phrase_boost: float


class DeepgramSttSettings(_Section):
    api_key: str
    model: str
    language: str
    smart_format: bool


class SttSettings(_Section):
    provider: SttProvider
    google: GoogleSttSettings
    deepgram: DeepgramSttSettings

    @field_validator("provider", mode="before")
    @classmethod
    def _lowercase(cls, raw: object) -> str:
        return str(raw).strip().lower()


class GoogleTtsSettings(_Section):
    voices: dict[Language, str]


class ElevenLabsTtsSettings(_Section):
    api_key: str
    voice_id: str
    model: str
    stability: float
    similarity_boost: float
    speed: float


class TtsSettings(_Section):
    provider: TtsProvider
    google: GoogleTtsSettings
    elevenlabs: ElevenLabsTtsSettings

    @field_validator("provider", mode="before")
    @classmethod
    def _lowercase(cls, raw: object) -> str:
        return str(raw).strip().lower()


class Settings(_Section):
    app: AppSettings
    google: GoogleSettings
    gemini: GeminiSettings
    embeddings: EmbeddingSettings
    calendar: CalendarSettings
    cli: CliSettings
    speech: SpeechSettings
    stt: SttSettings
    tts: TtsSettings

    @classmethod
    def load(
        cls, config_path: Path = CONFIG_PATH, environ: dict[str, str] | None = None
    ) -> Settings:
        """Read the TOML defaults, then apply the environment overrides."""
        if environ is None:
            load_dotenv(ENV_PATH)
            environ = dict(os.environ)
        data = tomllib.loads(config_path.read_text(encoding="utf-8"))
        for name, path in ENV_OVERRIDES.items():
            value = environ.get(name)
            if value:
                _set(data, path, value)
        try:
            return cls.model_validate(data)
        except ValidationError as exc:
            raise ConfigError(f"Invalid settings in {config_path.name} or .env:\n{exc}") from exc

    @classmethod
    def from_env(cls) -> Settings:
        return cls.load()

    def require_project(self) -> str:
        if not self.google.project:
            raise ConfigError(
                "GOOGLE_CLOUD_PROJECT is not set. Copy .env.example to .env and set it."
            )
        return self.google.project

    def require_elevenlabs(self) -> ElevenLabsTtsSettings:
        missing = [
            name
            for name, value in (
                ("ELEVENLABS_API_KEY", self.tts.elevenlabs.api_key),
                ("ELEVENLABS_VOICE_ID", self.tts.elevenlabs.voice_id),
            )
            if not value
        ]
        if missing:
            raise ConfigError(
                f"TTS_PROVIDER is elevenlabs, but {' and '.join(missing)} is not set. "
                "Set it in .env."
            )
        return self.tts.elevenlabs

    def require_deepgram_key(self) -> str:
        if not self.stt.deepgram.api_key:
            raise ConfigError(
                "STT_PROVIDER is deepgram, but DEEPGRAM_API_KEY is not set. Set it in .env."
            )
        return self.stt.deepgram.api_key


def _set(data: dict[str, Any], path: tuple[str, ...], value: str) -> None:
    for key in path[:-1]:
        data = data.setdefault(key, {})
    data[path[-1]] = value
