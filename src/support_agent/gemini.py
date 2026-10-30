"""Gemini on Vertex AI client factory."""

from __future__ import annotations

from google import genai
from google.genai import types

from support_agent.config import Settings

# Retry rate limits and server errors with exponential backoff.
_RETRY = types.HttpRetryOptions(attempts=3, initial_delay=1.0, max_delay=8.0)
_TIMEOUT_MS = 30_000


def make_client(settings: Settings) -> genai.Client:
    """Create a Vertex AI client. Auth uses Application Default Credentials."""
    return genai.Client(
        vertexai=True,
        project=settings.require_project(),
        location=settings.location,
        http_options=types.HttpOptions(timeout=_TIMEOUT_MS, retry_options=_RETRY),
    )
