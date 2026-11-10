"""Settings: TOML defaults and environment overrides (the real .env is never read)."""

from datetime import time

import pytest
from conftest import test_settings

from screening_agent.config import CONFIG_PATH, ENV_OVERRIDES, ConfigError, Settings
from screening_agent.domain.models import Language


def test_defaults_come_from_the_toml_file(settings):
    assert settings.gemini.temperature == 0.4
    assert settings.embeddings.min_score == 0.68
    assert settings.calendar.interview_times[0] == time(10, 0)
    assert settings.app.job_path.is_absolute()
    assert settings.app.job_path.exists()


def test_environment_overrides_the_toml_value():
    settings = test_settings(
        GEMINI_MODEL="gemini-x", DEFAULT_LANGUAGE="Spanish", GOOGLE_STT_PHRASE_HINTS="false"
    )
    assert settings.gemini.model == "gemini-x"
    assert settings.app.default_language is Language.ES
    assert settings.speech.phrase_hints is False


def test_an_empty_variable_keeps_the_default(settings):
    assert test_settings(GEMINI_MODEL="").gemini.model == settings.gemini.model


@pytest.mark.parametrize("name", list(ENV_OVERRIDES))
def test_every_override_points_to_a_real_setting(name):
    # A typo in ENV_OVERRIDES would create an unknown key, which the settings reject.
    path = ENV_OVERRIDES[name]
    value = {"DEFAULT_LANGUAGE": "es", "MAX_FIELD_ATTEMPTS": "2", "GOOGLE_STT_PHRASE_HINTS": "true",
             "STT_PROVIDER": "google", "TTS_PROVIDER": "google"}.get(name, "x")  # fmt: skip
    settings = test_settings(**{name: value})
    node = settings
    for key in path:
        node = getattr(node, key)
    assert str(getattr(node, "value", node)).lower().endswith(value.lower())


def test_an_unknown_provider_is_a_config_error():
    with pytest.raises(ConfigError, match="'google' or 'deepgram'"):
        test_settings(STT_PROVIDER="whisper")


def test_a_missing_project_is_a_config_error():
    settings = Settings.load(CONFIG_PATH, environ={})
    with pytest.raises(ConfigError, match="GOOGLE_CLOUD_PROJECT"):
        settings.require_project()
