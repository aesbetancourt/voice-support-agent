"""Speech services chosen by STT_PROVIDER and TTS_PROVIDER (no network, no connection)."""

import pytest
from conftest import test_settings

from screening_agent.adapters.inbound.speech import (
    build_stt,
    build_tts,
    google_tts_settings,
    stt_languages,
    stt_locale,
    tts_language_update,
)
from screening_agent.config import ConfigError, SttProvider
from screening_agent.domain.models import Language


@pytest.mark.parametrize("language", list(Language))
def test_every_language_has_a_locale_a_voice_and_terms(settings, language):
    locale = stt_locale(settings, language)
    assert locale.value.startswith(language.value)
    # The TTS voice and the STT locale must match, e.g. es-US.
    assert settings.tts.google.voices[language].startswith(f"{locale.value}-")
    assert "Harbor Logistics" in settings.speech.terms[language]


@pytest.mark.parametrize("language", list(Language))
def test_stt_listens_for_every_language_with_the_session_language_first(settings, language):
    languages = stt_languages(settings, language)
    assert languages[0] is stt_locale(settings, language)
    assert set(languages) == {stt_locale(settings, lang) for lang in Language}


def test_google_tts_follows_the_language(settings):
    update = google_tts_settings(settings, Language.ES)
    assert update.voice == settings.tts.google.voices[Language.ES]
    assert tts_language_update(settings, Language.ES) == update


def test_stt_provider_defaults_to_google(settings):
    assert settings.stt.provider is SttProvider.GOOGLE


def test_deepgram_stt_uses_the_configured_model_and_language():
    settings = test_settings(STT_PROVIDER="Deepgram", DEEPGRAM_API_KEY="test-key")
    stt = build_stt(settings, Language.ES)
    assert type(stt).__name__ == "DeepgramSTTService"
    assert stt._settings.model == settings.stt.deepgram.model
    assert stt._settings.language == settings.stt.deepgram.language


def test_deepgram_without_a_key_is_a_config_error():
    settings = test_settings(STT_PROVIDER="deepgram")
    with pytest.raises(ConfigError, match="DEEPGRAM_API_KEY"):
        build_stt(settings, Language.EN)


ELEVENLABS = {"TTS_PROVIDER": "elevenlabs", "ELEVENLABS_API_KEY": "k", "ELEVENLABS_VOICE_ID": "v1"}


def test_elevenlabs_tts_uses_the_configured_voice_model_and_language():
    settings = test_settings(**ELEVENLABS)
    tts = build_tts(settings, Language.ES)
    assert type(tts).__name__ == "ElevenLabsTTSService"
    assert tts._settings.voice == "v1"
    assert tts._settings.model == settings.tts.elevenlabs.model
    language = tts._settings.language
    assert str(getattr(language, "value", language)) == "es"


def test_elevenlabs_switch_changes_only_the_language():
    update = tts_language_update(test_settings(**ELEVENLABS), Language.EN)
    assert str(update.language.value) == "en"
    # A new voice or model would need a full reconnect for nothing.
    assert type(update.voice).__name__ == "NotGiven"


@pytest.mark.parametrize("missing", ["ELEVENLABS_API_KEY", "ELEVENLABS_VOICE_ID"])
def test_elevenlabs_without_a_key_or_voice_is_a_config_error(missing):
    env = {name: value for name, value in ELEVENLABS.items() if name != missing}
    with pytest.raises(ConfigError, match=missing):
        build_tts(test_settings(**env), Language.EN)
