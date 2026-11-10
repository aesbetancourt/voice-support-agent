"""Speech services of the voice channel: one builder per STT and TTS provider.

``STT_PROVIDER`` and ``TTS_PROVIDER`` choose the provider; ``config/settings.toml`` holds every
provider setting. Pipecat's services call the provider directly (ADR-0004), so they are built
here, in the inbound voice adapter. Add a provider: add its settings section, a builder, and
an entry in the builder table.
"""

from __future__ import annotations

from collections.abc import Callable

from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.elevenlabs.tts import ElevenLabsTTSService
from pipecat.services.google.stt import GoogleSTTService
from pipecat.services.google.tts import GoogleTTSService
from pipecat.services.settings import TTSSettings
from pipecat.services.stt_service import STTService
from pipecat.services.tts_service import TTSService
from pipecat.transcriptions.language import Language as PipecatLanguage

from screening_agent.config import Settings, SttProvider, TtsProvider
from screening_agent.domain.models import Language


def stt_locale(settings: Settings, language: Language) -> PipecatLanguage:
    return PipecatLanguage(settings.speech.locales[language])


def stt_languages(settings: Settings, language: Language) -> list[PipecatLanguage]:
    """Every supported language, the session language first (Google's primary language)."""
    others = [
        stt_locale(settings, lang) for lang in settings.speech.locales if lang is not language
    ]
    return [stt_locale(settings, language), *others]


def _all_terms(settings: Settings) -> list[str]:
    """The terms of every language, without duplicates, in a stable order."""
    return list(dict.fromkeys(term for terms in settings.speech.terms.values() for term in terms))


# --- STT -----------------------------------------------------------------------------------


def build_google_stt(settings: Settings, language: Language) -> GoogleSTTService:
    google = settings.stt.google
    adaptation = None
    if settings.speech.phrase_hints:
        phrases = [{"value": term, "boost": google.phrase_boost} for term in _all_terms(settings)]
        adaptation = {"phrase_sets": [{"phrases": phrases}]}
    return GoogleSTTService(
        location=google.location,
        settings=GoogleSTTService.Settings(
            languages=stt_languages(settings, language), model=google.model, adaptation=adaptation
        ),
    )


def build_deepgram_stt(settings: Settings, language: Language) -> DeepgramSTTService:
    # One multilingual stream: a language switch needs no reconnection (ADR-0012).
    deepgram = settings.stt.deepgram
    return DeepgramSTTService(
        api_key=settings.require_deepgram_key(),
        settings=DeepgramSTTService.Settings(
            model=deepgram.model,
            language=deepgram.language,
            smart_format=deepgram.smart_format,
            keyterm=_all_terms(settings) if settings.speech.phrase_hints else None,
        ),
    )


_STT_BUILDERS: dict[SttProvider, Callable[[Settings, Language], STTService]] = {
    SttProvider.GOOGLE: build_google_stt,
    SttProvider.DEEPGRAM: build_deepgram_stt,
}


def build_stt(settings: Settings, language: Language) -> STTService:
    return _STT_BUILDERS[settings.stt.provider](settings, language)


# --- TTS -----------------------------------------------------------------------------------


def google_tts_settings(settings: Settings, language: Language) -> GoogleTTSService.Settings:
    # Set the language too: the service defaults its language code to en-US.
    return GoogleTTSService.Settings(
        voice=settings.tts.google.voices[language], language=stt_locale(settings, language)
    )


def build_google_tts(settings: Settings, language: Language) -> GoogleTTSService:
    return GoogleTTSService(settings=google_tts_settings(settings, language))


def elevenlabs_tts_settings(
    settings: Settings, language: Language
) -> ElevenLabsTTSService.Settings:
    # One multilingual voice; the language code makes short replies ("Sí") sound right.
    # The language is in the WebSocket URL, so a switch reconnects once, between turns.
    elevenlabs = settings.require_elevenlabs()
    return ElevenLabsTTSService.Settings(
        voice=elevenlabs.voice_id,
        model=elevenlabs.model,
        language=PipecatLanguage(language.value),
        stability=elevenlabs.stability,
        similarity_boost=elevenlabs.similarity_boost,
        speed=elevenlabs.speed,
    )


def elevenlabs_language_update(
    settings: Settings, language: Language
) -> ElevenLabsTTSService.Settings:
    return ElevenLabsTTSService.Settings(language=PipecatLanguage(language.value))


def build_elevenlabs_tts(settings: Settings, language: Language) -> ElevenLabsTTSService:
    return ElevenLabsTTSService(
        api_key=settings.require_elevenlabs().api_key,
        settings=elevenlabs_tts_settings(settings, language),
    )


_TTS_BUILDERS: dict[TtsProvider, Callable[[Settings, Language], TTSService]] = {
    TtsProvider.GOOGLE: build_google_tts,
    TtsProvider.ELEVENLABS: build_elevenlabs_tts,
}

# The settings change that moves a running TTS service to another language.
_TTS_SWITCHES: dict[TtsProvider, Callable[[Settings, Language], TTSSettings]] = {
    TtsProvider.GOOGLE: google_tts_settings,
    TtsProvider.ELEVENLABS: elevenlabs_language_update,
}


def build_tts(settings: Settings, language: Language) -> TTSService:
    return _TTS_BUILDERS[settings.tts.provider](settings, language)


def tts_language_update(settings: Settings, language: Language) -> TTSSettings:
    return _TTS_SWITCHES[settings.tts.provider](settings, language)
