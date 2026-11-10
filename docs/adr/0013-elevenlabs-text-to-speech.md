# Add ElevenLabs as a text-to-speech option

Status: accepted. It extends ADR-0012 to the voice: `TTS_PROVIDER` chooses `google` (Chirp 3 HD, the default) or `elevenlabs`.

Review feedback asked for a more natural voice, and named ElevenLabs. The ElevenLabs adapter uses Pipecat's WebSocket service with the `eleven_flash_v2_5` model: the lowest-latency multilingual model. One voice speaks English and Spanish, so a language switch keeps the same voice. Every setting (model, stability, similarity, speed) is in `config/settings.toml`; the key and the voice ID are in `.env`.

## Language

The adapter sends the language code with each connection, because a fixed code makes short replies ("Sí", "Vale") sound right. In Pipecat's ElevenLabs service, the language is part of the WebSocket URL, so a live language switch reconnects the TTS once. The switch happens between turns, so the candidate does not hear a gap in a sentence.

## Consequences

- ElevenLabs needs `ELEVENLABS_API_KEY` and `ELEVENLABS_VOICE_ID`. A missing value stops the call at start with a clear configuration error.
- The key needs only the text-to-speech permission.
- Free plans can use only ElevenLabs' default (premade) voices through the API. A Voice Library voice returns HTTP 402 (`paid_plan_required`).
- The default stays Google, so an existing setup works with no new key.

## Considered Options

- **No language code (the model detects the language from the text).** Rejected: no reconnection on a switch, but short replies can sound in the wrong accent.
- **`eleven_multilingual_v2`.** Rejected for the default: more expressive, but slower. Set `ELEVENLABS_MODEL` to try it.
