# Fix the language at the start of each conversation, with a speech profile per language

> **Partly superseded by [ADR-0009](0009-detect-the-language-in-voice-calls.md):** voice calls now detect the language per utterance and switch live. The fixed language still applies to text chat and simulations.

Each conversation has one language (English or Spanish), chosen before it starts: `{"language": "es"}` in the client's session body, otherwise `DEFAULT_LANGUAGE` (or the CLI option `--language`). The language selects a voice profile (STT language, STT model, STT location, TTS voice, phrase hints), the prompt's language rules, and the fixed customer messages. The internal language stays English: tool descriptions, tool results, enum values, sentiment reasons, and the summary for the support team.

## Considered Options

- **Automatic language detection, with a switch during the call.** Rejected for now. The first turn would go through the wrong STT model before any switch, and Google STT multi-language support depends on the model. Pipecat can change STT and TTS settings during a call (`STTUpdateSettingsFrame`, `TTSUpdateSettingsFrame`), so this stays possible later.
- **Let Gemini translate the fixed messages at runtime.** Rejected. The closing messages carry the ticket ID and are fixed on purpose (ADR-0001), so they are translated once in `application/messages.py`.
- **Translate tool results and the summary to the customer's language.** Rejected. The support team works in one language, and English tool results keep the prompt and the tests the same for every language. Gemini translates when it speaks.
- **One STT model for all languages.** Rejected after a measurement. `latest_long` misrecognized a fluently spoken Spanish order number ("O R D, uno cero cuatro, dos tres tres" became "o Rd 1004 233"), while `chirp_3` recognized it correctly in streaming mode. English keeps `latest_long`, which already worked in live voice calls. Spanish uses `chirp_3`, which Google serves only from regional locations (`us`).

## Consequences

- **The order-number normalizer accepts spoken digits in every supported language at once** ("uno cero cuatro", "double three", "o erre de"). It does not depend on the session language, because customers mix languages and STT can write either form.
- **Spanish uses the formal "usted".** The prompt requires it, so Gemini's replies match the fixed closing messages.
- **To add a language**, you add an entry in `Language`, `LANGUAGE_NAMES`, the closing messages, `VOICE_PROFILES`, and the normalizer word lists. A test fails if a closing message or a voice profile is missing.
- **The old `GOOGLE_TTS_VOICE` and `GOOGLE_STT_LOCATION` settings are removed.** The voice profiles replace them.
- **The prebuilt browser UI cannot choose a language**, so each voice server runs with one default language (`support-agent voice --language es`). A custom client can choose it per session.
