# Detect the customer's language per utterance in voice calls, and switch live

Voice calls no longer need a language chosen before the call. One Chirp 3 recognizer (`chirp_3`, location `us`) gets both `en-US` and `es-US` and reports the detected language on every final transcript. A domain rule, `language_switch()`, switches when the detected language changes and the transcript has at least 3 words. A `LanguageSwitcher` processor between STT and the user aggregator then sends the new TTS voice and a developer note for Gemini before the customer's words, so the reply to that same turn is in the new language. The fixed closing messages and the saved record follow the switch. This partly supersedes ADR-0008: the starting language (`--language`, `DEFAULT_LANGUAGE`, `body.language`) now sets only the greeting in voice calls. Text chat and simulations still use one fixed language.

## Considered Options

- **Keep one fixed language per call (ADR-0008).** Rejected for voice. A demo with the prebuilt UI needed a server restart per language, and a real customer cannot be asked to pick a server.
- **Let Gemini follow the customer's language on its own.** Rejected. Gemini could change its words, but not the TTS voice: an English voice reading Spanish text sounds wrong, and the closing messages would stay in the old language. The code must own the switch, so the voice, the words and the fixed messages change together (ADR-0001).
- **Detect with `language_codes: ["auto"]`.** Not chosen. It worked in a streaming test, but it can return a language with no voice profile. Listing the supported locales keeps detection inside what Ava can speak, and returns full locales (`es-US`) that match the profiles.
- **Switch on any detected change.** Rejected. Short replies ("ok", "sí", "no") give unreliable detection, so a 3-word minimum stops one word from flipping the call.
- **Keep `latest_long` for English, and swap the STT model on a switch.** Rejected. A switch would need a new STT connection in a different location, in the middle of a call. `chirp_3` recognized the spoken order number in both languages, so one recognizer serves both.

## Consequences

- **English now uses `chirp_3` instead of `latest_long`.** It recognized "O R D one zero four two three three" as `ORD104233` in a streaming test, but it has fewer live English calls behind it than `latest_long` had.
- **The first utterance in a new language is recognized correctly**, because the recognizer listens for both languages from the start. Only the reply language and voice wait for that utterance.
- **The switch is visible:** a server log line and a `serverMessage` event in the browser's Events panel.
- **A mixed-language sentence is detected as one language.** The 3-word minimum keeps short mixed replies from changing the language.
- **A new language must be supported by `chirp_3`**, because every profile shares the recognizer.
