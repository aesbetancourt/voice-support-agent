# Add Deepgram as a speech-to-text option

Status: accepted. It changes ADR-0002, which kept one vendor and no provider switch.

Review feedback asked for a state-of-the-art voice stack: Deepgram for transcription and ElevenLabs for the voice. Deepgram Nova-3 is built for real-time calls, and Pipecat has a native service for it. The voice channel now chooses its speech-to-text service with `STT_PROVIDER`: `google` (Chirp 3, the default) or `deepgram` (Nova-3).

Deepgram runs in its multilingual mode (`language="multi"`). One stream transcribes English and Spanish, and Pipecat puts the detected language of each utterance on the `TranscriptionFrame`. `LanguageSwitcher` reads the language from either service (`detected_language`), so the live switch (ADR-0009) works the same with both. A switch needs no reconnection to Deepgram.

The service is chosen in the voice adapter, because Pipecat's STT services call the provider directly (ADR-0004). The domain, the tools and the text channel do not change.

## Consequences

- Deepgram needs `DEEPGRAM_API_KEY`. A missing key stops the call at start with a clear configuration error.
- Keyterm prompting replaces Google's phrase hints for the company and site names. `GOOGLE_STT_PHRASE_HINTS=false` turns off both.
- The default stays Google, so an existing setup works with no new key.
- The choice between the two needs a measurement on real calls: latency to the final transcript, and accuracy on names, dates and Spanish.

## Considered Options

- **Replace Google STT with Deepgram.** Rejected for now: a switch keeps a working default and lets the two be compared on the same calls.
- **Deepgram with one language per connection.** Rejected: a language change would need a reconnection in the middle of a call.
