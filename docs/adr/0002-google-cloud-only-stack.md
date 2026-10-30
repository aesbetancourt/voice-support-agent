# Use Google Cloud only: Gemini on Vertex AI, Google STT and Google TTS

Every AI service runs on Google Cloud: Gemini on Vertex AI for the conversation and the summary, Cloud Speech-to-Text, and Cloud Text-to-Speech (Chirp 3 HD). One vendor and one auth method (Application Default Credentials) keep setup and operations simple, and Pipecat has native services for all three. We deliberately keep **no provider switch and no local fallback**, to avoid extra complexity.

## Considered Options

- **Claude (Anthropic) as the LLM, with local Whisper (MLX) STT and Kokoro TTS.** This was the first plan, when only an Anthropic key was available. Replaced when GCP access became available and the goal changed to running the whole project on one platform.
- **Google STT/TTS with local models as a fallback, selected by environment variables.** Rejected by the project owner: "Let's not overcomplicate this. Let's focus only on GCP."
- **Claude through the Vertex AI Model Garden.** Not chosen: the owner preferred a Google model.

## Consequences

- Reviewers need their own GCP project with billing and three APIs enabled. The text tests run with no network.
- Changing vendor means writing new outbound adapters (see ADR-0005), not changing the core.
- With user ADC credentials, a quota project must be set, or Text-to-Speech calls can fail. The setup steps in the README include it.
