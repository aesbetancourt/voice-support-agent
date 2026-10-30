# Use a cascaded STT → LLM → TTS pipeline, not speech-to-speech

The voice channel uses three separate services: Google Speech-to-Text, Gemini, and Google Text-to-Speech. We did not use a speech-to-speech model. The brief asks for a transcription model and a TTS model. The case is scored mostly on extraction and validation, and a cascaded pipeline gives the most control there: STT phrase hints boost "ORD" and the category names, every transcript is visible and saved, and each stage can be measured separately.

## Considered Options

- **Gemini Live on Vertex AI (speech-to-speech).** Lower latency and a simpler pipeline with one service. Rejected for now: it has no phrase hints for order numbers, the transcription is internal to the model, and it fits the brief less directly. It is listed as a potential improvement, to compare latency against extraction accuracy.

## Consequences

- Latency adds up across three services and the tool-call rounds. Measure it with Pipecat metrics before tuning.
