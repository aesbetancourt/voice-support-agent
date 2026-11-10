# CLAUDE.md

Instructions for coding agents in this repository: a voice recruiting agent that runs candidate
screening calls (Pipecat, Google Cloud Speech-to-Text and Text-to-Speech, Gemini and embeddings on
Vertex AI). Read [`docs/how-it-works.md`](docs/how-it-works.md) for the system, and the matching
[`docs/adr/`](docs/adr/) record before you change a decision it covers.

## Gates

A change is done when all three pass. They need no network:

```bash
uv run ruff format src tests
uv run ruff check src tests
uv run pytest
```

A change to prompts, tools or the agent loop also needs a live run against Gemini, with a configured
`.env`: `uv run screening-agent simulate` (all eight personas pass, status and LLM judge), plus `--language es`.

## Architecture rules

The code is hexagonal. Imports point inward only: `adapters` → `application` → `domain`.
`tests/test_architecture.py` enforces it.

- **domain/** holds pure rules: models, validators, the `ScreeningSession` state machine, `FrustrationPolicy`.
  It imports nothing from the project and no SDK.
- **application/** holds use cases and the ports (`ports.py`). It imports the domain and its own ports.
- **adapters/** hold every SDK: Pipecat, Google and Deepgram in `inbound/voice.py`, `google-genai` in `outbound/gemini.py`
  and `outbound/embeddings_knowledge.py`.
- **bootstrap.py** builds each conversation: it wires the outbound adapters to the ports.
  The voice adapter also builds Pipecat's Google services, which call Google directly (ADR-0004).

A new external system gets a port in `application/ports.py` and an adapter in `adapters/outbound/`.

## Invariants

These are the design. Keep them true in every change:

- **The code owns the outcome.** Gemini proposes values only through the tools in `application/tools.py`.
  `ScreeningSession` validates every value, decides the stage and calculates the recommendation.
  A new rule goes in the domain, with a test.
- **Closing messages are fixed text** from `application/messages.py`, with the booking time and reference from code.
  Every language has every message.
- **Optional features never break a call.** Sentiment, the summary, the job search, the frustration display and
  the language display fail to a safe default. Each broad `except Exception` carries `# noqa: BLE001 - <reason>`.
- **The internal language is English**: tool results, enum values, frustration reasons and the summary.
  Only what the candidate hears is translated. Evidence quotes keep the candidate's own words.
- **Ava never says an assessment.** No score, recommendation or requirement result reaches the candidate.
  She never asks about protected topics (age, nationality, health, family).
- **Fakes, not network, in tests.** Unit tests use the fakes in `tests/`, so `pytest` runs offline.

## Python conventions

- Python 3.12. Annotate every function signature in `src/` (ruff `ANN` enforces it). Use built-in
  generics (`list[str]`) and `X | None`.
- Use `enum.StrEnum` for string enums. Use Pydantic models for data that is validated or saved
  (`domain/models.py`), and frozen dataclasses for plain values, such as the port types.
- Network calls are async. Give every optional call a timeout: `asyncio.wait_for` for sentiment and the job
  search, and the client timeout for Gemini. The JSON store writes small files synchronously, after each turn.
- Log with `loguru`'s `logger`. The CLI is the only module that prints.
- Re-raise with `from exc` to keep the cause.
- No magic values in adapters. Every tunable value (model, voice, location, timeout, threshold) goes in
  `config/settings.toml`, with a one-line reason, and reaches the adapter through `Settings`. Secrets
  and per-machine choices go in `.env`; add each new variable to `ENV_OVERRIDES` in `config.py`.
- Fixed protocol strings (Pipecat roles, RTVI message types) go in `adapters/constants.py`.
- The domain and application layers cannot import `config`. Their business rules and limits stay as
  named constants next to the code that uses them, with a one-line reason.
- A new speech provider gets a settings section, a builder in `adapters/inbound/speech.py`, and an entry
  in its builder table.
- Write comments that explain why. Keep them to one or two lines.

## Docs and commits

- Write docs, comments and commit messages in Simplified Technical English: short sentences, active voice.
- Record a decision that is hard to reverse as a new ADR in `docs/adr/`. Update the README, the feature
  docs in `docs/features/` and `docs/how-it-works.md` when the behaviour they describe changes.
- Commit messages: an imperative subject under 72 characters, then a short body that says why.
- Commit only what Git tracks on purpose. `.env`, `conversations/` and `docs/private/` stay local.
