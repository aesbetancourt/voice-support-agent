# Specification: Acme Store Support Agent

This document compares the original case requirements with what the project delivers.
It has a completion checklist for every requirement, the functionality added beyond the brief,
how the plan changed during the work, and what is still open.

| | |
|---|---|
| **Status** | All core requirements met. Bonus features: 3 of 4 (RAG is not delivered, see [section 7](#7-not-delivered-rag-over-a-knowledge-base)). The demo video URL is still to add. Open items listed in [section 8](#8-open-items) |
| **Code** | `src/support_agent/`, hexagonal architecture ([diagram](diagrams/architecture.html)) |
| **Tests** | 127 automated tests, no network. 7 live simulated personas, passing in English and Spanish |
| **History** | 12 commits on `main` in a local Git repository. No remote yet |
| **Decisions** | [ADR-0001 to ADR-0009](adr/) |

---

## 1. Summary

The brief asked for a conversational agent for a fictional business. The agent interviews customers, extracts structured information, validates it, stores the conversation and its data as JSON, and writes a summary. It must use Python, an LLM, and a speech-to-text and text-to-speech platform.

The project delivers **Ava**, a voice customer-support agent for the fictional online shop *Acme Store*:

- **Ticket:** Ava collects an order number, a problem category, a description and an urgency level, and opens a validated support ticket.
- **Channels:** customers talk to Ava by voice in the browser, or by text in the terminal.
- **Stack:** everything runs on Google Cloud (Speech-to-Text, Gemini on Vertex AI, Text-to-Speech), orchestrated by Pipecat.
- **Bonus features from the brief:** frustration detection, conversation memory, and English and Spanish with a live switch. RAG is not delivered.
- **Extras beyond the brief:** a customer simulator for automated evaluation, and an architecture enforced by tests.

---

## 2. Requirement checklist

Legend: ✅ done · 🟡 done with a caveat · ⬜ not done

### 2.1 Conversational flow

| # | Requirement | Status | How it is met | Evidence |
|---|---|---|---|---|
| 1.1 | A bot that holds a natural conversation | ✅ | Ava, a support persona on Gemini, keeps replies short and asks one question at a time. Voice runs in the browser; text runs in the terminal | `application/prompts.py`, `adapters/inbound/voice.py`, `adapters/inbound/cli.py` |
| 1.2 | Collect specific information (example: order number, problem category, problem description, urgency) | ✅ | Exactly the four example fields, plus the customer name taken from the order. Six categories and three urgency levels | `domain/models.py` (`Ticket`, `Category`, `Urgency`) |
| 1.3 | Extract and validate the information into a structured format | ✅ | Gemini proposes values only through tool calls. Pydantic models and validators check every value, and the order must exist in the order database | `application/tools.py`, `domain/state.py`, `domain/validators.py` · [ADR-0001](adr/0001-code-owns-the-ticket-state.md) |
| 1.4 | Handle basic errors and invalid input gracefully | ✅ | Each rejected value returns a reason, and Ava asks again. Three failures on one field escalate to a human. Off-topic input is redirected, garbled transcripts are re-asked, API errors are retried, and auth problems print the command to fix them | README "Error handling" table · `tests/domain/test_state.py` |
| 1.5 | Keep the conversation context throughout the interaction | ✅ | The full history stays in the Pipecat context (voice) or in `GeminiChatModel` (text). Corrections at any point work, and a correction after the read-back resets the confirmation | `domain/state.py` (`_reopen_if_confirming`) · `tests/domain/test_state.py` |

### 2.2 Data extraction and storage

| # | Requirement | Status | How it is met | Evidence |
|---|---|---|---|---|
| 2.1 | Store conversations and extracted data as JSON | ✅ | One file per conversation in `conversations/`: transcript with timestamps, ticket, order snapshot, tool calls, validation events, per-turn sentiment, language, status and summary. Saved after every turn, with atomic writes | `adapters/outbound/json_store.py`, `application/recorder.py` · [sample](../examples/sample_conversation.json) |
| 2.2 | Basic data validation for the collected information | ✅ | Order-number format and existence; category and urgency enums with aliases; description length; spoken forms in English and Spanish | `domain/validators.py` · `tests/domain/test_validators.py` |
| 2.3 | A summary of the conversation with its key points | ✅ | Gemini structured output: headline, key points, customer sentiment and next action, always in English for the support team. A rule-based fallback is used if the call fails | `application/summary.py`, `adapters/outbound/gemini.py` |

### 2.3 Technical requirements

| # | Requirement | Status | How it is met | Evidence |
|---|---|---|---|---|
| 3.1 | Use Python | ✅ | Python 3.12, managed with `uv`. Ruff for lint and format | `pyproject.toml` |
| 3.2 | Proper error handling | ✅ | Typed errors at each boundary: `ConfigError`, `GoogleAuthError`, `genai` `APIError`, `SessionClosedError`, timeouts on sentiment, fallback summary, partial saves on disconnect | README "Error handling" table |
| 3.3 | An LLM chosen for the use case | ✅ | Gemini 3.5 Flash on Vertex AI for conversation turns and summaries, chosen for latency (see section 4). Gemini 3.5 Flash-Lite for per-turn sentiment | [ADR-0002](adr/0002-google-cloud-only-stack.md) |
| 3.4 | A conversational AI platform or models for transcription and TTS | ✅ | Pipecat 1.15 with Google Cloud Speech-to-Text (`chirp_3`, detecting English or Spanish per utterance) and Text-to-Speech (Chirp 3 HD voices) | [ADR-0003](adr/0003-cascaded-voice-pipeline.md), [ADR-0004](adr/0004-pipecat-single-process-webrtc.md), [ADR-0008](adr/0008-language-fixed-per-conversation.md) |

### 2.4 Deliverables

| # | Deliverable | Status | How it is met | Evidence |
|---|---|---|---|---|
| 4.1 | Source code in a Git repository | 🟡 | A local Git repository with 12 commits. **No remote is created or pushed yet** | `git log` |
| 4.2 | README: setup instructions | ✅ | Prerequisites, `uv sync`, Google Cloud setup (APIs, ADC, quota project), the `.env` file, and the run commands | [README → Setup](../README.md#setup) |
| 4.3 | README: system architecture overview | ✅ | Hexagonal view, voice runtime view, state machine, tools and project structure | [README → Architecture](../README.md#architecture) · [diagram](diagrams/architecture.html) |
| 4.4 | README: key design decisions | ✅ | A decision table in the README, plus 8 ADRs that record the rejected alternatives | [README → Key design decisions](../README.md#key-design-decisions) · [ADRs](adr/) |
| 4.5 | README: potential improvements | ✅ | A table by area: speech, sentiment, latency, knowledge (RAG), identity, conversation, languages, data, evaluation, production, security | [README → Potential improvements](../README.md#potential-improvements) |
| 4.6 | URL to a demo video of sample conversations | ⬜ | To record. The script covers a full ticket, an invalid order number, frustration with escalation, and a live switch to Spanish | — |

**Result: 16 of 18 items done. Item 4.1 has a caveat: the repository exists, but it is not pushed to a remote. Item 4.6, the demo video URL, is still to add.**

### 2.5 Bonus features (optional)

| # | Bonus feature | Status | How it is met | Evidence |
|---|---|---|---|---|
| B.1 | RAG using a provided knowledge base (example: common customer issues) | ⬜ | Not part of this delivery. It needs cloud infrastructure and curated content outside the code. It is listed as a potential improvement, with its design | [Section 7](#7-not-delivered-rag-over-a-knowledge-base) |
| B.2 | Sentiment analysis to detect customer frustration | ✅ | Gemini Flash-Lite rates every customer turn in the background. A domain rule makes Ava acknowledge frustration, then offer a human once | [Feature doc](features/frustration-detection.md) · [ADR-0007](adr/0007-frustration-detection.md) |
| B.3 | Multi-turn conversation memory | ✅ | The full history stays in context for the whole call. Ava keeps every collected field across turns, accepts corrections at any point, and reads the ticket back before submitting. Memory **across** calls is not built (section 8) | `domain/state.py` · `tests/domain/test_state.py` |
| B.4 | Support for multiple languages | ✅ | English and Spanish. Voice calls detect the language per utterance and switch the voice and replies live | [Feature doc](features/multi-language.md) · [ADR-0009](adr/0009-detect-the-language-in-voice-calls.md) |

**Bonus result: 3 of 4.**

---

## 3. Functionality beyond the brief

The bonus features are in section 2.5. This table lists what the brief did not ask for.

| Feature | What it adds | Where it is documented |
|---|---|---|
| **Order lookup against data** | The order number must exist in a mock order database (10 orders). Ava uses the order status and items, and warns when the category does not fit the status (for example, a defect on an undelivered order) | README · `domain/state.py` |
| **Read-back and confirmation** | Ava reads the ticket back before she submits it. The code refuses to submit without a confirmed read-back | [ADR-0001](adr/0001-code-owns-the-ticket-state.md) |
| **Fixed closing messages** | The ticket ID and the closing words come from code, so the LLM cannot invent a reference number | [ADR-0001](adr/0001-code-owns-the-ticket-state.md) |
| **Escalation and end paths** | Transfer to a human on request, on a safety issue, or after three failures. The customer can also leave without a ticket | `application/tools.py` |
| **Text channel** | The same agent in the terminal, without audio, for fast and cheap testing | README → Run |
| **Customer simulator** | Gemini plays 7 difficult customer personas against the real agent. Each persona has an expected outcome, and the command exits non-zero on failure | README → Testing |
| **Hexagonal architecture, enforced** | Domain, application with ports, inbound and outbound adapters. A test fails if any layer imports outward | [ADR-0005](adr/0005-hexagonal-architecture.md) · `tests/test_architecture.py` |
| **Documentation set** | A system guide, 9 ADRs, 2 feature documents, an architecture diagram, and this specification | [`docs/`](.) |

---

## 4. How the plan changed

The requirements stayed the same. The way to meet them changed as we found new facts and made decisions.

| Topic | First plan | Final result | Why it changed | Record |
|---|---|---|---|---|
| Use case | Support, scheduling or claims intake | E-commerce support with order lookup | Maps one to one to the brief, and adds validation against data | — |
| LLM | Claude (only an Anthropic key was available) | Gemini on Vertex AI | GCP access became available, and the goal became one platform | [ADR-0002](adr/0002-google-cloud-only-stack.md) |
| STT / TTS | Local MLX Whisper and Kokoro | Google Cloud STT and TTS only, with no fallback | Owner decision: "focus only on GCP" | [ADR-0002](adr/0002-google-cloud-only-stack.md) |
| Voice pipeline | Cascaded, or Gemini Live | Cascaded STT → LLM → TTS | Matches the brief, and phrase hints help order numbers | [ADR-0003](adr/0003-cascaded-voice-pipeline.md) |
| Frontend | A custom frontend was considered | Pipecat prebuilt UI in one Python process | One command to run, with no frontend build | [ADR-0004](adr/0004-pipecat-single-process-webrtc.md) |
| Code structure | Flat package → three layers | Hexagonal (ports and adapters) | Owner preference; explicit boundaries | [ADR-0005](adr/0005-hexagonal-architecture.md) |
| Conversation control | Pipecat Flows was evaluated | Our own state machine | Flows works only inside Pipecat, so the text channel could not share it | [ADR-0006](adr/0006-own-state-machine-not-pipecat-flows.md) |
| Main model | `gemini-3.8-flash` (Pipecat's default) | `gemini-3.5-flash` | Voice replies were slow. Measured per model round: 2 to 20 s for 3.8 Flash, about 0.8 s for 3.5 Flash. 3.5 Flash passed 7/7 personas in English and Spanish | README → Known limitations |
| Language choice | Fixed per conversation, with `--language` | Detected per utterance in voice calls, switched live | The prebuilt UI cannot pick a language, so a demo needed a restart per language. Chirp 3 detects `en-US` and `es-US` while streaming | [ADR-0009](adr/0009-detect-the-language-in-voice-calls.md) |
| Spanish STT | Same model as English (`latest_long`) | `chirp_3` for Spanish, later for both languages (ADR-0009) | `latest_long` misheard fluent Spanish order numbers in a measurement | [ADR-0008](adr/0008-language-fixed-per-conversation.md) |

### Bugs found by live tests and fixed

| Found during | Problem | Fix |
|---|---|---|
| First voice call | The closing message sometimes missed the transcript, because the call ended before the transcript event | The tool handler records it, and the recorder skips duplicates |
| GCP setup | Text-to-Speech needs a quota project with user ADC credentials | The setup steps include `set-quota-project`. The CLI explains auth errors |
| Spanish run | Gemini used the informal "tú", but the fixed messages use "usted" | The prompt requires "usted", and a test checks it |
| Spanish speech check | Pipecat's Google TTS defaults its language code to `en-US` | Each voice profile sets both the voice and the language code |
| Spanish voice call | Gemini 3.5 Flash once started a reply with a literal `<thought>` tag, which reached the chat and the voice | `ThoughtTagStripper` removes the tags from streamed voice text and from text-channel replies |

---

## 5. The system as built

### Ticket fields

| Field | Values | Validation |
|---|---|---|
| `order_number` | `ORD-` + 6 digits | Spoken forms in English and Spanish are normalized, and the order must exist |
| `category` | `shipping`, `billing`, `product_defect`, `return_refund`, `account`, `other` | Enum, with English and Spanish aliases |
| `description` | Free text | 10 to 1000 characters, whitespace cleaned |
| `urgency` | `low`, `medium`, `high` | Enum, with aliases. Ava infers it, then confirms it |

### Conversation stages

`collecting → confirming → completed`, or `escalated` / `ended`. A dropped call is saved as `abandoned`.
Details: [README → State machine](../README.md#state-machine).

### Commands

| Command | Purpose |
|---|---|
| `uv run support-agent voice [--language es]` | Voice server, then open <http://localhost:7860> |
| `uv run support-agent chat [--language es]` | Text chat in the terminal |
| `uv run support-agent simulate [--persona NAME] [--language es]` | Simulated customers, with PASS/FAIL results |
| `uv run pytest` | Automated tests |

---

## 6. Verification

| Check | Result | Scope |
|---|---|---|
| Automated tests | 127 pass in under 1 second | Domain, application, adapters (with fakes), architecture rule |
| Lint | `ruff check` clean | All of `src/` and `tests/` |
| Simulated customers, English | 7/7 pass, live Gemini | Text channel, end to end |
| Simulated customers, Spanish | 7/7 pass, live Gemini | Text channel, end to end |
| Frustration calibration | Calm 0.0, repetition 0.6, anger 0.8 | Live Flash-Lite on sample messages |
| Gemini latency per model round | 3.8 Flash: 2 to 20 s, unstable. 3.5 Flash: about 0.8 s | One tool turn and one reply turn, `global` endpoint, 3 runs each |
| Language detection, streaming | Chirp 3 with `en-US` + `es-US` detected both samples, with phrase hints | TTS audio through Pipecat's Google STT and the switcher, no microphone |
| Spanish speech round trip | `chirp_3` streaming transcribed "ORD104233" correctly | TTS audio sent to STT, no microphone |
| Voice call, English | Completed with a ticket, heard by the owner | Real microphone and browser |
| Voice call, Spanish | Server starts. **Not yet tried with a microphone** | Startup only |

---

## 7. Not delivered: RAG over a knowledge base

**Bonus feature (optional):** "Implement RAG using a provided knowledge base. For example, for a customer support agent, knowledge of common customer issues."

**Status:** ⬜ Not delivered. The brief lists it as optional. This was a scope decision: RAG is a separate piece of work with its own infrastructure, content and evaluation, not one more feature in the code.

### Why it is not part of this delivery

| Reason | Detail |
|---|---|
| **It needs infrastructure outside the code** | On the Google-only stack ([ADR-0002](adr/0002-google-cloud-only-stack.md)), RAG means a Vertex AI RAG Engine corpus (or a Vertex AI Search data store), a Cloud Storage bucket for the documents, an embedding model, IAM roles and an import job. That is cloud provisioning and running cost, not a change in the repository |
| **There is no knowledge base content** | Retrieval is only as good as its documents. The project has no curated articles about common issues (late deliveries, returns, refunds, defects). Invented articles would show the plumbing, but not the value |
| **It changes Ava's role** | Today Ava does intake: she collects and validates a ticket. With RAG she also answers questions, and each answer is a claim about store policy. That needs guardrails: answer only from retrieved text, and say so when no article matches |
| **It needs its own evaluation** | The simulator checks ticket outcomes. RAG also needs retrieval tests (the right article for a question) and grounding tests (no invented policy). A wrong refund policy spoken by a voice agent is worse than no answer |
| **Voice latency** | Each retrieval adds a network round trip to the turn. Only the model latency is measured so far, not the full voice turn ([section 8](#8-open-items)), so this cost cannot be judged yet |

### Potential improvement: how to add it

RAG is the first candidate for the next iteration, and the README lists it under [Potential improvements](../README.md#potential-improvements).
The hexagonal architecture has a clear place for RAG, and the domain does not change.
Retrieval would be a tool behind a port, the same way as `lookup_order`, so the voice and text channels share it.
Gemini's built-in retrieval grounding was not chosen for this design, because it would bypass the port and the tool-call record.

| Layer | Addition |
|---|---|
| Application | A `KnowledgeBase` port (`search(query, language)` returns articles) and a seventh tool, `search_help_articles` |
| Outbound adapters | `VertexRagKnowledgeBase` for Vertex AI RAG Engine. A local `JsonKnowledgeBase` with keyword search for tests and offline runs |
| Prompt | Use the tool for policy and "how do I" questions. Answer only from the returned articles |
| Data | `data/knowledge/`: articles about common issues and policies, in English and Spanish |
| Tests | A fake port in unit tests. New simulator personas that ask policy questions and expect specific facts |

Steps to add it:

1. Write the knowledge base articles.
2. Add the port, the tool and the local adapter, with tests.
3. Create the RAG Engine corpus and import the articles.
4. Add the Vertex adapter, then measure the added latency.

---

## 8. Open items

| Item | State | Next step |
|---|---|---|
| Git remote | ⬜ Not created | Create the remote repository and push `main` |
| Case brief in history | 🟡 The first commit still contains the brief, which is now ignored in `docs/private/` | Rewrite history before the first push, if the brief must not be public |
| Live language switch with a microphone | ⬜ Tested with synthesized speech through the Pipecat pipeline only | Start `support-agent voice`, speak English, then switch to Spanish and complete a call |
| RAG over a knowledge base | ⬜ Not delivered. Planned as a potential improvement | Follow the steps in [section 7](#7-not-delivered-rag-over-a-knowledge-base) |
| Demo video URL | ⬜ To record | Record the video, upload it, and add the URL to the README |
| Memory across calls | ⬜ Explored, then deferred | Needs a decision on identity verification before history is shown (see README → Potential improvements) |
| Latency measurement | 🟡 Model latency measured and fixed by the switch to 3.5 Flash. The full voice turn is not measured | Read the Pipecat TTFB metrics per stage (STT, turn detection, LLM, TTS) from a real call |
| Sample record | 🟡 `examples/sample_conversation.json` predates the `language` and `sentiment` fields | Regenerate it from a current run |
