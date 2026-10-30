# Detect frustration with a background Gemini Flash-Lite classifier and a domain policy

Each customer turn is rated for frustration (0.0–1.0) by a small, fast Gemini model (`gemini-3.5-flash-lite`) through a new `SentimentAnalyzer` port. A deterministic `FrustrationPolicy` in the domain decides the reaction, following ADR-0001: the model measures, the code decides. The first high reading (≥ 0.7) makes Ava acknowledge the frustration. Two high readings in a row make her offer a human agent, once per conversation. The classifier runs in the background, so it never delays the current reply, and its guidance applies from the next turn.

## Considered Options

- **A. Gemini Flash-Lite classifier per turn.** Chosen. It understands context, sarcasm and repetition ("I already told you twice" scored 0.6), it stays on Google Cloud (ADR-0002), and it runs off the critical path.
- **B. Cloud Natural Language API (`analyzeSentiment`).** Rejected. It is cheap and fast, but it measures general polarity, not frustration: impatient but polite messages read as neutral.
- **C. A `report_sentiment` tool for the main LLM.** Rejected. It adds a tool round to every turn, which increases latency, and it lets the conversation model grade itself.
- **D. Acoustic analysis (tone of voice).** Rejected for now. It catches what text misses, but it needs a separate audio model and works only in the voice channel. It is listed as a potential improvement.

## Consequences

- **One extra LLM call per customer turn.** It takes about 1 second, runs in parallel with the reply, and has a 5-second timeout. If it fails or times out, the turn has no reading and the call continues normally.
- **The guidance is one turn late.** A sudden angry turn is acknowledged in the reply after the next customer turn, or immediately if the prompt rule ("if the customer is very upset, escalate") applies. This trade-off keeps the reply latency unchanged.
- **The guidance is an instruction, not customer speech.** Voice adds it to the context as a `developer` message, with `run_llm=False`. Text sends it as a marked note before the customer's words.
- **The offer is a suggestion, not an automatic transfer.** The customer decides, and `escalate_to_human` runs only if they accept.
- **Readings are saved on each transcript turn** (`sentiment.frustration`, `reason`, `action`). The summary input includes them, and the fallback summary derives `customer_sentiment` from the peak reading (≥ 0.85 angry, ≥ 0.5 frustrated).
- **The thresholds are a calibration choice**, made from a live sample: calm 0.0, repeating oneself 0.6, angry 0.8. Change them in `domain/frustration.py`, and update its tests.
