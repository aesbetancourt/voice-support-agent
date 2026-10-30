# Keep our own state machine instead of Pipecat Flows

Pipecat Flows, now part of Pipecat core as `pipecat.flows`, models a conversation as nodes. Each node has its own prompt and tools, and a function can return the next node. It fits this use case well. We did not adopt it, because `FlowManager` runs only inside a Pipecat pipeline: the text channel and the simulator could not use it, so the shared core would split in two (see ADR-0005).

## Considered Options

- **Pipecat Flows (`FlowManager`, `NodeConfig`).** Rejected for the reason above. Choose it if the product will only ever be voice.
- **Stage-scoped tools in our own domain.** Each stage declares its instructions and its allowed tools, so `finalize_ticket` does not exist while collecting. This is the Flows idea without the dependency. It is deferred: with four fields, the state machine already refuses invalid calls, so the gain is small. Revisit it when the flow grows, for example with identity verification or several problem types.
- **Multi-agent handoffs (OpenAI Agents SDK, Google ADK).** Rejected: there is only one intent. Revisit it if billing, returns or other intents need their own agents.
