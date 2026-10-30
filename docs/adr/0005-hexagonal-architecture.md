# Organize the code as hexagonal architecture (ports and adapters)

The package is split into `domain/` (pure rules), `application/` (use cases and `ports.py`), and `adapters/inbound` and `adapters/outbound`. `bootstrap.py` is the composition root. Dependencies point inward only, and `tests/test_architecture.py` fails if a layer imports outward or if the core imports `pipecat` or `google`. This keeps the business rules testable without audio or network. It also lets text, voice and simulations share one core, and it limits any vendor change to the adapters (see ADR-0002).

## Considered Options

- **A flat package.** This was the first version. Rejected: the layers were not visible, and SDK code sat next to the rules.
- **Three folders (domain, services, channels).** The second version. Replaced, because adapters still lived inside the core (JSON file access in `domain/`, Gemini code in `services/`) and the ports were not explicit.

## Consequences

- **The domain receives a lookup function, not a repository.** `TicketSession` takes `find_order: Callable[[str], Order | None]` instead of an `OrderRepository` port. The ports live in the application layer, and the domain must not import them. This is deliberate, so do not "fix" it by importing the port.
- The text channel uses the `google-genai` SDK through the `GeminiChatModel` adapter, while voice uses Pipecat's own LLM service. Both call the same `ToolExecutor`.
- Tests mirror the layers (`tests/domain`, `tests/application`, `tests/adapters`).
