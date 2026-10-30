# Code owns the ticket state; the LLM only proposes values through tools

An LLM is good at conversation but unreliable with facts and procedure: it can invent an order number, skip a question, or submit before the customer confirms. So Gemini never writes the ticket directly. It can only call tools, and a deterministic state machine (`TicketSession`) validates every value and decides the stage (`collecting → confirming → completed`, or `escalated`/`ended`). A ticket is submitted only after a read-back that the customer confirms, a correction after the read-back cancels the confirmation, and three failures on one field escalate to a human.

## Considered Options

- **The LLM drives the flow and outputs the ticket at the end.** Rejected: completeness and confirmation would depend on the prompt, so they could not be tested.
- **A separate extraction pass after each turn.** Rejected: an extra LLM call per turn adds latency, which hurts voice, and the extractor can still disagree with the conversation.

## Consequences

- The closing message, which includes the ticket ID, is fixed text from code. After a closing tool call, the LLM does not run again, so it cannot invent or change a reference number.
- Validation errors go back to the LLM as short reasons, so it can explain the problem and ask again.
- Spoken-form normalization of order numbers ("o r d one oh four…") happens in code, not in the LLM.
