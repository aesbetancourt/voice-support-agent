# Change the domain from customer support to candidate screening

The first version was a support agent: it collected an order number, a category, a description and an urgency, and opened a ticket. Review feedback said that this is too simple: the agent fills a form, and it has no room to do real work. The agent also told the customer the urgency, which a customer does not want to hear.

The agent is now **Ava, a recruiting assistant** who runs a first screening call for a warehouse operator role at the fictional company Harbor Logistics. The brief asks for an agent that "conducts short interviews", so a screening call is a direct fit. The call has real work in it:

- **Must-haves** (work permit, shifts, start date, commute) are domain rules. A failed answer must be read back and confirmed before it closes the call, because speech recognition can mishear a "yes" or a date.
- **Experience questions** get a follow-up when the answer is vague. Gemini rates each answer, and the domain checks that the evidence quote is in the candidate's own words, so a score cannot rest on an invented quote.
- **Job questions** get answers from the job information (ADR-0011).
- **The interview** is booked in a free slot, and the closing message says the time from code.

The architecture stays the same (ADR-0001, ADR-0005): Gemini proposes values through tools, and `ScreeningSession` validates them and decides the stage.

## Rules that come from the hiring domain

- **Ava never says a score, an assessment, or whether an answer meets a requirement.** The fixed closing message gives the outcome. This applies the lesson from the urgency feedback.
- **The code calculates the recommendation** (advance, review, reject) from fixed rules. A person makes the decision, and the greeting says so. Hiring AI is high-risk under the EU AI Act, so the agent informs a human and does not decide.
- **Competency scores do not block the booking.** Only objective must-haves end an application. Every eligible candidate books an interview, and the recruiter sees the scores.
- **Ava never asks about protected topics** (age, nationality, health, family). The prompt forbids it, and the simulator's LLM judge checks it.

## Considered Options

- **Keep support, and let Ava solve problems** (refunds, reschedules). Rejected: it improves the agent, but the domain stays "problem in, ticket out".
- **Candidate screening.** Chosen, for the reasons above.
- **Scheduling only.** Rejected: it collects a time, which is another form.
