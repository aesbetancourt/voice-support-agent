# Answer job questions with embeddings retrieval behind a port

Candidates ask about pay, shifts, location and benefits during a screening call. Ava must answer from the company's job information, and must never invent a detail. A question with no answer goes to the recruiter and is listed on the scorecard.

The `JobKnowledge` port has one method, `search(question, limit)`. The adapter `GeminiJobKnowledge` embeds each piece of job information once per process with `gemini-embedding-001`, and ranks the pieces by cosine similarity in memory. The tool `search_job_info` calls the port. A result below the score threshold counts as "no answer", and the question goes on the scorecard for the recruiter.

## Measured threshold

On English questions, on-topic questions scored 0.699 or more and off-topic questions ("can I bring my dog?") scored 0.662 or less. The threshold is 0.68. Gemini sends the question in English (the tool description asks for it), so a Spanish question gets the same scores. `RETRIEVAL_QUERY` separated the two groups better than `QUESTION_ANSWERING`.

## Consequences

- **Latency:** each question adds one tool round trip, about 0.5 to 1.5 s of silence in a voice call. The index is built when the call starts, so the first question does not also wait for it.
- **Failure:** a search error or a timeout (4 s) counts as "no answer". The call continues, and the recruiter answers the question.
- **Scale:** the index lives in memory. It suits one role with a few dozen pieces of information. For many roles or long documents, the adapter can move to a vector store (for example Vertex AI Vector Search) with no change to the port.

## Considered Options

- **Job facts in the system prompt.** No latency and no new code. Rejected for this version, so that the design shows a retrieval port that scales to more roles. It stays a good choice for a single short job description.
- **Keyword search.** Rejected: it misses synonyms ("salary" and "pay") and cross-language questions.
- **Managed search (Vertex AI Search).** Rejected: a new cloud service and its setup, for a few dozen pieces of text.
