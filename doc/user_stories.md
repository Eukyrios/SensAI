# User stories

Two user stories per feature, in the brief's format: **As a** type of user, **I want** an action,
**so that** an outcome — followed by acceptance criteria that name something observable in the
demo or in a test. Tick a criterion (`[x]`) once it can be shown.

Stories are written when a feature is specced, not at the end: they decide what "done" means.

| Code | Feature | Rating | Layer | Owner | Status |
|------|---------|--------|-------|-------|--------|
| [B1](#b1--functional-cli-chatbot) | Functional CLI chatbot | 👍 | 1–2 | A · D | ✅ built |
| [M1](#m1--session-persistence--profile) | Session persistence & profile | ★ | 1–2 | B | 🚧 storage built |
| [T5](#t5--structured-output) | Structured output | ★ | 3 | A | ⏳ |
| [M3](#m3--external-structured-memory) | External structured memory | ★★ | 3, 6 | B | ⏳ |
| [T1](#t1--tools-and-mcp) | Tools and MCP | ★★★ | 3, 5 | C | ⏳ |
| [EV3](#ev3--logging-and-monitoring) | Logging & monitoring | ★★ | 3–4 | D | ⏳ |
| [A2](#a2--multi-agent-orchestration) | Multi-agent orchestration | ★★★ | 4–5 | A | 🚧 contracts built |
| [A1](#a1--reasoning-loop-react) | Reasoning loop (ReAct) | ★★★ | 4 | C | ⏳ |
| [M2](#m2--token-budgeting--compression) | Token budgeting & compression | ★★ | 4–5 | B | ⏳ |
| [R1](#r1--basic-rag) | Basic RAG | ★★ | 5 | C | ⏳ |
| [EV1](#ev1--automated-eval--hallucination-detection) | Eval & hallucination detection | ★★★ | 5 | D | ⏳ |
| [R2](#r2--advanced-rag) | Advanced RAG | ★★★ | 6 | C | ⏳ |
| [EV2](#ev2--content--privacy-guardrails) | Content & privacy guardrails | ★★★ | 6 | D | ⏳ |
| [A5](#a5--persona) | Persona | ★ | 6 | A | ⏳ |
| [A4](#a4--prompt-versioning) | Prompt versioning | ★ | 6 | A | ⏳ |
| [X1](#x1--web-ui) | Web UI | ★ | 7 | A · D | ⏳ |
| [EV4](#ev4--adversarial-testing) | Adversarial testing | ★★ | 7 | C | ⏳ |
| [A6](#a6--semantic-cache) | Semantic cache (optional) | ★★ | 7 | B | ⏳ |

Owners: A = Core, B = Memory, C = Knowledge, D = Trust & surface (see the README's Team section).

---

## B1 — Functional CLI chatbot

### B1-1 · Follow-up questions in one conversation
**As a** beginner learner, **I want** to ask follow-up questions about the previous answer and see
each reply appear as it is written, **so that** I can dig into a grammar point without repeating
myself or waiting for the whole answer.

- [x] The reply is printed progressively, token by token, not as one block.
- [x] A follow-up such as "what did I just ask you?" is answered from the earlier turns.
- [x] `/reset` clears the history; the next question no longer knows the earlier turns.
- [x] Ctrl-C stops a reply mid-stream, the prompt comes back, and no half answer stays in the history.

**Verified by:** `tests/orchestration/test_chat.py`, `tests/surfaces/test_cli.py`; demo: two related questions, then `/reset`.

### B1-2 · Clear errors and a model I can choose
**As a** learner on a small laptop, **I want** to pick a lighter model and get clear messages when
something is wrong, **so that** I can fix the problem myself instead of facing a crash.

- [x] `--model <name>` at startup or `/model <name>` in the chat switches the model without editing code; the history is kept.
- [x] A model that isn't installed is reported before the prompt appears, with the exact `ollama pull` command.
- [x] When Ollama isn't running, Sensai says "Cannot reach Ollama" and exits cleanly.
- [x] An empty line is ignored, and an unknown `/command` is rejected instead of being sent to the model.

**Verified by:** `tests/surfaces/test_cli.py`, `tests/llm/test_ollama.py`; demo: `python -m sensai --model nope`.

---

## M1 — Session persistence & profile

### M1-1 · Pick up where I left off
**As a** returning learner, **I want** to resume my previous conversation after restarting Sensai,
**so that** I don't have to re-explain what we were working on.

- [ ] After quitting and restarting, the previous session's messages are reloaded from the database.
- [ ] A follow-up that refers to yesterday's lesson is answered correctly.
- [x] Every turn is stored with the question, the reply and its latency.

**Verified by:** `tests/memory/test_repository.py`; demo: quit, restart, ask "what were we studying?".

### M1-2 · A tutor that knows my level
**As a** learner preparing the JLPT N5, **I want** Sensai to remember my level and goals,
**so that** every answer is pitched at me without me repeating it each session.

- [ ] The profile (level, native language, goals) is stored once and injected automatically at the start of every session.
- [ ] Changing the level to N4 changes the difficulty of the examples in the next session.

**Verified by:** a test that the profile appears in the first context of a new session; demo: two sessions with different levels.

---

## T5 — Structured output

### T5-1 · Mistakes recorded in the same shape every time
**As a** learner, **I want** each correction to come as the same fields (my sentence, the error,
the correction, the rule), **so that** my mistakes can be stored and reviewed later.

- [ ] The corrector's output is validated against its JSON schema.
- [ ] An invalid output gets exactly one repair attempt with the validation error.
- [ ] If the repair also fails, an `Error` event appears in the trace and I still get a normal reply.

**Verified by:** tests with valid, repairable and broken model outputs.

### T5-2 · Agents that can't corrupt each other
**As a** developer of the tutor, **I want** every handoff between agents to be validated JSON,
**so that** one agent's malformed output never silently becomes another agent's input.

- [ ] A handoff payload that doesn't match the target agent's schema is rejected before the agent runs.
- [ ] The rejection is visible in the trace with the offending payload.

**Verified by:** orchestrator tests with a malformed handoff.

---

## M3 — External structured memory

### M3-1 · The tutor remembers my mistakes
**As a** learner, **I want** the tutor to remember the mistakes I make, **so that** it brings them
back until I've mastered them.

- [ ] Writing a wrong sentence adds a row to `errors` (sentence, correction, error type).
- [ ] In the next session, the tutor brings up that exact mistake.
- [ ] The agent writes these records itself through tools, not through hard-coded calls.

**Verified by:** ledger tests; demo: wrong sentence, restart, the tutor mentions it.

### M3-2 · Vocabulary reviewed right before I forget it
**As a** learner, **I want** new words to come back on a spaced-repetition schedule,
**so that** I review each word just before I would forget it.

- [ ] Words I learn become `srs_cards` with a next review date.
- [ ] Cards due today are proposed at the start of a session.
- [ ] Answering a card correctly pushes its next review further away; a wrong answer brings it closer.

**Verified by:** SRS scheduling tests with fixed dates.

---

## T1 — Tools and MCP

### T1-1 · Real dictionary answers, not guesses
**As a** learner, **I want** the tutor to look words, kanji and conjugations up with real tools,
**so that** readings, meanings and verb forms are correct.

- [ ] Asking the meaning of 食べる triggers a `dictionary` tool call, visible in the trace, and the answer uses its result.
- [ ] `conjugate` returns the requested form (e.g. the て-form) and `kanji` returns readings and stroke count.
- [ ] Tools work both with native tool calling and with the prompted fallback, chosen from `/api/show`.

**Verified by:** tool unit tests; demo: "what does 食べる mean, and its て-form?".

### T1-2 · Plug in an existing MCP server
**As a** developer of the tutor, **I want** to connect a pre-existing MCP server next to our own tools,
**so that** Sensai gains capabilities we didn't have to write.

- [ ] The external server's tools appear in the same tool list as our own.
- [ ] The model calls one of them during a turn and uses the result.
- [ ] If the server is unavailable, the turn continues without it and the trace says so.

**Verified by:** MCP client tests against a stub server; demo with the real server.

---

## EV3 — Logging and monitoring

### EV3-1 · Understand a bad answer after the fact
**As a** developer, **I want** every interaction logged — prompt, response, tools called, latency,
tokens — **so that** I can find out why an answer was wrong.

- [ ] Each event is written as one JSON line in `logs/sensai.jsonl`.
- [ ] All lines of a turn share its `turn_id`, and runs are linked by `parent_run_id`.
- [ ] The per-turn trace shows every agent run, tool call and handoff of that turn.

**Verified by:** logger tests; demo: `/trace` after a multi-agent turn.

### EV3-2 · Notice when a change makes things worse
**As** the team, **we want** a metrics view of latency, time to first token, tokens and error rate,
**so that** we notice when a prompt or model change degrades the tutor.

- [ ] The metrics view shows those numbers per turn and per agent, read from the logged rows.
- [ ] Failed turns are counted in the error rate.
- [ ] A change of model or prompt version is visible as a shift in the numbers.

**Verified by:** metrics tests on recorded rows; demo: the metrics view before and after `/model`.

---

## A2 — Multi-agent orchestration

### A2-1 · Precise corrections without breaking the lesson
**As a** beginner learner, **I want** my Japanese sentences checked by a dedicated corrector agent
while the sensei keeps teaching me, **so that** I get a precise correction without losing the flow
of the lesson.

- [ ] Sending a sentence with a mistake produces two agent runs in the trace: `sensei` and `corrector`, the corrector's `parent_run_id` pointing to the sensei's run.
- [ ] The handoff between them is a JSON payload (the sentence and the suspected error) shown in the trace, never free text.
- [ ] The reply shows the corrected sentence with its romaji and an explanation in English.

**Verified by:** orchestrator tests with fake agents; demo: "私は学校を行きます".

### A2-2 · Always an answer, even when an agent fails
**As a** learner, **I want** the tutor to keep answering even when one of its agents fails or loops,
**so that** a bug in one specialist never leaves me without a reply.

- [ ] An agent spec that hands work back and forth forever is stopped after the maximum number of hops, within the turn budget, and the trace shows which guard stopped it.
- [ ] When an agent times out or raises an error, the sensei answers instead and I see a normal reply.
- [ ] Both cases are covered by automated tests that run without a real model.

**Verified by:** guard tests with a looping and a failing agent; demo: the deliberate loop.

---

## A1 — Reasoning loop (ReAct)

### A1-1 · A correction that checks itself
**As a** learner, **I want** the corrector to verify its correction with a tool before answering,
**so that** I'm never taught a wrong correction.

- [ ] The trace shows the steps: think, tool call, verification, answer.
- [ ] When the verification contradicts the first idea, the final answer is revised.
- [ ] The loop stops after a fixed number of steps and works on a context of about 800 tokens.

**Verified by:** ReAct tests with scripted model outputs.

### A1-2 · Understand the rule, not just the fix
**As a** learner, **I want** to see why a sentence was wrong, **so that** I learn the rule
instead of memorising one fix.

- [ ] The answer names the grammar rule behind the correction.
- [ ] The reasoning steps can be expanded in the trace.

**Verified by:** demo: expand the corrector's run in `/trace`.

---

## M2 — Token budgeting & compression

### M2-1 · Long lessons that stay coherent
**As a** learner in an hour-long session, **I want** the tutor to still remember what we covered at
the start, **so that** long lessons don't lose their thread.

- [ ] When the context approaches the budget, the oldest turns are condensed into a summary.
- [ ] A question about a fact from the first turns is still answered correctly after compression.
- [ ] The summary is visible in the trace.

**Verified by:** compression tests with a long scripted session.

### M2-2 · Each agent gets only what it needs
**As a** developer, **I want** each agent to receive only the context it needs within a token budget,
**so that** small models stay fast and never overflow their `num_ctx`.

- [ ] Each context provider respects its `max_share` of the budget.
- [ ] The real `prompt_eval_count` never exceeds the agent's `num_ctx`.
- [ ] The token estimate is corrected with the real count, and the gap shows in the metrics.

**Verified by:** context-builder tests; metrics after a long session.

---

## R1 — Basic RAG

### R1-1 · Grammar answers with a source
**As a** learner, **I want** grammar explanations to cite where they come from, **so that** I can
trust them and read further.

- [ ] A grammar answer includes a citation (document and section) from the corpus.
- [ ] The retrieved chunks are visible in the trace.
- [ ] A question the corpus doesn't cover is answered without a fake citation.

**Verified by:** retrieval tests on a fixture corpus; demo: "when do I use は vs が?".

### R1-2 · Material at my level
**As a** learner preparing the N4, **I want** explanations drawn from material tagged with my JLPT
level, **so that** examples match what I'm studying.

- [ ] Every chunk carries its JLPT level and source as metadata.
- [ ] Filtering on N4 returns only N4 (or lower) chunks.
- [ ] A newly ingested document becomes searchable without code changes.

**Verified by:** ingest and filter tests.

---

## EV1 — Automated eval & hallucination detection

### EV1-1 · Warned when the tutor might be inventing
**As a** learner, **I want** answers checked against their sources before I rely on them,
**so that** I'm warned when the tutor may be making something up.

- [ ] Each claim of an answer is checked against the exact retrieved chunks.
- [ ] Unsupported claims are flagged visibly, never silently passed.
- [ ] A prepared demo case with an invented fact is caught live.

**Verified by:** judge tests with supported and unsupported claims.

### EV1-2 · Compare two versions objectively
**As** the team, **we want** a faithfulness and relevance score for every answer on a fixed test set,
**so that** we can tell whether a change made the tutor better or worse.

- [ ] Running the evaluation produces a score per answer and an average per system version.
- [ ] Two versions (e.g. two models) can be compared in one table.

**Verified by:** evaluation run on the bake-off test cases.

---

## R2 — Advanced RAG

### R2-1 · The best explanation first
**As a** learner, **I want** the most relevant explanation to reach the tutor, not a near-miss,
**so that** answers are precise.

- [ ] Retrieved chunks are reranked before entering the context.
- [ ] Chunks below the relevance threshold are dropped, and their scores show in the trace.
- [ ] On the same question, the reranked answer cites a better source than R1 did.

**Verified by:** reranking tests; demo: the same question with R1 and R2.

### R2-2 · No material above my level
**As** an N5 learner, **I want** advanced (N1) material kept out of my answers,
**so that** explanations stay within what I can understand.

- [ ] Chunks are filtered by the learner's JLPT level from the profile.
- [ ] Changing the profile level changes which chunks can be retrieved.

**Verified by:** filter tests with a mixed-level corpus.

---

## EV2 — Content & privacy guardrails

### EV2-1 · My personal data never reaches the model
**As a** learner, **I want** personal information such as my email or phone number masked before
the model sees it, **so that** practising with real sentences never leaks my data.

- [ ] An email or phone number in my message is masked before the model call; the trace shows the masked version.
- [ ] A `GuardrailTriggered` event records the rule and the action.
- [ ] Personal data in the model's reply is redacted before it is shown.

**Verified by:** guardrail tests with PII samples.

### EV2-2 · A tutor that stays a tutor
**As** the school, **we want** the tutor to refuse unsafe or out-of-scope requests,
**so that** it stays a Japanese teacher.

- [ ] An out-of-scope or unsafe prompt is refused before it reaches the model.
- [ ] Cheap checks run first (patterns, then embeddings); a model call is used only when they are unsure.
- [ ] The refusal is a clear message, not an error.

**Verified by:** guardrail tests; the adversarial suite (EV4).

---

## A5 — Persona

### A5-1 · Strict teacher or friendly partner, mid-lesson
**As a** learner, **I want** to switch between a strict teacher and a casual conversation partner
during a lesson, **so that** I can practise in the style I need without starting over.

- [ ] `/persona casual` switches the persona; the earlier messages stay in the context.
- [ ] The tone of the next reply changes.

**Verified by:** persona tests; demo: switch mid-conversation.

### A5-2 · New personas without code
**As a** teacher, **I want** to create a persona by adding a configuration file,
**so that** the same system can serve different classes without code changes.

- [ ] A new file in the personas folder appears in `/persona` without any code change.
- [ ] Its name, tone and scope are applied to the replies.

**Verified by:** a test loading a persona from a temporary folder.

---

## A4 — Prompt versioning

### A4-1 · Know which prompt produced an answer
**As** the team, **we want** every agent run to record the version of the prompt it used,
**so that** we can trace a bad answer back to the prompt that caused it.

- [ ] Prompts are stored by content hash; editing a prompt creates a new hash.
- [ ] Every `agent_runs` row stores the hash of its prompt.

**Verified by:** versioning tests.

### A4-2 · Compare and roll back prompts
**As** the team, **we want** to compare two prompt versions on quality and latency and roll back,
**so that** prompt changes are decisions backed by numbers.

- [ ] Metrics can be grouped by prompt hash.
- [ ] An agent can be pinned back to a previous prompt version by its hash.

**Verified by:** metrics grouped by version after two runs.

---

## X1 — Web UI

### X1-1 · Learn from the browser
**As a** learner who prefers a browser, **I want** a web page where replies stream as in the CLI,
**so that** I can study without a terminal.

- [ ] Messages stream token by token in the browser.
- [ ] A context bar shows the tokens used against the budget.
- [ ] The backend logic is unchanged: the web UI only subscribes to the event stream.

**Verified by:** SSE endpoint tests; demo in the browser.

### X1-2 · Show the agents at work
**As a** presenter at the keynote, **I want** the trace and an agent switcher in the browser,
**so that** the audience can see the agents collaborating.

- [ ] Each turn has a collapsible trace of its agent runs.
- [ ] Switching agent or model keeps the history.

**Verified by:** demo during the keynote.

---

## EV4 — Adversarial testing

### EV4-1 · Know where the guardrails break
**As** the team, **we want** an automated suite of jailbreaks and prompt injections,
**so that** we know where the guardrails hold and where they break.

- [ ] The suite runs with one command and reports each attack as blocked or passed through.
- [ ] The report is kept in the repository and discussed honestly in the keynote.

**Verified by:** the suite itself, run in CI or locally.

### EV4-2 · Hostile documents and broken input
**As a** learner, **I want** instructions hidden in documents and malformed input to have no effect,
**so that** the tutor behaves the same whatever it reads.

- [ ] Instructions hidden in a RAG document are ignored.
- [ ] Huge, empty, binary or badly encoded input never crashes the chat.

**Verified by:** adversarial tests with poisoned documents and malformed input.

---

## A6 — Semantic cache

### A6-1 · Instant answers to common questions
**As a** learner, **I want** common questions answered instantly, **so that** I don't wait for the
model on things it has already explained.

- [ ] A question similar to a cached one ("how do I say hello?" / "how to greet someone?") is answered from the cache without a model call.
- [ ] Cache hits show in the metrics with their latency.

**Verified by:** cache tests with paraphrases.

### A6-2 · Never the wrong cached answer
**As a** learner, **I want** the cache to answer only when the question really is the same,
**so that** I never get an answer to a different question.

- [ ] Questions below the similarity threshold always go to the model.
- [ ] Near-miss questions (e.g. "hello" vs "goodbye") are tested and never served from the cache.

**Verified by:** threshold tests with near-miss pairs.
