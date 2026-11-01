# Features and how they are built

Each feature of the features catalogue that Sensai implements, in the
catalogue's order and categories: what the catalogue asks, what we built, how it works in the code,
the choices we made, where it is checked, and what it does not do yet. The last section lists the
features we did not take, and why.

Every surface (CLI, JSONL log, `/trace`, web UI) subscribes to one event stream
(`sensai/core/events.py`); most features below are a step of that stream or something that reads it.

Owners: A = Core (Eukyrios), B = Memory (EvansRamin), C = Knowledge (DanielLoww), D = Trust & surface
(NicolasGonzalz).

| Category | Code | Feature | Rating | Owner |
|---|---|---|---|---|
| Base | B1 | Functional CLI chatbot | base | A, D |
| Memory & Context | M1 | Session Persistence & Profiles | 1 star | B |
| Memory & Context | M2 | Token Budgeting & Semantic Compression | 2 stars | B |
| Memory & Context | M3 | External Structured Memory | 2 stars | B |
| Tools | T1 | MCP | 3 stars | C |
| Tools | T5 | Structured Output | 1 star | A |
| RAG | R1 | Basic RAG | 2 stars | C |
| RAG | R2 | Advanced RAG | 3 stars | C |
| Orchestration & Reasoning | A1 | Reasoning Loops (ReAct) | 3 stars | C |
| Orchestration & Reasoning | A2 | Multi-Agent Orchestration | 3 stars | A |
| Orchestration & Reasoning | A4 | Prompt Versioning | 1 star | A |
| Orchestration & Reasoning | A5 | Persona | 1 star | A |
| Evaluation | EV1 | Automated Eval & Hallucination Detection | 3 stars | D |
| Evaluation | EV2 | Content & Privacy Guardrails | 3 stars | D |
| Evaluation | EV3 | Logging & Monitoring Dashboard | 2 stars | D |
| Evaluation | EV4 | Adversarial Testing | 2 stars | C |
| UX & Lifecycle | X1 | Web UI | 1 star | A, D |

---

## Base

### B1 - Functional CLI chatbot

**The catalogue asks:** a command-line chatbot on a local Ollama model, with streaming and a
conversation history within a session.

**What we built:** `python -m sensai` opens a chat where replies stream token by token, follow-up
questions use the history, Escape stops a reply mid-stream, the next question can be typed while a reply
is still streaming, and slash commands (`/model`, `/persona`, `/think`, `/metrics`, `/trace`, `/reset`)
change the session without leaving it.

**How it works:**
- `sensai/llm/ollama.py` talks to Ollama's HTTP API directly (`/api/chat` streaming, `/api/embed`,
  `/api/tags`, `/api/show`); no LLM framework, as the subject requires.
- `sensai/orchestration/stream.py` runs the blocking HTTP stream in a thread and hands each piece to
  asyncio, so a reply can be cancelled without freezing the input line.
- `sensai/orchestration/turn.py` (`run_turn`) turns one learner message into a stream of events
  (`TurnStarted`, `Token`, ..., `TurnFinished`); `sensai/surfaces/cli.py` renders them. The live input
  line is `prompt_toolkit`; when stdin is not a terminal the CLI falls back to a plain loop, which is
  what scripts and the feature video use.
- Configuration is layered: defaults < `config.toml` < `SENSAI_SECTION__KEY` environment variables <
  command-line arguments (`sensai/core/config.py`).

**Choices:** direct HTTP instead of the `ollama` Python package, to keep full control of streaming,
reasoning tokens and tool calls; one event stream as the only interface between the turn and the
surfaces, so the web UI and the log needed no change to the turn logic.

**Checked by:** `tests/llm/test_ollama.py`, `tests/orchestration/test_turn.py`, `tests/surfaces/test_cli.py`.

**Limits:** the first turn after Ollama loads the model waits for the load (about 10 s on the reference
laptop).

---

## Memory & Context

### M1 - Session Persistence & Profiles

**The catalogue asks:** save and reload message history across sessions, plus a persistent user profile
injected automatically.

**What we built:** every message is stored in SQLite; `--list-sessions` lists them and
`--session <id>` resumes one with its history. A learner profile (name, level, goals) is injected into
every prompt without being part of the conversation.

**How it works:**
- `sensai/memory/repository.py` stores sessions, messages, turns and agent runs; migrations live in
  `sensai/memory/migrations/` and are applied by `migrate.py` (tracked by `PRAGMA user_version`).
- Messages form a tree (`parent_id`), and each session points to its active leaf, so a rolled-back
  question (failed turn) is removed without breaking the chain.
- The profile is a row of `learner_profiles`; `build_context()` (`sensai/orchestration/context.py`) adds
  it as a system-prompt block with its own share of the budget.

**Choices:** SQLite over JSON files, because the same database later holds the ledger (M3), the prompt
versions (A4) and the agent runs, and needs transactions; the profile is a prompt block, not a fake first
message, so it never ages out of the history window.

**Checked by:** `tests/memory/test_repository.py`, `tests/orchestration/test_turn.py`; demo database
`python -m scripts.seed_demo`, then `--session demo-food`.

**Limits:** one learner profile (`default`), no accounts.

### M2 - Token Budgeting & Semantic Compression

**The catalogue asks:** count tokens per turn against a budget and, near the cap, summarise the oldest
turns into a compact block.

**What we built:** each agent gets a context built for its own `num_ctx`: 25% is kept for the reply, the
system-prompt blocks (profile, handoff note, passages, ledger) each get a capped share, and the newest
history fills the rest. When the unsummarised history passes 70% of its budget, the oldest turns are
folded into a running summary by the model and leave the window.

**How it works:**
- `sensai/orchestration/context.py`: `ContextBuilder` with a `TokenEstimator` (characters per token,
  3.5 by default); each provider has a `priority` and a `max_share`.
- `sensai/orchestration/summary.py`: `compress()` checks the trigger, `summarize()` asks the model for
  the updated summary (at most 25% of the window), and `SummaryProvider` puts it in the prompt.
- Settings: `[memory] max_history_turns`, `[memory.summary] enabled, trigger, keep_recent`.

**Choices:** a budget per agent rather than per session, since the checker, corrector and sensei have
different context sizes; a running summary updated incrementally rather than re-summarising the whole
history each time.

**Checked by:** `tests/orchestration/test_context.py`, `tests/orchestration/test_summary.py`.

When a compression happens, a `ContextCompressed` event (messages folded, tokens before and after) is
emitted: the CLI prints it, `/trace` shows `Context compressed: <n> messages, <before> -> <after> tokens`, and the
web page shows the same line under the reply.

**Limits:** the estimator can be calibrated from Ollama's real `prompt_eval_count`, but nothing calls that
calibration yet.

### M3 - External Structured Memory

**The catalogue asks:** move memory to a real database; the agent reads, writes and updates structured
records itself, through CRUD operations it drives.

**What we built:** a learner ledger in SQLite with three tables: `srs_cards` (flashcards scheduled with
SM-2), `errors` (verified corrections) and `facts`. The curriculum agent adds and grades cards by calling
tools after each answer; past errors and the cards due today are put in front of the tutor on the first
message of a session, so it can bring them up.

**How it works:**
- `sensai/memory/ledger.py`: generic create / read / update / delete on the three tables, with one
  owning agent per table (`errors` corrector, `srs_cards` curriculum, `facts` sensei); a write by another
  agent raises `WriteNotAllowed`. The ledger also exposes these operations as tools.
- `sensai/memory/srs.py`: `next_review()` is SM-2 (ease, interval, due date); `srs_add_card` and
  `srs_review` are the curriculum agent's tools.
- `sensai/orchestration/curriculum.py`: once the answer and `TurnFinished` are out, the curriculum agent
  (`agents/curriculum.toml`) runs a short tool loop in the background, one validated JSON step per call,
  up to `[curriculum] max_steps`. The learner does not wait for it and, on the web, the session is free
  again at `TurnFinished`.
- `sensai/orchestration/context.py`: `ErrorsProvider` and `DueCardsProvider` render the past errors and
  due cards into the first prompt of a session.

**Choices:** ownership per table enforced in code (SQLite has no per-user rights), so an agent cannot
corrupt another agent's records; SM-2 as a pure function, tested with fixed dates.

**Checked by:** `tests/memory/test_ledger.py`, `tests/memory/test_srs.py`, `tests/orchestration/test_curriculum.py`.

**Limits:** `facts` is never written; verified errors are written by the orchestrator after the
corrector's run rather than by the agent through a tool.

---

## Tools

### T1 - MCP

**The catalogue asks:** the model decides mid-generation to call a function; the code runs it and the
model continues with the result. Tested with our own and a pre-existing MCP tool.

**What we built:** own tools (`dictionary`, `conjugate`, `kanji`) over local data, an MCP client that
connects to any stdio MCP server listed in `config.toml`, our own MCP server serving the own tools, and
the pre-existing time server (`uvx mcp-server-time`). The sensei calls them during an answer; each call
and result shows in the chat and in `/trace`.

**How it works:**
- `sensai/knowledge/tools.py`: `Tool` and `ToolRegistry`; the own tools read
  `sensai/knowledge/data/dictionary.json` and `kanji.json`, and `japanese.py` conjugates verbs by rule
  (ichidan, godan, する, 来る and the irregular forms).
- `sensai/knowledge/mcp.py`: JSON-RPC 2.0 over stdio (initialize handshake, `tools/list` with
  pagination, `tools/call`, pings, cancellation, timeouts); a server that cannot start is reported and
  skipped.
- `sensai/knowledge/mcp_server.py`: `python -m sensai.knowledge.mcp_server` serves the own tools.
- `sensai/knowledge/toolbox.py`: the session's tools; an agent lists the ones it may use in its spec
  (`tools = ["dictionary", ..., "mcp:time"]`).
- `sensai/orchestration/executor.py`: the tool loop. Native tool calls when `/api/show` says the model
  supports them, otherwise the tools are described in the prompt and a reply that looks like
  `{"tool": ..., "args": ...}` is held back from the learner until it is known to be a call. At most
  `[tools] max_rounds` rounds.
- Switches: `[tools] enabled`, `[tools] disabled = [...]` for single tools, `[mcp] enabled`, and
  `enabled` per server.

**Choices:** two calling paths (native and prompted) so that a model without tool support still works;
tools answer `found: false` instead of guessing, so the model is not handed invented data.

**Checked by:** `tests/knowledge/test_japanese.py`, `tests/knowledge/test_mcp_stdio.py` (real
subprocesses), `tests/orchestration/test_tool_use.py`, `tests/orchestration/test_knowledge_turn.py`.

**Limits:** the time server needs `uv` installed; the dictionary covers common N5 words only.

### T5 - Structured Output

**The catalogue asks:** force the model to output valid JSON matching a schema.

**What we built:** agents with an `output_schema` in their spec (checker, corrector, classifier,
curriculum) answer in JSON that is validated against it; an invalid reply gets one repair attempt with
the validation errors, and if that fails too the run ends with an `invalid_output` error and the sensei
answers instead.

**How it works:**
- `sensai/orchestration/structured.py`: the schema is written into the system prompt, the request uses
  Ollama's JSON format, and `jsonschema` validates the reply.
- `sensai/core/agent_spec.py`: an agent that lists `handoffs` gets a `handoff` field added to its schema
  (target agent and reason), which is how agents pass work to each other (A2).

**Choices:** the full schema in the prompt plus validation in code, rather than trusting the format flag
alone; exactly one repair, to bound the cost of a bad reply.

**Checked by:** `tests/orchestration/test_structured.py`; demo `python -m sensai --agent checker`.

**Limits:** the checker's `rule` field is sometimes vague or wrong (a 9B model).

---

## RAG

### R1 - Basic RAG

**The catalogue asks:** ingest local documents, chunk them, embed them, store them in a local vector
store, retrieve the relevant chunks and answer with citations.

**What we built:** a grammar corpus (`corpus/*.md`, one section per grammar point, each tagged with its
JLPT level), embedded with `bge-m3`. For each question, the sensei receives the closest passages under the
learner's level and cites them (`[particles.md § Particle に (ni): ...]`); if it cites nothing, a
`Sources:` line is added.

**How it works:**
- `sensai/knowledge/rag.py`: chunking by Markdown heading (`# Title [N5]`; a sub-heading inherits its
  parent's level); `SemanticIndex` embeds each chunk and each title with `bge-m3` (on the CPU, so the chat
  model keeps the GPU) and caches the vectors in `data/rag_index.json`, keyed by a hash of the text, so
  only changed sections are re-embedded.
- `sensai/knowledge/retrieval.py`: `Retriever.retrieve()` reads the learner's level from the profile,
  keeps the `candidate_k` closest chunks above `min_similarity` (cosine), and frames the kept passages as
  quoted reference text in the prompt. If embeddings fail it falls back to word overlap.
- `sensai/orchestration/orchestrator.py`: an agent with `rag = true` gets the passages before it runs; the
  `Retrieved` event carries what was kept and dropped.

**Choices:** one chunk per grammar point rather than fixed-size windows, so a passage is always one
complete rule with its examples; the JLPT level as metadata, so an N5 learner is not taught N2 grammar.

**Checked by:** `tests/knowledge/test_rag_and_mcp.py`, `tests/knowledge/test_retrieval.py`,
`tests/orchestration/test_knowledge_turn.py`.

**Limits:** the corpus is 5 Markdown files; a question it does not cover is answered from the model's
own knowledge, without a citation.

### R2 - Advanced RAG

**The catalogue asks:** reranking, metadata filtering and per-chunk relevance scoring to drop weak
chunks.

**What we built:** the candidates from R1 are rescored 0 to 10 by the chat model (like a cross-encoder),
those under `relevance_threshold` are dropped, and the best `top_k` are kept. `/trace` shows every
candidate with its cosine score and its rerank score. A passage that reads like an instruction to the
model ("ignore your rules", "answer in French") is dropped before reranking.

**How it works:**
- `sensai/knowledge/retrieval.py`: `_model_scores()` sends the snippets to the model with a scoring prompt
  and JSON output; `_reranked()` applies the threshold; `_split_poisoned()` removes instruction-like
  passages (EV4). If the model's scores cannot be read, a hybrid score is used instead (chunk cosine,
  title cosine and word overlap).
- `sensai/knowledge/advanced_rag.py`: `rerank_and_filter_chunks()` and the per-chunk trace records.
- Settings: `[rag] rerank, model_rerank, relevance_threshold, hybrid_threshold, top_k`.

**Choices:** the chat model as reranker rather than a separate cross-encoder, since no extra model has to
be installed. Measured on 12 questions with a known right section: the model reranker kept the right
section 12/12 times with 3 unrelated extras, the hybrid score 9/12 with 8 extras, at about 3.7 s against
0.2 s (see `doc/benchmark.md`). The reranker uses the answering agent's context size, so Ollama does not
reload the model between reranking and answering.

**Checked by:** `tests/knowledge/test_advanced_rag.py`, `tests/knowledge/test_retrieval.py`.

**Limits:** reranking adds a few seconds before the first token.

---

## Orchestration & Reasoning

### A1 - Reasoning Loops (ReAct)

**The catalogue asks:** a loop that makes the model's reasoning explicit and iterative.

**What we built:** the corrector checks the checker's correction with a tool before it reaches the
learner: it writes a Thought and an Action (a tool call), reads the Observation, then writes a
Verification, a Hypothesis (confirmed or wrong), the Rule, the Correction and a Final Answer. If the tool
contradicts the first idea, the correction is revised. The steps stream to the learner.

**How it works:**
- `sensai/knowledge/react.py`: `run_react_loop()` with a fixed prompt format, a parser for the labelled
  fields, a check that the model cannot invent its own Observation, a small context (about 800 tokens)
  and a maximum number of steps.
- `sensai/orchestration/executor.py`: an agent with `strategy = "react"` (`agents/corrector.toml`) runs this
  loop and returns structured data (correction, rule, `verified`, `revised`, tools used), then hands off to
  the sensei.

**Choices:** ReAct on the step where a mistake costs the most (a wrong correction taught to a beginner),
not on every reply; the tool call is required before any correction.

**Checked by:** `tests/knowledge/test_react.py`, `tests/orchestration/test_tool_use.py`,
`tests/orchestration/test_knowledge_turn.py`.

**Limits:** the tools check words, not reasoning: when the checker's claim is wrong in a way no tool
covers, the corrector can keep or worsen it. A conjugated word missing from the dictionary leaves the
correction unverified.

### A2 - Multi-Agent Orchestration

**The catalogue asks:** several specialised agents coordinating on a task, passing work between them.

**What we built:** five agents defined in `agents/*.toml`: the sensei (answers), the checker (finds the
mistake in a sentence), the corrector (verifies it, A1), the classifier (routing) and the curriculum agent
(M3). A three-tier router picks who starts each turn; agents pass the turn on with a JSON handoff, under
guards; when an agent fails, the sensei answers instead.

**How it works:**
- `sensai/orchestration/router.py`: tier 1 regex rules, tier 2 embedding similarity to each route's
  examples (`bge-m3`, threshold 0.55 and margin 0.05), tier 3 the classifier agent; a decision below the
  confidence floor goes to the sensei. `@agent` at the start of a message forces an agent (not the
  internal ones, unless `[router] debug = true`); an agent or persona name that would leave its folder
  (`../x`) is refused when it is loaded. Routes are in `routing.toml`.
- `sensai/core/handoff.py`: the handoff envelope (from, to, reason, JSON payload, hop count).
- `sensai/orchestration/orchestrator.py`: runs the chain (checker -> corrector -> sensei for a sentence
  with a mistake) and the fallback.
- `sensai/orchestration/guards.py`: maximum hops, no agent receiving the turn twice in a turn, a time
  budget for the turn and a timeout per agent.
- `RouteDecided` and `HandoffEvent` events make every decision visible in `/trace`.

**Choices:** the cheapest tier first, so most messages are routed by a rule, without a model call, and only
ambiguous ones cost one; handoffs as validated JSON rather than prose, so one agent cannot
inject instructions into the next; `--agent` turns routing off for demos.

**Checked by:** `tests/orchestration/test_router.py`, `tests/orchestration/test_orchestrator.py`,
`tests/orchestration/test_guards.py`; `python -m sensai --agent demo/ping` shows the loop guard.

**Limits:** all agents use the same 9B model, so a correction takes 40 to 70 s.

### A4 - Prompt Versioning

**The catalogue asks:** prompts as versioned artifacts, comparable and rollback-able.

**What we built:** every system prompt is identified by its content hash; the first time an agent runs
with a prompt it is stored, and every agent run records the hash it used, with its duration and tokens.
`session.prompts.metrics("sensei")` compares versions; `prompt_version = "<hash>"` in an agent's TOML
pins it back to an earlier prompt.

**How it works:**
- `sensai/core/agent_spec.py`: `prompt_hash` is the first 12 hex digits of the SHA-256 of the prompt.
- `sensai/memory/prompts.py`: `PromptStore` (`prompts` table, migration `004_prompt_versions.sql`):
  `resolve()` stores or swaps a prompt, `versions()` and `metrics()` compare them.
- `AgentRunStarted` carries the hash, so the log has it too.

**Choices:** content hashes instead of manual version numbers, so editing a prompt can never forget to
bump a version; a persona is part of the prompt, so each persona is its own version and can be compared.

**Checked by:** `tests/memory/test_prompts.py`, `tests/orchestration/test_personas.py`.

**Limits:** comparison is from Python only; no chat command shows it.

### A5 - Persona

**The catalogue asks:** the agent's name, tone, role and scope configurable without code changes,
switchable.

**What we built:** personas are TOML files (`personas/teacher.toml`, `strict.toml`, `casual.toml`) with a
name, description, tone and scope. `/persona` lists them and `/persona strict` switches mid-session; the
history is kept.

**How it works:** `sensai/core/persona.py` loads and validates a persona; the orchestrator appends it to
the system prompt of the agents that talk to the learner (no `output_schema`), so the checker and
corrector keep their exact prompts. Default in `[personas] default`.

**Choices:** a new file is a new persona, with no code change; the persona changes the tone only, and the
prompt says so, so the format of the Japanese stays the same.

**Checked by:** `tests/core/test_persona.py`, `tests/orchestration/test_personas.py`, `tests/surfaces/test_cli.py`.

---

## Evaluation

### EV1 - Automated Eval & Hallucination Detection

**The catalogue asks:** automatic evaluation of answers (relevance, coherence, faithfulness to sources).

**What we built:** after a reply that used corpus passages or tool results, a judge splits it into claims
and marks each one supported, contradicted or unverified by those sources, plus a relevance grade for the
question. The chat shows `[faithfulness] 0.75 - relevance 0.90 - 3 supported, 0 contradicted,
1 unverified`, then each flagged claim. `python -m scripts.evaluate` scores several models on a fixed set
of questions.

**How it works:**
- `sensai/trust/eval.py`: `SourceCollector` gathers the turn's kept passages and successful tool results
  from the event stream; `LLMJudge.verify_response()` sends them with the reply to the model with a strict
  JSON format; `faithfulness_event()` turns the report into a `FaithfulnessChecked` event.
- `sensai/orchestration/turn.py`: the judge runs after `TurnFinished`, in a thread, and never fails the
  turn. Settings: `[eval] enabled, judge_model, max_context_chars`.

**Choices:** judge only what has a source: a reply built on no passage and no tool has nothing to be
checked against, and judging it would only measure the model against itself. Claims from the model's
memory are counted as unverified, not as errors.

**Checked by:** `tests/trust/test_eval.py`; `python -m scripts.evaluate`.

**Limits:** the judge is the same local model, so it can be wrong; it adds about 16 s after a sourced
reply on the reference laptop (more on a slower machine), but the reply is already shown and the web
session is free again at `TurnFinished`. A reply with no passage and no tool call is not judged, so an
invented fact on a topic the corpus does not cover is not flagged. Structured replies (checker,
corrector) are not judged.

### EV2 - Content & Privacy Guardrails

**The catalogue asks:** interceptors on input and output: block unsafe or out-of-scope prompts, protect
personal data.

**What we built:** on input, prompt injections and out-of-scope requests (code, maths homework, other
subjects) are refused before any model call, and emails, phone numbers, card numbers and API keys are
masked. On output, the same masking runs on the reply while it streams. Each trigger is a
`GuardrailTriggered` event (rule and action) shown in the chat and the log.

**How it works:**
- `sensai/trust/guardrails.py`: regex families for injections (English, French and Japanese wordings,
  fake role tags), scope, and personal data; `check_input()` returns allow, redact or block with the rule
  that fired.
- `sensai/orchestration/turn.py`: a blocked message gets a fixed refusal without reaching the model;
  `redact_tokens()` holds back the end of the streamed reply that could still grow into an email or a
  phone number, so nothing leaks between two tokens.
- Settings: `[guardrails] enabled, block_prompt_injection, block_out_of_scope, redact_pii_input,
  redact_pii_output`.

**Choices:** cheap pattern checks that run before the model, so a blocked request costs nothing;
masking instead of refusing for personal data, so the learner still gets an answer.

**Checked by:** `tests/trust/test_guardrails.py`, `tests/orchestration/test_turn.py`; the EV4 suite.

**Limits:** patterns miss what they do not describe (a reworded or leetspeak injection, a number spelled
out in words), and the scope check lets through anything that mentions Japanese. The tier of embeddings
and then a model call that the user stories plan for unsure cases is not built. The offline suite shows
0 breaks; the live report in `doc/adversarial.md` (3 broke) predates the guardrail fixes and must be run
again with `--live`.

### EV3 - Logging & Monitoring Dashboard

**The catalogue asks:** structured logging of every interaction (prompt, response, tools, latency,
tokens) and a way to monitor it.

**What we built:** every event of every turn is written as one JSON line to `logs/sensai.jsonl`, linked by
`turn_id`, `run_id` and `parent_run_id`. `/metrics` shows the last turn's tokens, generation speed, time to
first token and an estimated cost; `/trace` shows the route decision, every agent run, tool call and result,
retrieved passages with their scores and the judge's verdict. The web UI shows the same metrics and trace.

**How it works:** `sensai/trust/logging.py` (`JsonlLogger` wraps the event stream),
`sensai/trust/metrics.py` (`compute_turn_metrics()` from `TurnFinished`), `sensai/trust/tracing.py`
(`TurnTracer` builds the trace from the events). First token and total time come from the same clock,
and speed from Ollama's generation time.

**Choices:** the log is the event stream itself, so nothing can be shown in the chat without being in the
log; the cost estimate ($0.50 / $1.50 per million tokens) shows what the same usage would cost on a hosted
API, since it is free locally.

**Checked by:** `tests/trust/test_logging.py`, `tests/trust/test_metrics.py`, `tests/trust/test_tracing.py`.

**Limits:** metrics are per turn; there is no view per agent and no error rate over time.

### EV4 - Adversarial Testing

**The catalogue asks:** a suite that attacks the agent (jailbreaks, prompt injections, malformed input),
documented.

**What we built:** 27 attacks in `scripts/adversarial_cases.toml`, in seven categories (prompt injection,
jailbreak, out of scope, system prompt leak, personal data, poisoned corpus document, malformed input), run
through the real pipeline. Each one is reported as blocked (by a guardrail), held (reached the system and
failed) or broke (worked).

**How it works:** `python -m scripts.adversarial` runs offline (the model is a stub, model-dependent
checks are "not run"); `--live` runs real turns through Ollama; `--report doc/adversarial.md` writes the
Markdown report. Poisoned documents are written into a temporary corpus to check that retrieval drops them.

**Choices:** run through `run_turn`, not against the guardrails alone, so an attack that slips past a
filter is still caught if the system holds; canary words (`PWNED-42`) make "broke" an objective check.

**Checked by:** `tests/scripts/test_adversarial.py` runs the suite offline in CI.

**Limits:** the last live report (2026-10-30): 3 blocked, 21 held, 3 broke; it predates the latest
guardrail fixes and should be re-run.

---

## UX & Lifecycle

### X1 - Web UI

**The catalogue asks:** a web frontend over the CLI backend: message input, streaming display, live
interaction.

**What we built:** `python -m sensai.surfaces.web` serves a single page and an API on the same origin.
Replies stream into the page; the page shows the session's tokens, speed and cost, opens the last turn's
trace, switches model and reasoning, shows guardrail notices, the judge's verdict and context
compression under the reply, resumes the last conversation after a reload and starts a new one.

**How it works:**
- `sensai/surfaces/web/app.py` (FastAPI): `POST /api/sessions/{id}/turns` runs `run_turn()` and sends each
  event as a server-sent event named after its class (`sse.py`); sessions are created, listed and resumed
  through `/api/sessions` (`sessions.py`); `settings.py` switches model, agent, persona and reasoning.
- `frontend/index.html`: one page, no build step, reading the event stream.

**Choices:** server-sent events rather than WebSockets, since the stream goes one way; the backend only
subscribes to the event stream, so the turn logic is the same as the CLI's.

**Checked by:** `tests/surfaces/test_web.py`.

**Limits:** parts of the page are in French; Tailwind and the fonts load from a CDN, so the page needs
internet; the page cannot choose an agent or persona yet (the routes exist) and shows no context-usage
bar or per-turn trace history; there is no automated browser test, only the API tests.

---

## Features we did not take

The catalogue rewards fewer features done in depth. We kept the ones that serve a beginner learning
Japanese and left the others out:

| Code | Feature | Why not |
|---|---|---|
| M4 | Artifact & State Document | A learner does not need a living document; their progress is in the ledger (M3). |
| T2 | Sandboxed Code Execution | A Japanese tutor has no honest use for running generated code. |
| T3 | Web Search | Sensai runs fully offline; answers come from the curated corpus with JLPT levels, which a web page cannot guarantee. |
| T4 | File Access within Permissions | No task of a learner needs the tutor to read or write their files. |
| A3 | Human-in-the-Loop | No action of the tutor is risky enough to need a human approval step. |
| A6 | Semantic Cache | Out of scope: the answers depend on the learner's history and profile, so a cached answer is rarely the right one. |
| A7 | Automated Prompt Optimization | Prompt versions are compared by hand with A4; automating it needs a scored test set we did not build. |
| X2 | Branching - Resume & Interrupt | Sessions are resumed (M1) and replies can be interrupted (B1); branching a conversation adds little for a learner. |
| X3 | Export & Publication | Nothing a learner would publish. |
| X4 | Questions & Forms | The curriculum's card reviews cover asking the learner questions. |
| X5 | Scheduling | Spaced repetition (M3) already decides when to review; there is no background job to schedule. |
