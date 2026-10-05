# Sensai — Local Japanese Tutor

A multi-agent AI assistant for learning Japanese, running 100% locally through Ollama.
No data ever leaves the machine.

📄 Full work plan: [`doc/work_plan.pdf`](doc/work_plan.pdf)
🗺️ Architecture: [overview](doc/architecture_overview.png) · [detailed](doc/architecture_detailed.png)
🗂️ Previous README: [`doc/README_old.md`](doc/README_old.md)

## Architecture

Every learner turn goes through six stages:

| Stage | Role | Features |
|-------|------|----------|
| 1 · Input | CLI / Web UI, input guardrails (scope, PII) | B1, X1, EV2 |
| 2 · Decide | Orchestrator, 3-tier router (rules → embeddings → classifier), typed handoffs | A2 |
| 3 · Build the prompt | Persona, versioned prompts, token budgeting and compression | A5, A4, M2 |
| 4 · Run the loop | ReAct loop, tools + MCP, RAG, structured output | A1, T1, T5, R1 |
| 5 · Remember | Sessions, profile, structured memory (errors, SRS, facts) | M1, M3 |
| 6 · Check and show | Hallucination judge, output guardrails, logging and metrics | EV1, EV2, EV3 |

Stage 4 can loop back to stage 2 ("another hop"), and stage 5 is read back on the next turn.

**Overview**

![Architecture overview](doc/architecture_overview.png)

**Detailed** — same flow, with every component inside each stage

![Architecture detailed](doc/architecture_detailed.png)

## Requirements

- Python 3.10+
- [Ollama](https://ollama.com) installed and running (`ollama serve`)
- Node.js 18+ (only for the web UI, layer 7)

## Installation

```bash
git clone <repo-url>
cd G-AIA-500-PAR-5-1-sensai-4

python3 -m venv venv        # on Ubuntu, `python` only exists inside the venv
source venv/bin/activate
pip install -r requirements-dev.txt

ollama pull qwen3.5:9b      # default model (fits an 8 GB GPU); lighter option: qwen3.5:4b
ollama pull bge-m3          # RAG embeddings (layer 5)
```

## Usage

```bash
python -m sensai                          # start the chat
python -m sensai --model qwen3.5:4b       # use another model for every agent
python -m sensai --config my-config.toml  # use another config file
python -m sensai --list-sessions          # list the stored sessions (id, date, model, title)
python -m sensai --session <id>           # resume a stored session
python -m sensai --agent checker          # talk to another agent from agents/ (here: JSON corrections)
```

An unknown session id prints `[error] session not found` and exits with a non-zero code.

At startup Sensai checks that Ollama is reachable and the model is installed, and stops with the
exact `ollama pull` command if it isn't.

### Chat commands

| Command | What it does |
|---------|--------------|
| `/help` | list the commands |
| `/model` | show the current model and the installed ones |
| `/model <name>` | switch model mid-session; the history is kept |
| `/think` | show whether the model reasons before answering |
| `/think on` / `/think off` | turn the model's reasoning on or off (only for models that can reason) |
| `/reset` | clear the history and start a new session (the old one stays stored) |
| `/quit` | exit (Ctrl-D works too) |

Anything else starting with `/` is rejected as an unknown command and never sent to the model.

In a terminal, the input line stays usable while a reply streams:

- **Type ahead.** Write your next question while the model is answering; Enter queues it, and it runs
  as soon as the current reply ends (shown as `> question` above its own answer).
- **Status toolbar.** The bottom line shows `thinking…` until the first token, `replying…` while text
  streams, and how many questions are queued.
- **Token usage.** A dim line after each reply shows prompt and reply tokens, time to first token and
  total time.
- **Reasoning.** When the model reasons before answering (`/think on`, or `think = true` in the agent
  spec), its reasoning is shown dim before the answer. It is never part of the reply or the history.
  Reasoning is off by default: it can take several times longer than the answer itself.
- **Esc** stops the reply being generated and keeps the chat. The question and the part of the
  reply already written stay in the history, so the model knows what it had said ("continue" works).
- **Ctrl-D** (or `/quit`) quits, without running questions still queued. **Ctrl-C** does nothing,
  so it can't end a lesson by accident.

When input is piped (scripts, tests), Sensai reads one line at a time instead, and Ctrl-C stops a reply.

### Example

```
$ python -m sensai
Sensai (qwen3.5:9b) — type /help for commands

> how to say hi in japanese
Hello! In Japanese, you say "hi" as:

こんにちは (konnichiwa) — hello.

It is the most common way to greet people during the day.

> /model qwen3.5:4b
Model: qwen3.5:4b (history kept)
```

### Troubleshooting

| Symptom | Fix |
|---------|-----|
| `zsh: command not found: python` | create the venv with `python3`, then `python` works inside it |
| `[error] Model '…' not found` | run the `ollama pull …` command shown in the message |
| `[error] Cannot reach Ollama` | start it: `ollama serve` or `sudo systemctl start ollama` |
| answers are slow, `ollama ps` shows `100% CPU` | the NVIDIA driver isn't loaded; check `nvidia-smi` (after a kernel update, install the matching `linux-modules-nvidia-…` package and reboot) |
| `ollama pull` fails with `no space left on device` | models are stored on the root partition; point Ollama to a bigger one with `OLLAMA_MODELS` in a systemd override |

## Configuration

Settings are layered, each level overriding the previous one:

1. built-in defaults (`sensai/core/config.py`)
2. `config.toml`
3. environment variables `SENSAI_<SECTION>__<KEY>`, e.g. `SENSAI_OLLAMA__URL=http://gpu:11434`
4. CLI arguments, e.g. `--model`

Each agent is described in `agents/<name>.toml` (model, `num_ctx`, prompt, tools, output schema).
`model` and `num_ctx` are always explicit there: we never inherit Ollama's defaults.
`--model` overrides the model of every agent for one run.

The `[memory]` section controls what the chat remembers:

```toml
[memory]
max_history_turns = 20   # turns of history sent to the model
profile = "default"      # name of the learner profile injected into every prompt
```

## Persistence

`sensai/memory/` stores sessions and history in SQLite, at the path from `[storage].db_path`
in `config.toml` (`:memory:` for tests).

- **Connection** (`db.py`): one `connect(path)` with `PRAGMA foreign_keys = ON`,
  `journal_mode = WAL`, and `row_factory = sqlite3.Row`.
- **Schema** (`migrations/001_init.sql`, `002_active_leaf_and_profile.sql`): `sessions` → `messages` (FK, cascade) → `turns`
  (links a user and an assistant message, with `latency_ms`/`tokens`) → `agent_runs` (one row
  per sub-step of a turn — `rag`, `guardrail`, a tool call — with `status` and `duration_ms`).
  Ids are **text uuids** from `sensai.core.ids.new_id()`, the same ids the events carry
  (`session_id`, `turn_id`, `run_id`), so a turn can emit its events before any row exists and
  logs, trace and database all share one id. `turns.assistant_message_id` is nullable so a turn
  whose reply failed is still recorded.
  Indexed on `messages.session_id`, `turns.session_id` and `agent_runs.turn_id`. `messages.parent_id`
  and `agent_runs.parent_run_id` are self-referencing (nullable, cascade) — unused for now, but in
  place so a later layer can branch a message (edits/retries) or nest an agent run without a
  migration.
- **Migrations** (`migrate.py`): applies every `NNN_*.sql` file above the DB's
  `PRAGMA user_version`, in order; re-running is a no-op once every file is applied.
- **Access pattern** (`repository.py`): a synchronous `SessionRepository`
  (`create_session`, `add_message`, `list_messages`, `record_turn`, `record_agent_run`; each write
  takes an optional id, generates one otherwise, and returns it; plus the session and profile
  calls below) plus an
  `AsyncSessionRepository` facade that runs each call through `asyncio.to_thread`. We chose that
  over `aiosqlite` because it needs no new dependency, keeps the storage logic in one place that
  stays trivial to unit-test synchronously, and the thread-pool overhead is irrelevant at
  Sensai's scale (one learner, a handful of DB calls per turn) — the web UI (layer 7) can await
  the facade without blocking its event loop.

### Sessions, branches and the learner profile

- `add_message(session_id, role, content, parent_id)` inserts a message and moves
  `sessions.active_leaf_id` onto it in one transaction. Messages form a tree through `parent_id`.
- `load_history(session_id)` walks the tree from `active_leaf_id` up to the root (recursive CTE)
  and returns the messages oldest first. `set_active_leaf()` switches to another branch.
  `get_session()`, `load_history()` and `set_active_leaf()` raise `SessionNotFound` for an unknown
  session. `list_sessions()` returns the sessions, newest first.
- The learner profile lives in `learner_profiles`, read with `get_profile()`, written with
  `save_profile()`. `profile_system_message()` turns it into a system message.
- `Session` (`orchestration/session.py`) is the link with the turn loop. `run_turn()` saves the
  user message before calling the model and the reply after the stream; a failed turn is removed
  from the database too, and a stopped one keeps the question and the part already written. The
  session itself is only stored with its first message, and `Session.resume(config, id)` reloads a
  stored one.
- The profile named by `[memory].profile` goes in the prompt right after the system prompt on
  every turn, and is never stored in `messages`.

## Features

Committed: six of the seven ★★★ features (T2 is declined on purpose — a Japanese tutor has no honest use for running generated code).

| Code | Feature | Rating | Layer | Status |
|------|---------|--------|-------|--------|
| B1 | CLI chatbot (streaming, history, errors) | 👍 | 1–2 | ✅ |
| M1 | Session persistence & profile | ★ | 2 | ✅ |
| T5 | Structured output (JSON schema, one repair) | ★ | 3 | ✅ |
| M3 | Structured memory (errors, SRS, facts) | ★★ | 3, 6 | ⏳ |
| T1 | Tools (dictionary, conjugate, kanji) + MCP | ★★★ | 3, 5 | 🚧 own tools done, MCP in layer 5 |
| EV3 | Logging, metrics, per-turn trace | ★★ | 3–4 | 🚧 JSONL logger done |
| A2 | Multi-agent orchestration + router | ★★★ | 4–5 | ⏳ |
| A1 | ReAct (corrector) | ★★★ | 4 | ⏳ |
| M2 | Token budgeting & compression | ★★ | 4–5 | ⏳ |
| R1 | Basic RAG (citations, JLPT metadata) | ★★ | 5 | ⏳ |
| EV1 | Judge: claim-by-claim verification | ★★★ | 5 | ⏳ |
| R2 | Advanced RAG (reranking, scoring) | ★★★ | 6 | ⏳ |
| EV2 | Input / output guardrails | ★★★ | 6 | ⏳ |
| A5 | Switchable personas mid-session | ★ | 6 | ⏳ |
| A4 | Prompt versioning (content hash) | ★ | 6 | ⏳ |
| X1 | Web UI (FastAPI + SSE) | ★ | 7 | ⏳ |
| EV4 | Adversarial suite | ★★ | 7 | ⏳ |
| A6 | Semantic cache (optional) | ★★ | 7 | ⏳ |

✅ done · 🚧 in progress · ⏳ to do — **Never cut:** T1, R2, A1, A2, EV1, EV2.

## Layered plan

| Layer | Content |
|-------|---------|
| 1 | Foundations: contracts (events, AgentSpec, Handoff, config), SQLite, Ollama transport, CLI + logger |
| 2 | Base loop: `run_turn()`, persistence, model bake-off, error handling |
| 3 | Real memory: structured output, ledger, own tools, metrics v1 |
| 4 | Multi-agent: orchestrator, per-agent context, ReAct corrector, trace view |
| 5 | Knowledge & routing: router, compression, RAG + external MCP, judge |
| 6 | Three-star finish: personas & prompts, curriculum agent, advanced RAG, guardrails |
| 7 | Surface & proof: web UI, semantic cache, adversarial suite |
| 8 | Delivery: demo script, crew vs monolith measurement, fallback videos, README |

Rule: never start a layer until the previous layer's contracts are **merged**.

## Code architecture

### Packages

One package per domain, each owned by one member so parallel branches rarely touch the same files.
Each package's `__init__.py` lists the modules that belong in it and the layer that adds them,
so new code always has an obvious home.

```
sensai/
  core/           shared contracts: messages, errors, config, events, specs    (A)
  llm/            Ollama HTTP transport, the only code that calls the model    (C)
  orchestration/  turn loop, executor, orchestrator, router                     (A)
  memory/         SQLite storage, sessions, ledger, context budgeting           (B)
  knowledge/      tools, MCP client, RAG pipeline                               (C)
  trust/          guardrails, judge, logging, metrics                           (D)
  surfaces/       CLI and web UI                                                (D)
agents/           one TOML spec per agent (sensei, corrector, judge, ...)
tests/            one folder per package, mirroring sensai/
doc/              work plan, architecture diagrams, previous README
config.toml       project configuration
```

Runtime files (`data/*.db`, `logs/`) are created on first run and never committed.

### Rules that keep it stable

- **Dependencies point inward.** Every package may import `core`; `core` imports nothing from the
  others. `surfaces` never contains business logic.
- **Agents are data, not code.** An agent is a TOML file loaded into an `AgentSpec`
  (model, `num_ctx`, prompt, tools, output schema). Adding the corrector or the judge means adding a file.
- **One transport for every agent.** `OllamaClient.chat_stream(model, messages, num_ctx=...)` takes the
  model and options on each call, so agents with different models share one client. Errors carry a `kind`
  (`model_missing`, `connection`, `timeout`, `bad_request`, `stream`) that surfaces can display, including
  failures in the middle of a reply. The stream yields text only; its return value is a `ChatStats` with
  the final chunk's token counts and durations, which `run_turn()` puts into `TurnFinished`.
  `capabilities(model)` and `context_length(model)` read `/api/show` (cached), so tools can choose native
  or prompted tool calling and specs can be checked against the model's maximum context.
- **Surfaces only subscribe.** The turn loop emits events (`core/events.py`) and the CLI, the logger,
  the trace view and the web UI only consume them.

### How a turn flows

| Stage | Package |
|-------|---------|
| 1 · Input | `surfaces` → `trust` (input guardrails) |
| 2 · Decide | `orchestration` (orchestrator, router) |
| 3 · Build the prompt | `memory` (context builder, budget) with the agent's spec from `core` |
| 4 · Run the loop | `orchestration` (executor) → `llm`, `knowledge` (tools, RAG) |
| 5 · Remember | `memory` (sessions, ledger) |
| 6 · Check and show | `trust` (judge, output guardrails, logging) → events → `surfaces` |

Today a turn is `run_turn(session, text)` in `orchestration/turn.py`: an async generator that yields
`TurnStarted → AgentRunStarted → Token… → AgentRunFinished → TurnFinished` (with `Error` and
`GuardrailTriggered` when they apply) for one agent. `Session` (`orchestration/session.py`) holds the
active agent, the history and the client. The CLI renders those events; it never calls the model.

### Structured output (T5)

An agent with an `output_schema` in its spec (e.g. `agents/checker.toml`) replies with JSON data,
not text. `run_turn()` emits no `Token` for it, but one `StructuredOutput` event with the validated
data, or `Error(kind="invalid_output")` with the reply that failed.

- **The schema is written into the system prompt**, and `format: "json"` is sent so the reply is
  always parseable. We measured why: with qwen3.5 and `think = false`, Ollama 0.18 ignores a schema
  passed in `format` and answers in free prose; it only enforces it when the model reasons first,
  which takes about a minute per reply. With the schema in the prompt the model follows it in ~4 s.
  When an agent has `think = true`, the schema itself is sent as `format`.
- **The reply is validated** against the schema with `jsonschema`; a spec whose schema is itself
  invalid is rejected when it's loaded.
- **One repair attempt:** the validation errors are sent back once; if the second reply is still
  invalid, the turn fails loudly with `invalid_output`, and the question isn't kept in the history.
- The schema guarantees the shape of the reply, not that it's right: checking the content is the
  job of the corrector's tools (A1, layer 4) and the judge (EV1, layer 5).

## Organisation

### Branches

Every sub-issue has its own branch, named `layerN/<owner letter>-<task>` (e.g. `layer1/c-ollama-transport`).
Each layer also has an integration branch, `layer1/integration` through `layer8/integration`, where the four
tasks of the layer are merged and checked together before reaching `main`.

> Git does not allow a branch called `layer1` while branches like `layer1/a-…` exist, so each integration
> branch sits in its layer's folder next to the task branches.

### Workflow

```
layer1/a-core-contracts-…   ─┐
layer1/b-sqlite-schema-…    ─┤  PR  →  layer1/integration  ──  PR  →  main  →  Epitech repo
layer1/c-ollama-transport   ─┤         (merge all 4 and check)
layer1/d-cli-renderer-…     ─┘
```

1. Each person opens a PR from their task branch into `layerN/integration`.
2. When all four are merged, test the layer together on that branch and check its "Done when" condition.
3. Open one PR from `layerN/integration` into `main`. Once it merges, the CI pushes to the Epitech repo.

The CI runs the tests and the 85% coverage check on every push and every PR to any branch.
The push to Epitech only happens from `main`, so nothing on an integration branch reaches the Epitech repo early.

### Before starting a layer

The integration branches for layers 2–8 were created from an early `main`. Before starting a layer,
update its integration branch and the task branches from `main` so they include the previous layer's work:

```bash
git switch layer2/integration && git merge main && git push
```

## CI

Code is pushed to the mirror repo. Every push runs the tests with coverage
(**blocking under 85%**, reports kept as artifacts for 90 days); if everything passes on `main`,
the full history is pushed to the Epitech repo.

```bash
python -m pytest --cov          # locally
```

## User stories

Two user stories per feature, with acceptance criteria and how each one is verified: see [`doc/user_stories.md`](doc/user_stories.md).

## Team

| Member | Role | Owns |
|--------|------|------|
| A | Core | Ollama transport, event stream, agent executor, orchestrator, router, config |
| B | Memory | SQLite, profile, facts, SRS, context builder, budgeting, compression |
| C | Knowledge | Tool registry, own tools, MCP client, RAG pipeline, corrector spec |
| D | Trust & surface | Logging, metrics, trace view, judge spec, guardrails, CLI and Web UI |
