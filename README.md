# Sensai — Local Japanese Tutor

A multi-agent AI assistant for learning Japanese, running 100% locally through Ollama.
No data ever leaves the machine.

📄 Full work plan: [`doc/work_plan.pdf`](doc/work_plan.pdf)
🗺️ Architecture: [overview](doc/architecture_overview.png) · [detailed](doc/architecture_detailed.png)
🗂️ Previous README: [`doc/README_old.md`](doc/README_old.md)
📊 Model benchmark: [`doc/benchmark.md`](doc/benchmark.md)

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
- [uv](https://docs.astral.sh/uv/) for the pre-existing MCP time server (`uvx mcp-server-time`); without it the
  chat starts and reports the server as unavailable

## Installation

```bash
git clone <repo-url>
cd G-AIA-500-PAR-5-1-sensai-4

python3 -m venv venv        # on Ubuntu, `python` only exists inside the venv
source venv/bin/activate
pip install -r requirements-dev.txt

ollama pull qwen3.5:9b      # default model (see doc/benchmark.md); lighter option: qwen3.5:4b
ollama pull bge-m3          # embeddings: router tier 2 and RAG (layer 5)
```

## Usage

```bash
python -m sensai                          # start the chat; the router picks the agent for each message
python -m sensai --model qwen3.5:4b       # use another model for every agent
python -m sensai --config my-config.toml  # use another config file
python -m sensai --list-sessions          # list the stored sessions (id, date, model, title)
python -m sensai --session <id>           # resume a stored session
python -m sensai --agent checker          # always start with this agent (turns the router off)
python -m sensai --agent demo/ping        # infinite-loop demo: ping and pong hand off forever until the guards stop them
```

Other entry points:

```bash
python -m sensai.surfaces.demo            # renders and logs a scripted turn, no model needed
python -m sensai.surfaces.web             # web UI + backend on http://127.0.0.1:8000 (see Web UI below)
python -m scripts.benchmark               # compare the candidate models on 20 Japanese questions
python -m scripts.seed_demo               # build data/demo.db for the keynote (see doc/demo.md)
python -m scripts.adversarial --live      # attack Sensai (EV4); --report doc/adversarial.md keeps the results
python -m scripts.evaluate --models qwen3.5:4b qwen3.5:9b   # faithfulness and relevance of each model's replies (EV1)
```

An unknown session id prints `[error] session not found` and exits with a non-zero code.

At startup Sensai checks that Ollama is reachable and the model is installed, and stops with the
exact `ollama pull` command if it isn't.

### Chat commands

| Command | What it does |
|---------|--------------|
| `/help` | list the commands |
| `/persona` | list the personas (the active one marked `*`) |
| `/persona <name>` | switch persona mid-session; the history is kept |
| `/model` | show the current model and the installed ones |
| `/model <name>` | switch model mid-session; the history is kept |
| `/think` | show whether the model reasons before answering |
| `/think on` / `/think off` | turn the model's reasoning on or off (only for models that can reason) |
| `/metrics` | token usage, speed and estimated cost of the last turn |
| `/trace` | the agent runs of the last turn (tokens, reasoning, ok or failed) |
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

> how do I say cat?
猫 (neko) — cat.
370 prompt + 14 reply tokens · first token 10.3s · 11.1s

> /model qwen3.5:4b
Model: qwen3.5:4b (history kept)
```

Every Japanese word or sentence comes as `characters (romaji) — translation`
(set in `agents/sensei.toml`). The first reply after starting waits for Ollama to load the model
(here ~10 s); later replies start in under a second.

With `--agent checker`, a sentence comes back as data:

```
$ python -m sensai --agent checker
> 私は学校を行きます
{
  "sentence": "私は学校を行きます",
  "is_correct": false,
  "correction": "私は学校に行きます",
  "error_type": "vocabulary",
  "rule": "The verb for going to a place is 行く (iku), …"
}
```

The reply always has this shape, but its content can still be wrong: this mistake is a particle
(を → に), not vocabulary. Checking content comes with the corrector's tools and the judge.

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

[memory.summary]
enabled = true           # fold older turns into a running summary
trigger = 0.7            # share of the history budget that starts a summary
keep_recent = 6          # newest messages kept as they are
```

When the history passes `trigger`, the model summarises the older turns; the summary is added to
the system prompt (`orchestration/summary.py`) and those turns leave the window. If Ollama fails
while summarising, the turn goes on with the plain window. The summary lives in the session and is
rebuilt after `Session.resume`.

## Persistence

`sensai/memory/` stores sessions and history in SQLite, at the path from `[storage].db_path`
in `config.toml` (`:memory:` for tests).

- **Connection** (`db.py`): one `connect(path)` with `PRAGMA foreign_keys = ON`,
  `journal_mode = WAL`, and `row_factory = sqlite3.Row`. The connection is shared by every thread
  (the async facade, the web UI), so it is a `LockedConnection`: a reentrant lock guards each
  statement and fetch, and `with conn:` holds it for the whole transaction, so two threads can never
  interleave statements or commit a half-finished transaction.
- **Schema** (`migrations/001_init.sql`, `002_active_leaf_and_profile.sql`, `003_ledger.sql`,
  `004_prompt_versions.sql`): `sessions` → `messages` (FK, cascade) → `turns`
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
  The chat writes `sessions` and `messages` (plus the profile and the ledger tables). A turn whose
  answer is kept also writes its `turns` row and one `agent_runs` row per agent that ran (`kind =
  "agent"`, with `agent`, `prompt_hash` and `tokens`); a failed or stopped turn records neither.
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

### Learner ledger

`migrations/003_ledger.sql` adds three tables, handled by `Ledger` (`memory/ledger.py`, opened with
`open_ledger(config)`):

- `errors` (sentence, correction, error_type, resolved, optional session), `srs_cards` (front, back,
  ease, interval_days, due_date) and `facts` (key, value).
- `create / read / list / update / delete`, with a whitelist of writable columns per table (an agent
  cannot set an id or a timestamp).
- **Write ownership is enforced in the data layer**: each table has one owning agent
  (default: `errors` → `corrector`, `srs_cards` → `curriculum`, `facts` → `sensei`, overridable with
  `Ledger(conn, owners=...)`). Any other agent gets `WriteNotAllowed`; reads are open to everyone.
- `ledger.tools(agent)` returns four tools (`ledger_create`, `ledger_read`, `ledger_update`,
  `ledger_delete`) bound to that agent, so the model drives the CRUD itself and the ownership check
  still applies. They are not registered on an agent yet, and nothing writes `facts` (reserved).
- **Read back at the start of a session** (`orchestration/context.py`): for a conversational agent (no
  output schema), the first message of a session gets two context blocks, each capped at 10% of the
  budget: the last 3 unresolved `errors` from earlier sessions ("Last time you wrote …") and the
  `srs_cards` due today (`srs.due_cards`), so the tutor can bring a mistake up and propose a review.
- **SRS cards** are scheduled by `memory/srs.py` (SM-2: a right answer pushes the next review further,
  a wrong one brings it back to tomorrow). The `curriculum` agent (`agents/curriculum.toml`) starts once the answer and `TurnFinished` are
  out, in the background: the turn does not wait for it, and the next question can be asked while it
  works. It calls its own tools `srs_add_card` and `srs_review` (one validated step per reply,
  at most `[curriculum] max_steps`) to add new cards and grade the ones the learner practised. A failure there is swallowed. Turn it off with `[curriculum] enabled = false`.

## Features

Who uses these features and how, step by step: [`doc/scenarios.md`](doc/scenarios.md).
How they are checked (tests, benchmark, scenario runs, live metrics): [`doc/evaluation.md`](doc/evaluation.md).

Committed: six of the seven ★★★ features (T2 is declined on purpose — a Japanese tutor has no honest use for running generated code).

| Code | Feature | Rating | Layer | Status |
|------|---------|--------|-------|--------|
| B1 | CLI chatbot (streaming, history, errors) | 👍 | 1–2 | ✅ |
| M1 | Session persistence & profile | ★ | 2 | ✅ |
| T5 | Structured output (JSON schema, one repair) | ★ | 3 | ✅ checker, corrector, classifier and handoffs |
| M3 | Structured memory (errors, SRS, facts) | ★★ | 3, 6 | 🚧 SRS cards by the curriculum agent; verified corrections written to `errors`; past errors and due cards injected at session start; `facts` reserved, unused |
| T1 | Tools (dictionary, conjugate, kanji) + MCP | ★★★ | 3, 5 | ✅ sensei calls them (native or prompted); MCP stdio client, own MCP server, pre-existing time server configured |
| EV3 | Logging, metrics, per-turn trace | ★★ | 3-4 | ✅ JSONL event log, `/metrics`, `/trace` |
| A2 | Multi-agent orchestration + router | ★★★ | 4-5 | ✅ handoffs, guards, fallback, three-tier router |
| A1 | ReAct (corrector) | ★★★ | 4 | ✅ checker → corrector → sensei |
| M2 | Token budgeting & compression | ★★ | 4-5 | ✅ per-agent context budget, running summary |
| R1 | Basic RAG (citations, JLPT metadata) | ★★ | 5 | ✅ bge-m3 embeddings, cosine, JLPT filter, citations |
| EV1 | Judge: claim-by-claim verification | ★★★ | 5 | ✅ judge runs after each reply that used passages or tools; `scripts/evaluate` compares models |
| R2 | Advanced RAG (reranking, scoring) | ★★★ | 6 | ✅ model reranker, threshold drop, scores in `/trace` |
| EV2 | Input / output guardrails | ★★★ | 6 | ✅ injection and off-topic refusal, PII masked on input and output, rules in `config.toml` |
| A5 | Switchable personas mid-session | ★ | 6 | ✅ |
| A4 | Prompt versioning (content hash) | ★ | 6 | ✅ |
| X1 | Web UI (FastAPI + SSE) | ★ | 7 | ✅ backend and single-page frontend (`frontend/index.html`, no build step) |
| EV4 | Adversarial suite | ★★ | 7 | ✅ 27 attacks, one command, report in [`doc/adversarial.md`](doc/adversarial.md) |
| A6 | Semantic cache (optional) | ★★ | 7 | out of the scope |

✅ done · 🚧 in progress · ⏳ to do — **Never cut:** T1, R2, A1, A2, EV1, EV2.

## Limits

What fails or falls short today, seen in tests and rehearsals on the reference laptop (RTX 4060
Laptop, 8 GB VRAM, 16 GB RAM, `qwen3.5:9b`). Each item says how it shows up.

- **Retrieval delays the first token.** With `[rag] model_rerank = true` (the default) the chat
  model grades the candidate passages before the answer starts: the first token comes after about
  5 s on a warm model (median of 6 turns); with `model_rerank = false` a warm turn started in 1.3 s,
  but the hybrid score keeps the right passage less often (9/12 against 12/12, see
  [`doc/benchmark.md`](doc/benchmark.md)). The first turn after Ollama loads the model adds about 10 s.
- **The model does not fit in VRAM.** About 28% of `qwen3.5:9b` runs on the CPU, which caps it near
  20 tokens/s (see [`doc/benchmark.md`](doc/benchmark.md)). `qwen3.5:4b` is about 3x faster but scored
  75-80% against 90% on the benchmark.
- **Corrections can be wrong.** The checker, corrector and sensei are the same 9B model. In one
  rehearsal `私は駅を行きます` was flagged as a vocabulary error ("行 is used for walk"), and the
  corrector, whose tools check words and not reasoning, revised it to an unrelated te-form; the next run
  of the same sentence was right. The `rule` the checker writes is often vague or off.
- **Romaji is the weakest part.** The benchmark's remaining errors are mostly wrong romaji with the
  right Japanese, and the sensei sometimes answers in romaji only ("Sushi ga suki desu") despite its
  `characters (romaji) - translation` format. Explanations can contain grammar mistakes too: one reply
gave the stem of 飲む as "飲ん (non)" instead of 飲み. The checker also marks a correct sentence typed in romaji
(`kinou eiga wo mimashita`) as an error, only because it isn't written in Japanese characters.
- **The adversarial suite (EV4) found 3 breaks out of 27 attacks** ([`doc/adversarial.md`](doc/adversarial.md),
  live run before the guardrail fix): a fake `<system>` tag made the sensei answer only the attacker's word,
  and phone numbers reached the model. Both are now stopped by the guardrails; poisoned corpus documents,
  system prompt leaks, out-of-scope requests and malformed input held; a 28,000-character message ends with
  a checker timeout, not a crash.
- **Guardrails are pattern-based (EV2).** Emails, French/Japanese/international phone numbers, cards and
  prefixed API keys (`sk-`, `ghp_`, `AKIA`) are masked before the model call and again in the reply, while it
  streams (the last word and any digit run are held back until they are complete). An injection (English,
  French, Japanese, or a fake `<system>` tag) is refused before the call: nothing is sent to the model or
  stored; the learner gets a plain refusal, not an error. A request that is not about Japanese (code, maths
  homework, "forget Japanese") is refused the same way, by patterns only: anything that mentions Japanese is
  let through, and the embedding tier planned in the user stories is not built. Rules are set in
  `[guardrails]` of `config.toml`. Patterns are easy to get around (a reworded or leetspeak injection, a number
  spelled out in words). The offline adversarial run shows 0 breaks; the live report in `doc/adversarial.md`
  predates this change and must be run again with `--live`.
- **The judge only checks what has a source (EV1).** After a reply, `trust/eval.py` checks each claim against the
  passages the agent kept and the tool results of the turn, and flags what they do not support
  (`[unverified]`, `[contradicted]`). A reply built on no passage and no tool (the model answering from memory)
  has nothing to be checked against and is not judged. The judge is the same local model, so it can be wrong,
  and it adds a model call after each sourced reply (about 16 s on the reference laptop). The turn is
  already over when it runs: in the web UI the session takes the next message at `TurnFinished`, and the
  verdict follows on the previous reply's stream. Turn it off with `[eval] enabled = false`. Structured replies (checker, corrector) are not judged.
- **The learner ledger is mostly used (M3).** Verified corrections are written to `errors`, the
  curriculum agent keeps the `srs_cards`, and both are read back at the start of a session; `facts` is
  never written (no agent has the `ledger_*` tools).
- **The time MCP server needs `uv`.** Without `uvx` on the PATH, Sensai says the server is unavailable
  and goes on without it.
- **One learner, one machine.** SQLite, one learner profile (`default`), no accounts; the knowledge
  base is 5 Markdown files and a dictionary of common N5 words. A word outside it comes back as
  `found: false`, and the model answers from its own knowledge.

## Layered plan

| Layer | Content |
|-------|---------|
| 1 | Foundations: contracts (events, AgentSpec, Handoff, config), SQLite, Ollama transport, CLI + logger |
| 2 | Base loop: `run_turn()`, persistence, model benchmark, error handling |
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
The folders have no `__init__.py`: Python imports them as namespace packages. The module map below
says where each piece of code lives, and where the planned ones go, so new code always has an
obvious home.

```
sensai/
  core/           shared contracts: messages, errors, config, events, specs    (A)
  llm/            Ollama HTTP transport, the only code that calls the model    (C)
  orchestration/  turn loop, executor, orchestrator, router                     (A)
  memory/         SQLite storage, sessions, ledger                              (B)
  knowledge/      tools, conjugator, MCP client and server, RAG, ReAct          (C)
  trust/          event log, metrics, tracing, guardrails, judge                (D)
  surfaces/       CLI and its renderer, demo script, web backend                (D)
frontend/         index.html: the web chat page                                 (D)
agents/           one TOML spec per agent: sensei (default), checker (T5, hands off to the corrector or the sensei),
                  corrector (A1, ReAct), curriculum (SRS),
                  classifier (router tier 3),
                  demo/ping and demo/pong (loop stopped by the guards)
scripts/          benchmark.py: model benchmark, results in doc/benchmark.md    (C)
                  adversarial.py + adversarial_cases.toml: EV4 suite, report in doc/adversarial.md (C)
tests/            one folder per package, mirroring sensai/
doc/              work plan, architecture diagrams, previous README
config.toml       project configuration
routing.toml      routes: which agent starts a turn, with the router's rules and examples
corpus/           grammar notes for RAG, one "# Title [N5]" header per section
personas/         one TOML per persona: teacher (default), strict, casual
```

Runtime files (`data/*.db`, `logs/`) are created on first run and never committed.

### Module map

Built modules, then planned ones with the layer that adds them.

| Package | Module | What it does |
|---------|--------|--------------|
| `core/` | `events.py` | the event stream that surfaces subscribe to |
| | `agent_spec.py` | `AgentSpec`, loaded from `agents/<name>.toml` |
| | `handoff.py` | the envelope passed between agents |
| | `messages.py` | chat message format sent to the model |
| | `errors.py` | error categories (`ErrorKind`) |
| | `config.py` | layered config: defaults < `config.toml` < env < CLI |
| | `persona.py` | personas from `personas/<name>.toml`, applied to the learner-facing agents |
| | `ids.py` | uuid ids for sessions, turns and runs |
| `llm/` | `ollama.py` | chat streaming, embeddings, `/api/tags`, `/api/show` |
| `orchestration/` | `session.py` | `Session`: active agent, history, profile, client |
| | `turn.py` | `run_turn()`: guardrails → orchestrator → keep or roll back the history |
| | `orchestrator.py` | runs the agents of a turn: handoff chain, guards, fallback to the sensei |
| | `guards.py` | max hops, handoff cycles, turn budget, per-agent timeout |
| | `executor.py` | `AgentRun`: runs one `AgentSpec` and emits its events |
| | `context.py` | per-agent context: providers (profile, handoff note) and history fitted to `num_ctx` |
| | `stream.py` | thread bridge from the blocking Ollama stream to asyncio |
| | `structured.py` | structured output: schema in the prompt, validation, one repair |
| | `router.py` | picks the agent that starts a turn: rules, embeddings, then the classifier agent |
| `memory/` | `db.py` | SQLite connection: WAL, foreign keys, `:memory:` for tests |
| | `migrations/`, `migrate.py` | numbered SQL migrations, tracked by `PRAGMA user_version` |
| | `repository.py` | `SessionRepository` and its async facade, sessions, profile |
| | `ledger.py` | `errors`, `srs_cards`, `facts` with per-agent write ownership |
| | `prompts.py` | `PromptStore`: prompts by content hash, pins, run metrics per prompt |
| | `summary.py` (orchestration) | sliding-window summarisation (M2) |
| | `srs.py` | SM-2 spaced-repetition scheduling of the `srs_cards` |
| `knowledge/` | `tools.py` | tool registry, native or prompted tool calling, own tools on `data/*.json` |
| | `japanese.py` | rule-based verb conjugator and kana → romaji |
| | `mcp.py` | MCP client, JSON-RPC 2.0 over stdio |
| | `mcp_server.py` | our own MCP server, serving the own tools |
| | `rag.py`, `advanced_rag.py` | JLPT-tagged chunks, word-overlap and bge-m3 indexes, reranking |
| | `retrieval.py` | the turn's retrieval: candidates, model reranker, fallbacks |
| | `react.py` | the corrector's ReAct loop |
| | `toolbox.py` | a session's tools (own + MCP) and retriever |
| `trust/` | `logging.py` | JSON-lines event log |
| | `metrics.py` | tokens, speed, TTFT and cost per turn (`/metrics`) |
| | `tracing.py` | per-turn trace of the agent runs (`/trace`) |
| | `guardrails.py` | input filter (prompt injection, PII) and output filter |
| | `eval.py` | LLM judge: claim-by-claim verification against retrieved chunks |
| `surfaces/` | `cli.py` | command-line chat and its renderer |
| | `demo.py` | scripted turn through the logger and the renderer |
| | `web/app.py` | FastAPI app: sessions and turns streamed as server-sent events |
| | `web/settings.py` | switch a session's model, agent, persona or reasoning (`think`) |
| | `web/sessions.py` | the open sessions, resumed from storage on demand |
| | `web/sse.py` | one SSE message per event; `TurnFinished` also carries the turn's metrics and trace |
| | `web/frontend.py` | serves `frontend/index.html` at `/` |
| `sensai/` | `__main__.py` | entry point for `python -m sensai` |

### Rules that keep it stable

- **Dependencies point inward.** Every package may import `core`; `core` imports nothing from the
  others. `surfaces` never contains business logic.
- **Agents are data, not code.** An agent is a TOML file loaded into an `AgentSpec`
  (model, `num_ctx`, prompt, tools, output schema). Adding the corrector or the judge means adding a file.
- **One transport for every agent.** `OllamaClient.chat_stream(model, messages, num_ctx=..., format=...)`
  takes the model and options on each call, so agents with different models share one client. Errors carry a
  `kind` (`model_missing`, `connection`, `timeout`, `bad_request`, `stream`; the turn loop adds `empty_input`,
  `invalid_output`, and the guards `max_hops`, `handoff_cycle`, `turn_budget`) that surfaces can display, including failures in the middle of a reply. The stream
  yields the reply as text tokens, and the model's reasoning as `Thought` items when thinking is on; its
  return value is a `ChatStats` with the final chunk's token counts and durations, which `run_turn()` puts
  into `TurnFinished`.
  `capabilities(model)` and `context_length(model)` read `/api/show` (cached), so tools can choose native
  or prompted tool calling and specs can be checked against the model's maximum context.
- **Surfaces only subscribe.** The turn loop emits events (`core/events.py`) and the CLI, the logger,
  the trace view and the web UI only consume them.

### How a turn flows

| Stage | Package |
|-------|---------|
| 1 · Input | `surfaces` → `trust` (input guardrails) |
| 2 · Decide | `orchestration` (orchestrator, router) |
| 3 · Build the prompt | `orchestration` (context builder, budget) with the agent's spec from `core` |
| 4 · Run the loop | `orchestration` (executor) → `llm`, `knowledge` (tools, RAG) |
| 5 · Remember | `memory` (sessions, ledger) |
| 6 · Check and show | `trust` (judge, output guardrails, logging) → events → `surfaces` |

Today a turn is `run_turn(session, text)` in `orchestration/turn.py`: an async generator that yields
`TurnStarted`, then each agent run as `AgentRunStarted → [Thinking…] → Token… → AgentRunFinished`
(`StructuredOutput` instead of `Token` for an agent that has an `output_schema`), then `TurnFinished`, with
`Error` or `GuardrailTriggered` when they apply. `Session` (`orchestration/session.py`) holds the active
agent, the history, the learner profile and the client, and stores each message. Every turn goes `run_turn() → JsonlLogger → CLIRenderer`:
the logger writes each event to `[logging].path` (`logs/sensai.jsonl`) and the renderer prints it and
keeps the last turn's metrics and trace for `/metrics` and `/trace`.

### Multi-agent turns (A2)

The `Orchestrator` (`orchestration/orchestrator.py`) starts each turn with the session's agent. An agent
that lists `handoffs = ["sensei"]` in its spec gets a required `handoff` field in its output schema:
`{"to": "sensei", "reason": "..."}` passes the turn on, `null` ends it. The orchestrator then emits a
`HandoffEvent` and runs the named agent under the first one (`parent_run_id`), with the first agent's
validated output (minus `handoff`) as the payload, shown to it as a note in its context. Each agent's
context is built by `ContextBuilder` (`orchestration/context.py`): its system prompt, then the learner
profile and the handoff note as budgeted blocks, then as much recent history as fits in 75% of its
`num_ctx` (the rest is kept for the answer). Only the last agent's answer is stored in the history.

Guards (`orchestration/guards.py`, `[orchestration]` in `config.toml`) stop a crew that loops or hangs:

| Guard | Stops the turn when | Setting |
|-------|---------------------|---------|
| Max hops | more than `max_hops` handoffs happen in one turn | `max_hops = 3` |
| Cycle | an agent receives the turn after it already ran in this turn | always on |
| Turn budget | all the agents of the turn take longer than the budget | `turn_budget_s = 180` |
| Agent timeout | one agent takes longer than its spec's `timeout_s` | in `agents/<name>.toml` |

When the chain fails (an agent error, invalid output, a timeout or a guard), the fallback agent
(`[agents] fallback = "sensei"`) answers the learner instead, with the last handoff's findings. There
is no fallback when Ollama is down, when the turn budget is spent, or when the sensei already ran.
`python -m sensai --agent demo/ping` shows it: ping and pong hand off to each other, the cycle guard
stops them, the sensei answers, and `/trace` lists the three runs.

### Routing (A2)

Without `--agent`, the `Router` (`orchestration/router.py`) picks the agent that starts each turn from
the routes in `routing.toml`, tried in file order. Tier 1 matches each route's regex rules: a message
written only in Japanese, or a request to check next to some Japanese, goes to the checker; an English
question goes to the sensei. Tier 2 handles what the rules leave alone (e.g. a sentence in romaji): it
embeds the message with `bge-m3` and picks the route whose closest example is at least
`embedding_threshold` similar (0.55) and `embedding_margin` (0.05) ahead of the next route. It runs on
the CPU (~0.2 s per message) so the chat model keeps the GPU; without `bge-m3` the tier stays silent.
Tier 3 runs only when both are unsure: the classifier agent (`agents/classifier.toml`, same model as
the sensei, so nothing is swapped) gets the route descriptions and answers `{agent, confidence, reason}`
in JSON, with `agent` limited to the route names (~2 s). An invalid reply or an Ollama error decides
nothing. When no tier decides, the default agent (`[agents] default`) answers.

Starting a message with `@<agent>` (e.g. `@checker kinou eiga wo mimashita`) overrides every tier; the
prefix is removed before the message is stored. Internal agents (`[router] internal_agents`: the
curriculum, the classifier, `demo/*`) can't be reached this way unless `[router] debug = true`;
`@curriculum hello` is routed like any other message. A tier decision below `confidence_floor` (0.5) goes to
the fallback agent (the sensei). Each decision is a `RouteDecided` event (agent, tier, confidence,
reason): it is written to the event log and `/trace` shows it, e.g.
`Routed to 'checker' by embeddings (0.57): closest example 'sumimasen, eki wa doko desu ka'`.
Every decision records its tier, its confidence and why. `--agent <name>` turns routing off.

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

### Personas (A5)

A persona is the voice of the agents that talk to the learner: one TOML file in `personas/` with a
`name`, a `description`, a `tone` and a `scope`. Three ship with the project: `teacher` (the default,
`[personas] default`), `strict` (every mistake corrected, polite Japanese only) and `casual` (a
friendly conversation partner using plain forms). Adding a file adds a persona, with no code change.

- **Applied to every learner-facing run.** Before each run, the orchestrator appends the session's
  persona to the system prompt of agents without an `output_schema` (the sensei); the checker and the
  classifier keep their exact prompts.
- **Switched mid-session.** `/persona casual` changes the persona for the next turns; the history and
  the session stay as they are, so the conversation carries on in the new tone.
- **Versioned like any prompt.** The persona is part of the prompt, so each persona gets its own
  prompt hash (A4) and their runs can be compared with `session.prompts.metrics("sensei")`.

### Tools and MCP (T1)

An agent lists its tools in its spec: `tools = ["dictionary", "conjugate", "kanji", "mcp:time"]`
(`mcp:<server>` stands for every tool of that server). The executor offers them natively when `/api/show`
says the model supports tools, otherwise it writes them into the system prompt and holds back a reply that
starts like a JSON `{"tool": ..., "args": ...}` block until it knows whether it is a call. Each call is a
`ToolCall` and a `ToolResult` event, shown in the chat and in `/trace`; at most `[tools] max_rounds` rounds.

Switches, in `config.toml`: `[tools] enabled = false` turns every tool off; `[tools] disabled = ["conjugate"]`
turns off the tools listed, for every agent (own tools and MCP tools by name); `[mcp] enabled = false` stops
every MCP server, and `enabled = false` on one `[[mcp.servers]]` entry stops that server only.

- **Own tools** read local data (`sensai/knowledge/data/`): a dictionary of common N5 words with particle
  notes, the N5 kanji with readings and stroke counts, and a rule-based conjugator (ichidan, godan, する,
  来る, the 行く and ある exceptions) for 16 forms. An unknown word comes back as `found: false`, never a guess.
- **MCP** (`knowledge/mcp.py`): JSON-RPC 2.0 over stdio (initialize handshake, `tools/list` with pagination,
  `tools/call`, pings). Servers come from `[[mcp.servers]]` in `config.toml` and start when the chat starts;
  the banner lists their tools, and one that cannot start is reported and skipped. A name already taken gets
  the server's name as prefix.
- **Our own MCP server**: `python -m sensai.knowledge.mcp_server` serves the own tools to any MCP client.
- **Pre-existing server**: the reference time server (`uvx mcp-server-time`, needs `uv`; first start downloads
  it) gives the sensei the time in Japan.

### Corrections (A1)

A sentence routed to the checker comes back as data. With a mistake it goes to the corrector
(`agents/corrector.toml`, `strategy = "react"`): on a 1024-token context it writes a Thought, calls a tool,
compares the Observation with its idea (`Revised: yes/no`), then gives the rule and the correction. A final
answer before any tool is sent back, and an Observation the model invents is cut. The steps stream as dim
reasoning lines. Its data (`correction`, `rule`, `verified`, `verified_with`) is handed to the sensei, which
explains it; a verified correction is written to the `errors` table.

### Retrieval (R1, R2)

An agent with `rag = true` (the sensei) gets corpus passages for each question:

1. **Candidates (R1).** `corpus/*.md` is split at each `# Title [N5]` header; every chunk and its title are
   embedded with `bge-m3` (cached in `[rag].index_path`, keyed by content hash), and the closest
   `candidate_k` chunks at or below the learner's JLPT level (read from the profile, e.g. "Level: N5") above
   `min_similarity` are kept.
2. **Rerank (R2).** The chat model reads the question with the candidates and scores each 0–1; those under
   `relevance_threshold` are dropped and `top_k` reach the prompt. Embeddings alone score "how do I say cat?"
   almost as high as a grammar question, so they only pick candidates. If the model fails, a hybrid score
   (chunk + title similarity + word overlap) is used; without `bge-m3`, word overlap.
3. **Citations.** The passages come with `[file § section]`; an answer that cites none gets a `Sources:` line.
   `/trace` shows each candidate's score, kept or dropped.

Reranking costs one short model call (~2–3 s) per sensei turn; `model_rerank = false` uses the hybrid score.

### Prompt versioning (A4)

Every agent's system prompt is versioned by its content: `AgentSpec.prompt_hash` is the first 12 hex
digits of its SHA-256, so editing a prompt gives it a new hash and nothing has to be numbered by hand.

- **Stored on first use.** Before each run, the orchestrator hands the spec to the `PromptStore`
  (`memory/prompts.py`), which saves the prompt in the `prompts` table under its hash (agent, content,
  date). `AgentRunStarted` carries the hash, so the JSONL log has it too.
- **Recorded on every run.** When a turn's answer is kept, each of its agent runs is written to
  `agent_runs` with its prompt hash, status, duration and tokens.
- **Compared by numbers.** `session.prompts.metrics("sensei")` groups that agent's runs by prompt
  hash: runs, successful runs, average duration (ms) and average tokens. `versions("sensei")` lists
  its hashes, oldest first.
- **Rolled back by hash.** `prompt_version = "<hash>"` in `agents/<name>.toml` pins the agent to
  that stored prompt: it runs with it instead of the one in the file. An unknown hash fails the turn
  with a `SpecError`.

### Web UI (X1)

`python -m sensai.surfaces.web` serves the same sessions as the CLI over HTTP (`--host`, `--port`, or
the `[web]` section of `config.toml`). It only subscribes to the event stream: a turn runs through
`run_turn()` exactly as in the CLI, and each event goes to the browser as one server-sent event, named
after its class, with its fields as JSON.

Open `http://127.0.0.1:8000` for the chat page (`frontend/index.html`, served by the same app, so no
CORS setup and no build step). It streams tokens, switches model and reasoning, shows session metrics and
the `/trace` of the last turn, and resumes the last conversation after a reload (session id kept in
`localStorage`). Tailwind and the fonts load from a CDN, so the page needs internet access.

| Method | Path | Does |
|---|---|---|
| GET | `/` | the chat page |
| GET | `/api/health` | `{"status": "ok"}` |
| GET | `/api/sessions` | stored sessions: id, date, model, title |
| POST | `/api/sessions` | a new session, with its settings |
| GET | `/api/sessions/{id}` | a session's settings and history; a stored one is resumed |
| POST | `/api/sessions/{id}/turns` | `{"text": "..."}` -> the turn's events as `text/event-stream` |
| PUT | `/api/sessions/{id}/model` | `{"model": "qwen3.5:4b"}`; 400 when it isn't installed |
| PUT | `/api/sessions/{id}/agent` | `{"agent": "checker"}` pins an agent (routing off, like `--agent`); `{}` routes again |
| PUT | `/api/sessions/{id}/persona` | `{"persona": "strict"}` |
| PUT | `/api/sessions/{id}/thinking` | `{"think": true}`; 400 when the model can't reason |
| GET | `/api/models` | installed Ollama models; 503 when Ollama is down |
| GET | `/api/personas` | persona names and descriptions |

```
$ curl -N -X POST localhost:8000/api/sessions/<id>/turns -H 'content-type: application/json' -d '{"text": "how do I say cat?"}'
event: TurnStarted
data: {"turn_id": "...", "ts": 1793..., "session_id": "...", "user_input": "how do I say cat?"}

event: RouteDecided
...
event: Token
data: {"turn_id": "...", "ts": 1793..., "run_id": "...", "text": "猫"}
...
event: TurnFinished
data: {..., "metrics": {"total_tokens": 412, "tokens_per_sec": 38.2, "cost_usd": 0.000321, "duration_s": 9.4}, "trace": "[trace] Turn ID: ..."}
```

`TurnFinished` is the one event with extra fields: `metrics` (from `trust/metrics.py`) and `trace`
(from `trust/tracing.py`), which the page shows in its footer and in the trace window.

A session runs one turn at a time (a second one gets 409). Closing the connection mid-reply cancels
the turn as Escape does in the CLI. Every change keeps the history. `allow_origins` lists the browser
origins the frontend may call from (CORS).

## Organisation

### Branches

Every task has its own branch, named `layerN/<owner letter>-<task>` (e.g. `layer1/c-ollama-transport`),
linked to its issue on the board. Smaller pieces of a task are commits on that branch.

### Workflow

```
layerN/<letter>-<task>  ──  PR  →  main  ──  CI  →  Epitech repo
```

1. Before starting, update the task branch from `main`: `git switch <branch> && git merge origin/main`.
2. Open a PR into `main`, with `Closes #<task>` so merging closes the task.
3. CI must be green (tests, coverage ≥ 85%) before merging. Once merged, CI pushes `main` to the Epitech repo.

The `layerN/integration` branches from the original plan exist but aren't used: every PR goes straight
to `main`.

When resolving a merge conflict in a test file, keep the tests from both sides: a deleted test doesn't
fail, it silently stops protecting the code.

## CI

Code is pushed to the mirror repo. Every push runs the tests with coverage on Python 3.10 (the
oldest version the subject allows) and 3.12 (**blocking under 85%**, reports kept as artifacts for
90 days); if everything passes on `main`,
the full history is pushed to the Epitech repo.

```bash
python -m pytest --cov          # locally
```

## User stories

Two user stories per feature, with acceptance criteria and how each one is verified: see [`doc/user_stories.md`](doc/user_stories.md).

## Team

| Member | Role | GitHub | Owns |
|--------|------|--------|------|
| A | Core | Eukyrios | Ollama transport, event stream, agent executor, orchestrator, router, config |
| B | Memory | EvansRamin | SQLite, profile, facts, SRS, context builder, budgeting, compression |
| C | Knowledge | DanielLoww | Tool registry, own tools, MCP client, RAG pipeline, corrector spec |
| D | Trust & surface | NicolasGonzalz | Logging, metrics, trace view, judge spec, guardrails, CLI and Web UI |
