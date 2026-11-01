# Memory, context and curriculum: member B's work

Everything the learner's data goes through: storage, sessions, the learner ledger, spaced repetition,
the token budget and the compression of long conversations. This page says what each piece does, why it
is built that way, how it is checked, and what it does not do.

Features covered: **M1** (session persistence and profile), **M2** (token budgeting and compression),
**M3** (structured memory). It also supports **A4** (prompt runs are stored in the same database) and
**EV3** (`turns` and `agent_runs` hold the numbers behind `/metrics` and `/trace`).

## 1. Storage (`sensai/memory/`)

| File | Role |
|------|------|
| `db.py` | one SQLite connection: WAL, `foreign_keys = ON`, rows as `sqlite3.Row`, serialised across threads |
| `migrate.py`, `migrations/00N_*.sql` | numbered migrations, tracked by `PRAGMA user_version` |
| `repository.py` | `SessionRepository` (sync) and `AsyncSessionRepository` (facade) |
| `ledger.py` | `errors`, `srs_cards`, `facts` with per-agent write ownership |
| `srs.py` | SM-2 scheduling and the curriculum agent's two tools |

**Schema.** `001_init.sql`: `sessions` → `messages` → `turns` → `agent_runs`, all linked by foreign keys
with `ON DELETE CASCADE`. Ids are text uuids from `core/ids.py`, the same ids the events carry, so a turn
can name its `turn_id` and run ids before any row exists and the logs, the trace and the database share
one id. `turns.assistant_message_id` is nullable so a failed reply is still recorded.
`002_active_leaf_and_profile.sql`: `sessions.active_leaf_id` and `learner_profiles`.
`003_ledger.sql`: `errors`, `srs_cards`, `facts`, with indexes on `resolved`, `due_date` and `key`.
(`004_prompt_versions.sql` belongs to member A's prompt versioning.)

**Migrations are idempotent.** The runner applies every `NNN_*.sql` file above the database's
`user_version`, in order; running it again changes nothing. Each file is one transaction.

**Sync repository plus async facade.** The repository is plain synchronous SQLite, trivial to test.
`AsyncSessionRepository` runs each call through `asyncio.to_thread`. We did not use `aiosqlite`: no new
dependency, the storage logic stays in one place, and the thread overhead does not matter for one learner
and a handful of queries per turn.

### Thread safety (commit `47fdb92`)

The web backend and `asyncio.to_thread` use the same connection from several threads, and
`check_same_thread=False` alone does not make that safe. `db.py` now returns a `LockedConnection`
that owns an `RLock`:

- every `execute`, `executemany`, `executescript` and every `fetch*` of its cursors takes the lock;
- `with conn:` holds the lock for the whole transaction, so another thread can neither interleave a
  statement nor commit a half-finished transaction;
- the lock is re-entrant, so code that nests `with conn:` blocks keeps working.

The ledger, the curriculum agent and the prompt store use the same connection, so they are covered too.
Limit: iterating a cursor with `for row in cursor` is not locked (nothing in the code does it).

## 2. Sessions, branches and the learner profile (M1)

- `add_message(session_id, role, content, parent_id)` inserts a message and moves
  `sessions.active_leaf_id` onto it in **one transaction**. Messages form a tree through `parent_id`.
- `load_history(session_id)` walks the tree from `active_leaf_id` up to the root with a recursive CTE and
  returns the messages oldest first. `set_active_leaf()` switches branch. An unknown session raises
  `SessionNotFound`.
- `discard_leaf()` removes a failed turn's messages and puts the leaf back on its parent.
- `learner_profiles` holds the profile (`get_profile`, `save_profile`); `profile_system_message()` turns it
  into a system message. The profile is injected into every prompt and **never stored in `messages`**.
- `Session.resume(config, id)` reloads a stored session (`python -m sensai --session <id>`). The session row
  is created with its first message, so an empty session leaves nothing behind.
- Why a tree and not a list: `parent_id` and `active_leaf_id` are the basis for editing a past message and
  forking a branch. The schema allows it; the chat does not expose it.

## 3. The learner ledger (M3)

Three tables, one class (`Ledger`, opened with `open_ledger(config)`).

- **CRUD with a whitelist.** Each table has a list of writable columns; an agent cannot set an id or a
  timestamp. `create`, `read`, `list`, `update`, `delete`.
- **Write ownership is enforced in the data layer**, not in a prompt: `errors` → `corrector`,
  `srs_cards` → `curriculum`, `facts` → `sensei` (overridable). Any other agent gets `WriteNotAllowed`.
  Reads are open to everyone. SQLite has no per-user rights, so the check lives in `ledger.py`.
- **`ledger.tools(agent)`** returns four tools (`ledger_create`, `ledger_read`, `ledger_update`,
  `ledger_delete`) bound to an agent, so a model can drive the CRUD itself with the ownership check still
  applied. They are not given to any agent (see section 7).

### What writes and reads it today

| Table | Written by | Read by |
|-------|-----------|---------|
| `errors` | the orchestrator, after a verified correction from the corrector (A1); marked resolved when the learner later writes the correction | `ErrorsProvider`, first message of a session |
| `srs_cards` | the curriculum agent, through `srs_add_card` and `srs_review` | `DueCardsProvider`, and the curriculum agent's own prompt (known cards) |

### Read back at the start of a session (commit `d971e87`)

For a conversational agent (no output schema) and only on the **first message of a session**, the context
gets two blocks, each capped at 10% of the budget:

- **Past mistakes**: the last 3 unresolved `errors` from *earlier* sessions, as
  "Last time you wrote … it should be …", with the instruction to remind the learner of one, briefly, once,
  when it fits.
- **Due cards**: the `srs_cards` due today (at most 6), with the instruction to offer a review, one word at
  a time.

Errors from the current session are excluded, a failing query returns nothing instead of an error, and
structured agents (checker, curriculum) never get these blocks. Try it with `python -m scripts.seed_demo`
(it seeds one error and three cards, two of them due), then `python -m sensai --session demo-food`.

## 4. Spaced repetition and the curriculum agent (M3)

**SM-2 (`memory/srs.py`).** `next_review(ease, interval_days, grade, today)` is a pure function, so it is
tested with fixed dates. Grades: `again` 1, `hard` 3, `good` 4, `easy` 5. Ease is updated by the SM-2
formula and never goes below 1.3. A grade under 3 brings the card back to **tomorrow**; otherwise the
interval goes 1 day → 6 days → `round(interval × ease)`.

**The model never computes a date.** It only picks a tool and a grade; the code applies the scheduling.

**The curriculum agent** (`agents/curriculum.toml`, `orchestration/curriculum.py`) runs **after** the
visible answer and `TurnFinished`, never before, so it cannot delay or alter the reply.

- One tool call per reply, validated against the agent's output schema (`srs_add_card`, `srs_review`,
  or `done`), at most `[curriculum] max_steps` (8) steps. It reads the last exchange and the known cards.
- It is the only agent allowed to write `srs_cards` (ownership, section 3).
- Two guards keep a small model honest: a new card's word (or its reading) must appear in the exchange,
  and a card is graded only when the tutor's previous message asked a question.
- Best effort: any failure is swallowed and never reaches the learner. Turn it off with
  `[curriculum] enabled = false`.

## 5. Token budgeting and context (M2) (`orchestration/context.py`)

- **One budget per agent.** `ContextBuilder` gives each agent `num_ctx × 0.75` (25% is kept for the
  reply), minus its system prompt.
- **Providers.** Anything that adds context (profile, handoff note, summary, past errors, due cards)
  is a provider with a `priority` and a `max_share` of the budget. A block that is over its share is cut;
  one that does not fit is dropped.
- **History fits what is left**, newest messages first, and always keeps the last one. The ReAct corrector
  works on a small context and takes no history.
- **Estimate.** Tokens are estimated at 3.5 characters each. `TokenEstimator.calibrate()` can correct that
  ratio from Ollama's real `prompt_eval_count`.

### Compression (`orchestration/summary.py`)

When the unsummarised history passes `[memory.summary] trigger` (70%) of the history budget, the model folds
the oldest turns into a **running summary** (the new turns are merged into the previous summary, at most 25%
of the window), keeping the 6 newest messages as they are. The summary goes into the prompt through
`SummaryProvider`; the folded turns leave the window. The summary lives in the session and is rebuilt after
`Session.resume`. If Ollama fails while summarising, the turn goes on with the plain window.

## 6. How it is checked

| Area | Tests |
|------|-------|
| connection, migrations | `tests/memory/test_db.py`, `test_migrate.py` |
| repository, branches, profile | `tests/memory/test_repository.py` (29), `test_async_repository.py` |
| ledger, ownership, tools | `tests/memory/test_ledger.py` |
| SM-2 with fixed dates | `tests/memory/test_srs.py` (14) |
| context builder, providers, session-start recall | `tests/orchestration/test_context.py` (16), `test_knowledge_turn.py` |
| compression | `tests/orchestration/test_summary.py` (9) |
| curriculum pass | `tests/orchestration/test_curriculum.py` (13) |
| session loop and persistence | `tests/orchestration/test_session.py`, `test_chat.py` |

The whole suite is 697 tests at 95.6% coverage (the CI bar is 85%). Coverage of this area: `srs.py`,
`migrate.py`, `prompts.py` 100%; `repository.py` 97%; `ledger.py` 96%; `curriculum.py` 99%;
`summary.py` 98%; `db.py` 85% (the locking error path).

User stories: M1-1, M1-2 (persistence and profile), M2-1, M2-2 (budget and compression), M3-1, M3-2
(mistakes and spaced repetition) in [`user_stories.md`](user_stories.md).

## 7. Limits

- **No agent is given the `ledger_*` tools**, so the `facts` table has no writer. Giving them to the sensei
  would add four tools to a 9B model that already has `dictionary`, `conjugate`, `kanji`, `time` and RAG
  passages, with a risk of worse answers and malformed tool calls; we chose not to.
- **`errors` are written by code, not by an agent's tool** (`Orchestrator.record_errors`), and an error is
  resolved by a simple text match (the correction appears in the learner's message, the mistake does not).
  A learner who only quotes the correction in a question resolves it too early.
- **The recall happens on the first message only**, and the sensei is told to offer a review whatever the
  question is, so a very specific first question can get an unrelated offer.
- **Token counts are estimated, not measured.** The budget uses 3.5 characters per token for every text;
  Ollama's real `prompt_eval_count` is not fed back. The 25% reply reserve keeps the budget safe, and
  Japanese text, which has more tokens per character, is the case where the estimate is weakest.
- **Branches are stored, not exposed.** The tree and `set_active_leaf()` exist; there is no command or
  button to edit a past message.
- **One learner, one database file.** A single profile (`default`), no accounts.
