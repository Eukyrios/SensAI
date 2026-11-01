# Evaluation method

How we check that Sensai works and that a change made it better rather than worse. Four levels, from
the cheapest (no model, every push) to the most expensive (a person reading the replies). Each one
says what it checks, the pass criteria, and how to run it.

| Level | Checks | Needs a model | When |
|---|---|---|---|
| 1. Automated tests | the code does what it should, model replies faked | no | every push (CI) |
| 2. Model benchmark | the quality and speed of a model on 20 fixed questions | yes | when a model or the sensei's prompt changes |
| 3. Scenario runs | the whole crew on the usage scenarios | yes | before a merge that touches a turn, and before the demo |
| 4. Live measurements | every real turn: tokens, speed, first token, errors, per prompt version | yes | always on |

## 1. Automated tests

**What:** about 600 pytest tests over every module: config, event stream, executor, handoffs and
guards, router tiers, context budget, tools, RAG, storage, guardrails, CLI and web routes. Model
replies come from fakes (`tests/fakes.py`), so the tests are fast and give the same result every
time, including on failure paths (Ollama down, timeouts, invalid JSON, handoff loops).

**Pass criteria:** every test passes and total coverage is at least 85%; CI blocks the branch
otherwise.

**Run:** `python -m pytest --cov`. CI runs it on every push and pull request.

## 2. Model benchmark

**What:** 20 questions a beginner asks (translation both ways, particles, conjugation, kanji readings,
corrections, keigo, counters, a self-introduction), sent to the sensei with its real settings.

**Criteria:**
- **Automatic score:** each question lists what a right answer must contain (e.g. 食べませんでした for
  the polite negative past of 食べる); a reply passes when it contains one of them, case-insensitive.
- **Manual review:** a person reads every reply and grades it correct, partly correct or wrong,
  because a keyword match can pass a wrong explanation or miss a right one written differently.
  Correct: the Japanese, the romaji and the explanation are all right. Partly correct: the Japanese
  is right but the romaji or the explanation has a mistake. Wrong: the Japanese answer is wrong.
- **Speed:** generation speed (tokens/s from Ollama's `eval_duration`), first token and load time.

**Run:** `python -m scripts.benchmark`; results go to `doc/benchmark_results.json` and are written up
in [`benchmark.md`](benchmark.md). It measures the sensei alone, not the crew.

## 3. Scenario runs

**What:** the usage scenarios of [`scenarios.md`](scenarios.md) typed into the real chat, on the
seeded demo database (`python -m scripts.seed_demo`, then `SENSAI_STORAGE__DB_PATH=data/demo.db`).

**Pass criteria:** each step shows its "Expected result": the right agents in `/trace`, the handoffs,
the tool calls, the guardrail event, a correct Japanese answer in the
`characters (romaji) - translation` format. Since the model is not deterministic, a scenario is run
twice; a step that fails once is written down in the README's Limits section.

**Run:** by hand, or piped: `printf '%s\n' "私は映画に見ます" "/trace" "/quit" | python -m sensai`.

## 4. Live measurements

**What:** every turn emits its events to the JSONL log (`logs/sensai.jsonl`) and its runs are stored
in SQLite with their prompt hash.

**Criteria and where to read them:**
- `/metrics`: tokens in and out, generation speed, time to first token, estimated cost of the last turn.
- `/trace`: the route decision, each agent run, tool calls and their results, retrieved passages
  with their scores.
- `session.prompts.metrics("sensei")`: per prompt version, runs, successful runs, average duration
  and tokens. A prompt or persona change shows as a new hash, so two versions are compared on the
  same numbers; `prompt_version` rolls back to the better one.

## Faithfulness across models (EV1)

`python -m scripts.evaluate --models qwen3.5:4b qwen3.5:9b --limit 5 --report doc/evaluation_results.json`
asks both models the same questions through the real pipeline (tools and corpus included) and has the
same judge (`qwen3.5:9b`) check every reply against the sources it had. Run on 2026-11-01:

| Model | Replies judged | Faithfulness | Relevance |
|---|---|---|---|
| `qwen3.5:4b` | 5/5 | 0.78 | 1.00 |
| `qwen3.5:9b` | 4/5 | 1.00 | 1.00 |

The 4B answers every question but invents details its sources do not hold (a reading of 鬱積 given as
"fusaaku", a meaning added from memory); the judge flags them as unverified. One 9B reply used no source,
so it was not scored. Every reply and flagged claim is in [`evaluation_results.json`](evaluation_results.json).

## What it does not measure

- **Correction accuracy:** the benchmark grades the sensei; the checker and corrector are checked by unit
  tests and by the scenario runs, not by a scored set of wrong sentences.
