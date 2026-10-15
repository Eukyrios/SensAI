# Model benchmark

Which model the sensei should run, measured on the team's reference laptop: RTX 4060 Laptop GPU
(8 GB VRAM), 16 GB RAM, Ollama 0.18.2. Run twice on 2026-10-14 with `python -m scripts.benchmark`,
once with `num_ctx = 2048` and once with `num_ctx = 8192`; the raw replies of the 8192 run (the
current setting) are in [`benchmark_results.json`](benchmark_results.json).

## Method

- **20 questions** covering what a beginner asks: translation both ways, particles (は/が, を/に),
  conjugation, kanji readings, corrections of wrong sentences, keigo, counters, a self-introduction.
- **The sensei's real settings**, read from `agents/sensei.toml`: its system prompt (pronunciation
  format, Japanese-only rules), `num_ctx`, temperature 0.3, reasoning off. Each question asks for a
  short answer (2 sentences at most).
- **Automatic score:** each question lists the answer the reply must contain (e.g. 食べませんでした
  for the polite negative past of 食べる), matched without regard to case.
- **Manual review:** every reply was then read and graded correct, partly correct or wrong, because
  the automatic check is a keyword match (see *Limits*).
- **Speed** is generation speed (`eval_count / eval_duration` from Ollama), so model loading doesn't
  count; first token and load time are measured separately.

## Results

| Model | `num_ctx` | Automatic score | Manual review: correct / partial / wrong | Speed | First token | Load |
|-------|-----------|-----------------|------------------------------------------|-------|-------------|------|
| `qwen3.5:4b` | 2048 | 15/20 (75%) | **9** / 5 / 6 | **60.1 t/s** | 0.43 s | 5.4 s |
| `qwen3.5:4b` | 8192 | 16/20 (80%) | **7** / 7 / 6 | 50.6 t/s | 0.50 s | 8.5 s |
| `qwen3.5:9b` | 2048 | 18/20 (90%) | **13** / 5 / 2 | 18.3 t/s | 0.68 s | 5.3 s |
| `qwen3.5:9b` | 8192 | 15/20 (75%) | **11** / 4 / 5 | 11.7 t/s | 0.82 s | 7.2 s |
| `gemma4:12b-it-qat` | — | not run | — | — | — | — |

The speeds of the 8192 run are not caused by the context size: the laptop was busier during that
run (load average ~4.4, the 9B swung between 6 and 20 t/s from one question to the next). See
*Context size* for a controlled measurement.

`gemma4:12b-it-qat` couldn't be downloaded: it needs a newer Ollama than 0.18.2. The script skips
missing models, so it can be added by upgrading Ollama and pulling it.

### What the mistakes look like

- **qwen3.5:4b** gets the grammar wrong where it matters most for a learner: 食べる in the polite
  negative past given as *tabemashinai* (it's 食べませんでした), wrong readings for 鬱 (*o*, *iku*),
  "3 examples of the te-form" answered with 食べます three times, *itadakemasen ka* turned into
  いただきます, 枚 read *hitoh*, and invalid forms like *yamesen* / *akiramesen*.
- **qwen3.5:9b** is right on most of the core grammar in both runs (を vs に, あります vs います,
  〜てしまう, 〜たい, 枚, keigo, 聞く vs 聞こえる). Its repeated mistakes: the readings of 鬱 (wrong in
  both runs), a self-introduction that doesn't follow the usual はじめまして… pattern, and, in the
  8192 run, 食べなかった given as the polite negative past and a wrong *itadakemasen ka* request.
  Its partial answers mostly have the right Japanese with a wrong romaji (たべませんでした written
  *tabemashimashita*).
- **Both** produce a correct Japanese sentence more reliably than a correct romaji or pronunciation
  guide: the romaji is where most remaining errors are.

### Context size (`num_ctx`)

The first run used `num_ctx = 2048`, the sensei's setting from #200. Measured on `qwen3.5:9b`:

| `num_ctx` | Speed | Loaded size | Processor |
|-----------|-------|-------------|-----------|
| 2048 | 22.9 t/s | 8.7 GB | 29% CPU / 71% GPU |
| 4096 | 22.7 t/s | 8.8 GB | 28% CPU / 72% GPU |
| 8192 | 20.5 t/s | 8.9 GB | 28% CPU / 72% GPU |

To separate the context size from the machine's load, the same prompt was then sent three times
at each size, alternating 2048 and 8192: **20.4 t/s at 2048 against 20.5 t/s at 8192** (median).
The context size makes no measurable difference to the speed.

The 9B model never fits entirely in 8 GB of VRAM: about 28% of it runs on the CPU whatever the
context. A smaller context saves nothing, but it holds four times less conversation: the sensei's
system prompt alone is ~370 tokens, so at 2048 Ollama starts dropping the oldest messages after a
few exchanges, long before `max_history_turns = 20`.

## Decision

- **`qwen3.5:9b` stays the sensei's model.** Over the two runs it's right on 24 questions against
  16 for the 4B, with 7 outright mistakes against 12. For a tutor, a wrong conjugation taught with
  confidence is the worst failure, so accuracy beats speed. At ~20 t/s it still writes faster than
  anyone reads.
- **`qwen3.5:4b` is the fallback** for laptops with less than 8 GB of VRAM (`--model qwen3.5:4b`):
  three times faster, but its answers need checking.
- **`num_ctx` is back to 8192** in `agents/sensei.toml` (it was lowered to 2048 in #200). It costs
  no measurable speed and gives the model the whole recent conversation.
- **To do:** run `gemma4:12b-it-qat` once Ollama is upgraded, and measure the other team laptops
  (rule 4 of the plan: the same settings must behave the same everywhere).

## Limits

- **The automatic score is a keyword match**, so it is generous: it passed the 9B's 鬱 reply because
  *yuuutsu* contains "utsu" although its readings are wrong, and the 4B's 枚 reply although it reads
  it *hitoh*. It also missed the 9B's correct たべませんでした (kana, not the expected kanji). The manual
  review is the number to use; the automatic score is only a quick regression signal.
- **Two runs per model**, at temperature 0.3: replies vary between runs (the 9B went from 13 to 11
  correct, the 4B from 9 to 7, on the same questions), so differences of one or two questions are
  within noise. The gap between the models holds in both runs.
- **Speed depends on what else the laptop is doing**: compare models within one run, and use the
  alternating measurement above to compare settings.
- **20 questions** are a small sample, chosen for a beginner. They don't test long conversations,
  corrections in context, or the structured-output agents (see the README: Ollama ignores a JSON
  schema with reasoning off, so it goes in the prompt).
