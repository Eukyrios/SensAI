# Model benchmark

Which model the sensei should run, measured on the team's reference laptop: RTX 4060 Laptop GPU
(8 GB VRAM), 16 GB RAM, Ollama 0.18.2. Run on 2026-10-14 with `python -m scripts.benchmark`;
the raw replies are in [`benchmark_results.json`](benchmark_results.json).

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

| Model | Automatic score | Manual review: correct / partial / wrong | Speed | First token | Load |
|-------|-----------------|------------------------------------------|-------|-------------|------|
| `qwen3.5:4b` | 15/20 (75%) | **9** / 5 / 6 | **60.1 t/s** | 0.43 s | 5.4 s |
| `qwen3.5:9b` | 18/20 (90%) | **13** / 5 / 2 | 18.3 t/s | 0.68 s | 5.3 s |
| `gemma4:12b-it-qat` | not run | — | — | — | — |

`gemma4:12b-it-qat` couldn't be downloaded: it needs a newer Ollama than 0.18.2. The script skips
missing models, so it can be added by upgrading Ollama and pulling it.

### What the mistakes look like

- **qwen3.5:4b** gets the grammar wrong where it matters most for a learner: 食べる in the polite
  negative past given as *tabemashinai* (it's 食べませんでした), wrong readings for 鬱 (*o*, *iku*),
  "3 examples of the te-form" answered with 食べます three times, *itadakemasen ka* turned into
  いただきます, 枚 read *hitoh*, and invalid forms like *yamesen* / *akiramesen*.
- **qwen3.5:9b** is right on all the core grammar (を vs に, あります vs います, 〜てしまう, 〜たい, 枚,
  keigo, 聞く vs 聞こえる). Its two clear mistakes: the readings of 鬱 (*ou*, *utsutaushii*) and a
  self-introduction that doesn't follow the usual はじめまして… pattern. Its partial answers have the
  right Japanese with a wrong romaji (たべませんでした written *tabemashimashita*).
- **Both** produce a correct Japanese sentence more reliably than a correct romaji or pronunciation
  guide: the romaji is where most remaining errors are.

### Context size (`num_ctx`)

The sensei runs with `num_ctx = 2048` since #200. Measured on `qwen3.5:9b`:

| `num_ctx` | Speed | Loaded size | Processor |
|-----------|-------|-------------|-----------|
| 2048 | 22.9 t/s | 8.7 GB | 29% CPU / 71% GPU |
| 4096 | 22.7 t/s | 8.8 GB | 28% CPU / 72% GPU |
| 8192 | 20.5 t/s | 8.9 GB | 28% CPU / 72% GPU |

The 9B model never fits entirely in 8 GB of VRAM: about 28% of it runs on the CPU whatever the
context. A smaller context saves almost nothing (2048 is ~10% faster than 8192), but it holds four
times less conversation: the sensei's system prompt alone is ~370 tokens, so at 2048 Ollama starts
dropping the oldest messages after a few exchanges, long before `max_history_turns = 20`.

## Decision

- **`qwen3.5:9b` stays the sensei's model.** It's right on 13 questions against 9 and makes about a
  third as many outright mistakes. For a tutor, a wrong conjugation taught with confidence is the
  worst failure, so accuracy beats speed. At 18–23 t/s it still writes faster than anyone reads.
- **`qwen3.5:4b` is the fallback** for laptops with less than 8 GB of VRAM (`--model qwen3.5:4b`):
  three times faster, but its answers need checking.
- **Recommended: set `num_ctx` back to 8192** in `agents/sensei.toml`. It costs ~10% speed and gives
  the model the whole recent conversation. To be agreed with the team (it was lowered in #200).
- **To do:** run `gemma4:12b-it-qat` once Ollama is upgraded, and measure the other team laptops
  (rule 4 of the plan: the same settings must behave the same everywhere).

## Limits

- **The automatic score is a keyword match**, so it is generous: it passed the 9B's 鬱 reply because
  *yuuutsu* contains "utsu" although its readings are wrong, and the 4B's 枚 reply although it reads
  it *hitoh*. It also missed the 9B's correct たべませんでした (kana, not the expected kanji). The manual
  review is the number to use; the automatic score is only a quick regression signal.
- **One run per model**, at temperature 0.3: replies vary between runs, so differences of one or two
  questions are within noise. The 4 vs 9 gap is not.
- **20 questions** are a small sample, chosen for a beginner. They don't test long conversations,
  corrections in context, or the structured-output agents (see the README: Ollama ignores a JSON
  schema with reasoning off, so it goes in the prompt).
