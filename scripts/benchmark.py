# Model benchmark: asks each candidate model 20 Japanese questions with the sensei's real settings
# (system prompt, num_ctx, temperature), checks each reply for the answer it must contain, and
# records accuracy and speed. Results are printed and saved to doc/benchmark_results.json.

import json
import time
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

from tabulate import tabulate

from sensai.core.agent_spec import AgentSpec, load_spec
from sensai.core.config import resolve
from sensai.llm.ollama import ChatStats, OllamaClient, OllamaError

OLLAMA_URL = "http://localhost:11434/"
MODELS = ["qwen3.5:4b", "qwen3.5:9b", "gemma4:12b-it-qat"]
SPEC_PATH = resolve("agents/sensei.toml")
RESULTS_PATH = resolve("doc/benchmark_results.json")
CONCISE_INSTRUCTION = "\n(Answer in at most 2 sentences, be very concise.)"


@dataclass(frozen=True)
class Case:
    prompt: str
    expected: tuple[str, ...]


TEST_CASES = [
    Case("Translate into formal Japanese: 'I would like to book a table for two people at 7 pm.'", ("予約",)),
    Case("Explain the difference between the particles 'wa' (は) and 'ga' (が) with simple examples.", ("topic",)),
    Case("Conjugate the verb 'taberu' (食べる) in the polite negative past.", ("食べませんでした", "tabemasen deshita")),
    Case("What does this kanji mean: 鬱? Give its main readings.", ("utsu", "うつ", "ウツ")),
    Case("Correct this sentence: 'Watashi wa eiga ni mimasu.'", ("eiga o", "eiga wo", "映画を")),
    Case("How do you say 'It's my first time coming to Japan'?", ("初めて", "はじめて", "hajimete")),
    Case("Give 3 examples of using the '-te' form (て).", ("て",)),
    Case("What is the difference between 'arimasu' and 'imasu'?", ("animate", "living", "alive")),
    Case("Translate: 'It has been raining since this morning, so I don't want to go out.'", ("朝から", "asa kara")),
    Case("Explain the grammar of '~te shimau' (〜てしまう).", ("regret", "complet", "accident")),
    Case("Give the reading and meaning of the word: 飛行機", ("ひこうき", "hikouki", "hikōki")),
    Case("Make a polite request with 'itadakemasen ka'.", ("いただけませんか", "itadakemasen")),
    Case("Translate into English: '明日は忙しいから、明後日にしましょう。'", ("day after tomorrow",)),
    Case("How do you express a wish (I want to do...) in Japanese?", ("たい", "-tai")),
    Case("Correct this sentence: 'Kono hon wa oishii desu.'", ("omoshiroi", "面白い", "おもしろい")),
    Case("Briefly explain the concept of 'Keigo' (honorific language).", ("respect", "honorific", "polite")),
    Case("Translate: 'Even if it's difficult, I won't give up.'", ("諦め", "あきらめ", "akirame")),
    Case("What is the right counter for sheets of paper?", ("枚", "mai")),
    Case("Give the difference between 'kiku' (聞く) and 'kikoeru' (聞こえる).", ("audible", "can hear", "be heard")),
    Case("Write a short, classic self-introduction (jikoshoukai).", ("はじめまして", "hajimemashite", "よろしく")),
]


@dataclass
class CaseResult:
    prompt: str
    reply: str
    correct: bool
    tokens_per_second: float | None
    first_token_s: float | None


@dataclass
class ModelResult:
    model: str
    cases: list[CaseResult] = field(default_factory=list)
    load_s: float = 0.0

    @property
    def accuracy(self) -> float:
        return sum(case.correct for case in self.cases) / len(self.cases) if self.cases else 0.0

    @property
    def average_speed(self) -> float:
        speeds = [case.tokens_per_second for case in self.cases if case.tokens_per_second]
        return sum(speeds) / len(speeds) if speeds else 0.0

    @property
    def median_first_token_s(self) -> float:
        times = sorted(case.first_token_s for case in self.cases[1:] if case.first_token_s is not None)
        return times[len(times) // 2] if times else 0.0

    def table_row(self) -> list[str]:
        correct = sum(case.correct for case in self.cases)
        return [self.model, f"{correct}/{len(self.cases)} ({self.accuracy:.0%})",
                f"{self.average_speed:.1f} t/s", f"{self.median_first_token_s:.2f} s", f"{self.load_s:.1f} s"]


def main() -> None:
    client = OllamaClient(OLLAMA_URL)
    spec = load_spec(SPEC_PATH)
    print(f"Benchmark: {len(TEST_CASES)} questions per model, num_ctx={spec.num_ctx}, sensei prompt\n")
    results = [result for model in MODELS if (result := benchmark_model(client, spec, model)) is not None]
    print_results(results)
    save_results(results, spec)


def benchmark_model(client: OllamaClient, spec: AgentSpec, model: str) -> ModelResult | None:
    print(f"\n--- {model} ---")
    try:
        client.require_model(model)
    except OllamaError:
        print(f"Skipped: model not installed. Run 'ollama pull {model}'")
        return None
    model_spec = replace(spec, model=model)
    result = ModelResult(model)
    for number, case in enumerate(TEST_CASES, 1):
        case_result, stats = run_case(client, model_spec, number, case)
        result.cases.append(case_result)
        if number == 1 and stats:
            result.load_s = stats.load_duration_ns / 1e9
    return result


def run_case(client: OllamaClient, spec: AgentSpec, number: int, case: Case) -> tuple[CaseResult, ChatStats | None]:
    print(f"  {number:02d}. ", end="", flush=True)
    started = time.perf_counter()
    reply, first_token_s, stats = collect_reply(client, spec, case.prompt + CONCISE_INSTRUCTION, started)
    correct = contains_expected(reply, case.expected)
    speed = tokens_per_second(stats)
    print(f"{'OK  ' if correct else 'MISS'} {speed or 0:5.1f} t/s  {reply[:70]!r}")
    return CaseResult(case.prompt, reply, correct, speed, first_token_s), stats


def collect_reply(client: OllamaClient, spec: AgentSpec, prompt: str,
                  started: float) -> tuple[str, float | None, ChatStats | None]:
    messages = [{"role": "system", "content": spec.system_prompt}, {"role": "user", "content": prompt}]
    stream = client.chat_stream(spec.model, messages, num_ctx=spec.num_ctx,
                                temperature=spec.temperature, think=spec.think)
    pieces, first_token_s = [], None
    while True:
        try:
            piece = next(stream)
        except StopIteration as end:
            return "".join(pieces), first_token_s, end.value
        if isinstance(piece, str):
            first_token_s = first_token_s if first_token_s is not None else time.perf_counter() - started
            pieces.append(piece)


def contains_expected(reply: str, expected: tuple[str, ...]) -> bool:
    lowered = reply.lower()
    return any(answer.lower() in lowered for answer in expected)


def tokens_per_second(stats: ChatStats | None) -> float | None:
    if not stats:
        return None
    duration_s = (stats.eval_duration_ns or stats.total_duration_ns) / 1e9
    return stats.eval_count / duration_s if duration_s > 0 else None


def print_results(results: list[ModelResult]) -> None:
    print("\n" + "=" * 70)
    print(tabulate([result.table_row() for result in results],
                   headers=["Model", "Accuracy", "Speed", "First token", "Load"], tablefmt="github"))


def save_results(results: list[ModelResult], spec: AgentSpec, path: Path = RESULTS_PATH) -> None:
    report = {
        "settings": {"num_ctx": spec.num_ctx, "temperature": spec.temperature, "think": spec.think,
                     "system_prompt": "agents/sensei.toml", "questions": len(TEST_CASES)},
        "models": [_model_report(result) for result in results],
    }
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"\nSaved to {path}")


def _model_report(result: ModelResult) -> dict:
    return {"model": result.model, "accuracy": round(result.accuracy, 3),
            "average_tokens_per_second": round(result.average_speed, 1),
            "median_first_token_s": round(result.median_first_token_s, 2), "load_s": round(result.load_s, 1),
            "cases": [asdict(case) for case in result.cases]}


if __name__ == "__main__":
    main()
