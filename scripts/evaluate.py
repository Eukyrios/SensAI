# Answer evaluation (EV1): asks each model a fixed set of questions through the real pipeline (tools and corpus
# passages included), lets the judge check every reply against the sources it had, and prints one table to
# compare the models: faithfulness (share of claims the sources support) and relevance (0-1).
#
#   python -m scripts.evaluate --models qwen3.5:4b qwen3.5:9b
#   python -m scripts.evaluate --models qwen3.5:9b --limit 3 --report doc/evaluation_results.json
#
# The same judge model grades every model, so the scores are comparable. A question whose reply had no
# source to check against (nothing retrieved, no tool called) is counted apart, not scored.

import argparse
import asyncio
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from tabulate import tabulate

from sensai.core.config import load_config
from sensai.core.events import FaithfulnessChecked, Token
from sensai.llm.ollama import OllamaClient, OllamaError
from sensai.memory import db
from sensai.memory.migrate import migrate
from sensai.memory.repository import SessionRepository
from sensai.orchestration.session import Session
from sensai.orchestration.turn import run_turn

JUDGE_MODEL = "qwen3.5:9b"
MODELS = ["qwen3.5:9b"]
QUESTIONS = [
    "Conjugate 食べる in the polite negative past.",
    "What does the word 猫 mean?",
    "Give the reading and meaning of the word 飛行機.",
    "What is the て-form of 飲む?",
    "What are the readings of the kanji 鬱?",
    "When do I use the particle に?",
    "What is the difference between は and が?",
    "How do I make a polite request with いただけませんか?",
    "What is the right counter for sheets of paper?",
    "Explain the grammar of 〜てしまう.",
    "Who invented hiragana?",  # nothing in the corpus says: any claim here is invented
]


@dataclass
class QuestionResult:
    question: str
    reply: str = ""
    faithfulness: float | None = None
    relevance: float | None = None
    flagged: list[dict] = field(default_factory=list)
    status: str = "judged"  # judged | no sources | judge error | failed


@dataclass
class ModelResult:
    model: str
    questions: list[QuestionResult] = field(default_factory=list)

    def scored(self, attribute: str) -> list[float]:
        return [value for q in self.questions if q.status == "judged" and (value := getattr(q, attribute)) is not None]

    def average(self, attribute: str) -> float | None:
        values = self.scored(attribute)
        return sum(values) / len(values) if values else None

    def count(self, status: str) -> int:
        return sum(q.status == status for q in self.questions)

    def table_row(self) -> list[str]:
        return [self.model, f"{self.count('judged')}/{len(self.questions)}", _percent(self.average("faithfulness")),
                _percent(self.average("relevance")), self.count("no sources"),
                self.count("judge error") + self.count("failed")]


def _percent(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.2f}"


def memory_repository() -> SessionRepository:
    conn = db.connect(":memory:")
    migrate(conn)
    return SessionRepository(conn)


def evaluation_config(model: str, judge: str) -> dict:
    return load_config(env={}, cli={
        "agents": {"model_override": model},
        "router": {"enabled": False},
        "curriculum": {"enabled": False},
        "mcp": {"enabled": False},
        "eval": {"enabled": True, "judge_model": judge},
        "logging": {"path": "logs/evaluate.jsonl"},
    })


async def ask(client: OllamaClient, config: dict, question: str) -> QuestionResult:
    result = QuestionResult(question)
    session = Session(config, client, repo=memory_repository())
    verdict: FaithfulnessChecked | None = None
    try:
        async for event in run_turn(session, question):
            if isinstance(event, Token):
                result.reply += event.text
            elif isinstance(event, FaithfulnessChecked):
                verdict = event
    except OllamaError as error:
        result.status, result.reply = "failed", str(error)
        return result
    if verdict is None:
        result.status = "no sources"
    elif verdict.error:
        result.status = "judge error"
        result.reply += f"\n[judge: {verdict.error}]"
    else:
        result.faithfulness, result.relevance, result.flagged = verdict.faithfulness, verdict.relevance, verdict.flagged
    return result


def evaluate_model(client: OllamaClient, model: str, judge: str, questions: list[str]) -> ModelResult | None:
    print(f"\n--- {model} (judge: {judge}) ---")
    try:
        client.require_model(model)
        client.require_model(judge)
    except OllamaError as error:
        print(f"Skipped: {error}")
        return None
    config = evaluation_config(model, judge)
    result = ModelResult(model)
    for number, question in enumerate(questions, 1):
        answered = asyncio.run(ask(client, config, question))
        result.questions.append(answered)
        score = "-" if answered.faithfulness is None else f"{answered.faithfulness:.2f}"
        print(f"  {number:02d}. faithfulness {score:>4}  [{answered.status}]  {question}")
        for claim in answered.flagged:
            print(f"        [{claim['status'].lower()}] {claim['claim']}")
    return result


def print_results(results: list[ModelResult]) -> None:
    headers = ["model", "judged", "faithfulness", "relevance", "no sources", "errors"]
    print("\n" + tabulate([result.table_row() for result in results], headers=headers, tablefmt="github", disable_numparse=True))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score the answers of one or more models with the judge")
    parser.add_argument("--models", nargs="+", default=MODELS, help="models to compare")
    parser.add_argument("--judge", default=JUDGE_MODEL, help="model that grades every reply")
    parser.add_argument("--limit", type=int, default=0, help="only the first N questions")
    parser.add_argument("--report", type=Path, help="write the results as JSON to this file")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    questions = QUESTIONS[:args.limit] if args.limit else QUESTIONS
    client = OllamaClient(load_config(env={})["ollama"]["url"], timeout_s=180)
    results = [result for model in args.models if (result := evaluate_model(client, model, args.judge, questions))]
    if not results:
        raise SystemExit("No model could be evaluated.")
    print_results(results)
    if args.report:
        args.report.write_text(json.dumps([asdict(result) for result in results], ensure_ascii=False, indent=2),
                               encoding="utf-8")
        print(f"\nResults written to {args.report}")


if __name__ == "__main__":
    main()
