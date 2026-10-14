# Model bake-off: runs 20 Japanese prompts on each candidate model and prints their speed.

from collections.abc import Iterator

from tabulate import tabulate

from sensai.llm.ollama import ChatStats, OllamaClient, OllamaError

OLLAMA_URL = "http://localhost:11434/"
MODELS = ["qwen3.5:4b", "qwen3.5:9b", "gemma4:12b-it-qat"]
NUM_CTX = 2048
CONCISE_INSTRUCTION = "\n(Answer in at most 2 sentences, be very concise.)"

TEST_CASES = [
    "Translate into formal Japanese: 'I would like to book a table for two people at 7 pm.'",
    "Explain the difference between the particles 'wa' (は) and 'ga' (が) with simple examples.",
    "Conjugate the verb 'taberu' (食べる) in the polite negative past.",
    "What does this kanji mean: 鬱? Give its main readings.",
    "Correct this sentence: 'Watashi wa eiga ni mimasu.'",
    "How do you say 'It's my first time coming to Japan'?",
    "Give 3 examples of using the '-te' form (て).",
    "What is the difference between 'arimasu' and 'imasu'?",
    "Translate: 'It has been raining since this morning, so I don't want to go out.'",
    "Explain the grammar of '~te shimau' (〜てしまう).",
    "Give the reading and meaning of the word: 飛行機",
    "Make a polite request with 'itadakemasen ka'.",
    "Translate into English: '明日は忙しいから、明後日にしましょう。'",
    "How do you express a wish (I want to do...) in Japanese?",
    "Correct this sentence: 'Kono hon wa oishii desu.'",
    "Briefly explain the concept of 'Keigo' (honorific language).",
    "Translate: 'Even if it's difficult, I won't give up.'",
    "What is the right counter for sheets of paper?",
    "Give the difference between 'kiku' (聞く) and 'kikoeru' (聞こえる).",
    "Write a short, classic self-introduction (jikoshoukai).",
]


def main() -> None:
    client = OllamaClient(OLLAMA_URL)
    print(f"Starting the bake-off on {len(TEST_CASES)} prompts with num_ctx={NUM_CTX}...\n")
    results = [row for model in MODELS if (row := benchmark_model(client, model)) is not None]
    print_results(results)


def benchmark_model(client: OllamaClient, model: str) -> list[str] | None:
    print(f"\n--- Evaluating {model} ---")
    try:
        client.require_model(model)
    except OllamaError:
        print(f"Error: model missing. Run 'ollama pull {model}'")
        return None
    speeds = [speed for number, prompt in enumerate(TEST_CASES, 1)
              if (speed := run_prompt(client, model, number, prompt)) is not None]
    average_speed = sum(speeds) / len(speeds) if speeds else 0
    return [model, f"{average_speed:.2f} t/s", "not measured"]


def run_prompt(client: OllamaClient, model: str, number: int, prompt: str) -> float | None:
    print(f"\n  Prompt {number:02d}: ", end="", flush=True)
    messages = [{"role": "user", "content": prompt + CONCISE_INSTRUCTION}]
    stats = print_reply(client.chat_stream(model, messages, num_ctx=NUM_CTX, temperature=0.1))
    speed = tokens_per_second(stats)
    if speed is not None:
        print(f"\n  -> [Speed: {speed:.2f} t/s]")
    return speed


def print_reply(stream: Iterator) -> ChatStats | None:
    while True:
        try:
            piece = next(stream)
        except StopIteration as end:
            return end.value
        if isinstance(piece, str):
            print(piece, end="", flush=True)


def tokens_per_second(stats: ChatStats | None) -> float | None:
    if not stats:
        return None
    duration_s = (stats.eval_duration_ns or stats.total_duration_ns) / 1e9
    return stats.eval_count / duration_s if duration_s > 0 else None


def print_results(results: list[list[str]]) -> None:
    print("\n" + "=" * 50)
    print("BAKE-OFF RESULTS (layer 2, task C)")
    print("=" * 50)
    print(tabulate(results, headers=["Model", "Average speed", "Accuracy"], tablefmt="grid"))


if __name__ == "__main__":
    main()
