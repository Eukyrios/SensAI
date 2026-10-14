# Per-turn metrics: tokens, speed, time to first token and estimated cost.

from dataclasses import dataclass
from ..core.events import TurnFinished

INPUT_COST_PER_M = 0.50
OUTPUT_COST_PER_M = 1.50


@dataclass(frozen=True)
class TurnMetrics:
    prompt_tokens: int
    eval_tokens: int
    total_tokens: int
    duration_s: float
    ttft_ms: float | None
    tokens_per_sec: float
    estimated_cost_usd: float

    def format_summary(self) -> str:
        ttft_str = f"{self.ttft_ms:.1f}ms" if self.ttft_ms is not None else "N/A"
        return (
            f"[metrics] Tokens: {self.total_tokens} ({self.prompt_tokens} in / {self.eval_tokens} out) | "
            f"Speed: {self.tokens_per_sec:.1f} t/s | "
            f"TTFT: {ttft_str} | "
            f"Cost: ${self.estimated_cost_usd:.6f}"
        )


def compute_turn_metrics(event: TurnFinished) -> TurnMetrics:
    duration_s = (event.total_duration_ns or 0) / 1e9
    prompt_tokens = event.prompt_eval_count or 0
    eval_tokens = event.eval_count or 0
    total_tokens = prompt_tokens + eval_tokens

    tokens_per_sec = (eval_tokens / duration_s) if duration_s > 0 else 0.0

    prompt_cost = (prompt_tokens / 1_000_000) * INPUT_COST_PER_M
    eval_cost = (eval_tokens / 1_000_000) * OUTPUT_COST_PER_M
    total_cost = prompt_cost + eval_cost

    return TurnMetrics(
        prompt_tokens=prompt_tokens,
        eval_tokens=eval_tokens,
        total_tokens=total_tokens,
        duration_s=duration_s,
        ttft_ms=event.ttft_ms,
        tokens_per_sec=tokens_per_sec,
        estimated_cost_usd=total_cost,
    )
