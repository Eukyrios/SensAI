# Tests for the metrics and cost estimation.

from sensai.core.events import TurnFinished
from sensai.trust.metrics import compute_turn_metrics


def test_compute_turn_metrics_calculates_correctly():
    event = TurnFinished(
        turn_id="t1",
        model="qwen3.5:9b",
        prompt_eval_count=1000,
        eval_count=200,
        total_duration_ns=2_000_000_000,
        ttft_ms=150.0,
    )

    metrics = compute_turn_metrics(event)

    assert metrics.prompt_tokens == 1000
    assert metrics.eval_tokens == 200
    assert metrics.total_tokens == 1200
    assert metrics.duration_s == 2.0
    assert metrics.tokens_per_sec == 100.0
    assert metrics.ttft_ms == 150.0

    assert abs(metrics.estimated_cost_usd - 0.0008) < 1e-6

    summary = metrics.format_summary()
    assert "Tokens: 1200" in summary
    assert "100.0 t/s" in summary
    assert "$0.000800" in summary


def test_speed_counts_only_the_generation_time():
    event = TurnFinished(turn_id="t1", model="qwen3.5:9b", eval_count=200, eval_duration_ns=1_000_000_000,
                         total_duration_ns=5_000_000_000)
    assert compute_turn_metrics(event).tokens_per_sec == 200.0
