"""Tests for metrics and cost estimation logic."""
from sensai.core.events import TurnFinished
from sensai.trust.metrics import compute_turn_metrics


def test_compute_turn_metrics_calculates_correctly():
    event = TurnFinished(
        turn_id="t1",
        model="qwen3.5:9b",
        prompt_eval_count=1000,
        eval_count=200,
        total_duration_ns=2_000_000_000,  # 2.0s
        ttft_ms=150.0,
    )

    metrics = compute_turn_metrics(event)

    assert metrics.prompt_tokens == 1000
    assert metrics.eval_tokens == 200
    assert metrics.total_tokens == 1200
    assert metrics.duration_s == 2.0
    assert metrics.tokens_per_sec == 100.0
    assert metrics.ttft_ms == 150.0

    # Coût : (1000/1M * 0.50) + (200/1M * 1.50) = 0.0005 + 0.0003 = 0.0008 $
    assert abs(metrics.estimated_cost_usd - 0.0008) < 1e-6

    summary = metrics.format_summary()
    assert "Tokens: 1200" in summary
    assert "100.0 t/s" in summary
    assert "$0.000800" in summary