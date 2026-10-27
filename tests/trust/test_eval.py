"""Tests for automated evaluation & hallucination detection (EV1)."""
from unittest.mock import MagicMock
import pytest

from sensai.trust.eval import LLMJudge, VerificationReport, ClaimVerification


def test_verification_report_metrics():
    report = VerificationReport(
        claims=[
            ClaimVerification("Fact A", "SUPPORTED", "Match"),
            ClaimVerification("Fact B", "CONTRADICTED", "Mismatch"),
            ClaimVerification("Fact C", "UNVERIFIED", "Not in context"),
        ]
    )
    assert report.total_claims == 3
    assert report.supported_claims == 1
    assert report.contradicted_claims == 1
    assert report.unverified_claims == 1
    assert report.faithfulness_score == 0.33


def test_llm_judge_verify_response_success():
    mock_client = MagicMock()
    json_response = """
    ```json
    {
      "claims": [
        {
          "claim": "Tokyo is the capital of Japan.",
          "status": "SUPPORTED",
          "reasoning": "Stated in context"
        }
      ]
    }
    ```
    """
    mock_chunk = MagicMock()
    mock_chunk.text = json_response
    mock_client.generate_stream.return_value = [mock_chunk]

    judge = LLMJudge(client=mock_client)
    report = judge.verify_response(
        response_text="Tokyo is the capital of Japan.",
        context="Tokyo is the capital city of Japan."
    )

    assert report.total_claims == 1
    assert report.claims[0].status == "SUPPORTED"
    assert report.faithfulness_score == 1.0


def test_llm_judge_empty_input():
    judge = LLMJudge(client=MagicMock())
    report = judge.verify_response("", "")
    assert report.total_claims == 0
    assert report.faithfulness_score == 1.0