"""Tests for automated evaluation & hallucination detection (EV1)."""
import json

import pytest

from sensai.core import events as ev
from sensai.llm.ollama import OllamaError
from sensai.trust.eval import (ClaimVerification, LLMJudge, SourceCollector, VerificationReport,
                               faithfulness_event)
from tests.fakes import FakeClient, ScriptedClient


def judge_reply(*claims, relevance=8):
    return json.dumps({"claims": [{"claim": c, "status": s, "reasoning": "r"} for c, s in claims],
                       "relevance": relevance})


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
    assert [c.claim for c in report.flagged] == ["Fact B", "Fact C"]


def test_judge_talks_to_the_real_chat_api():
    client = ScriptedClient(judge_reply(("Tokyo is the capital of Japan.", "SUPPORTED")))
    report = LLMJudge(client, "judge-model").verify_response(
        "Tokyo is the capital of Japan.", "Tokyo is the capital city of Japan.", "What is the capital?")
    model, messages, options = client.calls[0]
    assert model == "judge-model" and options["format"] == "json" and options["temperature"] == 0.0
    assert "Tokyo is the capital city of Japan." in messages[1]["content"]
    assert "What is the capital?" in messages[1]["content"]
    assert report.total_claims == 1 and report.claims[0].status == "SUPPORTED"
    assert report.faithfulness_score == 1.0 and report.relevance == 0.8 and report.error is None


def test_judge_flags_an_invented_fact():
    client = ScriptedClient(judge_reply(("Hiragana comes from man'yōgana.", "SUPPORTED"),
                                        ("Hiragana was invented in 1650.", "UNVERIFIED"),
                                        ("Katakana is for foreign words.", "CONTRADICTED")))
    report = LLMJudge(client, "m").verify_response("reply", "sources")
    assert report.faithfulness_score == 0.33
    assert [c.claim for c in report.flagged] == ["Hiragana was invented in 1650.", "Katakana is for foreign words."]


def test_judge_reply_with_fences_and_unknown_status_is_read():
    reply = "```json\n" + judge_reply(("x", "maybe"), relevance=99) + "\n```"
    report = LLMJudge(ScriptedClient(reply), "m").verify_response("reply", "sources")
    assert report.claims[0].status == "UNVERIFIED" and report.relevance == 1.0


def test_judge_failures_are_errors_not_scores():
    for client in (ScriptedClient("no json here"), FakeClient(error=OllamaError("connection", "down"))):
        report = LLMJudge(client, "m").verify_response("reply", "sources")
        assert report.error and report.total_claims == 0


def test_judge_needs_a_model_and_something_to_check():
    assert LLMJudge(FakeClient()).verify_response("reply", "sources").error == "no judge model set"
    assert LLMJudge(FakeClient(), "m").verify_response("", "").total_claims == 0
    assert LLMJudge(FakeClient(), "m").verify_response("reply", "  ").error is None


def test_judge_cuts_the_sources():
    client = ScriptedClient(judge_reply())
    LLMJudge(client, "m", max_context_chars=10).verify_response("reply", "a" * 50)
    assert "a" * 10 in client.calls[0][1][1]["content"] and "a" * 11 not in client.calls[0][1][1]["content"]


def test_sources_collect_passages_and_successful_tool_results():
    sources = SourceCollector()
    assert not sources
    sources.observe(ev.Retrieved(turn_id="t", run_id="r", query="q", level=None, mode="lexical", dropped=[],
                                 kept=[{"source": "a.md", "section": "に", "content": "に marks a destination"}]))
    sources.observe(ev.ToolResult(turn_id="t", run_id="r", name="dictionary", result='{"word": "猫"}'))
    sources.observe(ev.ToolResult(turn_id="t", run_id="r", name="kanji", result="boom", ok=False))
    sources.observe(ev.Token(turn_id="t", run_id="r", text="x"))
    assert sources and len(sources.passages) == 2
    assert "[a.md § に]\nに marks a destination" in sources.context and "[tool dictionary]" in sources.context


def test_faithfulness_event_carries_the_verdict():
    report = VerificationReport([ClaimVerification("a", "SUPPORTED", ""), ClaimVerification("b", "UNVERIFIED", "why")], 0.5)
    event = faithfulness_event("t", report, "m")
    assert (event.faithfulness, event.relevance, event.supported, event.unverified) == (0.5, 0.5, 1, 1)
    assert event.flagged == [{"claim": "b", "status": "UNVERIFIED", "reasoning": "why"}]
    failed = faithfulness_event("t", VerificationReport(error="down"), "m")
    assert failed.faithfulness is None and failed.error == "down"
