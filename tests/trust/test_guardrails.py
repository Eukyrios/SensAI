# Tests for the guardrail stubs.

from sensai.trust import guardrails


def test_passthrough():
    assert guardrails.filter_input("a", {}) == "a"
    assert guardrails.filter_output("b", {}) == "b"
