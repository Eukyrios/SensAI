# Tests for the per-agent context: token estimate, calibration, providers, budget and build_context().

from dataclasses import dataclass, replace

import pytest

from sensai.core.agent_spec import AgentSpec
from sensai.core.handoff import Handoff
from sensai.orchestration.context import ContextBuilder, ContextProvider, TokenEstimator, build_context
from sensai.orchestration.session import Session
from tests.conftest import memory_repository
from tests.fakes import FakeClient, make_config

SPEC = AgentSpec(name="a", model="m", num_ctx=1000, system_prompt="sys")


@dataclass
class Fake:
    name: str
    priority: int
    max_share: float
    text: str | None = "x" * 70

    def render(self, budget: int) -> str | None:
        return self.text


def user(text):
    return {"role": "user", "content": text}


def test_provider_protocol():
    assert isinstance(Fake("p", 1, 0.5), ContextProvider)


def test_estimate():
    est = TokenEstimator(chars_per_token=2)
    assert est.estimate("") == 0
    assert est.estimate("a") == 1
    assert est.estimate("abcd") == 2
    assert est.estimate_messages([user("abcd"), user("ab")]) == 3
    with pytest.raises(ValueError):
        TokenEstimator(chars_per_token=0)


def test_calibration_moves_toward_observed_ratio():
    est = TokenEstimator(chars_per_token=4, smoothing=0.5)
    est.calibrate([user("a" * 100)], 50)            # really 2 chars per token
    assert est.chars_per_token == 3
    for _ in range(30):
        est.calibrate([user("a" * 100)], 50)
    assert est.chars_per_token == pytest.approx(2, abs=0.01)


def test_calibration_ignores_missing_counts():
    est = TokenEstimator()
    est.calibrate([user("abc")], 0)
    est.calibrate([user("")], 10)
    assert est.chars_per_token == 3.5


def test_budget_keeps_room_for_the_answer():
    assert ContextBuilder().budget(SPEC) == 750


def test_providers_follow_priority_and_join_the_system_message():
    builder = ContextBuilder([Fake("low", 1, 1, "LOW"), Fake("high", 9, 1, "HIGH")])
    system = builder.build(SPEC, [user("hi")])[0]["content"]
    assert system == "sys\n\nHIGH\n\nLOW"
    assert [b.provider for b in builder.blocks] == ["high", "low"]


def test_max_share_truncates_a_block():
    builder = ContextBuilder([Fake("big", 1, 0.1, "x" * 5000)], TokenEstimator(chars_per_token=1))
    builder.build(SPEC, [user("hi")])
    block, = builder.blocks
    assert block.truncated and block.tokens <= 75


def test_empty_provider_is_skipped():
    builder = ContextBuilder([Fake("none", 1, 1, None)])
    assert builder.build(SPEC, [])[0]["content"] == "sys"
    assert builder.blocks == []


def test_history_keeps_the_newest_that_fit_and_always_the_last():
    builder = ContextBuilder(estimator=TokenEstimator(chars_per_token=1))
    history = [user("a" * 300), user("b" * 300), user("c" * 300)]
    kept = builder.build(SPEC, history)[1:]
    assert [m["content"][0] for m in kept] == ["b", "c"]
    huge = builder.build(SPEC, [user("z" * 5000)])[1:]
    assert len(huge) == 1


def test_no_provider_gets_more_than_what_is_left():
    builder = ContextBuilder([Fake("a", 2, 1, "x" * 600), Fake("b", 1, 1, "y" * 600)],
                             TokenEstimator(chars_per_token=1))
    builder.build(SPEC, [])
    assert sum(b.tokens for b in builder.blocks) <= builder.budget(SPEC) - 1


def test_builder_calibrate_updates_the_estimator():
    builder = ContextBuilder()
    builder.calibrate([user("a" * 70)], 10)
    assert builder.estimator.chars_per_token > 3.5


def test_build_context_fits_the_profile_note_and_history_in_the_agent_budget():
    repo = memory_repository()
    repo.save_profile("default", "N5 learner")
    session = Session(make_config(), client=FakeClient(), repo=repo)
    session.history = [user("old " * 2000), user("question")]
    handoff = Handoff("checker", "sensei", "particle", {"rule": "use に"}, hop=1, run_id="r1")
    messages = build_context(session, replace(session.spec, num_ctx=2048), handoff)
    system = messages[0]["content"]
    assert system.index("N5 learner") < system.index("checker agent looked at this turn first")
    assert '"rule": "use に"' in system
    assert messages[1:] == [user("question")]
