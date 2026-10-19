# Tests for the router: routes from routing.toml, tier 1 rules, the default decision and the wiring.

import json
import re

import pytest

from sensai.core import events as ev
from sensai.core.config import resolve
from sensai.orchestration.router import (Route, RouteDecision, Router, RoutingError, RuleTier, load_router,
                                         load_routes)
from sensai.orchestration.session import Session
from sensai.orchestration.turn import run_turn
from tests.fakes import FakeClient, ScriptedClient, make_config

ROUTES = load_routes(resolve("routing.toml"))
CHECKED = json.dumps({"sentence": "私は学校に行きます", "is_correct": True, "correction": "私は学校に行きます",
                      "error_type": "none", "rule": "", "handoff": None}, ensure_ascii=False)


def routed_config(**router):
    return make_config(router={"enabled": True, "routes": "routing.toml", **router})


def write_routes(tmp_path, text):
    path = tmp_path / "routes.toml"
    path.write_text(text, encoding="utf-8")
    return path


async def agents_run(session, text):
    return [event.agent async for event in run_turn(session, text) if isinstance(event, ev.AgentRunStarted)]


def test_routing_file_has_the_checker_then_the_sensei():
    assert [route.agent for route in ROUTES] == ["checker", "sensei"]
    assert all(route.rules and route.examples and route.description for route in ROUTES)


@pytest.mark.parametrize("text, agent", [
    ("私は学校を行きます", "checker"),
    ("ありがとう。", "checker"),
    ("Is this correct: 昨日、映画を見ました", "checker"),
    ("Can you check my sentence? 猫が好きです", "checker"),
    ("How do you say good morning?", "sensei"),
    ("What does 飛行機 mean", "sensei"),
    ("explain the te-form", "sensei"),
    ("is wa a particle?", "sensei"),
])
def test_rules_route_the_obvious_cases(text, agent):
    decision = RuleTier(ROUTES).decide(text)
    assert (decision.agent, decision.tier, decision.confidence) == (agent, "rules", 1.0)
    assert decision.reason.startswith("matched ")


@pytest.mark.parametrize("text", ["watashi wa sushi ga suki", "I like anime", "konnichiwa"])
def test_rules_stay_out_of_unclear_cases(text):
    assert RuleTier(ROUTES).decide(text) is None


def test_first_matching_route_wins():
    routes = [Route("checker", "first", (re.compile("a"),)), Route("sensei", "second", (re.compile("a"),))]
    assert RuleTier(routes).decide("a").agent == "checker"


def test_router_falls_back_to_the_default_agent():
    decision = Router(ROUTES, "sensei").route("konnichiwa")
    assert decision == RouteDecision("sensei", "default", 0.0, "no tier decided")


def test_load_router_is_off_when_disabled():
    assert load_router(make_config()) is None
    assert isinstance(load_router(routed_config()), Router)


def test_routes_must_point_to_existing_agents(tmp_path):
    path = write_routes(tmp_path, '[[routes]]\nagent = "nobody"\ndescription = "x"\n')
    with pytest.raises(RoutingError, match="unknown agent.*nobody"):
        load_router(routed_config(routes=str(path)))


@pytest.mark.parametrize("text, problem", [
    ("", "no \\[\\[routes\\]\\]"),
    ('[[routes]]\nagent = "sensei"\n', "missing 'description'"),
    ('[[routes]]\nagent = "sensei"\ndescription = "x"\nrules = ["("]\n', "invalid rule in route 'sensei'"),
])
def test_bad_routing_files_are_rejected(tmp_path, text, problem):
    with pytest.raises(RoutingError, match=problem):
        load_routes(write_routes(tmp_path, text))


async def test_a_japanese_sentence_starts_with_the_checker():
    session = Session(routed_config(), client=ScriptedClient(CHECKED))
    assert await agents_run(session, "私は学校に行きます") == ["checker"]


async def test_a_question_starts_with_the_sensei():
    session = Session(routed_config(), client=FakeClient())
    assert await agents_run(session, "How do you say thank you?") == ["sensei"]
    assert session.history[-1]["content"] == "Konnichiwa"
