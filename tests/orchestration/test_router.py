# Tests for the router: routes, the three tiers, the override, the floor, the default and the wiring.

import json
import re

import pytest

from sensai.core import events as ev
from sensai.core.config import resolve
from sensai.llm.ollama import OllamaError
from sensai.core.agent_spec import AgentSpec
from sensai.orchestration.router import (ClassifierTier, EmbeddingTier, Route, RouteDecision, Router, RoutingError,
                                         RuleTier, classifier_schema, cosine, load_router, load_routes)
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
    decision = Router([RuleTier(ROUTES)], "sensei").route("konnichiwa")
    assert decision == RouteDecision("sensei", "default", 0.0, "no tier decided")


def test_load_router_is_off_when_disabled():
    assert load_router(make_config(), FakeClient()) is None
    assert isinstance(load_router(routed_config(), FakeClient()), Router)


def test_routes_must_point_to_existing_agents(tmp_path):
    path = write_routes(tmp_path, '[[routes]]\nagent = "nobody"\ndescription = "x"\n')
    with pytest.raises(RoutingError, match="unknown agent.*nobody"):
        load_router(routed_config(routes=str(path)), FakeClient())


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


class VectorClient(FakeClient):

    def __init__(self, vectors, error=None):
        super().__init__()
        self.vectors = vectors
        self.error = error
        self.embed_calls = []

    def embed_many(self, texts, model, **options):
        self.embed_calls.append((list(texts), model, options))
        if self.error:
            raise self.error
        return [self.vectors.get(text, [0.0, 0.0]) for text in texts]


EMBED_SETTINGS = {"embed_model": "bge-m3", "embed_on_cpu": True, "embedding_threshold": 0.55,
                  "embedding_margin": 0.05}
TWO_ROUTES = [Route("checker", "check", examples=("watashi wa", "kare wa")),
              Route("sensei", "teach", examples=("how do you say",))]
VECTORS = {"watashi wa": [1.0, 0.0], "kare wa": [0.9, 0.1], "how do you say": [0.0, 1.0]}


def embedding_tier(query_vector, error=None, **settings):
    client = VectorClient({**VECTORS, "query": query_vector}, error)
    return EmbeddingTier(TWO_ROUTES, client, {**EMBED_SETTINGS, **settings}), client


def test_embeddings_route_to_the_closest_examples():
    tier, client = embedding_tier([1.0, 0.05])
    decision = tier.decide("query")
    assert (decision.agent, decision.tier) == ("checker", "embeddings")
    assert decision.confidence > 0.99 and "'watashi wa'" in decision.reason
    assert client.embed_calls == [(["query"], "bge-m3", {"num_gpu": 0}),
                                  (["watashi wa", "kare wa", "how do you say"], "bge-m3", {"num_gpu": 0})]


def test_examples_are_embedded_once():
    tier, client = embedding_tier([0.0, 1.0], embed_on_cpu=False)
    assert tier.decide("query").agent == "sensei" and tier.decide("query").agent == "sensei"
    assert [texts for texts, _, _ in client.embed_calls].count(["watashi wa", "kare wa", "how do you say"]) == 1
    assert client.embed_calls[0][2] == {}


@pytest.mark.parametrize("query_vector", [[0.5, 0.5], [-1.0, 0.0]])
def test_embeddings_stay_out_when_unsure(query_vector):
    tier, _ = embedding_tier(query_vector, embedding_threshold=0.95)
    assert tier.decide("query") is None


def test_close_runner_up_means_no_decision():
    tier, _ = embedding_tier([1.0, 1.2], embedding_threshold=0.1)
    assert tier.decide("query") is None


def test_a_single_route_needs_only_the_threshold():
    client = VectorClient({"watashi wa": [1.0, 0.0], "query": [1.0, 0.0]})
    tier = EmbeddingTier([Route("checker", "check", examples=("watashi wa",))], client, EMBED_SETTINGS)
    assert tier.decide("query").agent == "checker"


def test_embedding_errors_leave_the_decision_to_the_next_tier():
    tier, _ = embedding_tier([1.0, 0.0], error=OllamaError("model_missing", "pull bge-m3"))
    assert tier.decide("query") is None


def test_cosine_of_an_empty_vector_is_zero():
    assert cosine([0.0, 0.0], [1.0, 0.0]) == 0.0
    assert cosine([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)


def test_rules_decide_before_embeddings():
    client = VectorClient({})
    router = Router([RuleTier(ROUTES), EmbeddingTier(ROUTES, client, EMBED_SETTINGS)], "sensei")
    assert router.route("私は学校を行きます").tier == "rules"
    assert client.embed_calls == []


def test_embeddings_decide_what_rules_leave_alone():
    tier, _ = embedding_tier([1.0, 0.0])
    router = Router([RuleTier(TWO_ROUTES), tier], "sensei")
    assert router.route("query").tier == "embeddings"


def tier_types(router):
    return [type(tier).__name__ for tier in router.tiers]


def test_the_embedding_and_classifier_tiers_are_optional():
    assert tier_types(load_router(routed_config(), FakeClient())) == ["RuleTier", "EmbeddingTier", "ClassifierTier"]
    assert tier_types(load_router(routed_config(embed_model="", classifier=""), FakeClient())) == ["RuleTier"]


CLASSIFIER = AgentSpec(name="classifier", model="qwen3.5:9b", num_ctx=2048, temperature=0,
                       system_prompt="You route messages.")


def classifier_reply(agent="checker", confidence=0.8):
    return json.dumps({"agent": agent, "confidence": confidence, "reason": "a romaji sentence"})


def test_classifier_picks_a_route_with_its_confidence():
    decision = ClassifierTier(ROUTES, ScriptedClient(classifier_reply()), CLASSIFIER).decide("ore wa")
    assert decision == RouteDecision("checker", "classifier", 0.8, "a romaji sentence")


def test_classifier_sees_the_routes_and_answers_in_json():
    client = ScriptedClient(classifier_reply())
    ClassifierTier(ROUTES, client, CLASSIFIER).decide("ore wa")
    model, messages, options = client.calls[0]
    assert model == "qwen3.5:9b" and messages[1] == {"role": "user", "content": "ore wa"}
    system = messages[0]["content"]
    assert system.startswith("You route messages.") and "- checker: The learner wrote" in system
    assert '"enum": ["checker", "sensei"]' in system
    assert options == {"num_ctx": 2048, "temperature": 0, "think": False, "format": "json"}


@pytest.mark.parametrize("reply", ["not json", classifier_reply(agent="nobody"), classifier_reply(confidence=3)])
def test_an_invalid_classifier_reply_decides_nothing(reply):
    assert ClassifierTier(ROUTES, ScriptedClient(reply), CLASSIFIER).decide("ore wa") is None


def test_classifier_errors_decide_nothing():
    client = FakeClient(error=OllamaError("timeout", "slow"))
    assert ClassifierTier(ROUTES, client, CLASSIFIER).decide("ore wa") is None


def test_classifier_only_runs_when_the_other_tiers_are_unsure():
    client = ScriptedClient(classifier_reply(agent="sensei", confidence=0.6))
    router = Router([RuleTier(ROUTES), ClassifierTier(ROUTES, client, CLASSIFIER)], "sensei")
    assert router.route("私は学校を行きます").tier == "rules" and client.calls == []
    assert router.route("I like anime").tier == "classifier" and len(client.calls) == 1


def test_classifier_schema_limits_the_agents():
    schema = classifier_schema(["checker", "sensei"])
    assert schema["properties"]["agent"]["enum"] == ["checker", "sensei"]
    assert schema["required"] == ["agent", "confidence", "reason"]


class FixedTier:

    def __init__(self, decision):
        self.decision = decision

    def decide(self, text):
        return self.decision


AGENTS = frozenset({"sensei", "checker", "demo/ping"})


def floor_router(decision, floor=0.5):
    return Router([FixedTier(decision)], "sensei", "sensei", floor, AGENTS)


def test_at_agent_overrides_every_tier():
    router = floor_router(RouteDecision("sensei", "rules", 1.0, "x"))
    decision = router.route("@checker kinou eiga wo mimashita")
    assert decision == RouteDecision("checker", "override", 1.0, "the learner asked for @checker")
    assert router.route("@demo/ping hello").agent == "demo/ping"


@pytest.mark.parametrize("text", ["@nobody hello", "@checker", "mail me @checker please", "email@checker.com hi"])
def test_only_a_known_agent_at_the_start_overrides(text):
    router = floor_router(RouteDecision("sensei", "rules", 1.0, "x"))
    assert router.route(text).tier == "rules" and router.strip_override(text) == text


def test_strip_override_keeps_the_message():
    router = floor_router(None)
    assert router.strip_override("  @checker 猫が\n好きです ") == "猫が\n好きです"


def test_unsure_decisions_fall_back_to_the_sensei():
    decision = floor_router(RouteDecision("checker", "classifier", 0.3, "maybe a sentence")).route("x")
    assert (decision.agent, decision.tier, decision.confidence) == ("sensei", "floor", 0.3)
    assert decision.reason == "checker by classifier is below the 0.5 floor (maybe a sentence)"


def test_sure_decisions_pass_the_floor():
    decision = RouteDecision("checker", "embeddings", 0.5, "close")
    assert floor_router(decision).route("x") == decision


def test_the_router_knows_every_agent_for_the_override():
    agents = load_router(routed_config(), FakeClient()).agents
    assert {"sensei", "checker", "classifier", "demo/ping", "demo/pong"} <= agents


async def test_the_turn_emits_the_decision_and_stores_the_message_without_the_override():
    session = Session(routed_config(), client=ScriptedClient(CHECKED, "Perfect!"))
    events = [event async for event in run_turn(session, "@checker kinou eiga wo mimashita")]
    route = next(event for event in events if isinstance(event, ev.RouteDecided))
    assert (route.agent, route.tier) == ("checker", "override")
    assert type(events[1]).__name__ == "RouteDecided"
    assert session.history[0] == {"role": "user", "content": "kinou eiga wo mimashita"}


async def test_no_decision_event_when_routing_is_off():
    session = Session(make_config(), client=FakeClient())
    events = [event async for event in run_turn(session, "hello")]
    assert not any(isinstance(event, ev.RouteDecided) for event in events)
