# Router: decides which agent starts a learner turn, with a cascade of tiers from the cheapest to the
# most expensive. Tier 1 matches each route's regex rules from routing.toml; tier 2 compares the message
# with each route's examples by embedding similarity; tier 3 asks the classifier agent to pick a route.
# When no tier decides, the default agent answers. A message starting with @<agent> overrides every
# tier, and a decision below the confidence floor falls back to the sensei. Every decision says which
# tier made it, how sure it is and why.

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol

from ..core.agent_spec import AgentSpec, spec_for
from ..core.config import load_toml, resolve
from ..core.messages import Message
from ..llm.ollama import OllamaClient, OllamaError
from . import structured

TierName = Literal["override", "rules", "embeddings", "classifier", "floor", "default"]
OVERRIDE = re.compile(r"^@([\w/-]+)\s+(.+)$", re.DOTALL)


class RoutingError(ValueError):
    pass


@dataclass(frozen=True)
class Route:
    agent: str
    description: str
    rules: tuple[re.Pattern, ...] = ()
    examples: tuple[str, ...] = ()


@dataclass(frozen=True)
class RouteDecision:
    agent: str
    tier: TierName
    confidence: float
    reason: str


class RuleTier:

    def __init__(self, routes: list[Route]):
        self.routes = routes

    def decide(self, text: str) -> RouteDecision | None:
        for route in self.routes:
            rule = _first_matching_rule(route, text)
            if rule is not None:
                return RouteDecision(route.agent, "rules", 1.0, f"matched {rule.pattern!r}")
        return None


@dataclass(frozen=True)
class Example:
    agent: str
    text: str
    vector: list[float]


class EmbeddingTier:

    def __init__(self, routes: list[Route], client: OllamaClient, settings: dict):
        self.routes = routes
        self.client = client
        self.model = settings["embed_model"]
        self.threshold = settings["embedding_threshold"]
        self.margin = settings["embedding_margin"]
        self.options = {"num_gpu": 0} if settings["embed_on_cpu"] else {}
        self.examples: list[Example] = []

    def decide(self, text: str) -> RouteDecision | None:
        try:
            ranking = self._ranking(text)
        except OllamaError:
            return None
        best, best_score = ranking[0]
        runner_up_score = ranking[1][1] if len(ranking) > 1 else 0.0
        if best_score < self.threshold or best_score - runner_up_score < self.margin:
            return None
        return RouteDecision(best.agent, "embeddings", round(best_score, 3),
                             f"closest example {best.text!r} ({best_score:.2f} vs {runner_up_score:.2f})")

    def _ranking(self, text: str) -> list[tuple[Example, float]]:
        query = self.client.embed(text, self.model, **self.options)
        best_by_agent: dict[str, tuple[Example, float]] = {}
        for example in self._embedded_examples():
            score = cosine(query, example.vector)
            if example.agent not in best_by_agent or score > best_by_agent[example.agent][1]:
                best_by_agent[example.agent] = (example, score)
        return sorted(best_by_agent.values(), key=_score, reverse=True)

    def _embedded_examples(self) -> list[Example]:
        if not self.examples:
            pairs = [(route.agent, text) for route in self.routes for text in route.examples]
            vectors = self.client.embed_many([text for _, text in pairs], self.model, **self.options)
            self.examples = [Example(agent, text, vector) for (agent, text), vector in zip(pairs, vectors)]
        return self.examples


class ClassifierTier:

    def __init__(self, routes: list[Route], client: OllamaClient, spec: AgentSpec):
        self.client = client
        self.spec = spec
        self.schema = classifier_schema([route.agent for route in routes])
        self.system_prompt = "\n\n".join([spec.system_prompt, _route_list(routes),
                                           structured.schema_instructions(self.schema)])

    def decide(self, text: str) -> RouteDecision | None:
        try:
            reply = self._ask(text)
        except OllamaError:
            return None
        data, problems = structured.validate_reply(reply, self.schema)
        if problems:
            return None
        return RouteDecision(data["agent"], "classifier", round(data["confidence"], 3), data["reason"])

    def _ask(self, text: str) -> str:
        messages: list[Message] = [{"role": "system", "content": self.system_prompt},
                                   {"role": "user", "content": text}]
        stream = self.client.chat_stream(self.spec.model, messages, num_ctx=self.spec.num_ctx,
                                         temperature=self.spec.temperature, think=False, format="json")
        return "".join(piece for piece in stream if isinstance(piece, str))


class Tier(Protocol):

    def decide(self, text: str) -> RouteDecision | None: ...


class Router:

    def __init__(self, tiers: list[Tier], default_agent: str, fallback_agent: str = "sensei",
                 confidence_floor: float = 0.0, agents: frozenset[str] = frozenset()):
        self.tiers = tiers
        self.default_agent = default_agent
        self.fallback_agent = fallback_agent
        self.confidence_floor = confidence_floor
        self.agents = agents

    def route(self, text: str) -> RouteDecision:
        override = self.override(text)
        if override is not None:
            return override
        for tier in self.tiers:
            decision = tier.decide(text)
            if decision is not None:
                return self._above_floor(decision)
        return RouteDecision(self.default_agent, "default", 0.0, "no tier decided")

    def override(self, text: str) -> RouteDecision | None:
        match = OVERRIDE.match(text.strip())
        if match is None or match.group(1) not in self.agents:
            return None
        return RouteDecision(match.group(1), "override", 1.0, f"the learner asked for @{match.group(1)}")

    def strip_override(self, text: str) -> str:
        return OVERRIDE.match(text.strip()).group(2) if self.override(text) else text

    def _above_floor(self, decision: RouteDecision) -> RouteDecision:
        if decision.confidence >= self.confidence_floor:
            return decision
        return RouteDecision(self.fallback_agent, "floor", decision.confidence,
                             f"{decision.agent} by {decision.tier} is below the {self.confidence_floor} floor "
                             f"({decision.reason})")


def load_router(config: dict, client: OllamaClient) -> Router | None:
    settings = config["router"]
    if not settings["enabled"]:
        return None
    routes = load_routes(resolve(settings["routes"]))
    _check_agents_exist(routes, config)
    tiers: list[Tier] = [RuleTier(routes)]
    if settings["embed_model"]:
        tiers.append(EmbeddingTier(routes, client, settings))
    if settings["classifier"]:
        tiers.append(ClassifierTier(routes, client, spec_for(settings["classifier"], config)))
    agents = config["agents"]
    return Router(tiers, agents["default"], agents["fallback"], settings["confidence_floor"], _agent_names(config))


def load_routes(path: str | Path) -> list[Route]:
    routes = load_toml(path).get("routes", [])
    if not routes:
        raise RoutingError(f"{path}: no [[routes]] defined")
    return [_route_from(entry, path) for entry in routes]


def _route_from(entry: dict, path: str | Path) -> Route:
    try:
        rules = tuple(re.compile(rule) for rule in entry.get("rules", ()))
        return Route(entry["agent"], entry["description"], rules, tuple(entry.get("examples", ())))
    except KeyError as missing:
        raise RoutingError(f"{path}: a route is missing {missing}") from None
    except re.error as error:
        raise RoutingError(f"{path}: invalid rule in route '{entry.get('agent')}': {error}") from None


def classifier_schema(agents: list[str]) -> dict:
    return {
        "type": "object",
        "properties": {"agent": {"type": "string", "enum": agents},
                       "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                       "reason": {"type": "string", "description": "Why, in a few words."}},
        "required": ["agent", "confidence", "reason"],
        "additionalProperties": False,
    }


def cosine(first: list[float], second: list[float]) -> float:
    norms = math.sqrt(sum(value * value for value in first)) * math.sqrt(sum(value * value for value in second))
    return sum(a * b for a, b in zip(first, second)) / norms if norms else 0.0


def _score(ranked: tuple[Example, float]) -> float:
    return ranked[1]


def _route_list(routes: list[Route]) -> str:
    return "Routes:\n" + "\n".join(f"- {route.agent}: {route.description}" for route in routes)


def _agent_names(config: dict) -> frozenset[str]:
    agents_dir = resolve(config["agents"]["dir"])
    return frozenset(path.relative_to(agents_dir).with_suffix("").as_posix() for path in agents_dir.rglob("*.toml"))


def _check_agents_exist(routes: list[Route], config: dict) -> None:
    agents_dir = resolve(config["agents"]["dir"])
    unknown = [route.agent for route in routes if not (agents_dir / f"{route.agent}.toml").exists()]
    if unknown:
        raise RoutingError(f"routes point to unknown agent(s): {', '.join(unknown)}")


def _first_matching_rule(route: Route, text: str) -> re.Pattern | None:
    for rule in route.rules:
        if rule.search(text):
            return rule
    return None
