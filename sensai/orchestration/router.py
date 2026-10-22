# Router: decides which agent starts a learner turn, with a cascade of tiers from the cheapest to the
# most expensive. Tier 1 matches each route's regex rules from routing.toml; tier 2 compares the message
# with each route's examples by embedding similarity; when no tier decides, the default agent answers.
# Every decision says which tier made it and how sure it is.

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol

from ..core.config import load_toml, resolve
from ..llm.ollama import OllamaClient, OllamaError

TierName = Literal["rules", "embeddings", "classifier", "default"]


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


class Tier(Protocol):

    def decide(self, text: str) -> RouteDecision | None: ...


class Router:

    def __init__(self, tiers: list[Tier], default_agent: str):
        self.tiers = tiers
        self.default_agent = default_agent

    def route(self, text: str) -> RouteDecision:
        for tier in self.tiers:
            decision = tier.decide(text)
            if decision is not None:
                return decision
        return RouteDecision(self.default_agent, "default", 0.0, "no tier decided")


def load_router(config: dict, client: OllamaClient) -> Router | None:
    settings = config["router"]
    if not settings["enabled"]:
        return None
    routes = load_routes(resolve(settings["routes"]))
    _check_agents_exist(routes, config)
    tiers: list[Tier] = [RuleTier(routes)]
    if settings["embed_model"]:
        tiers.append(EmbeddingTier(routes, client, settings))
    return Router(tiers, config["agents"]["default"])


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


def cosine(first: list[float], second: list[float]) -> float:
    norms = math.sqrt(sum(value * value for value in first)) * math.sqrt(sum(value * value for value in second))
    return sum(a * b for a, b in zip(first, second)) / norms if norms else 0.0


def _score(ranked: tuple[Example, float]) -> float:
    return ranked[1]


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
