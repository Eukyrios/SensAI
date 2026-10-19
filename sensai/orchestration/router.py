# Router: decides which agent starts a learner turn, with a cascade of tiers from the cheapest to the
# most expensive. Tier 1 matches each route's regex rules from routing.toml; when no tier decides,
# the default agent answers. Every decision says which tier made it and how sure it is.

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from ..core.config import load_toml, resolve

Tier = Literal["rules", "embeddings", "classifier", "default"]


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
    tier: Tier
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


class Router:

    def __init__(self, routes: list[Route], default_agent: str):
        self.routes = routes
        self.default_agent = default_agent
        self.tiers = [RuleTier(routes)]

    def route(self, text: str) -> RouteDecision:
        for tier in self.tiers:
            decision = tier.decide(text)
            if decision is not None:
                return decision
        return RouteDecision(self.default_agent, "default", 0.0, "no tier decided")


def load_router(config: dict) -> Router | None:
    settings = config["router"]
    if not settings["enabled"]:
        return None
    routes = load_routes(resolve(settings["routes"]))
    _check_agents_exist(routes, config)
    return Router(routes, config["agents"]["default"])


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
