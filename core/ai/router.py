"""Task classification and model ranking. Pure functions over ModelSpecs, no
network: the router decides, the hub executes."""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from . import types as T
from .types import ModelSpec, Request

_COMPLEX = re.compile(
    r"\b(debug|refactor|implement|architecture|design|prove|derive|optimi[sz]e|"
    r"analy[sz]e|compare|trade-?offs?|step[- ]by[- ]step|algorithm|strategy|"
    r"root cause|why (does|is|did)|explain (how|why)|migrate|review)\b", re.I)
_PLAN = re.compile(r"\b(plan|roadmap|break (it )?down|schedule|itinerary|multi-?step)\b", re.I)
_CODE = re.compile(r"```|\bdef \w+\(|\bclass \w+|function\s*\w*\(|=>|#include|"
                   r"\bimport \w+|\bSELECT\b.+\bFROM\b|Traceback|\bnpm\b|\bpip\b", re.I)

_TASK_TIER = {"classify": T.LIGHT, "extract": T.LIGHT, "translate": T.LIGHT,
              "summarize": T.LIGHT, "rewrite": T.LIGHT, "ack": T.LIGHT,
              "command": T.LIGHT, "reasoning": T.STRONG, "plan": T.STRONG,
              "code": T.STANDARD, "vision": T.STANDARD}

# Used only to order models whose price is unknown: USD per 1M tokens, blended.
_TIER_PROXY_COST = {T.LIGHT: 0.2, T.STANDARD: 2.0, T.STRONG: 10.0}


@dataclass
class Profile:
    tier: int
    needs: set = field(default_factory=set)
    reason: str = ""


def classify(req: Request) -> Profile:
    """Choose the weakest tier that should do this job well. Heuristic and
    free: a model call to decide which model to call would defeat the point."""
    needs = req.needs()
    last = ""
    for m in reversed(req.messages):
        if m.get("role") == "user":
            last = str(m.get("content", ""))
            break
    total_chars = len(last) + sum(len(str(m.get("content", ""))) for m in req.messages[:-1]) // 4

    if req.task in _TASK_TIER:
        tier, why = _TASK_TIER[req.task], f"task={req.task}"
        if req.task in ("code", "vision") and (len(last) > 3000 or _COMPLEX.search(last)):
            tier, why = T.STRONG, why + "+complex"
    else:
        tier, why = T.LIGHT, "short/simple"
        if _CODE.search(last):
            tier, why = T.STANDARD, "contains code"
        if _COMPLEX.search(last) or _PLAN.search(last):
            tier, why = (T.STRONG if len(last) > 280 else T.STANDARD), "needs reasoning"
        if len(last) > 3000 or total_chars > 12_000:
            tier, why = max(tier, T.STANDARD), "long input"
        if req.images:
            tier, why = max(tier, T.STANDARD), "image input"
        if req.tools and tier == T.LIGHT and len(last) > 160:
            tier, why = T.STANDARD, "tool use"
    if req.min_tier is not None:
        tier = max(tier, req.min_tier)
    return Profile(tier=tier, needs=needs, reason=why)


def est_cost(m: ModelSpec, tokens_in: int, tokens_out: int) -> float:
    c = m.cost(tokens_in, tokens_out)
    if c is not None:
        return c
    return _TIER_PROXY_COST[m.tier] * (tokens_in + tokens_out) / 1_000_000


def rank(models: list[ModelSpec], profile: Profile, *, strategy: str,
         priority: dict, latency, available, tokens_in: int, tokens_out: int,
         allow_downgrade: bool, exclude: frozenset = frozenset()) -> tuple[list[ModelSpec], bool]:
    """Ordered candidates plus a flag saying whether they are weaker than asked.

    Models lacking a needed capability are removed outright. The remaining
    adequate ones (tier >= wanted) come first, cheapest-sufficient first;
    stronger tiers follow; weaker ones are appended only when allowed, or when
    nothing adequate exists at all (a weaker answer beats no answer)."""
    ok = [m for m in models
          if profile.needs <= set(m.caps) and m.provider not in exclude and available(m.key)]

    def key(m: ModelSpec):
        c = est_cost(m, tokens_in, tokens_out)
        lat = latency(m.key)
        lat = lat if lat is not None else (1.0 + m.tier)
        p = priority.get(m.provider, 100)
        if strategy == "priority":
            return (p, m.tier, c)
        if strategy == "speed":
            return (lat, c, p)
        return (m.tier, c, p, lat)

    adequate = sorted((m for m in ok if m.tier >= profile.tier), key=key)
    weaker = sorted((m for m in ok if m.tier < profile.tier), key=lambda m: (-m.tier,) + key(m))
    if adequate:
        return adequate + (weaker if allow_downgrade else []), False
    return weaker, bool(weaker)
