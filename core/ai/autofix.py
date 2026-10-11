"""Repair a provider whose configured models do not exist (the other common
HTTP 404): ask the provider what it really offers, keep what is valid, and fill
the rest from its list with sensible tiers."""
from __future__ import annotations

import re

from . import config as C

_LIGHT = re.compile(r"(mini|lite|small|haiku|instant|flash-8b|8b|nano|tiny|1b|3b|7b)", re.I)
_STRONG = re.compile(r"(opus|large|70b|405b|ultra|pro\b|o1|o3|reason|r1|sonnet-?\d.*max)", re.I)
_SKIP = re.compile(r"(embed|whisper|tts|audio|image|moderation|guard|rerank|vision-preview|dall)", re.I)


def guess_tier(model_id: str) -> int:
    if _STRONG.search(model_id):
        return 2
    if _LIGHT.search(model_id):
        return 0
    return 1


def autofix_models(hub, provider: str, limit: int = 4) -> str:
    """Rewrite `provider`'s model list from what it actually serves. Returns a
    one-line report. Never touches keys."""
    cfg = C.load(force=True)
    p = cfg["providers"].get(provider)
    if not p:
        return f"No provider called '{provider}'."
    offered = [m for m in hub._adapter(provider).list_models() if not _SKIP.search(m)]
    if not offered:
        return (f"{provider} did not return a model list. Check the key and the base URL "
                f"({p.get('base_url', 'default')}).")
    have = {m["id"]: m for m in p.get("models", [])}
    keep = [m for mid, m in have.items() if mid in offered]
    dropped = [mid for mid in have if mid not in offered]
    if not keep:
        by_tier: dict[int, list[str]] = {0: [], 1: [], 2: []}
        for mid in offered:
            by_tier[guess_tier(mid)].append(mid)
        picks = []
        for t in (1, 0, 2):                       # one per tier first, standard before the rest
            if by_tier[t]:
                picks.append(sorted(by_tier[t], key=len)[0])
        picks += [m for m in offered if m not in picks]
        caps = ["text", "stream"] + (["tools"] if p.get("kind") != "ollama" else [])
        keep = [{"id": m, "tier": guess_tier(m), "caps": caps} for m in picks[:limit]]
    p["models"] = keep
    hub.health.reset()
    C.save(cfg)
    msg = f"{provider}: using {', '.join(m['id'] for m in keep)}."
    if dropped:
        msg += f" Removed models it does not offer: {', '.join(dropped)}."
    return msg
