"""Provider doctor: send one tiny real request to each AI and say, in plain words, why it
does or does not respond — and repair what can be repaired.

    python -m core.ai check            every provider
    python -m core.ai check groq       one provider

Typical results: ✓ works · key rejected · model name not offered (repaired automatically) ·
quota used up · cannot reach the server · wrong URL · no key yet · switched off.
"""
from __future__ import annotations

from . import config as C
from .autofix import autofix_models
from .types import AIError

_FRIENDLY = {
    "auth": "{p} rejected the key. Re-copy the WHOLE key from the provider's site (no spaces or quotes) and use Set key again.",
    "rate_limit": "{p} accepted the key but its quota / rate limit is used up right now. Wait a bit, or lean on another provider.",
    "network": "This computer cannot reach {p}'s servers (network, VPN or firewall). Try another network.",
    "timeout": "{p} did not answer in time. The connection is slow or the service is busy; try again.",
    "unavailable": "{p} is having problems on its side (server error). Try again in a few minutes.",
    "bad_request": "{p} refused the request: {detail}",
    "not_found": "{p} says the model or address does not exist: {detail}",
    "invalid_response": "{p} answered with something unreadable. Check the base URL points at the API, not a website.",
    "unsupported": "{detail}",
}


def check(hub, provider: str, auto_fix: bool = True) -> tuple[str, str]:
    """(status, message). status: ok | fixed | no_provider | no_key | disabled | no_models | <error kind>."""
    cfg = C.load(force=True)
    p = cfg["providers"].get(provider)
    if p is None:
        return "no_provider", hub.why_unavailable(provider)
    if p.get("requires_key", True) is not False and not C.get_api_key(provider):
        return "no_key", f"{provider} has no API key yet. Use Set key (Settings → AI providers) or  python -m core.ai key {provider}"
    if p.get("user_disabled"):
        return "disabled", f"{provider} is switched off. Tick On in Settings → AI providers."
    if not p.get("models"):
        if auto_fix:
            note = autofix_models(hub, provider)
            p = C.load(force=True)["providers"][provider]
            if not p.get("models"):
                return "no_models", note
        else:
            return "no_models", f"{provider} has no models listed. Press Auto-fix models."
    hub.health.reset()                                   # test for real, not from a remembered failure
    specs = sorted(hub.models(), key=lambda m: m.tier)
    spec = next((m for m in specs if m.provider == provider), None)
    if spec is None:                                     # configured but filtered out (should not happen once keyed)
        C.update_provider(provider, enabled=True, user_disabled=False)
        spec = next((m for m in sorted(hub.models(), key=lambda m: m.tier) if m.provider == provider), None)
        if spec is None:
            return "unavailable", hub.why_unavailable(provider)
    r = hub._ask_one(spec, "", "Reply with the single word: ok", "", 20, 20.0, None)
    if r.ok:
        if not p.get("enabled"):
            C.update_provider(provider, enabled=True, user_disabled=False)
        return "ok", f"✓ {provider} works ({spec.id}, {r.latency:.1f}s)"
    kind, _, detail = r.error.partition(": ")
    if kind == "not_found" and auto_fix:
        note = autofix_models(hub, provider)
        again = [m for m in hub.models() if m.provider == provider]
        if again:
            r2 = hub._ask_one(sorted(again, key=lambda m: m.tier)[0], "", "Reply with the single word: ok", "", 20, 20.0, None)
            if r2.ok:
                return "fixed", f"✓ {provider} works now. {note}"
            kind, _, detail = r2.error.partition(": ")
    msg = _FRIENDLY.get(kind, "{p} failed: {detail}").format(p=provider, detail=detail[:160])
    return kind, msg


def check_all(hub, auto_fix: bool = True) -> list[tuple[str, str, str]]:
    """[(provider, status, message)] for every provider that has a key or needs none."""
    out = []
    for name, p in C.load(force=True)["providers"].items():
        keyed = p.get("requires_key", True) is False or bool(C.get_api_key(name))
        if not keyed and not p.get("enabled"):
            out.append((name, "no_key", "no key stored (skipped)"))
            continue
        out.append((name, *check(hub, name, auto_fix)))
    return out
