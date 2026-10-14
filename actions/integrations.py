"""integrations — call any connected web API (weather, FX, Wikipedia, your own…).
Reading is automatic; anything that changes something elsewhere needs your approval."""
from __future__ import annotations

import json

from core import confirm
from core.integrations import Integrations, IntegrationError, NeedsApproval


def integrations(parameters: dict, player=None) -> str:
    mode = str(parameters.get("mode", "list")).lower()
    reg = Integrations()
    try:
        if mode == "list":
            a = reg.all()
            return "; ".join(f"{n}: {s.get('description', '')[:60]}" for n, s in sorted(a.items()))
        name = str(parameters.get("name", "")).strip().lower()
        if mode == "describe":
            return reg.describe(name)
        try:
            raw = parameters.get("params") or "{}"
            params = raw if isinstance(raw, dict) else json.loads(str(raw))
            if not isinstance(params, dict):
                raise ValueError
        except ValueError:
            return "Sir, 'params' must be a JSON object like {\"city\": \"Nairobi\"}."
        if mode == "call":
            action = str(parameters.get("action", "")).strip()
            try:
                return reg.call(name, action, params)
            except NeedsApproval:
                return confirm.request("integration_call", f"Send to {name}: {action}", json.dumps(params)[:300],
                                       lambda: reg.call(name, action, params, approved=True))
        if mode == "add":
            spec = json.loads(str(parameters.get("spec", "")))
            reg.validate(spec)
            return confirm.request("integration_add", f"Connect a new service: {name}",
                                   f"{spec.get('base_url', '(secret URL)')} — {', '.join(spec['actions'])}",
                                   lambda: (reg.save(name, spec), f"{name} connected.")[1])
        if mode == "remove":
            return f"Removed {name}." if reg.remove(name) else f"{name} is not one of your own integrations."
        return "Modes: list, describe (name), call (name, action, params), add (name, spec JSON), remove (name)."
    except (IntegrationError, ValueError, KeyError) as e:
        return f"Sir, the integration failed: {e}."
    except Exception as e:
        return f"Sir, integrations failed: {type(e).__name__}: {str(e)[:100]}"


TOOL = {
    "name": "integrations",
    "description": (
        "Use connected web APIs: weather, geocode, wikipedia, fx (currency), crypto, dictionary, hackernews, nasa, "
        "countries, books, plus the user's own. mode: list | describe (name) | call (name, action, params JSON) | "
        "add (name, spec JSON; user confirms) | remove. Reading is automatic; posting needs approval."),
    "parameters": {
        "type": "OBJECT",
        "properties": {
            "mode": {"type": "STRING", "description": "list|describe|call|add|remove"},
            "name": {"type": "STRING", "description": "integration name"},
            "action": {"type": "STRING", "description": "action name"},
            "params": {"type": "STRING", "description": "JSON object of values, e.g. {\"name\":\"Nairobi\"}"},
            "spec": {"type": "STRING", "description": "JSON definition for add"},
        },
        "required": ["mode"],
    },
    "handler": integrations,
}
