import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.ai.adapters.base import Adapter           # noqa: E402
from core.ai.health import Health                     # noqa: E402
from core.ai.hub import Hub                           # noqa: E402
from core.ai.types import AIError, Response, Usage    # noqa: E402
from core.ai.usage import UsageTracker                # noqa: E402


class FakeAdapter(Adapter):
    """Scripted adapter: `script` is a list of Response | AIError consumed in
    order (the last item repeats). No network, no cost."""
    calls = 0

    def __init__(self, name, cfg, api_key=""):
        super().__init__(name, cfg, api_key)
        self.script = cfg.get("_script") or [Response(text=f"hello from {name}",
                                                      usage=Usage(10, 5))]
        self.seen = []

    def complete(self, req, model):
        self.calls += 1
        self.seen.append((model.id, req))
        item = self.script[min(self.calls - 1, len(self.script) - 1)]
        if isinstance(item, AIError):
            raise item
        return Response(text=item.text, tool_calls=list(item.tool_calls),
                        usage=Usage(item.usage.input, item.usage.output))


def cfg_for(*providers, **settings):
    base = {"strategy": "cost", "mode": "auto", "manual_model": "", "fallback": True,
            "max_attempts": 4, "deadline_s": 45, "request_timeout_s": 25,
            "allow_downgrade": False, "cross_provider_fallback": True, "cache": True,
            "cache_ttl_s": 900, "max_input_tokens": 6000, "max_output_tokens": 800,
            "monthly_cost_cap_usd": None, "share_memory_default": False}
    base.update(settings)
    return {"settings": base, "providers": {p["_name"]: p for p in providers}, "agents": {}}


def prov(name, models, priority=10, **kw):
    d = {"_name": name, "kind": "openai_compat", "enabled": True, "priority": priority,
         "requires_key": False, "models": models}
    d.update(kw)
    return d


def model(id, tier, caps=("text", "stream"), pi=None, po=None, **kw):
    m = {"id": id, "tier": tier, "caps": list(caps)}
    if pi is not None:
        m["price_in"], m["price_out"] = pi, po
    m.update(kw)
    return m


@pytest.fixture
def make_hub(tmp_path):
    def _make(cfg, scripts=None):
        scripts = scripts or {}
        for n, p in cfg["providers"].items():
            if n in scripts:
                p["_script"] = scripts[n]
        sleeps = []
        hub = Hub(config_loader=lambda: cfg, key_getter=lambda n: "k",
                  adapter_factory=lambda n, p, k: FakeAdapter(n, p, k),
                  health=Health(), usage=UsageTracker(tmp_path / "u.jsonl"),
                  sleep=sleeps.append)
        hub._sleeps = sleeps
        return hub
    return _make
