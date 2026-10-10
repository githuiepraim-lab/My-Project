import json

import pytest
from conftest import FakeAdapter, cfg_for, model, prov
from core.ai import hub as hubmod
from core.ai.health import Health
from core.ai.usage import UsageTracker


@pytest.fixture
def action(monkeypatch, tmp_path):
    cfg = cfg_for(prov("alpha", [model("a", 1)], priority=1), prov("beta", [model("b", 1)], priority=2))
    h = hubmod.Hub(config_loader=lambda: cfg, key_getter=lambda n: "k",
                   adapter_factory=lambda n, p, k: FakeAdapter(n, p, k), health=Health(),
                   usage=UsageTracker(tmp_path / "u.jsonl"), sleep=lambda s: None)
    import actions.multi_ai as m
    monkeypatch.setattr(m, "get_hub", lambda: h)
    return m


def test_tool_is_valid_for_the_action_loader(tmp_path):
    from pathlib import Path
    from core.action_loader import discover_actions
    reg = discover_actions(Path("actions"), logger=lambda s: None)
    assert reg.has("multi_ai")
    decl = [d for d in reg.get_tool_declarations() if d["name"] == "multi_ai"][0]
    # No Live-only fields: an unsupported `behavior` can make the server reject the whole
    # session at connect, which looked like the assistant flickering thinking/sleeping.
    assert "behavior" not in decl and len(decl["description"]) < 600


def test_compare_returns_each_answer(action):
    out = action.multi_ai({"mode": "compare", "prompt": "q", "models": "alpha, beta"})
    assert "alpha:" in out and "beta:" in out


def test_discuss_returns_summary(action):
    out = action.multi_ai({"mode": "discuss", "prompt": "tabs?", "models": "alpha, beta", "rounds": 1})
    assert "turns" in out


def test_status_and_empty_prompt(action):
    assert "alpha: ready" in action.multi_ai({"mode": "status"})
    assert action.multi_ai({"mode": "ask", "prompt": ""}).startswith("What")


def test_cli_status_runs(capsys, tmp_path, monkeypatch):
    from core.ai import config as C
    monkeypatch.setattr(C, "CONFIG_DIR", tmp_path)
    monkeypatch.setattr(C, "PROVIDERS_FILE", tmp_path / "providers.json")
    monkeypatch.setattr(C, "LEGACY_FILE", tmp_path / "api_keys.json")
    monkeypatch.setattr(C, "_cache", None)
    from core.ai.__main__ import main
    assert main(["status"]) == 0
    assert "gemini" in capsys.readouterr().out
