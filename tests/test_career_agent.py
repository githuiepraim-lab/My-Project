import pytest
from conftest import FakeAdapter, cfg_for, model, prov
from core.ai import hub as hubmod
from core.ai.health import Health
from core.ai.types import Response, Usage
from core.ai.usage import UsageTracker


@pytest.fixture
def ca(monkeypatch, tmp_path):
    cfg = cfg_for(prov("trusted", [model("v", 1, caps=("text", "vision"))], share_memory=True, priority=2),
                  prov("cloud", [model("c", 0, caps=("text", "vision"))], share_memory=False, priority=1))
    h = hubmod.Hub(config_loader=lambda: cfg, key_getter=lambda n: "k",
                   adapter_factory=lambda n, p, k: FakeAdapter(n, p, k), health=Health(),
                   usage=UsageTracker(tmp_path / "u.jsonl"), sleep=lambda s: None)
    import actions.career_agent as m
    monkeypatch.setattr(m, "get_hub", lambda: h)
    monkeypatch.setattr(m, "_capture", lambda src: m.Image(b"\xff\xd8img", "image/jpeg"))
    m._h = h
    return m


def test_camera_frame_only_goes_to_trusted_provider(ca):
    out = ca.career_agent({"mode": "coach", "source": "camera"})
    assert out == "hello from trusted"
    assert ca._h._adapter("trusted").seen[0][1].images
    assert "cloud" not in ca._h._adapters or ca._h._adapters["cloud"][1].calls == 0


def test_draft_never_sends_and_returns_text(ca, monkeypatch):
    sent = []
    import core.confirm as cf
    monkeypatch.setattr(cf, "request", lambda *a, **k: sent.append(a) or "x")
    assert ca.career_agent({"mode": "draft", "to": "Dr Mwangi", "goal": "ask for advice"})
    assert sent == []


def test_send_is_parked_behind_confirmation(ca, monkeypatch):
    import core.confirm as cf
    got = {}
    monkeypatch.setattr(cf, "request", lambda key, title, detail, run: got.update(t=title, run=run) or "Ask the user to confirm")
    out = ca.career_agent({"mode": "send", "platform": "WhatsApp", "to": "Amina", "text": "Hi Amina"})
    assert "confirm" in out.lower() and "Amina" in got["t"]       # nothing was sent yet


def test_send_needs_all_fields(ca):
    assert "need" in ca.career_agent({"mode": "send", "platform": "x"}).lower()


def test_tool_registers(tmp_path):
    from pathlib import Path
    from core.action_loader import discover_actions
    assert discover_actions(Path("actions"), logger=lambda s: None).has("career_agent")
