"""core/gemini.py <-> the hub, with the Gemini SDK's own types but no network."""
import pytest

pytest.importorskip("google.genai")
from conftest import FakeAdapter, cfg_for, model, prov
from core import gemini
from core.ai import hub as hubmod
from core.ai.adapters.gemini import GeminiAdapter
from core.ai.health import Health
from core.ai.types import Request, Response, Usage
from core.ai.usage import UsageTracker


def make_hub(cfg, tmp_path):
    return hubmod.Hub(config_loader=lambda: cfg, key_getter=lambda n: "k",
                      adapter_factory=lambda n, p, k: FakeAdapter(n, p, k),
                      health=Health(), usage=UsageTracker(tmp_path / "u.jsonl"),
                      sleep=lambda s: None)


class Boom:
    """genai.Client stand-in whose every model errors out."""
    def __init__(self, *a, **k):
        self.models = self

    def generate_content(self, **k):
        raise RuntimeError("503 UNAVAILABLE")


@pytest.fixture(autouse=True)
def reset_gemini(monkeypatch):
    gemini._cooldown.clear()
    monkeypatch.setattr(gemini, "api_key", lambda refresh=False: "AIza-test-key-0000000000")
    monkeypatch.setattr(gemini, "_live_call", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("no live")))
    import google.genai as genai
    monkeypatch.setattr(genai, "Client", Boom)


def test_other_provider_answers_when_every_gemini_rung_fails(monkeypatch, tmp_path):
    cfg = cfg_for(prov("backup", [model("b", 0)]))
    h = make_hub(cfg, tmp_path)
    monkeypatch.setattr("core.ai.get_hub", lambda: h)
    r = gemini.call("say hi", tier=gemini.FAST)
    assert r is not None and r.text == "hello from backup"
    sent = h._adapter("backup").seen[0][1]
    assert "data-processing function" in sent.system        # one-shot contract preserved


def test_no_fallback_when_nothing_else_configured(monkeypatch, tmp_path):
    h = make_hub(cfg_for(), tmp_path)
    monkeypatch.setattr("core.ai.get_hub", lambda: h)
    assert gemini.call("say hi") is None                    # exactly the old behaviour


def test_fallback_can_be_switched_off(monkeypatch, tmp_path):
    h = make_hub(cfg_for(prov("backup", [model("b", 0)]), cross_provider_fallback=False), tmp_path)
    monkeypatch.setattr("core.ai.get_hub", lambda: h)
    assert gemini.call("say hi") is None


def test_adapter_requests_only_the_routed_model(monkeypatch):
    seen = []
    monkeypatch.setattr(gemini, "call", lambda contents, **k: seen.append(k) or None)
    a = GeminiAdapter("gemini", {"kind": "gemini"}, "key")
    from core.ai.types import AIError, ModelSpec
    with pytest.raises(AIError):
        a.complete(Request(messages=[{"role": "user", "content": "hi"}], system="s",
                           tools=[{"name": "t", "description": "d",
                                   "parameters": {"type": "object", "properties": {
                                       "x": {"type": "string"}}}}]),
                   ModelSpec("gemini", "gemini-2.5-flash", 1, frozenset({"text", "tools"})))
    assert seen[0]["models"] == ("gemini-2.5-flash",) and seen[0]["cross_fallback"] is False


def test_adapter_builds_valid_sdk_objects():
    a = GeminiAdapter("gemini", {"kind": "gemini"}, "key")
    from core.ai.types import Image
    r = Request(messages=[
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "1", "name": "t", "arguments": {"x": "1"}}]},
        {"role": "tool", "name": "t", "tool_call_id": "1", "content": "res"},
        {"role": "user", "content": "now this"}],
        images=[Image(b"\x89PNG", "image/png")], json_mode=True, max_output_tokens=50,
        tools=[{"name": "t", "description": "d", "parameters": {"type": "object", "properties": {
            "x": {"type": "string"}}}}])
    contents = a._contents(r)
    cfg = a._config(r)
    assert len(contents) == 4 and contents[0].role == "user" and contents[1].role == "model"
    assert cfg.response_mime_type == "application/json" and cfg.max_output_tokens == 50


def test_successful_call_is_recorded(monkeypatch, tmp_path):
    h = make_hub(cfg_for(), tmp_path)
    monkeypatch.setattr("core.ai.get_hub", lambda: h)

    class OK(Boom):
        def generate_content(self, **k):
            class R:
                text = "fine"
                usage_metadata = type("U", (), {"prompt_token_count": 11, "candidates_token_count": 4,
                                                "cached_content_token_count": 0})()
            return R()
    import google.genai as genai
    monkeypatch.setattr(genai, "Client", OK)
    assert gemini.call("x").text == "fine"
    row = list(h.usage.summary().values())[0]
    assert row["in"] == 11 and row["out"] == 4
