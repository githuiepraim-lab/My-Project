import pytest
from core.ai.adapters import make_adapter
from core.ai.adapters.base import normalize_base
from core.ai.autofix import autofix_models, guess_tier


@pytest.mark.parametrize("kind,raw,want", [
    ("openai_compat", "https://api.groq.com", "https://api.groq.com/openai/v1"),
    ("openai_compat", "api.groq.com/openai/v1/chat/completions", "https://api.groq.com/openai/v1"),
    ("openai_compat", "https://openrouter.ai", "https://openrouter.ai/api/v1"),
    ("openai_compat", "https://api.openai.com/", "https://api.openai.com/v1"),
    ("openai_compat", "https://api.openai.com/v1/", "https://api.openai.com/v1"),
    ("openai_compat", "http://localhost:1234", "http://localhost:1234/v1"),
    ("openai_compat", "https://my.host/custom/v2", "https://my.host/custom/v2"),
    ("anthropic", "https://api.anthropic.com/v1", "https://api.anthropic.com"),
    ("anthropic", "https://api.anthropic.com/v1/messages", "https://api.anthropic.com"),
    ("ollama", "http://localhost:11434/api", "http://localhost:11434"),
    ("ollama", "http://localhost:11434/v1", "http://localhost:11434")])
def test_base_urls_are_repaired(kind, raw, want):
    assert normalize_base(kind, raw) == want


def test_adapters_use_the_repaired_url():
    a = make_adapter("g", {"kind": "openai_compat", "base_url": "https://api.groq.com"}, "k")
    assert a.base == "https://api.groq.com/openai/v1"


def test_404_message_explains_itself():
    class Resp:
        status_code, text, headers = 404, '{"error":"model not found"}', {}
        def close(self): pass
    class Sess:
        def post(self, *a, **k): return Resp()
    a = make_adapter("g", {"kind": "openai_compat", "base_url": "https://api.openai.com/v1"}, "k")
    a._session = Sess()
    from core.ai.types import AIError, ModelSpec, Request
    with pytest.raises(AIError) as e:
        a.complete(Request(messages=[{"role": "user", "content": "x"}]), ModelSpec("g", "nope"))
    assert e.value.kind == "not_found" and "model name" in str(e.value) and "/chat/completions" in str(e.value)


def test_autofix_replaces_missing_models_with_real_ones(tmp_path, monkeypatch):
    from core.ai import config as C
    for n, v in (("CONFIG_DIR", tmp_path), ("PROVIDERS_FILE", tmp_path / "p.json"),
                 ("LEGACY_FILE", tmp_path / "l.json"), ("_cache", None)):
        monkeypatch.setattr(C, n, v)
    cfg = C.default_config()
    cfg["providers"]["groq"]["enabled"] = True
    C.save(cfg)
    class Hub:
        class health:
            reset = staticmethod(lambda: None)
        def _adapter(self, n):
            return type("A", (), {"list_models": lambda s: ["llama-3.1-8b-instant", "llama-3.3-70b-versatile",
                                                              "whisper-large-v3", "gemma2-9b-it"]})()
    msg = autofix_models(Hub(), "groq")
    ids = [m["id"] for m in C.load(force=True)["providers"]["groq"]["models"]]
    assert "whisper-large-v3" not in ids and "llama-3.1-8b-instant" in ids
    assert "using" in msg
    assert guess_tier("claude-opus-9") == 2 and guess_tier("gpt-4o-mini") == 0 and guess_tier("foo") == 1
