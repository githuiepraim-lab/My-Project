import json

import pytest
from core.ai import config as C


@pytest.fixture
def cfgdir(tmp_path, monkeypatch):
    monkeypatch.setattr(C, "CONFIG_DIR", tmp_path)
    monkeypatch.setattr(C, "PROVIDERS_FILE", tmp_path / "providers.json")
    monkeypatch.setattr(C, "SECRETS_FILE", tmp_path / "secrets.json")
    monkeypatch.setattr(C, "LEGACY_FILE", tmp_path / "api_keys.json")
    monkeypatch.setattr(C, "_keyring", lambda: None)
    monkeypatch.setattr(C, "_cache", None)
    for n in ("GEMINI", "OPENAI", "GROQ", "ANTHROPIC", "OPENROUTER"):
        monkeypatch.delenv(f"{n}_API_KEY", raising=False)
        monkeypatch.delenv(f"EPHRAIM_{n}_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    return tmp_path


def test_migrates_legacy_gemini_and_ollama(cfgdir):
    (cfgdir / "api_keys.json").write_text(json.dumps(
        {"gemini_api_key": "AIzaLEGACY", "os_system": "linux", "llm_provider": "ollama",
         "llm_url": "http://localhost:11434", "llm_model": "qwen2.5"}))
    cfg = C.load(force=True)
    assert cfg["providers"]["gemini"]["enabled"] is True
    assert cfg["providers"]["ollama"]["enabled"] and \
        cfg["providers"]["ollama"]["models"][0]["id"] == "qwen2.5"
    assert C.get_api_key("gemini") == "AIzaLEGACY"          # legacy key still honoured
    legacy = json.loads((cfgdir / "api_keys.json").read_text())
    assert legacy["gemini_api_key"] == "AIzaLEGACY"          # legacy file untouched
    assert "AIzaLEGACY" not in (cfgdir / "providers.json").read_text()   # no secret in config


def test_migrates_openai_compatible_local(cfgdir):
    (cfgdir / "api_keys.json").write_text(json.dumps(
        {"llm_provider": "lmstudio", "llm_url": "http://localhost:1234", "llm_model": "q"}))
    cfg = C.load(force=True)
    assert cfg["providers"]["lmstudio"]["base_url"] == "http://localhost:1234/v1"
    assert cfg["providers"]["lmstudio"]["enabled"]


def test_fresh_install_everything_disabled(cfgdir):
    cfg = C.load(force=True)
    assert not any(p["enabled"] for p in cfg["providers"].values())


def test_env_beats_file_and_is_never_written(cfgdir, monkeypatch):
    C.set_api_key("groq", "gsk_fromfile123456")
    monkeypatch.setenv("GROQ_API_KEY", "gsk_fromenv123456")
    assert C.get_api_key("groq") == "gsk_fromenv123456"
    assert C.key_source("groq") == "env:EPHRAIM_GROQ_API_KEY" or C.key_source("groq").startswith("env:")


def test_secret_file_and_not_in_providers(cfgdir):
    where = C.set_api_key("openai", "sk-abcdefghijklmnopqrstuv")
    assert where == "secrets.json"
    C.load(force=True)
    C.update_provider("openai", enabled=True)
    assert "sk-abcdefghijklmnopqrstuv" not in (cfgdir / "providers.json").read_text()
    assert C.get_api_key("openai") == "sk-abcdefghijklmnopqrstuv"
    C.delete_api_key("openai")
    assert C.get_api_key("openai") == ""


def test_bad_hand_edit_is_repaired_not_fatal(cfgdir):
    (cfgdir / "providers.json").write_text(json.dumps({
        "providers": {"good": {"kind": "openai_compat", "enabled": True,
                               "models": [{"id": "m"}, {"nope": 1}]},
                      "evil name!": {"kind": "openai_compat"},
                      "weird": {"kind": "telepathy"}}}))
    cfg = C.load(force=True)
    assert list(cfg["providers"]) == ["good"]
    assert len(cfg["providers"]["good"]["models"]) == 1


def test_corrupt_json_falls_back(cfgdir):
    (cfgdir / "providers.json").write_text("{not json")
    assert C.load(force=True)["providers"]
