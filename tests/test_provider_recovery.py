"""Adding an API key must actually make the AI work: the three real-world failures
(provider left switched off, locked out by an earlier wrong key, a badly pasted key)
and the doctor that explains the rest. Real hub + real adapters + local servers."""
import json
import os
import types

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from conftest import cfg_for, model, prov
from core.ai import config as C
from core.ai.health import Health
from core.ai.hub import Hub
from core.ai.provider_doctor import check, check_all
from core.ai.types import AIError, Request
from core.ai.usage import UsageTracker
from test_e2e_http import AnthropicLike, KEY, OllamaLike, OpenAILike, serve


@pytest.fixture
def env(tmp_path, monkeypatch):
    for n, v in (("CONFIG_DIR", tmp_path), ("PROVIDERS_FILE", tmp_path / "providers.json"), ("SECRETS_FILE", tmp_path / "secrets.json"),
                 ("LEGACY_FILE", tmp_path / "api_keys.json"), ("_cache", None)):
        monkeypatch.setattr(C, n, v)
    monkeypatch.setattr(C, "_keyring", lambda: None)
    for n in ("GROQ", "OPENAI", "ANTHROPIC", "GEMINI", "OPENROUTER"):
        monkeypatch.delenv(f"{n}_API_KEY", raising=False)
    srv, url = serve(type("S", (OpenAILike,), {}))
    cfg = C.default_config()
    cfg["providers"]["groq"].update(base_url=url, models=[{"id": "fast-mini", "tier": 0, "caps": ["text", "stream"]}])
    C.save(cfg)
    hub = Hub(health=Health(), usage=UsageTracker(tmp_path / "u.jsonl"))
    yield types.SimpleNamespace(hub=hub, url=url, srv=srv, dir=tmp_path)
    srv.shutdown()


def ask(hub):
    return hub.ask(Request(messages=[{"role": "user", "content": "hi"}])).provider


def test_adding_a_key_switches_the_provider_on_without_ticking_anything(env):
    assert "groq" not in {m.provider for m in env.hub.models()}              # absent: no key yet
    C.set_api_key("groq", KEY)
    assert ask(env.hub) == "groq"                                             # works at once, no restart
    assert C.load(force=True)["providers"]["groq"]["enabled"] is True


def test_a_key_in_the_environment_is_enough_too(env, monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", KEY)
    assert ask(env.hub) == "groq"


def test_a_provider_you_switched_off_stays_off_even_with_a_key(env):
    C.set_api_key("groq", KEY)
    C.update_provider("groq", enabled=False, user_disabled=True)
    assert "groq" not in {m.provider for m in env.hub.models()}
    assert "switched off" in env.hub.why_unavailable("groq")


def test_a_wrong_key_does_not_lock_the_provider_out_after_you_fix_it(env):
    C.set_api_key("groq", "sk-WRONG-0000000000000000")
    with pytest.raises(AIError) as e:
        ask(env.hub)
    assert e.value.kind == "auth"
    C.set_api_key("groq", KEY)                                                # fixed, no restart, no waiting
    assert ask(env.hub) == "groq"


def test_changing_the_model_list_clears_a_remembered_404(env):
    C.set_api_key("groq", KEY)
    C.update_provider("groq", models=[{"id": "no-such-model", "tier": 0, "caps": ["text", "stream"]}])
    with pytest.raises(AIError):
        ask(env.hub)
    C.update_provider("groq", models=[{"id": "fast-mini", "tier": 0, "caps": ["text", "stream"]}])
    assert ask(env.hub) == "groq"


@pytest.mark.parametrize("pasted", [f"  {KEY}\n", f'"{KEY}"', f"'{KEY}'", f"Bearer {KEY}", f"{KEY[:8]} {KEY[8:]}\r\n"])
def test_badly_pasted_keys_are_cleaned(env, pasted):
    C.set_api_key("groq", pasted)
    assert C.get_api_key("groq") == KEY and ask(env.hub) == "groq"


def test_a_gemini_key_added_after_setup_is_noticed(env):
    assert "gemini" not in {m.provider for m in env.hub.models()}
    (env.dir / "api_keys.json").write_text(json.dumps({"gemini_api_key": 'AIza' + 'x' * 35 + "\n"}))
    assert "gemini" in {m.provider for m in env.hub.models()}


def test_the_reason_is_spelled_out_not_a_bare_not_found(env):
    assert "no API key" in env.hub.why_unavailable("groq") and "not in your provider list" in env.hub.why_unavailable("nope")
    res = env.hub.ask_many("hi", ["groq"])[0]
    assert not res.ok and "no API key" in res.error
    C.set_api_key("groq", "sk-WRONG-0000000000000000")
    with pytest.raises(AIError):
        ask(env.hub)
    assert "resting after errors" in env.hub.why_unavailable("groq") or "available" in env.hub.why_unavailable("groq")


# ── the doctor ─────────────────────────────────────────────────────────────
def test_doctor_each_failure_gets_its_own_plain_explanation(env):
    assert check(env.hub, "groq")[0] == "no_key" and check(env.hub, "zzz")[0] == "no_provider"
    C.set_api_key("groq", "sk-WRONG-0000000000000000")
    st, msg = check(env.hub, "groq")
    assert st == "auth" and "rejected the key" in msg and "Re-copy the WHOLE key" in msg
    C.set_api_key("groq", KEY)
    st, msg = check(env.hub, "groq")
    assert st == "ok" and "works" in msg and "fast-mini" in msg
    C.update_provider("groq", base_url="http://127.0.0.1:9")
    assert check(env.hub, "groq")[0] == "network"
    C.update_provider("groq", base_url=env.url, models=[{"id": "ratelimited", "tier": 0, "caps": ["text"]}])
    st, msg = check(env.hub, "groq", auto_fix=False)
    assert st == "rate_limit" and "quota" in msg


def test_doctor_repairs_a_wrong_model_name_by_itself(env):
    C.set_api_key("groq", KEY)
    C.update_provider("groq", models=[{"id": "gpt-imaginary-9", "tier": 1, "caps": ["text", "stream"]}])
    st, msg = check(env.hub, "groq")
    assert st == "fixed" and "works now" in msg and "gpt-imaginary-9" in msg
    assert ask(env.hub) == "groq"


def test_doctor_switches_on_a_provider_that_passes(env):
    C.set_api_key("groq", KEY)
    C.update_provider("groq", enabled=False, user_disabled=False)
    assert check(env.hub, "groq")[0] == "ok" and C.load(force=True)["providers"]["groq"]["enabled"]


def test_check_all_reports_every_provider(env):
    C.set_api_key("groq", KEY)
    rows = {n: st for n, st, _ in check_all(env.hub)}
    assert rows["groq"] == "ok" and rows["openai"] == "no_key" and rows["anthropic"] == "no_key"


def test_cli_key_command_cleans_enables_and_tests(env, monkeypatch, capsys):
    import getpass
    from core.ai.__main__ import main
    monkeypatch.setattr(getpass, "getpass", lambda prompt="": f' "{KEY}"\n')
    monkeypatch.setattr("core.ai.__main__.get_hub", lambda: env.hub)
    assert main(["key", "groq"]) == 0
    out = capsys.readouterr().out
    assert "switched on" in out and "works" in out


def test_dialog_set_key_ticks_on_and_tests(env, monkeypatch):
    pytest.importorskip("PyQt6")
    from PyQt6.QtWidgets import QApplication, QInputDialog
    app = QApplication.instance() or QApplication([])
    from core.ai.settings_dialog import AISettingsDialog
    dlg = AISettingsDialog(hub=env.hub)
    names = [dlg.table.item(r, 1).text() for r in range(dlg.table.rowCount())]
    r = names.index("groq")
    assert not dlg.table.cellWidget(r, 0).isChecked()
    dlg.table.selectRow(r)
    monkeypatch.setattr(QInputDialog, "getText", staticmethod(lambda *a, **k: (f'"{KEY}"\n', True)))
    dlg._set_key()
    assert dlg.table.cellWidget(r, 0).isChecked() and "✓" in dlg.table.item(r, 3).text()
    import time
    for _ in range(100):
        app.processEvents(); time.sleep(0.03)
        if "works" in dlg.status.text():
            break
    assert "works" in dlg.status.text()
    dlg.table.cellWidget(r, 0).setChecked(False); dlg._save()                     # switching off is remembered
    assert C.load(force=True)["providers"]["groq"]["user_disabled"] is True


def test_multi_ai_tells_you_why_nobody_answered(env, monkeypatch):
    import actions.multi_ai as m
    monkeypatch.setattr(m, "get_hub", lambda: env.hub)
    out = m.multi_ai({"mode": "ask", "prompt": "hi", "models": "groq, openai"})
    assert "None of the AIs could answer" in out and "no API key" in out
