import json
import socket
import types

import pytest
from core.ai.doctor import clean_key, diagnose

GOOD = "AIza" + "x" * 35


class R:
    def __init__(self, code, text=""):
        self.status_code, self.text = code, text


def run(code=200, text="", dns_fail=False, exc=None, key=GOOD):
    def dns(h, p):
        if dns_fail:
            raise socket.gaierror("no")
    def get(url, headers, timeout):
        if exc:
            raise exc
        return R(code, text)
    return diagnose(key, dns=dns, http_get=get)[0]


@pytest.mark.parametrize("raw", [GOOD, f" {GOOD}\n", f'"{GOOD}"', f"'{GOOD}'", f"key={GOOD}",
                                 f"{GOOD}\r\n", f"{GOOD[:10]} {GOOD[10:]}"])
def test_pasted_key_is_cleaned(raw):
    assert clean_key(raw) == GOOD


def test_each_problem_gets_its_own_diagnosis():
    import requests
    assert run(key="") == "no_key"
    assert run(key="hello world") == "bad_format"
    assert run(dns_fail=True) == "no_dns"
    assert run(code=200) == "ok"
    assert run(code=400, text="API key not valid") == "bad_key"
    assert run(code=403) == "bad_key"
    assert run(code=429) == "quota"
    assert run(code=400, text="User location is not supported for the API use") == "region"
    assert run(exc=requests.exceptions.ConnectionError("x")) == "no_network"
    assert run(exc=requests.exceptions.SSLError("x")) == "ssl"
    assert run(code=500) == "other"


def test_key_goes_in_a_header_not_the_url():
    seen = {}
    def get(url, headers, timeout):
        seen.update(url=url, headers=headers)
        return R(200)
    diagnose(GOOD, dns=lambda h, p: None, http_get=get)
    assert GOOD not in seen["url"] and seen["headers"]["x-goog-api-key"] == GOOD


def test_gemini_key_is_cleaned_and_empty_is_not_cached(tmp_path, monkeypatch):
    from core import gemini
    f = tmp_path / "api_keys.json"
    monkeypatch.setattr(gemini, "_KEY_FILE", f)
    monkeypatch.setattr(gemini, "_cached_key", None)
    assert gemini.api_key() == ""                              # no file yet
    f.write_text(json.dumps({"gemini_api_key": f' "{GOOD}"\n'}))
    assert gemini.api_key() == GOOD                            # picked up WITHOUT a restart


def test_main_key_getter_cleans_and_explains(tmp_path):
    """Runs the REAL _get_api_key from main.py (main itself needs audio hardware libs)."""
    import ast
    from pathlib import Path
    src = Path("main.py").read_text(encoding="utf-8")
    fn = next(n for n in ast.parse(src).body if isinstance(n, ast.FunctionDef) and n.name == "_get_api_key")
    f = tmp_path / "k.json"
    ns = {"json": json, "API_CONFIG_PATH": f}
    exec(compile(ast.Module([fn], []), "main.py", "exec"), ns)
    with pytest.raises(RuntimeError):
        ns["_get_api_key"]()
    f.write_text(json.dumps({"gemini_api_key": f"{GOOD}\n"}))
    assert ns["_get_api_key"]() == GOOD


def test_setup_save_keeps_the_other_settings(tmp_path, monkeypatch):
    ui = pytest.importorskip("ui")
    f = tmp_path / "api_keys.json"
    f.write_text(json.dumps({"gemini_api_key": "old", "camera_index": 2, "avatar_style": "holo",
                             "llm_model": "qwen"}))
    monkeypatch.setattr(ui, "API_FILE", f)
    monkeypatch.setattr(ui, "CONFIG_DIR", tmp_path)
    monkeypatch.setattr(ui, "_read_full_config", lambda: {})
    fake = types.SimpleNamespace(_ready=False, _overlay=None, _apply_state=lambda s: None,
                                 _log=types.SimpleNamespace(append_log=lambda m: None),
                                 _assistant_name="")
    cls = next(v for v in vars(ui).values() if isinstance(v, type) and hasattr(v, "_on_setup_done"))
    cls._on_setup_done(fake, f' "{GOOD}"\n', "windows")
    data = json.loads(f.read_text())
    assert data["gemini_api_key"] == GOOD and data["os_system"] == "windows"
    assert data["camera_index"] == 2 and data["avatar_style"] == "holo" and data["llm_model"] == "qwen"
