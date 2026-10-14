import json
import socket
import threading
import types
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
from core import integrations as I
from core.integrations import BUILTIN, IntegrationError, Integrations, NeedsApproval, public_host


class Srv(BaseHTTPRequestHandler):
    log = []
    def log_message(self, *a): pass
    def _send(self, code, body, ctype="application/json", extra=None):
        data = body if isinstance(body, bytes) else json.dumps(body).encode()
        self.send_response(code); self.send_header("Content-Type", ctype)
        for k, v in (extra or {}).items(): self.send_header(k, v)
        self.send_header("Content-Length", str(len(data))); self.end_headers(); self.wfile.write(data)
    def do_GET(self):
        type(self).log.append(("GET", self.path, dict(self.headers)))
        if self.path.startswith("/big"): return self._send(200, b"x" * 1_200_000, "text/plain")
        if self.path.startswith("/redir"): return self._send(302, b"", extra={"Location": "http://169.254.169.254/latest/meta-data"})
        if self.path.startswith("/fail"): return self._send(500, {"error": "boom sk-live-ABCDEFGHIJKLMNOPQRSTUVWX"})
        if self.path.startswith("/text"): return self._send(200, b"plain words", "text/plain")
        if self.path.startswith("/long"): return self._send(200, {"items": list(range(5000))})
        self._send(200, {"result": {"items": [1, 2, 3]}, "echo": self.path})
    def do_POST(self):
        n = int(self.headers.get("Content-Length") or 0)
        type(self).log.append(("POST", self.path, json.loads(self.rfile.read(n) or b"{}")))
        self._send(200, {"ok": True})


@pytest.fixture
def srv():
    Srv.log = []
    s = ThreadingHTTPServer(("127.0.0.1", 0), Srv)
    threading.Thread(target=s.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{s.server_address[1]}"
    s.shutdown()


def reg(tmp_path, srv, key=lambda n: "SECRETKEY123456", **spec):
    r = Integrations(tmp_path / "i.json", key_getter=key)
    base = {"description": "t", "base_url": srv, "local": True, "auth": {"type": "none"}, "actions": {
        "get": {"method": "GET", "path": "/items/{id}", "query": {"q": "{q|default}"}, "pick": "result.items"},
        "send": {"method": "POST", "path": "/post", "body": {"text": "{text}"}}}}
    base.update(spec)
    r.save("svc", base)
    return r


def test_builtins_are_well_formed_and_cover_the_free_apis():
    assert set(BUILTIN) >= {"weather", "wikipedia", "fx", "crypto", "dictionary", "hackernews", "nasa", "countries", "books", "geocode"}
    for n, spec in BUILTIN.items():
        Integrations.validate(spec)
        assert spec["auth"]["type"] == "none" and all(a["method"] == "GET" for a in spec["actions"].values()), n


def test_every_builtin_builds_a_correct_public_request(tmp_path, monkeypatch):
    monkeypatch.setattr(I, "public_host", lambda h: True)
    calls = []
    class H:
        def request(self, method, url, params=None, headers=None, json=None, timeout=None, stream=None, allow_redirects=None):
            calls.append((method, url, params))
            return types.SimpleNamespace(status_code=200, headers={}, encoding="utf-8", iter_content=lambda n: iter([b'{"ok":1}']))
    r = Integrations(tmp_path / "x.json", http=H())
    r.call("weather", "forecast", {"latitude": "-1.29", "longitude": "36.82"})
    r.call("wikipedia", "summary", {"title": "Python (programming language)"})
    r.call("fx", "latest", {"base": "USD", "symbols": "KES,EUR"})
    r.call("fx", "history", {"date": "2025-01-02"})
    r.call("dictionary", "define", {"word": "naïve"})
    r.call("hackernews", "item", {"id": "8863"})
    r.call("countries", "info", {"name": "kenya"})
    r.call("geocode", "search", {"name": "Nairobi"})
    assert calls[0][1] == "https://api.open-meteo.com/v1/forecast" and calls[0][2]["latitude"] == "-1.29" and calls[0][2]["current"]
    assert calls[1][1] == "https://en.wikipedia.org/api/rest_v1/page/summary/Python%20%28programming%20language%29"
    assert calls[2][2] == {"base": "USD", "symbols": "KES,EUR"} and calls[3][1].endswith("/v1/2025-01-02")
    assert calls[4][1].endswith("/en/na%C3%AFve") and calls[5][1].endswith("/v0/item/8863.json")
    assert calls[7][2] == {"name": "Nairobi", "count": "3"}                             # default filled in
    with pytest.raises(IntegrationError, match="missing value 'word'"):
        r.call("dictionary", "define", {})


def test_get_with_defaults_encoding_and_pick(tmp_path, srv):
    out = reg(tmp_path, srv).call("svc", "get", {"id": "a b/c", "q": "x&y"})
    assert out == "[1,2,3]"
    method, path, _ = Srv.log[0]
    assert path.startswith("/items/a%20b%2Fc") and "q=x%26y" in path
    reg(tmp_path, srv).call("svc", "get", {"id": "1"})
    assert "q=default" in Srv.log[1][1]


@pytest.mark.parametrize("auth,check", [
    ({"type": "bearer"}, lambda log: log[0][2]["Authorization"] == "Bearer SECRETKEY123456"),
    ({"type": "header", "name": "X-Api-Key"}, lambda log: log[0][2]["X-Api-Key"] == "SECRETKEY123456"),
    ({"type": "query", "name": "key"}, lambda log: "key=SECRETKEY123456" in log[0][1])])
def test_auth_styles(tmp_path, srv, auth, check):
    reg(tmp_path, srv, auth=auth).call("svc", "get", {"id": "1"})
    assert check(Srv.log)


def test_missing_key_is_a_clear_instruction_not_a_crash(tmp_path, srv):
    with pytest.raises(IntegrationError, match="python -m core.integrations key svc"):
        reg(tmp_path, srv, key=lambda n: "", auth={"type": "bearer"}).call("svc", "get", {"id": "1"})


def test_posting_needs_approval_then_sends_json(tmp_path, srv):
    r = reg(tmp_path, srv)
    with pytest.raises(NeedsApproval):
        r.call("svc", "send", {"text": "hi"})
    assert Srv.log == []                                                                # nothing left the machine
    assert r.is_mutating("svc", "send") and not r.is_mutating("svc", "get")
    r.call("svc", "send", {"text": "hello Ephraim"}, approved=True)
    assert Srv.log[0][0] == "POST" and Srv.log[0][2] == {"text": "hello Ephraim"}


def test_webhook_url_is_a_secret_not_in_the_config_file(tmp_path, srv):
    r = Integrations(tmp_path / "i.json", key_getter=lambda n: srv + "/hook" if n == "int_discord_url" else "")
    with pytest.raises(IntegrationError):
        r.call("slack", "post", {"text": "x"}, approved=True)                           # no URL stored for slack
    with pytest.raises(IntegrationError, match="private/local"):                         # webhook to localhost: blocked
        r.call("discord", "post", {"text": "x"}, approved=True)


@pytest.mark.parametrize("ip,public", [("8.8.8.8", True), ("1.1.1.1", True), ("127.0.0.1", False), ("10.0.0.5", False),
                                       ("192.168.1.1", False), ("172.16.9.9", False), ("169.254.169.254", False),
                                       ("0.0.0.0", False), ("::1", False), ("fe80::1", False), ("fd00::1", False)])
def test_ssrf_guard_classifies_addresses(monkeypatch, ip, public):
    monkeypatch.setattr(socket, "getaddrinfo", lambda h, p: [(2, 1, 6, "", (ip, 0))])
    assert public_host("example.test") is public


def test_a_name_that_resolves_to_both_public_and_private_is_blocked(monkeypatch):
    monkeypatch.setattr(socket, "getaddrinfo", lambda h, p: [(2, 1, 6, "", ("8.8.8.8", 0)), (2, 1, 6, "", ("10.0.0.9", 0))])
    assert public_host("rebind.test") is False


def test_private_targets_are_refused_unless_marked_local(tmp_path, srv):
    spec = {"description": "t", "base_url": srv, "auth": {"type": "none"}, "actions": {"g": {"method": "GET", "path": "/x"}}}
    r = Integrations(tmp_path / "i.json")
    r.save("nolocal", spec)
    with pytest.raises(IntegrationError, match="private/local"):
        r.call("nolocal", "g")
    assert Srv.log == []


def test_redirect_into_the_private_network_is_stopped(tmp_path, srv):
    r = reg(tmp_path, srv, actions={"r": {"method": "GET", "path": "/redir"}})
    with pytest.raises(IntegrationError, match="private/local"):
        r.call("svc", "r")                                                              # 302 -> cloud metadata address


def test_size_error_and_truncation_limits(tmp_path, srv):
    r = reg(tmp_path, srv, actions={"big": {"method": "GET", "path": "/big"}, "fail": {"method": "GET", "path": "/fail"},
                                    "long": {"method": "GET", "path": "/long"}, "text": {"method": "GET", "path": "/text"}})
    with pytest.raises(IntegrationError, match="larger than 1 MB"): r.call("svc", "big")
    with pytest.raises(IntegrationError) as e: r.call("svc", "fail")
    assert "HTTP 500" in str(e.value) and "ABCDEFGHIJKLMNOP" not in str(e.value)       # a key in an error body is redacted
    out = r.call("svc", "long")
    assert len(out) < 3700 and "more chars" in out
    assert r.call("svc", "text") == "plain words"


def test_unreachable_host_is_a_clean_error(tmp_path):
    r = Integrations(tmp_path / "i.json")
    r.save("dead", {"description": "t", "base_url": "http://127.0.0.1:9", "local": True, "auth": {"type": "none"},
                    "actions": {"g": {"method": "GET", "path": "/"}}})
    with pytest.raises(IntegrationError, match="could not reach"): r.call("dead", "g")


@pytest.mark.parametrize("spec", [{}, {"actions": {}}, {"base_url": "ftp://x", "actions": {"a": {}}},
                                  {"base_url": "https://x.io", "actions": {"a": {"method": "TRACE"}}},
                                  {"base_url": "https://x.io", "auth": {"type": "magic"}, "actions": {"a": {}}}])
def test_bad_definitions_are_rejected(tmp_path, spec):
    with pytest.raises(IntegrationError): Integrations(tmp_path / "i.json").save("x", spec)


def test_user_integrations_persist_override_and_remove(tmp_path):
    r = Integrations(tmp_path / "i.json")
    r.save("mine", {"description": "mine", "base_url": "https://api.example.com", "actions": {"ping": {"method": "GET", "path": "/ping", "doc": "d"}}})
    assert "mine" in Integrations(tmp_path / "i.json").names() and "ping (GET)" in r.describe("mine")
    with pytest.raises(IntegrationError): r.save("Bad Name!", {"base_url": "https://x.io", "actions": {"a": {}}})
    assert r.remove("mine") and not r.remove("weather") and "weather" in r.names()      # built-ins cannot be removed


def test_no_secret_ever_lands_in_the_config_file(tmp_path, srv):
    r = reg(tmp_path, srv, auth={"type": "bearer"})
    r.call("svc", "get", {"id": "1"})
    assert "SECRETKEY123456" not in (tmp_path / "i.json").read_text()


# ── the tool ───────────────────────────────────────────────────────────────
@pytest.fixture
def tool(tmp_path, srv, monkeypatch):
    import actions.integrations as ai
    r = reg(tmp_path, srv)
    monkeypatch.setattr(ai, "Integrations", lambda: r)
    return ai, r


def test_tool_reads_automatically_and_posts_only_via_the_gate(tool):
    ai, r = tool
    assert ai.integrations({"mode": "call", "name": "svc", "action": "get", "params": '{"id": "1"}'}) == "[1,2,3]"
    got = {}
    ai.confirm.request = lambda k, t, d, run: got.update(t=t, d=d, run=run) or "[CONFIRMATION_PENDING]"
    assert ai.integrations({"mode": "call", "name": "svc", "action": "send", "params": '{"text": "hi"}'}) == "[CONFIRMATION_PENDING]"
    assert Srv.log[-1][0] == "GET" and "svc" in got["t"]                                # not sent yet
    got["run"]()
    assert Srv.log[-1][0] == "POST"


def test_tool_add_is_gated_and_list_describe_work(tool):
    ai, r = tool
    got = {}
    ai.confirm.request = lambda k, t, d, run: got.update(run=run) or "pending"
    spec = {"description": "x", "base_url": "https://api.example.com", "actions": {"a": {"method": "GET", "path": "/a"}}}
    assert ai.integrations({"mode": "add", "name": "newone", "spec": json.dumps(spec)}) == "pending" and "newone" not in r.names()
    got["run"](); assert "newone" in r.names()
    assert "weather" in ai.integrations({"mode": "list"}) and "forecast" in ai.integrations({"mode": "describe", "name": "weather"})
    assert "must be a JSON object" in ai.integrations({"mode": "call", "name": "svc", "action": "get", "params": "not json"})
    assert "no integration called" in ai.integrations({"mode": "call", "name": "zzz", "action": "a"})


def test_tool_registers_cleanly():
    from pathlib import Path
    from core.action_loader import discover_actions
    d = {x["name"]: x for x in discover_actions(Path("actions"), logger=lambda s: None).get_tool_declarations()}
    assert set(d["integrations"]) <= {"name", "description", "parameters"} and len(d["integrations"]["description"]) < 600
