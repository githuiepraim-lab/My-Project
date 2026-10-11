"""End-to-end: the REAL hub and REAL adapters talking real HTTP to local servers that
behave like OpenAI-compatible, Anthropic and Ollama APIs (including their error
shapes). Nothing in core.ai is mocked; only the remote service is simulated."""
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
from core.ai import config as C
from core.ai.autofix import autofix_models
from core.ai.health import Health
from core.ai.hub import Hub
from core.ai.types import AIError, Image, Request
from core.ai.usage import UsageTracker

KEY = "sk-test-KEY1234567890abcdef"


def serve(handler_cls):
    srv = ThreadingHTTPServer(("127.0.0.1", 0), handler_cls)
    handler_cls.log = []
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, f"http://127.0.0.1:{srv.server_address[1]}"


class Base(BaseHTTPRequestHandler):
    log = []
    def log_message(self, *a): pass
    def _json(self, code, obj, headers=None):
        body = json.dumps(obj).encode()
        self.send_response(code); self.send_header("Content-Type", "application/json")
        for k, v in (headers or {}).items(): self.send_header(k, v)
        self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)
    def _body(self):
        n = int(self.headers.get("Content-Length") or 0)
        return json.loads(self.rfile.read(n) or b"{}")
    def _sse(self, events):
        self.send_response(200); self.send_header("Content-Type", "text/event-stream"); self.end_headers()
        for e in events:
            self.wfile.write(e.encode()); self.wfile.flush()


class OpenAILike(Base):
    GOOD = ("fast-mini", "smart-large")
    strict_new = False                     # True: behave like newer OpenAI models (max_tokens rejected)
    def _auth(self):
        if self.headers.get("Authorization") != f"Bearer {KEY}":
            self._json(401, {"error": {"message": f"Incorrect API key provided: {self.headers.get('Authorization')}"}}); return False
        return True
    def do_GET(self):
        if not self._auth(): return
        if self.path == "/v1/models":
            return self._json(200, {"data": [{"id": m} for m in (*self.GOOD, "whisper-1", "text-embedding-3")]})
        self._json(404, {"error": {"message": "not found"}})
    def do_POST(self):
        if self.path != "/v1/chat/completions":
            return self._json(404, {"error": {"message": f"Invalid URL ({self.path})"}})
        if not self._auth(): return
        b = self._body(); type(self).log.append(b)
        m = b.get("model")
        if self.strict_new and "max_tokens" in b:
            return self._json(400, {"error": {"message": "Unsupported parameter: 'max_tokens'. Use 'max_completion_tokens'."}})
        if m == "ratelimited":
            return self._json(429, {"error": {"message": "Rate limit reached"}}, {"Retry-After": "7"})
        if m == "slow":
            time.sleep(2.5)
        if m not in (*self.GOOD, "slow"):
            return self._json(404, {"error": {"message": f"The model `{m}` does not exist", "code": "model_not_found"}})
        last = b["messages"][-1]
        content = last["content"] if isinstance(last["content"], str) else json.dumps(last["content"])
        usage = {"prompt_tokens": 21, "completion_tokens": 6, "prompt_tokens_details": {"cached_tokens": 4}}
        if b.get("stream"):
            ev = lambda d: "data: " + json.dumps(d) + "\n\n"
            if b.get("tools") and "weather" in content:
                return self._sse([ev({"choices": [{"delta": {"tool_calls": [{"index": 0, "id": "c1", "function": {"name": "get_weather", "arguments": '{"city":'}}]}}]}),
                                  ev({"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"arguments": '"Nairobi"}'}}]}, "finish_reason": "tool_calls"}]}),
                                  ev({"choices": [], "usage": usage}), "data: [DONE]\n\n"])
            return self._sse([ev({"choices": [{"delta": {"content": "Hel"}}]}), ev({"choices": [{"delta": {"content": "lo!"}, "finish_reason": "stop"}]}),
                              ev({"choices": [], "usage": usage}), "data: [DONE]\n\n"])
        if b.get("tools") and "weather" in content:
            return self._json(200, {"choices": [{"message": {"content": None, "tool_calls": [
                {"id": "c1", "type": "function", "function": {"name": "get_weather", "arguments": '{"city": "Nairobi"}'}}]}, "finish_reason": "tool_calls"}], "usage": usage})
        txt = '{"ok": true}' if b.get("response_format") else f"{m} says hi"
        self._json(200, {"choices": [{"message": {"content": txt}, "finish_reason": "stop"}], "usage": usage})


class AnthropicLike(Base):
    def do_GET(self):
        if self.headers.get("x-api-key") != KEY: return self._json(401, {"type": "error", "error": {"type": "authentication_error", "message": "invalid x-api-key"}})
        if self.path == "/v1/models":
            return self._json(200, {"data": [{"id": "haiku-x"}, {"id": "sonnet-x"}]})
        self._json(404, {})
    def do_POST(self):
        if self.path != "/v1/messages":
            return self._json(404, {"type": "error", "error": {"type": "not_found_error", "message": "Not Found"}})
        if self.headers.get("x-api-key") != KEY: return self._json(401, {"type": "error", "error": {"type": "authentication_error", "message": "invalid x-api-key"}})
        if not self.headers.get("anthropic-version"): return self._json(400, {"type": "error", "error": {"type": "invalid_request_error", "message": "anthropic-version header required"}})
        b = self._body(); type(self).log.append(b)
        if not isinstance(b.get("max_tokens"), int): return self._json(400, {"type": "error", "error": {"type": "invalid_request_error", "message": "max_tokens: Field required"}})
        roles = [m["role"] for m in b["messages"]]
        if any(roles[i] == roles[i + 1] for i in range(len(roles) - 1)) or roles[0] != "user":
            return self._json(400, {"type": "error", "error": {"type": "invalid_request_error", "message": "roles must alternate"}})
        if b["model"] not in ("haiku-x", "sonnet-x"):
            return self._json(404, {"type": "error", "error": {"type": "not_found_error", "message": f"model: {b['model']}"}})
        text = json.dumps(b["messages"][-1]["content"])
        usage = {"input_tokens": 15, "output_tokens": 5, "cache_read_input_tokens": 0}
        if b.get("stream"):
            ev = lambda n, d: f"event: {n}\ndata: " + json.dumps(d) + "\n\n"
            return self._sse([ev("message_start", {"type": "message_start", "message": {"usage": usage}}),
                              ev("content_block_delta", {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "Hi from "}}),
                              ev("content_block_delta", {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "Claude"}}),
                              ev("message_delta", {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 5}}),
                              ev("message_stop", {"type": "message_stop"})])
        if b.get("tools") and "weather" in text:
            return self._json(200, {"content": [{"type": "text", "text": "Checking."}, {"type": "tool_use", "id": "t1", "name": "get_weather", "input": {"city": "Nairobi"}}], "usage": usage, "stop_reason": "tool_use"})
        self._json(200, {"content": [{"type": "text", "text": f"{b['model']} here"}], "usage": usage, "stop_reason": "end_turn"})


class OllamaLike(Base):
    def do_GET(self):
        self._json(200, {"models": [{"name": "llama-local"}]}) if self.path == "/api/tags" else self._json(404, {})
    def do_POST(self):
        if self.path != "/api/chat": return self._json(404, {"error": "not found"})
        b = self._body(); type(self).log.append(b)
        if b["model"] != "llama-local": return self._json(404, {"error": f"model '{b['model']}' not found, try pulling it first"})
        if b.get("stream"):
            self.send_response(200); self.end_headers()
            for part in ({"message": {"content": "Hel"}}, {"message": {"content": "lo"}},
                         {"message": {"content": ""}, "done": True, "done_reason": "stop", "prompt_eval_count": 9, "eval_count": 2}):
                self.wfile.write((json.dumps(part) + "\n").encode()); self.wfile.flush()
            return
        self._json(200, {"message": {"content": "local reply"}, "done": True, "prompt_eval_count": 9, "eval_count": 2})


@pytest.fixture
def servers():
    s = {"oa": serve(type("OA", (OpenAILike,), {})), "an": serve(type("AN", (AnthropicLike,), {})),
         "ol": serve(type("OL", (OllamaLike,), {}))}
    yield s
    for srv, _ in s.values():
        srv.shutdown()


def build(tmp_path, servers, **settings):
    cfg = C.default_config()
    st = {"strategy": "priority", "cache": False, "deadline_s": 20, "request_timeout_s": 5, "max_attempts": 4}
    st.update(settings)
    cfg["settings"].update(st)
    cfg["providers"] = {
        "openai": {"kind": "openai_compat", "enabled": True, "priority": 1, "base_url": servers["oa"][1],
                   "max_tokens_param": "max_tokens", "stream_usage": True,
                   "models": [{"id": "fast-mini", "tier": 0, "caps": ["text", "stream", "tools", "json", "vision"]},
                              {"id": "smart-large", "tier": 2, "caps": ["text", "stream", "tools", "json", "vision"]}]},
        "claude": {"kind": "anthropic", "enabled": True, "priority": 2, "base_url": servers["an"][1],
                   "models": [{"id": "haiku-x", "tier": 0, "caps": ["text", "stream", "tools", "vision"]},
                              {"id": "sonnet-x", "tier": 1, "caps": ["text", "stream", "tools", "vision"]}]},
        "ollama": {"kind": "ollama", "enabled": True, "priority": 3, "base_url": servers["ol"][1], "requires_key": False,
                   "models": [{"id": "llama-local", "tier": 0, "caps": ["text", "stream", "tools"], "local": True}]}}
    return Hub(config_loader=lambda: cfg, key_getter=lambda n: "" if n == "ollama" else KEY,
               health=Health(), usage=UsageTracker(tmp_path / "u.jsonl"), sleep=lambda s: None), cfg


TOOL = {"name": "get_weather", "description": "weather", "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}}


def test_openai_compatible_roundtrip_with_real_headers_and_body(tmp_path, servers):
    hub, _ = build(tmp_path, servers)
    r = hub.ask(Request(messages=[{"role": "user", "content": "hello"}], system="be brief", max_output_tokens=50))
    assert r.provider == "openai" and r.text == "fast-mini says hi"
    assert (r.usage.input, r.usage.output, r.usage.cached) == (21, 6, 4) and r.cost is None
    sent = servers["oa"][0].RequestHandlerClass.log[0]
    assert sent["messages"][0] == {"role": "system", "content": "be brief"} and sent["max_tokens"] == 50


def test_new_openai_models_need_max_completion_tokens(tmp_path, servers):
    servers["oa"][0].RequestHandlerClass.strict_new = True
    hub, cfg = build(tmp_path, servers)
    r = hub.ask(Request(messages=[{"role": "user", "content": "hi"}]))                     # max_tokens rejected -> fails over
    assert r.provider != "openai"
    cfg["providers"]["openai"]["max_tokens_param"] = "max_completion_tokens"              # what the OpenAI default uses
    hub.health.reset()
    assert hub.ask(Request(messages=[{"role": "user", "content": "hi"}])).provider == "openai"


def test_anthropic_roundtrip_headers_system_blocks_and_alternation(tmp_path, servers):
    hub, cfg = build(tmp_path, servers)
    cfg["providers"]["openai"]["enabled"] = False
    r = hub.ask(Request(messages=[{"role": "user", "content": "q1"}, {"role": "user", "content": "q2"}], system="s", max_output_tokens=40))
    assert r.provider == "claude" and r.text == "haiku-x here"
    sent = servers["an"][0].RequestHandlerClass.log[0]
    assert sent["system"][0]["text"] == "s" and sent["max_tokens"] == 40 and [m["role"] for m in sent["messages"]] == ["user"]


def test_ollama_roundtrip_and_stream(tmp_path, servers):
    hub, cfg = build(tmp_path, servers)
    cfg["providers"]["openai"]["enabled"] = cfg["providers"]["claude"]["enabled"] = False
    assert hub.ask(Request(messages=[{"role": "user", "content": "hi"}])).text == "local reply"
    assert "".join(c.text for c in hub.stream(Request(messages=[{"role": "user", "content": "hi"}]))) == "Hello"


@pytest.mark.parametrize("which", ["openai", "claude", "ollama"])
def test_streaming_every_provider_yields_text_and_real_usage(tmp_path, servers, which):
    hub, cfg = build(tmp_path, servers)
    for n in cfg["providers"]:
        cfg["providers"][n]["enabled"] = (n == which)
    chunks = list(hub.stream(Request(messages=[{"role": "user", "content": "hi"}])))
    assert "".join(c.text for c in chunks).replace("Hi from Claude", "Hello").replace("Hello!", "Hello") == "Hello"
    assert chunks[-1].done and chunks[-1].usage.input > 0
    assert hub.usage.summary()


@pytest.mark.parametrize("which", ["openai", "claude"])
def test_tool_calling_roundtrip_nonstream_and_stream(tmp_path, servers, which):
    hub, cfg = build(tmp_path, servers)
    for n in cfg["providers"]:
        cfg["providers"][n]["enabled"] = (n == which)
    req = lambda: Request(messages=[{"role": "user", "content": "what is the weather in Nairobi"}], tools=[TOOL])
    r = hub.ask(req())
    assert r.tool_calls == [{"id": r.tool_calls[0]["id"], "name": "get_weather", "arguments": {"city": "Nairobi"}}]
    if which == "openai":
        last = list(hub.stream(req()))[-1]
        assert last.tool_calls[0]["arguments"] == {"city": "Nairobi"}


def test_image_and_json_mode_reach_the_server_in_the_right_shape(tmp_path, servers):
    hub, cfg = build(tmp_path, servers)
    cfg["providers"]["claude"]["enabled"] = cfg["providers"]["ollama"]["enabled"] = False
    r = hub.ask(Request(messages=[{"role": "user", "content": "what is this"}], images=[Image(b"\x89PNGdata", "image/png")], json_mode=True))
    sent = servers["oa"][0].RequestHandlerClass.log[0]
    assert sent["messages"][-1]["content"][1]["image_url"]["url"].startswith("data:image/png;base64,")
    assert sent["response_format"] == {"type": "json_object"} and json.loads(r.text) == {"ok": True}


def test_404_unknown_model_is_explained_then_autofix_repairs_it(tmp_path, servers, monkeypatch):
    hub, cfg = build(tmp_path, servers)
    cfg["providers"]["claude"]["enabled"] = cfg["providers"]["ollama"]["enabled"] = False
    cfg["providers"]["openai"]["models"] = [{"id": "gpt-imaginary-9", "tier": 1, "caps": ["text", "stream"]}]
    with pytest.raises(AIError) as e:
        hub.ask(Request(messages=[{"role": "user", "content": "hi"}]))
    assert e.value.kind == "not_found" and "model name" in str(e.value) and "gpt-imaginary-9" in str(e.value)
    saved = {}
    import core.ai.autofix as af
    monkeypatch.setattr(af.C, "load", lambda force=False: cfg)
    monkeypatch.setattr(af.C, "save", lambda c: saved.update(c))
    msg = autofix_models(hub, "openai")
    ids = [m["id"] for m in cfg["providers"]["openai"]["models"]]
    assert "using" in msg and "gpt-imaginary-9" in msg and "whisper-1" not in ids and "text-embedding-3" not in ids
    hub.health.reset()
    assert hub.ask(Request(messages=[{"role": "user", "content": "hi"}])).provider == "openai"


def test_base_url_pasted_wrong_still_works(tmp_path, servers):
    hub, cfg = build(tmp_path, servers)
    cfg["providers"]["claude"]["enabled"] = cfg["providers"]["ollama"]["enabled"] = False
    for pasted in ("/v1/chat/completions", "/", "/v1/"):
        cfg["providers"]["openai"]["base_url"] = servers["oa"][1] + pasted
        hub.health.reset()
        assert hub.ask(Request(messages=[{"role": "user", "content": "hi"}])).provider == "openai", pasted


def test_wrong_key_fails_over_and_the_key_never_leaks(tmp_path, servers):
    hub, cfg = build(tmp_path, servers)
    hub._key = lambda n: "sk-WRONG-KEY-0987654321zzzz" if n == "openai" else ("" if n == "ollama" else KEY)
    r = hub.ask(Request(messages=[{"role": "user", "content": "hi"}]))
    assert r.provider == "claude" and r.attempts[0] == {"model": "openai/fast-mini", "error": "auth", "cooldown_s": 21600.0}
    log = (tmp_path / "u.jsonl").read_text()
    assert "WRONG-KEY" not in log and KEY not in log
    st = json.dumps(hub.status())
    assert "WRONG-KEY" not in st and KEY not in st


def test_rate_limit_uses_retry_after_and_switches_provider(tmp_path, servers):
    hub, cfg = build(tmp_path, servers)
    cfg["providers"]["openai"]["models"] = [{"id": "ratelimited", "tier": 0, "caps": ["text", "stream"]}]
    r = hub.ask(Request(messages=[{"role": "user", "content": "hi"}]))
    assert r.provider == "claude" and r.attempts[0]["error"] == "rate_limit" and r.attempts[0]["cooldown_s"] == 7.0
    assert hub.health.remaining("openai/ratelimited") > 0


def test_slow_server_times_out_and_fails_over_within_bounds(tmp_path, servers):
    hub, cfg = build(tmp_path, servers, request_timeout_s=0.6)
    cfg["providers"]["openai"]["models"] = [{"id": "slow", "tier": 0, "caps": ["text", "stream"]}]
    t0 = time.time()
    r = hub.ask(Request(messages=[{"role": "user", "content": "hi"}]))
    assert r.provider == "claude" and time.time() - t0 < 6
    assert [a["error"] for a in r.attempts] == ["timeout", "timeout"]            # one retry, then switched


def test_dead_port_is_a_network_error_and_fails_over(tmp_path, servers):
    hub, cfg = build(tmp_path, servers)
    cfg["providers"]["openai"]["base_url"] = "http://127.0.0.1:9"                 # nothing listens here
    r = hub.ask(Request(messages=[{"role": "user", "content": "hi"}]))
    assert r.provider == "claude" and r.attempts[0]["error"] == "network"


def test_everything_down_is_a_bounded_clear_failure(tmp_path, servers):
    hub, cfg = build(tmp_path, servers, max_attempts=3, request_timeout_s=0.5)
    for n in ("openai", "claude", "ollama"):
        cfg["providers"][n]["base_url"] = "http://127.0.0.1:9"
    t0 = time.time()
    with pytest.raises(AIError):
        hub.ask(Request(messages=[{"role": "user", "content": "hi"}]))
    assert time.time() - t0 < 8


def test_ask_many_and_discuss_over_real_http(tmp_path, servers):
    hub, _ = build(tmp_path, servers)
    res = hub.ask_many("compare tabs and spaces", ["openai", "claude", "ollama"])
    assert [r.ok for r in res] == [True, True, True] and {r.provider for r in res} == {"openai", "claude", "ollama"}
    d = hub.discuss("tabs or spaces", ["openai", "claude"], rounds=1)
    assert len(d.transcript) == 2 and d.summary
    reqs = [r for r in servers["an"][0].RequestHandlerClass.log if "<ai_message" in json.dumps(r["messages"])]
    assert reqs and all("tools" not in r for r in reqs)                              # AI-to-AI turns carry no tools


def test_provider_listing_matches_what_the_server_offers(tmp_path, servers):
    hub, _ = build(tmp_path, servers)
    assert hub._adapter("openai").list_models() == ["fast-mini", "smart-large", "text-embedding-3", "whisper-1"]
    assert hub._adapter("claude").list_models() == ["haiku-x", "sonnet-x"]
    assert hub._adapter("ollama").list_models() == ["llama-local"]


def test_routing_picks_cheapest_adequate_across_real_providers(tmp_path, servers):
    hub, cfg = build(tmp_path, servers, strategy="cost")
    cfg["providers"]["openai"]["models"][0].update(price_in=0.5, price_out=1.5)
    cfg["providers"]["claude"]["models"][0].update(price_in=0.25, price_out=1.25)
    easy = hub.ask(Request(messages=[{"role": "user", "content": "what time is it"}]))
    assert easy.provider == "ollama"                                               # local = free, adequate
    hard = hub.ask(Request(messages=[{"role": "user", "content": "debug and explain why this architecture is slow " * 12}]))
    assert hard.provider == "openai" and hard.model == "smart-large"               # only tier-2 model
