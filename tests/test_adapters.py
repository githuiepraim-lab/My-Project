import json

import pytest
from core.ai.adapters import make_adapter
from core.ai.adapters.base import classify_http
from core.ai.redact import redact, register_secret
from core.ai.types import AIError, Image, ModelSpec, Request


class FakeResp:
    def __init__(self, status=200, body=None, lines=None, headers=None):
        self.status_code, self._body, self._lines = status, body, lines or []
        self.headers = headers or {}
        self.text = json.dumps(body) if body is not None else ""

    def json(self):
        return self._body

    def iter_lines(self):
        return iter(self._lines)

    def close(self):
        pass


class FakeSession:
    def __init__(self, resp):
        self.resp, self.sent = resp, []

    def post(self, url, json=None, headers=None, timeout=None, stream=False):
        self.sent.append({"url": url, "json": json, "headers": headers, "timeout": timeout})
        return self.resp

    def close(self):
        pass


def adapter(kind, cfg, resp, key="sk-test-123456789012345"):
    a = make_adapter("x", {"kind": kind, **cfg}, key)
    a._session = FakeSession(resp)
    return a


SPEC = ModelSpec("x", "m", 1, frozenset({"text", "vision", "tools", "json", "stream"}))
TOOL = {"name": "get_weather", "description": "weather", "parameters":
        {"type": "object", "properties": {"city": {"type": "string"}}}}


def req(**kw):
    return Request(messages=[{"role": "user", "content": "hi"}], **kw)


# ── OpenAI-compatible ──────────────────────────────────────────────────────
def test_openai_request_shape_and_parse():
    body = {"choices": [{"message": {"content": "yo", "tool_calls": [
        {"id": "c1", "function": {"name": "get_weather", "arguments": '{"city":"Nairobi"}'}}]},
        "finish_reason": "tool_calls"}],
        "usage": {"prompt_tokens": 12, "completion_tokens": 3,
                  "prompt_tokens_details": {"cached_tokens": 8}}}
    a = adapter("openai_compat", {"base_url": "https://api.openai.com/v1",
                                  "max_tokens_param": "max_completion_tokens"}, FakeResp(body=body))
    r = a.complete(req(system="be brief", tools=[TOOL], json_mode=True, max_output_tokens=50,
                       images=[Image(b"abc", "image/png")]), SPEC)
    sent = a._session.sent[0]
    assert sent["url"].endswith("/chat/completions")
    assert sent["headers"]["Authorization"].startswith("Bearer ")
    p = sent["json"]
    assert p["max_completion_tokens"] == 50 and "max_tokens" not in p
    assert p["messages"][0] == {"role": "system", "content": "be brief"}
    assert p["messages"][1]["content"][1]["type"] == "image_url"
    assert p["tools"][0]["function"]["name"] == "get_weather"
    assert p["response_format"] == {"type": "json_object"}
    assert r.tool_calls == [{"id": "c1", "name": "get_weather", "arguments": {"city": "Nairobi"}}]
    assert (r.usage.input, r.usage.output, r.usage.cached) == (12, 3, 8)


def test_openai_image_not_sent_to_non_vision_model():
    a = adapter("openai_compat", {}, FakeResp(body={"choices": [{"message": {"content": "x"}}]}))
    spec = ModelSpec("x", "m", 0, frozenset({"text"}))
    a.complete(req(images=[Image(b"abc")]), spec)
    assert isinstance(a._session.sent[0]["json"]["messages"][0]["content"], str)


def test_openai_stream_sentences_tool_fragments_and_usage():
    ev = lambda d: "data: " + json.dumps(d)
    lines = [ev({"choices": [{"delta": {"content": "Hel"}}]}),
             ev({"choices": [{"delta": {"content": "lo."}}]}),
             ev({"choices": [{"delta": {"tool_calls": [{"index": 0, "id": "c", "function":
                 {"name": "get_weather", "arguments": '{"ci'}}]}}]}),
             ev({"choices": [{"delta": {"tool_calls": [{"index": 0, "function":
                 {"arguments": 'ty":"X"}'}}]}, "finish_reason": "tool_calls"}]}),
             ev({"choices": [], "usage": {"prompt_tokens": 5, "completion_tokens": 2}}),
             "data: [DONE]"]
    a = adapter("openai_compat", {"stream_usage": True}, FakeResp(lines=lines))
    chunks = list(a.stream(req(tools=[TOOL]), SPEC))
    assert "".join(c.text for c in chunks) == "Hello."
    last = chunks[-1]
    assert last.done and last.tool_calls[0]["arguments"] == {"city": "X"}
    assert last.usage.input == 5
    assert a._session.sent[0]["json"]["stream_options"] == {"include_usage": True}


def test_openai_stream_cancel():
    from core.ai.types import CancelToken
    tok = CancelToken()
    lines = ["data: " + json.dumps({"choices": [{"delta": {"content": "a"}}]})] * 3
    a = adapter("openai_compat", {}, FakeResp(lines=lines))
    it = a.stream(req(cancel=tok), SPEC)
    next(it)
    tok.cancel()
    with pytest.raises(AIError) as e:
        next(it)
    assert e.value.kind == "cancelled"


def test_malformed_response_is_invalid_response():
    a = adapter("openai_compat", {}, FakeResp(body={"nope": 1}))
    with pytest.raises(AIError) as e:
        a.complete(req(), SPEC)
    assert e.value.kind == "invalid_response"


# ── Anthropic ──────────────────────────────────────────────────────────────
def test_anthropic_request_shape_and_parse():
    body = {"content": [{"type": "text", "text": "ok"},
                        {"type": "tool_use", "id": "t1", "name": "get_weather",
                         "input": {"city": "Nairobi"}}],
            "usage": {"input_tokens": 10, "output_tokens": 4, "cache_read_input_tokens": 90},
            "stop_reason": "tool_use"}
    a = adapter("anthropic", {}, FakeResp(body=body))
    r = a.complete(req(system="s" * 3500, tools=[TOOL], images=[Image(b"abc", "image/jpeg")]), SPEC)
    sent = a._session.sent[0]
    assert sent["url"] == "https://api.anthropic.com/v1/messages"
    assert sent["headers"]["x-api-key"] and sent["headers"]["anthropic-version"]
    p = sent["json"]
    assert p["max_tokens"] == 800                               # required by the API
    assert p["system"][0]["cache_control"] == {"type": "ephemeral"}   # long stable prompt cached
    assert p["tools"][0]["input_schema"]["type"] == "object"
    assert p["messages"][0]["content"][0]["type"] == "image"
    assert r.tool_calls[0]["arguments"] == {"city": "Nairobi"}
    assert r.usage.cached == 90 and r.usage.input == 100


def test_anthropic_merges_consecutive_roles_and_tool_results():
    a = adapter("anthropic", {}, FakeResp(body={"content": [{"type": "text", "text": "k"}]}))
    r = Request(messages=[
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "t1", "name": "get_weather",
                                                              "arguments": {"city": "X"}}]},
        {"role": "tool", "tool_call_id": "t1", "content": "sunny"},
        {"role": "user", "content": "thanks"}])
    a.complete(r, SPEC)
    msgs = a._session.sent[0]["json"]["messages"]
    assert [m["role"] for m in msgs] == ["user", "assistant", "user"]
    assert msgs[2]["content"][0]["type"] == "tool_result"


def test_anthropic_stream():
    ev = lambda d: "data: " + json.dumps(d)
    lines = ["event: x", ev({"type": "message_start", "message": {"usage": {"input_tokens": 7}}}),
             ev({"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "Hi"}}),
             ev({"type": "content_block_start", "index": 1, "content_block":
                 {"type": "tool_use", "id": "t", "name": "get_weather"}}),
             ev({"type": "content_block_delta", "index": 1, "delta":
                 {"type": "input_json_delta", "partial_json": '{"city":"Y"}'}}),
             ev({"type": "message_delta", "delta": {"stop_reason": "tool_use"},
                 "usage": {"output_tokens": 9}})]
    a = adapter("anthropic", {}, FakeResp(lines=lines))
    chunks = list(a.stream(req(tools=[TOOL]), SPEC))
    assert chunks[0].text == "Hi"
    assert chunks[-1].tool_calls[0]["arguments"] == {"city": "Y"}
    assert (chunks[-1].usage.input, chunks[-1].usage.output) == (7, 9)


# ── Ollama ─────────────────────────────────────────────────────────────────
def test_ollama_shape():
    body = {"message": {"content": "hey", "tool_calls": [
        {"function": {"name": "get_weather", "arguments": {"city": "Z"}}}]},
        "prompt_eval_count": 6, "eval_count": 2, "done_reason": "stop"}
    a = adapter("ollama", {"base_url": "http://localhost:11434"}, FakeResp(body=body), key="")
    r = a.complete(req(system="s", images=[Image(b"abc")]), SPEC)
    p = a._session.sent[0]["json"]
    assert a._session.sent[0]["url"] == "http://localhost:11434/api/chat"
    assert p["keep_alive"] and p["options"]["num_predict"] == 800
    assert "images" in p["messages"][1]
    assert r.tool_calls[0]["arguments"] == {"city": "Z"} and r.usage.input == 6


# ── Gemini (via a fake core.gemini; the SDK is not needed) ─────────────────
def test_gemini_adapter_asks_for_exactly_the_routed_model(monkeypatch):
    pytest.importorskip("google.genai")


def test_gemini_schema_upper():
    from core.ai.adapters.gemini import _upper_types
    out = _upper_types({"type": "object", "properties": {"a": {"type": "string"}}})
    assert out["type"] == "OBJECT" and out["properties"]["a"]["type"] == "STRING"


# ── errors and secrets ─────────────────────────────────────────────────────
@pytest.mark.parametrize("status,kind", [(429, "rate_limit"), (401, "auth"), (403, "auth"),
                                         (404, "not_found"), (503, "unavailable"),
                                         (504, "timeout"), (400, "bad_request")])
def test_http_classification(status, kind):
    assert classify_http(status, "x").kind == kind


def test_retry_after_parsed():
    assert classify_http(429, "", "12").retry_after == 12.0


def test_http_error_body_never_leaks_key():
    key = "sk-live-ABCDEFGHIJKLMNOPQRSTUVWX"
    register_secret(key)
    e = classify_http(401, f"bad key {key}")
    assert key not in str(e)


def test_redact_patterns():
    for s in ["sk-ant-api03-abcdefghijklmnop", "gsk_abcdefghijklmnopqrstu",
              "AIzaSyA1234567890abcdefghijklmn", "Authorization: Bearer abcdefghijklmnopqrs",
              "https://x.test/v1?key=SECRETVALUE123&a=1"]:
        out = redact(f"error with {s} in it")
        assert "[REDACTED]" in out
    assert "SECRETVALUE123" not in redact("https://x.test/v1?key=SECRETVALUE123&a=1")
