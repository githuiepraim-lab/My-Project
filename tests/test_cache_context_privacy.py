import pytest
from conftest import cfg_for, model, prov
from core.ai import context as ctx
from core.ai.types import Request, Response, Usage


def one():
    return cfg_for(prov("a", [model("m", 0)]))


def test_cache_hit_skips_provider_and_invalidates(make_hub):
    hub = make_hub(one())
    hub.ask("translate this", task="translate")
    r2 = hub.ask("translate this", task="translate")
    assert r2.cached and hub._adapter("a").calls == 1
    hub.invalidate_cache()
    assert not hub.ask("translate this", task="translate").cached
    assert hub._adapter("a").calls == 2


def test_chat_is_not_cached(make_hub):
    hub = make_hub(one())
    hub.ask("hello")
    hub.ask("hello")
    assert hub._adapter("a").calls == 2


def test_private_context_withheld_from_untrusted_provider(make_hub):
    hub = make_hub(cfg_for(prov("cloud", [model("m", 0)], share_memory=False)))
    hub.ask(Request(messages=[{"role": "user", "content": "hi"}],
                    private_context="SECRET MEMORY"))
    _id, seen = hub._adapter("cloud").seen[0]
    assert "SECRET MEMORY" not in seen.system


def test_private_context_sent_to_authorised_provider(make_hub):
    hub = make_hub(cfg_for(prov("home", [model("m", 0)], share_memory=True)))
    hub.ask(Request(messages=[{"role": "user", "content": "hi"}],
                    private_context="MY MEMORY"))
    assert "MY MEMORY" in hub._adapter("home").seen[0][1].system


def test_history_is_trimmed_and_summarised():
    msgs = [{"role": "user" if i % 2 == 0 else "assistant",
             "content": f"Message {i}. We decided the deadline is June {i}. " + "filler " * 80}
            for i in range(30)]
    out = ctx.fit_history(msgs, 600)
    assert sum(ctx.message_tokens(m) for m in out) <= 650
    assert out[-1] == msgs[-1]
    assert "summarised" in out[0]["content"] and "deadline" in out[0]["content"]


def test_summarizer_callback_used():
    msgs = [{"role": "user", "content": "x " * 400} for _ in range(10)]
    out = ctx.fit_history(msgs, 300, summarize=lambda older: "SUMMARY-OK")
    assert "SUMMARY-OK" in out[0]["content"]


def test_duplicate_tool_output_collapsed():
    big = "r" * 500
    msgs = [{"role": "tool", "content": big, "tool_call_id": "1"},
            {"role": "tool", "content": big, "tool_call_id": "2"}]
    assert "same output" in ctx.dedupe_tool_results(msgs)[1]["content"]


def test_tool_result_compaction():
    s = ctx.compact_tool_result('{"a": [' + ",".join(["1"] * 3000) + "]}", 300)
    assert len(s) < 420 and "omitted" in s


def test_select_tools_by_relevance():
    tools = [{"name": f"tool_{i}", "description": "unrelated thing"} for i in range(20)]
    tools.append({"name": "weather_report", "description": "weather forecast for a city"})
    picked = ctx.select_tools(tools, "what's the weather in Nairobi", 5)
    assert any(t["name"] == "weather_report" for t in picked) and len(picked) == 5


def test_output_budget_applied(make_hub):
    hub = make_hub(cfg_for(prov("a", [model("m", 0)]), max_output_tokens=123))
    hub.ask("hi")
    assert hub._adapter("a").seen[0][1].max_output_tokens == 123
