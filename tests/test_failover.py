import pytest
from conftest import cfg_for, model, prov
from core.ai.types import AIError, CancelToken, Request, Response, Usage


def pair(**kw):
    return cfg_for(prov("a", [model("ma", 0, pi=0.1, po=0.1)], priority=1),
                   prov("b", [model("mb", 0, pi=0.2, po=0.2)], priority=2), **kw)


def test_rate_limit_switches_provider_without_retrying_same(make_hub):
    hub = make_hub(pair(), {"a": [AIError("rate_limit", "429", retry_after=30)]})
    r = hub.ask("hi")
    assert r.model == "mb"
    assert hub._adapter("a").calls == 1                # no same-model retry on 429
    assert not hub.health.available("a/ma")            # cooling


def test_timeout_retries_once_then_switches(make_hub):
    hub = make_hub(pair(), {"a": [AIError("timeout", "slow")]})
    r = hub.ask("hi")
    assert r.model == "mb"
    assert hub._adapter("a").calls == 2                # exactly one retry
    assert len(hub._sleeps) == 1


def test_transient_error_recovers_on_retry(make_hub):
    hub = make_hub(pair(), {"a": [AIError("unavailable", "x"), Response(text="ok", usage=Usage(1, 1))]})
    assert hub.ask("hi").model == "ma"


def test_all_failing_is_bounded_not_endless(make_hub):
    boom = [AIError("unavailable", "down")]
    hub = make_hub(pair(max_attempts=3), {"a": boom, "b": boom})
    with pytest.raises(AIError):
        hub.ask("hi")
    total = hub._adapter("a").calls + hub._adapter("b").calls
    assert total <= 3


def test_cooldown_skips_dead_model_on_next_call(make_hub):
    hub = make_hub(pair(), {"a": [AIError("auth", "no")]})
    hub.ask("one")
    hub.ask("two")
    assert hub._adapter("a").calls == 1                # not paid for twice


def test_bad_request_does_not_cool_model(make_hub):
    hub = make_hub(pair(), {"a": [AIError("bad_request", "bad param")]})
    hub.ask("hi")
    assert hub.health.available("a/ma")


def test_invalid_empty_reply_fails_over(make_hub):
    hub = make_hub(pair(), {"a": [Response(text="   ", usage=Usage(1, 0))]})
    assert hub.ask("hi").model == "mb"


def test_fallback_disabled_means_single_attempt(make_hub):
    hub = make_hub(pair(fallback=False), {"a": [AIError("auth", "x")]})
    with pytest.raises(AIError):
        hub.ask("hi")
    assert hub._adapter("b").calls == 0 if "b" in hub._adapters else True


def test_cancel_before_start(make_hub):
    hub = make_hub(pair())
    tok = CancelToken()
    tok.cancel()
    with pytest.raises(AIError) as e:
        hub.ask(Request(messages=[{"role": "user", "content": "hi"}], cancel=tok))
    assert e.value.kind == "cancelled"


def test_invented_tool_call_is_dropped(make_hub):
    bad = Response(text="ok", tool_calls=[{"id": "1", "name": "rm_rf", "arguments": {}}],
                   usage=Usage(1, 1))
    cfg = cfg_for(prov("a", [model("ma", 0, caps=("text", "tools"))]))
    hub = make_hub(cfg, {"a": [bad]})
    tool = {"name": "safe", "description": "d", "parameters": {"type": "object", "properties": {}}}
    r = hub.ask(Request(messages=[{"role": "user", "content": "hi"}], tools=[tool]))
    assert r.tool_calls == []


def test_budget_cap_blocks_paid_models(make_hub, tmp_path):
    cfg = cfg_for(prov("paid", [model("p", 0, pi=1, po=1)]), monthly_cost_cap_usd=0.01)
    hub = make_hub(cfg)
    hub.usage._month_cost = 5.0
    with pytest.raises(AIError) as e:
        hub.ask("hi")
    assert e.value.kind == "budget"


def test_usage_logged_with_cost(make_hub):
    cfg = cfg_for(prov("a", [model("m", 0, pi=1.0, po=2.0)]))
    hub = make_hub(cfg, {"a": [Response(text="x", usage=Usage(1000, 500))]})
    r = hub.ask("hi")
    assert r.cost == pytest.approx((1000 * 1 + 500 * 2) / 1e6)
    assert hub.usage.summary()["a/m"]["calls"] == 1
