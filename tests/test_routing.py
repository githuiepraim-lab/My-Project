import pytest
from conftest import cfg_for, model, prov
from core.ai import router
from core.ai.types import AIError, Image, Request, Response, Usage


def two_tier():
    return cfg_for(
        prov("cheap", [model("small", 0, pi=0.1, po=0.2)], priority=20),
        prov("big", [model("large", 2, pi=5, po=15), model("mid", 1, pi=1, po=3)], priority=10))


def test_simple_question_uses_light_model(make_hub):
    hub = make_hub(two_tier())
    r = hub.ask("what time is it in Tokyo?")
    assert r.model == "small"


def test_complex_task_uses_strong_model(make_hub):
    hub = make_hub(two_tier())
    q = "Please debug this and explain why the algorithm is slow: " + "x " * 200
    assert hub.ask(q).model == "large"


def test_code_uses_standard_not_strong(make_hub):
    hub = make_hub(two_tier())
    assert hub.ask("fix:\n```\nprint(1)\n```").model == "mid"


def test_task_hint_classify_is_light(make_hub):
    hub = make_hub(two_tier())
    assert hub.ask("debug architecture " * 40, task="classify").model == "small"


def test_vision_only_to_vision_models(make_hub):
    cfg = cfg_for(prov("a", [model("text-only", 0)]),
                  prov("b", [model("seer", 1, caps=("text", "vision"))]))
    hub = make_hub(cfg)
    r = hub.ask(Request(messages=[{"role": "user", "content": "what is this"}],
                        images=[Image(b"\x89PNG")]))
    assert r.model == "seer"


def test_tools_only_to_tool_models(make_hub):
    cfg = cfg_for(prov("a", [model("plain", 0)]),
                  prov("b", [model("tooly", 1, caps=("text", "tools"))]))
    hub = make_hub(cfg)
    tool = {"name": "t", "description": "d", "parameters": {"type": "object", "properties": {}}}
    r = hub.ask(Request(messages=[{"role": "user", "content": "go"}], tools=[tool]),)
    assert r.model == "tooly"


def test_no_capable_model_is_error(make_hub):
    hub = make_hub(cfg_for(prov("a", [model("plain", 0)])))
    with pytest.raises(AIError):
        hub.ask(Request(messages=[{"role": "user", "content": "x"}], images=[Image(b"x")]))


def test_cheapest_sufficient_beats_priority_in_cost_strategy(make_hub):
    cfg = cfg_for(prov("pricey", [model("p", 1, pi=10, po=30)], priority=1),
                  prov("thrifty", [model("t", 1, pi=0.1, po=0.2)], priority=99))
    assert make_hub(cfg).ask("hello there, tell me a joke").model == "t"


def test_priority_strategy_respects_user_priority(make_hub):
    cfg = cfg_for(prov("pricey", [model("p", 1, pi=10, po=30)], priority=1),
                  prov("thrifty", [model("t", 1, pi=0.1, po=0.2)], priority=99),
                  strategy="priority")
    assert make_hub(cfg).ask("hello there").model == "p"


def test_local_model_preferred_when_sufficient(make_hub):
    cfg = cfg_for(prov("cloud", [model("c", 0, pi=0.1, po=0.2)]),
                  prov("home", [model("l", 0, local=True)]))
    assert make_hub(cfg).ask("hi").model == "l"


def test_manual_pin_and_fallback(make_hub):
    cfg = cfg_for(prov("a", [model("m1", 0)]), prov("b", [model("m2", 0)]))
    hub = make_hub(cfg, {"b": [AIError("unavailable", "down")]})
    r = hub.ask(Request(messages=[{"role": "user", "content": "hi"}], meta={"model": "b/m2"}))
    assert r.model == "m1"                      # pinned model failed -> fell over
    assert r.attempts[0]["model"] == "b/m2"


def test_weaker_model_used_only_if_nothing_adequate(make_hub):
    hub = make_hub(cfg_for(prov("a", [model("tiny", 0)])))
    r = hub.ask("debug and explain why the architecture " + "y " * 200)
    assert r.model == "tiny" and "downgraded" in r.finish_reason


def test_no_downgrade_after_adequate_model_fails(make_hub):
    cfg = cfg_for(prov("a", [model("tiny", 0)]), prov("b", [model("large", 2)]))
    hub = make_hub(cfg, {"b": [AIError("auth", "bad key")]})
    with pytest.raises(AIError):
        hub.ask("debug and explain why the architecture " + "y " * 200)


def test_classify_unit():
    r = Request(messages=[{"role": "user", "content": "hi"}])
    assert router.classify(r).tier == 0
