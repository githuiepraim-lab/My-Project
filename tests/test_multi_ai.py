import pytest
from conftest import cfg_for, model, prov
from core.ai.types import AIError, Response, Usage


def trio():
    return cfg_for(prov("alpha", [model("a1", 1)], priority=1),
                   prov("beta", [model("b1", 1)], priority=2),
                   prov("gamma", [model("g1", 1)], priority=3))


def test_ask_many_parallel_and_isolated_failure(make_hub):
    hub = make_hub(trio(), {"beta": [AIError("unavailable", "down")]})
    res = hub.ask_many("what is 2+2?", ["alpha", "beta", "gamma"])
    assert [r.provider for r in res] == ["alpha", "beta", "gamma"]
    assert res[0].ok and res[2].ok and not res[1].ok


def test_ask_many_unknown_target(make_hub):
    res = make_hub(trio()).ask_many("hi", ["nobody"])
    assert not res[0].ok


def test_discuss_turn_taking_and_untrusted_framing(make_hub):
    hub = make_hub(trio())
    d = hub.discuss("tabs vs spaces", ["alpha", "beta"], rounds=2)
    speakers = [w for w, _t in d.transcript]
    assert speakers == ["alpha", "beta", "alpha", "beta"]
    second_prompt = hub._adapter("beta").seen[0][1]
    assert "<ai_message from=\"alpha\">" in second_prompt.messages[0]["content"]
    assert "untrusted" in second_prompt.system.lower()
    assert second_prompt.tools == []                   # words only, never actions
    assert d.summary


def test_discuss_stops_on_consensus(make_hub):
    agree = [Response(text="AGREED: use spaces", usage=Usage(5, 5))]
    hub = make_hub(trio(), {"alpha": agree, "beta": agree})
    d = hub.discuss("x", ["alpha", "beta"], rounds=5)
    assert d.stopped == "consensus" and len(d.transcript) == 2


def test_discuss_token_budget(make_hub):
    big = [Response(text="long", usage=Usage(900, 900))]
    hub = make_hub(trio(), {"alpha": big, "beta": big})
    d = hub.discuss("x", ["alpha", "beta"], rounds=6, max_total_tokens=2000)
    assert d.stopped == "token budget reached" and len(d.transcript) < 12


def test_discuss_needs_two(make_hub):
    with pytest.raises(AIError):
        make_hub(trio()).discuss("x", ["alpha"])


def test_chain_passes_output_forward(make_hub):
    hub = make_hub(trio(), {"alpha": [Response(text="DRAFT", usage=Usage(1, 1))]})
    out = hub.chain("write a haiku", [{"ai": "alpha", "instruction": "draft"},
                                      {"ai": "beta", "instruction": "review"}])
    assert "DRAFT" in hub._adapter("beta").seen[0][1].messages[0]["content"]
    assert len(out) == 2


def test_agent_persona_from_config(make_hub):
    cfg = trio()
    cfg["agents"] = {"critic": {"model": "beta", "role": "You are a harsh critic."}}
    hub = make_hub(cfg)
    hub.ask_many("review this", ["critic"])
    assert "harsh critic" in hub._adapter("beta").seen[0][1].system


def test_submit_can_be_cancelled(make_hub):
    hub = make_hub(trio())
    job = hub.submit("hi")
    assert job.result(5).text
