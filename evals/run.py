"""
Regression evaluation for the AI layer.

    python -m evals.run            offline: deterministic mock providers, free, fast (CI-safe)
    python -m evals.run --live     real providers from your settings (spends tokens)
    python -m evals.run --history  show how pass-rate and latency have moved

What it checks: routing decisions, memory recall, answer content, structured
output, tool execution and latency budgets. Every run is appended to
logs/evals.jsonl and compared with the previous run of the same mode, so a
regression shows up as a named case, not a feeling.

This measures the SYSTEM around the model. It does not train or fine-tune
anything, and nothing here changes a model's weights.
"""
from __future__ import annotations

import json
import re
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
CASES = Path(__file__).parent / "cases"
RESULTS = ROOT / "logs" / "evals.jsonl"


def _mock_hub():
    """A hub over scripted providers that answer by simple rules."""
    from core.ai.adapters.base import Adapter
    from core.ai.health import Health
    from core.ai.hub import Hub
    from core.ai.types import Response, Usage
    from core.ai.usage import UsageTracker

    class Rules(Adapter):
        def complete(self, req, model):
            q = req.messages[-1]["content"].lower()
            if "17 + 25" in q:
                t = "42"
            elif "json" in q:
                t = '{"ok": true}'
            elif "single word" in q:
                t = "ready"
            else:
                t = f"mock answer from {model.id}"
            return Response(text=t, usage=Usage(20, 5))

    def mdl(i, t, **k):
        return {"id": i, "tier": t, "caps": ["text", "stream", "tools", "json"], **k}
    cfg = {"settings": {"strategy": "cost", "mode": "auto", "manual_model": "", "fallback": True,
                        "max_attempts": 4, "deadline_s": 45, "request_timeout_s": 25,
                        "allow_downgrade": False, "cache": False, "cache_ttl_s": 900,
                        "max_input_tokens": 6000, "max_output_tokens": 800,
                        "monthly_cost_cap_usd": None, "share_memory_default": False,
                        "cross_provider_fallback": True},
           "providers": {"mock": {"kind": "openai_compat", "enabled": True, "priority": 1,
                                  "requires_key": False,
                                  "models": [mdl("mock-small", 0, price_in=.1, price_out=.2),
                                             mdl("mock-mid", 1, price_in=1, price_out=2),
                                             mdl("mock-large", 2, price_in=5, price_out=10)]}},
           "agents": {}}
    return Hub(config_loader=lambda: cfg, key_getter=lambda n: "k",
               adapter_factory=lambda n, p, k: Rules(n, p, k), health=Health(),
               usage=UsageTracker(Path(tempfile.mkdtemp()) / "u.jsonl"), sleep=lambda s: None)


def run_case(kind: str, c: dict, hub) -> dict:
    t0 = time.perf_counter()
    ok, detail = True, ""
    try:
        if kind == "route":
            from core.ai import router
            from core.ai.types import Request
            prompt = c["prompt"] + " filler" * int(c.get("pad", 0))
            tier = router.classify(Request(messages=[{"role": "user", "content": prompt}],
                                           task=c.get("task", ""))).tier
            ok, detail = tier == c["expect_tier"], f"tier={tier} want={c['expect_tier']}"
        elif kind == "memory":
            from memory import memory_manager as M
            with tempfile.TemporaryDirectory() as d:
                old = (M.MEMORY_PATH, M._memory_changed)
                M.MEMORY_PATH, M._memory_changed = Path(d) / "m.json", lambda: None
                try:
                    M.update_memory(c["seed"])
                    top = [r["key"] for r in M.search_entries(c["query"], limit=3)]
                finally:
                    M.MEMORY_PATH, M._memory_changed = old
            ok, detail = c["expect_key"] in top, f"top3={top}"
        elif kind == "answer":
            from core.ai.types import Request
            r = hub.ask(Request(messages=[{"role": "user", "content": c["prompt"]}],
                                task=c.get("task", ""), json_mode=bool(c.get("json")),
                                max_output_tokens=120, cacheable=False))
            txt = r.text
            ok = True
            if "expect_contains" in c:
                ok = c["expect_contains"].lower() in txt.lower()
            if ok and "expect_regex" in c:
                ok = re.search(c["expect_regex"], txt) is not None
            if ok and "expect_json_key" in c:
                try:
                    ok = c["expect_json_key"] in json.loads(txt[txt.find("{"):txt.rfind("}") + 1])
                except ValueError:
                    ok = False
            lat = time.perf_counter() - t0
            if ok and "max_latency_s" in c and lat > c["max_latency_s"]:
                ok = False
            detail = f"{r.label} {lat:.2f}s {txt[:50]!r}"
        elif kind == "tool":
            import importlib
            mod = importlib.import_module(f"actions.{c['action']}")
            if c["action"] == "multi_ai":
                mod.get_hub = lambda: hub
            out = mod.TOOL["handler"](c["args"], None)
            ok, detail = c["expect_contains"].lower() in str(out).lower(), str(out)[:60]
    except Exception as e:
        ok, detail = False, f"{type(e).__name__}: {e}"[:120]
    return {"id": c["id"], "kind": kind, "ok": ok, "ms": round((time.perf_counter() - t0) * 1000, 1),
            "detail": detail}


def main(argv: list[str]) -> int:
    live = "--live" in argv
    mode = "live" if live else "mock"
    if "--history" in argv:
        for line in RESULTS.read_text().splitlines()[-10:] if RESULTS.exists() else []:
            r = json.loads(line)
            print(f"{r['ts']}  {r['mode']:4}  {r['passed']}/{r['total']}  median {r['median_ms']} ms")
        return 0
    hub = None
    if live:
        from core.ai import get_hub
        hub = get_hub()
        if not hub.is_configured():
            print("No provider configured — add one in Settings, AI providers.")
            return 2
    else:
        hub = _mock_hub()
    results = []
    for f in sorted(CASES.glob("*.json")):
        data = json.loads(f.read_text(encoding="utf-8"))
        for kind, cases in data.items():
            results += [run_case(kind, c, hub) for c in cases]
    passed = sum(r["ok"] for r in results)
    ms = sorted(r["ms"] for r in results)
    prev = None
    if RESULTS.exists():
        for line in RESULTS.read_text().splitlines():
            row = json.loads(line)
            if row["mode"] == mode:
                prev = row
    for r in results:
        print(f"{'PASS' if r['ok'] else 'FAIL'}  {r['id']:22} {r['ms']:8.1f} ms  {r['detail']}")
    print(f"\n{passed}/{len(results)} passed ({mode})")
    regress = []
    if prev:
        was = {x["id"]: x for x in prev["results"]}
        regress = [r["id"] for r in results if not r["ok"] and was.get(r["id"], {}).get("ok")]
        slow = [r["id"] for r in results if r["id"] in was and was[r["id"]]["ms"] > 5
                and r["ms"] > was[r["id"]]["ms"] * 2]
        if regress:
            print("REGRESSIONS (passed last run):", ", ".join(regress))
        if slow:
            print("SLOWER than last run (>2x):", ", ".join(slow))
    RESULTS.parent.mkdir(exist_ok=True)
    with RESULTS.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps({"ts": time.strftime("%Y-%m-%d %H:%M:%S"), "mode": mode,
                             "passed": passed, "total": len(results),
                             "median_ms": ms[len(ms) // 2] if ms else 0, "results": results}) + "\n")
    return 0 if passed == len(results) and not regress else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
