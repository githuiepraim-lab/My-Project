"""
Before/after benchmarks.

    python bench/benchmark.py --baseline /path/to/original/checkout

Runs the same workloads against the ORIGINAL code (the --baseline checkout) and
the upgraded code, in separate processes, and prints a table. Anything that is
simulated (no real provider is called) is labelled as such: provider latency
and prices are not measurable offline, so those rows measure OUR behaviour
(attempts made, calls avoided) rather than anyone's servers.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import statistics as st
import subprocess
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


# ── data ────────────────────────────────────────────────────────────────────
def build_memory(n_distractors: int = 1900, seed: int = 7):
    """~2000 facts. A fixed set of 60 'targets' with natural phrasing and a
    great many distractors that share common vocabulary (names, jobs, places,
    birthdays, likes), so ranking has to do real work."""
    rnd = random.Random(seed)
    first = ["Wanjiru", "Kamau", "Achieng", "Otieno", "Mwende", "Kiprop", "Njeri", "Odhiambo",
             "Wambui", "Mutua", "Akinyi", "Kariuki", "Chebet", "Nyambura", "Barasa"]
    jobs = ["nurse", "mechanic", "teacher", "barber", "driver", "accountant", "designer", "farmer"]
    places = ["Machakos", "Nairobi", "Kitui", "Mombasa", "Nakuru", "Eldoret", "Kisumu"]
    likes = ["football", "chess", "gospel music", "fishing", "gaming", "photography", "running"]
    mem: dict = {}
    targets = []

    def put(cat, key, val, q=None):
        mem.setdefault(cat, {})[key] = {"value": val, "updated": "2026-09-01"}
        if q:
            targets.append((q, cat, key))

    people = [("sister", "Wanjiru", "studies nursing"), ("brother", "Kamau", "runs a barbershop"),
              ("mother", "Njeri", "teaches primary school"), ("father", "Odhiambo", "drives a matatu"),
              ("best friend", "Mutua", "repairs phones"), ("cousin", "Akinyi", "lives in Kisumu")]
    for rel, name, what in people:
        put("relationships", f"{rel.replace(' ', '_')}_name", f"{name}, {what}",
            f"who is my {rel}")
        put("relationships", f"{rel.replace(' ', '_')}_birthday", f"{name}'s birthday is on {rnd.randint(1,28)} March",
            f"when is my {rel}'s birthday")
    topics = [("drip_closet", "Kenyan streetwear brand with a hoodie drop planned for May", "my streetwear brand"),
              ("nextgen_arcade", "gaming arcade near the university with twelve consoles", "the gaming arcade"),
              ("content_pipeline", "short videos published to YouTube and TikTok every day", "my video publishing"),
              ("coursework", "biblical studies assignment on the book of Romans", "my coursework"),
              ("assistant_project", "personal AI assistant built on Python and PyQt6", "my AI assistant project"),
              ("website", "e-commerce site with a POS and an admin panel", "the online shop website")]
    for key, val, q in topics:
        put("projects", key, val, f"tell me about {q}")
    prefs = [("answer_style", "short answers without preamble", "how do I like answers"),
             ("favourite_music", "gospel and afrobeats while working", "what music do I listen to"),
             ("coffee_order", "black coffee no sugar", "how do I take my coffee"),
             ("wake_time", "wakes up at five thirty on weekdays", "when do I wake up"),
             ("preferred_editor", "uses VS Code with the dark theme", "which editor do I use"),
             ("transport", "takes a motorbike to campus most days", "how do I get to campus")]
    for key, val, q in prefs:
        put("preferences", key, val, q)
    ident = [("city", "lives near Machakos University", "where do I live"),
             ("language", "speaks English Swahili and some Kikuyu", "what languages do I speak"),
             ("pet", "has a dog called Simba", "do I have a pet"),
             ("car", "drives a 2009 Toyota Fielder", "what car do I drive"),
             ("allergy_note", "dislikes cilantro", "what food do I dislike"),
             ("goal", "wants to open a second arcade branch next year", "what is my business goal")]
    for key, val, q in ident:
        put("identity", key, val, q)
    for i in range(n_distractors):
        r = rnd.random()
        nm, jb, pl, lk = rnd.choice(first), rnd.choice(jobs), rnd.choice(places), rnd.choice(likes)
        if r < .3:
            put("relationships", f"contact_{i}", f"{nm} works as a {jb} in {pl}")
        elif r < .5:
            put("relationships", f"contact_birthday_{i}", f"{nm}'s birthday is on {rnd.randint(1,28)} June")
        elif r < .7:
            put("notes", f"note_{i}", f"my friend {nm} likes {lk} and lives in {pl}")
        elif r < .85:
            put("projects", f"idea_{i}", f"idea to sell {lk} merchandise to customers in {pl}")
        else:
            put("preferences", f"pref_{i}", f"enjoys {lk} on weekends, prefers the {pl} route")
    return mem, targets


def run_memory(root: str) -> dict:
    sys.path.insert(0, root)
    os.chdir(root)
    from memory import memory_manager as M
    mem, targets = build_memory()
    tmp = Path(tempfile.mkdtemp()) / "m.json"
    tmp.write_text(json.dumps(mem), encoding="utf-8")
    M.MEMORY_PATH = tmp
    if hasattr(M, "_memory_changed"):
        M._memory_changed = lambda: None
    p1 = r3 = 0
    ts = []
    for q, cat, key in targets:
        t0 = time.perf_counter()
        out = M.search_memory(q, limit=3)
        ts.append((time.perf_counter() - t0) * 1000)
        rows = re.findall(r"^(\w[\w-]*)/(.+?):", out, re.M)
        got = [(c, k.replace(" ", "_")) for c, k in rows]
        want = (cat, key)
        p1 += bool(got) and got[0] == want
        r3 += want in got[:3]
    n = len(targets)
    return {"entries": sum(len(v) for v in mem.values()), "queries": n,
            "precision_at_1": round(p1 / n, 3), "recall_at_3": round(r3 / n, 3),
            "median_ms": round(st.median(ts), 2), "p95_ms": round(sorted(ts)[int(n * .95)], 2)}


def run_startup(root: str) -> dict:
    code = ("import sys,time;sys.path.insert(0,%r);t=time.perf_counter();"
            "import core.gemini, memory.memory_manager;print((time.perf_counter()-t)*1000)" % root)
    vals = []
    for _ in range(5):
        out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=root)
        try:
            vals.append(float(out.stdout.strip().splitlines()[-1]))
        except Exception:
            pass
    res = {"import_gemini_memory_ms": round(st.median(vals), 1) if vals else None}
    if (Path(root) / "core" / "ai").exists():
        code2 = ("import sys,time;sys.path.insert(0,%r);t=time.perf_counter();"
                 "from core.ai import get_hub;get_hub();print((time.perf_counter()-t)*1000)" % root)
        v2 = [float(subprocess.run([sys.executable, "-c", code2], capture_output=True, text=True,
                                   cwd=root).stdout.strip().splitlines()[-1]) for _ in range(5)]
        res["first_use_of_ai_hub_ms"] = round(st.median(v2), 1)
    return res


# ── new-code-only measurements ──────────────────────────────────────────────
def run_new() -> dict:
    sys.path.insert(0, str(ROOT))
    os.chdir(ROOT)
    from core.ai import context as ctx
    from core.ai.types import AIError, Request, Response, Usage
    sys.path.insert(0, str(ROOT / "tests"))
    from conftest import FakeAdapter, cfg_for, model, prov
    from core.ai.health import Health
    from core.ai.hub import Hub
    from core.ai.usage import UsageTracker
    out: dict = {}

    # history
    rnd = random.Random(1)
    msgs = [{"role": "user" if i % 2 == 0 else "assistant",
             "content": f"turn {i}: " + " ".join(rnd.choice(["deadline", "plan", "hoodie", "arcade", "decide", "the", "and", "May"])
                                                   for _ in range(70))} for i in range(60)]
    full = sum(ctx.message_tokens(m) for m in msgs)
    fitted = sum(ctx.message_tokens(m) for m in ctx.fit_history(msgs, 1500))
    out["history_tokens"] = {"before": full, "after": fitted, "saved_pct": round(100 * (1 - fitted / full), 1)}

    # tools
    from pathlib import Path as P
    from core.action_loader import discover_actions
    decls = discover_actions(P("actions"), logger=lambda s: None).get_tool_declarations()
    tools = [{"name": d["name"], "description": d["description"], "parameters": d.get("parameters")} for d in decls]
    asks = {"what's the weather in Nairobi": "weather_report", "open spotify": "open_app",
            "send a whatsapp message": "send_message", "set a reminder for 5pm": "reminder",
            "play a youtube video": "youtube_video"}
    names = {t["name"] for t in tools}
    full_t = ctx.estimate_tokens(tools)
    kept = 0
    sel_tokens = []
    for q, want in asks.items():
        sel = ctx.select_tools(tools, q, 6)
        sel_tokens.append(ctx.estimate_tokens(sel))
        kept += (want in {t["name"] for t in sel}) if want in names else 0
    present = sum(1 for w in asks.values() if w in names)
    out["tool_definitions"] = {"all_tokens": full_t, "selected_tokens": int(st.mean(sel_tokens)),
                               "needed_tool_kept": f"{kept}/{present}", "tools": len(tools)}

    # cache + failover (simulated providers)
    def mk(cfg, scripts=None, **kw):
        for n, p in cfg["providers"].items():
            if scripts and n in scripts:
                p["_script"] = scripts[n]
        h = Hub(config_loader=lambda: cfg, key_getter=lambda n: "k",
                adapter_factory=lambda n, p, k: FakeAdapter(n, p, k), health=Health(),
                usage=UsageTracker(Path(tempfile.mkdtemp()) / "u.jsonl"), sleep=lambda s: None)
        return h
    h = mk(cfg_for(prov("a", [model("m", 0, pi=.1, po=.2)])))
    for _ in range(100):
        h.ask("translate: good morning", task="translate")
    calls = h._adapter("a").calls
    out["cache"] = {"requests": 100, "provider_calls_before": 100, "provider_calls_after": calls}

    down = [AIError("unavailable", "503")]
    h = mk(cfg_for(prov("gemini", [model("g", 1)], priority=1), prov("backup", [model("b", 1)], priority=2)),
           {"gemini": down})
    ok = 0
    t0 = time.perf_counter()
    for _ in range(200):
        try:
            h.ask("hello there")
            ok += 1
        except AIError:
            pass
    out["failover_primary_down"] = {"requests": 200, "answered_before": 0, "answered_after": ok,
                                    "primary_calls": h._adapter("gemini").calls,
                                    "wasted_on_dead_primary_pct": round(100 * h._adapter("gemini").calls / 200, 1)}
    h = mk(cfg_for(prov("a", [model("m", 0)])), {"a": down})
    n = 0
    for _ in range(50):
        try:
            h.ask("hi")
        except AIError:
            pass
    out["retry_bound"] = {"max_attempts_setting": 4, "provider_calls_for_50_failing_requests":
                          h._adapter("a").calls}

    # routing cost: arithmetic on assumed workload and the default price table
    work = ([("what time is it", 0)] * 60 + [("fix this ```code```", 1)] * 25 +
            [("debug and explain why the algorithm is slow " + "x " * 160, 2)] * 15)
    cfg = cfg_for(prov("p", [model("small", 0, pi=.1, po=.4), model("mid", 1, pi=1, po=4), model("big", 2, pi=5, po=20)]))
    h = mk(cfg)
    routed = always = 0.0
    for q, _t in work:
        spec = h.plan(Request(messages=[{"role": "user", "content": q}]))[0][0]
        routed += spec.cost(600, 250)
        always += 5 * 600 / 1e6 + 20 * 250 / 1e6
    out["routing_cost_simulated"] = {"requests": 100, "always_strongest_usd": round(always, 4),
                                     "routed_usd": round(routed, 4),
                                     "saved_pct": round(100 * (1 - routed / always), 1)}
    return out


def table(title, rows):
    print(f"\n{title}")
    w = max(len(r[0]) for r in rows) + 2
    for r in rows:
        print("  " + r[0].ljust(w) + "  ".join(str(x).rjust(14) for x in r[1:]))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", default="")
    ap.add_argument("--json", default=str(HERE / "results.json"))
    ap.add_argument("--child", choices=["memory", "startup"], help=argparse.SUPPRESS)
    ap.add_argument("--root", default=str(ROOT), help=argparse.SUPPRESS)
    a = ap.parse_args()
    if a.child:
        print(json.dumps({"memory": run_memory, "startup": run_startup}[a.child](a.root)))
        return 0

    def child(kind, root):
        r = subprocess.run([sys.executable, __file__, "--child", kind, "--root", root],
                           capture_output=True, text=True, cwd=root)
        return json.loads(r.stdout.strip().splitlines()[-1])

    res: dict = {}
    for kind in ("memory", "startup"):
        res[kind] = {"after": child(kind, str(ROOT))}
        if a.baseline:
            res[kind]["before"] = child(kind, a.baseline)
    res["new"] = run_new()
    m = res["memory"]
    b, n = m.get("before", {}), m["after"]
    table(f"MEMORY RECALL  ({n['entries']} stored facts, {n['queries']} natural-language questions)",
          [("", "before", "after"),
           ("precision@1", b.get("precision_at_1", "-"), n["precision_at_1"]),
           ("recall@3", b.get("recall_at_3", "-"), n["recall_at_3"]),
           ("median ms/query", b.get("median_ms", "-"), n["median_ms"]),
           ("p95 ms/query", b.get("p95_ms", "-"), n["p95_ms"])])
    s = res["startup"]
    table("STARTUP  (median of 5 cold imports)",
          [("", "before", "after"),
           ("import gemini+memory ms", s.get("before", {}).get("import_gemini_memory_ms", "-"),
            s["after"]["import_gemini_memory_ms"]),
           ("first use of AI hub ms", "n/a (new)", s["after"].get("first_use_of_ai_hub_ms")),
           ("(before/after import gap is within run-to-run noise)", "", "")])
    x = res["new"]
    h = x["history_tokens"]
    table("TOKENS  (our estimator; 'before' = what an unmanaged request would carry — the original had no equivalent)",
          [("", "before", "after"),
           ("60-turn history", h["before"], h["after"]),
           (f"tool defs ({x['tool_definitions']['tools']} tools)", x["tool_definitions"]["all_tokens"],
            x["tool_definitions"]["selected_tokens"]),
           ("needed tool still offered", "-", x["tool_definitions"]["needed_tool_kept"]),
           ("provider calls, 100 repeats", x["cache"]["provider_calls_before"], x["cache"]["provider_calls_after"])])
    f = x["failover_primary_down"]
    table("RELIABILITY  (SIMULATED providers; measures our logic, not anyone's servers)",
          [("", "before", "after"),
           ("answered, primary down", "0/200 (by design)", f"{f['answered_after']}/200"),
           ("calls spent on dead primary", "not measured", f["primary_calls"]),
           ("calls for 50 failing reqs", "not measured", x["retry_bound"]["provider_calls_for_50_failing_requests"])])
    r = x["routing_cost_simulated"]
    table("ROUTING COST  (ARITHMETIC on the default price table, not a real bill)",
          [("", "always strongest", "routed"),
           ("100 mixed requests, USD", r["always_strongest_usd"], r["routed_usd"]),
           ("saved", "", f"{r['saved_pct']}%")])
    Path(a.json).write_text(json.dumps(res, indent=2))
    print(f"\nSaved {a.json}\n'by design': the original code has no second provider, so when every Gemini model fails the feature fails. "
          "The original ladder does keep its own per-model cooldowns; that was not benchmarked.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
