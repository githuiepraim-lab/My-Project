"""
multi_ai — use several AIs at once, or let them talk to each other.

One tool with a `mode` instead of four, because every tool definition is sent
to the model on every connection: one short declaration costs a quarter of the
tokens, and the model picks the mode from the user's words.

Everything here is text-only. No AI taking part can call a tool or touch the
computer, and other AIs' answers are handed to each other as quoted data.
"""
from __future__ import annotations

from core.ai import get_hub
from core.ai.types import AIError


def _log(player, msg: str) -> None:
    if player:
        try:
            player.write_log(msg)
        except Exception:
            pass


def _split(models: str) -> list[str]:
    return [m.strip() for m in (models or "").replace(" and ", ",").split(",") if m.strip()]


def multi_ai(parameters: dict, player=None) -> str:
    mode = str(parameters.get("mode", "ask")).lower()
    prompt = str(parameters.get("prompt", "")).strip()
    targets = _split(parameters.get("models", ""))
    hub = get_hub()
    try:
        if mode == "status":
            st = hub.status()
            lines = []
            for n, p in st["providers"].items():
                if p["enabled"]:
                    h = st["health"].get(f"{n}/{p['models'][0]}", {}) if p["models"] else {}
                    lines.append(f"{n}: {'ready' if p['key'] else 'no key'}"
                                 + (f", cooling {h['cooling_s']}s" if h.get("cooling_s") else ""))
            return "; ".join(lines) or "No AI provider is configured yet. Open Settings, AI providers."
        if not prompt:
            return "What should I ask them?"

        if mode == "discuss":
            rounds = int(parameters.get("rounds") or 2)
            _log(player, f"JARVIS: Starting a discussion between {', '.join(targets) or 'your AIs'}…")
            d = hub.discuss(prompt, targets or None, rounds=rounds,
                            on_turn=lambda who, t: _log(player, f"{who.upper()}: {t}"))
            return f"After {len(d.transcript)} turns ({d.stopped}): {d.summary}"

        if mode == "chain":
            names = targets or hub.default_targets(2)
            steps = [{"ai": n, "instruction": "Draft an answer." if i == 0 else
                      "Review the previous answer, fix errors and give the improved final answer."}
                     for i, n in enumerate(names)]
            out = hub.chain(prompt, steps)
            last = next((r for r in reversed(out) if r.ok), None)
            return last.text if last else "None of them could answer."

        # ask / compare: the same question to several AIs in parallel
        results = hub.ask_many(prompt, targets or None)
        good = [r for r in results if r.ok]
        for r in results:
            _log(player, f"{(r.provider or '?').upper()}: {r.text if r.ok else '(' + r.error.split(':')[0] + ')'}")
        if not good:
            return "None of the AIs could answer right now."
        if len(good) == 1 or mode == "ask" and len(targets) <= 1:
            return good[0].text
        return " | ".join(f"{r.provider}: {r.text}" for r in good)
    except AIError as e:
        return f"Sir, the AI hub could not do that: {e}"
    except Exception as e:                                   # never raise into the voice loop
        return f"Sir, multi_ai failed: {type(e).__name__}"


TOOL = {
    "name": "multi_ai",
    "description": (
        "Use several AI services at once. mode: ask (one or more AIs answer), compare "
        "(same question to many, show each), discuss (the AIs debate, then summarise), "
        "chain (first drafts, next reviews), status (which are available). "
        "models: comma list of provider or agent names (omit = all configured). "
        "Not for ordinary questions you can answer yourself."),
    "parameters": {
        "type": "OBJECT",
        "properties": {
            "mode": {"type": "STRING", "description": "ask | compare | discuss | chain | status"},
            "prompt": {"type": "STRING", "description": "The question or topic"},
            "models": {"type": "STRING", "description": "e.g. 'anthropic, groq'"},
            "rounds": {"type": "INTEGER", "description": "discuss rounds, 1-6"},
        },
        "required": ["mode"],
    },
    "handler": multi_ai,
}
