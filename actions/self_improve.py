"""self_improve — the assistant writes changes to its own code. Proposals are
tested in a copy first; applying needs your on-screen CONFIRM; every change can
be rolled back. There is no unattended auto-apply."""
from __future__ import annotations

import re

from core import confirm
from core.selfdev import SelfDev


def _split(v) -> list[str]:
    return [p.strip() for p in re.split(r"[\n;,]+", str(v or "")) if p.strip()]


def self_improve(parameters: dict, player=None) -> str:
    mode = str(parameters.get("mode", "list")).lower()
    sd = SelfDev()
    try:
        if mode == "propose":
            task = str(parameters.get("task", "")).strip()
            if not task:
                return "Tell me what to change or improve."
            p = sd.propose(task, _split(parameters.get("files")), str(parameters.get("upload", "")))
            if p.error:
                return f"I could not make a safe change: {p.error}"
            verdict = "tests PASS" if p.tests_passed else f"tests FAIL ({p.test_tail[-150:]})"
            return (f"Proposal {p.id}: {p.summary} — {len(p.files)} file(s), {verdict}. "
                    f"{'Say apply ' + p.id + ' to review and confirm it.' if p.tests_passed else 'It will not be applied.'}")
        if mode == "show":
            p = sd.load(str(parameters.get("id", "")))
            return p.diff[:3500] or "Empty proposal."
        if mode == "apply":
            p = sd.load(str(parameters.get("id", "")))
            return confirm.request("self_improve", f"Change my own code: {p.summary}"[:90],
                                   f"{len(p.files)} file(s): {', '.join(p.files)} — tests "
                                   f"{'passed' if p.tests_passed else 'FAILED'}",
                                   lambda: sd.apply(p.id))
        if mode == "rollback":
            return sd.rollback(str(parameters.get("id", "")))
        if mode == "review":
            from core.ai import get_hub
            from core.ai.types import Request
            ctx = sd._context(_split(parameters.get("files")), "")
            r = get_hub().ask(Request(messages=[{"role": "user", "content": ctx}], task="code", min_tier=2,
                                      system="Review this code. List the 5 most valuable concrete improvements "
                                             "(bugs, speed, clarity, missing tests), most important first, one line each.",
                                      max_output_tokens=500, cacheable=False))
            return r.text
        rows = sd.list()
        return ("; ".join(f"{p.id}: {p.summary or p.error[:40]} [{'ok' if p.tests_passed else 'no'}]" for p in rows[-5:])
                or "No proposals yet.")
    except ValueError as e:
        return f"Sir, I cannot do that: {e}."
    except Exception as e:
        return f"Sir, self_improve failed: {type(e).__name__}: {str(e)[:100]}"


TOOL = {
    "name": "self_improve",
    "description": (
        "Improve the assistant's own code. mode: propose (task, files, upload=path to a code file to "
        "integrate) writes + tests a change in a copy | show (id) | apply (id; user confirms on screen) | "
        "rollback (id) | review (files) lists improvements | list. Never applies without the user's confirmation."),
    "parameters": {
        "type": "OBJECT",
        "properties": {
            "mode": {"type": "STRING", "description": "propose|show|apply|rollback|review|list"},
            "task": {"type": "STRING", "description": "what to change"},
            "files": {"type": "STRING", "description": "comma-separated relative paths for context"},
            "upload": {"type": "STRING", "description": "path of an uploaded code file"},
            "id": {"type": "STRING", "description": "proposal id"},
        },
        "required": ["mode"],
    },
    "handler": self_improve,
}
