"""system_doctor — check this computer's setup for the assistant, explain what is wrong,
and fix it on command (each fix is shown to you and needs your confirmation)."""
from __future__ import annotations

from core import confirm, sysdoctor

_last: list = []


def system_doctor(parameters: dict, player=None) -> str:
    global _last
    mode = str(parameters.get("mode", "diagnose")).lower()
    try:
        if mode == "diagnose":
            _last = sysdoctor.diagnose()
            bad = [f for f in _last if f.severity in ("bad", "warn")]
            if not bad:
                return "Everything looks healthy: " + "; ".join(f.title for f in _last[:6])
            fixable = sorted({f.fix for f in bad if f.fix})
            return ("Found " + f"{len(bad)} problem(s). " + " | ".join(f.line() for f in bad[:6])
                    + (f" I can fix: {', '.join(fixable)}. Say 'fix <name>' or 'fix all'." if fixable else ""))
        if mode == "fix":
            what = str(parameters.get("id", "")).strip().lower()
            if what in ("all", ""):
                ids = sorted({f.fix for f in (_last or sysdoctor.diagnose()) if f.fix})
            else:
                ids = [what]
            bad = [i for i in ids if i not in sysdoctor.FIXES]
            if bad or not ids:
                return f"Fixes I know: {', '.join(sysdoctor.FIXES)}." if bad else "There is nothing I can fix automatically."
            def run() -> str:
                return " ".join(f"[{i}] {sysdoctor.FIXES[i]()}" for i in ids)
            return confirm.request("system_fix", "Fix: " + ", ".join(sysdoctor.FIX_DOC[i] for i in ids)[:100],
                                   "; ".join(f"{i}: {sysdoctor.FIX_DOC[i]}" for i in ids), run)
        return "Modes: diagnose, fix (id or all)."
    except Exception as e:
        return f"Sir, the system check failed: {type(e).__name__}: {str(e)[:100]}"


TOOL = {
    "name": "system_doctor",
    "description": "Diagnose this computer's setup (disk, memory, CPU, internet, Gemini key, packages, audio, AI providers, "
                   "jobs) and fix problems on command. mode: diagnose | fix (id: install_missing, clear_cache, reset_ai, "
                   "repair_config, recover_jobs, flush_dns, or all). Each fix needs the user's confirmation.",
    "parameters": {"type": "OBJECT", "properties": {
        "mode": {"type": "STRING", "description": "diagnose|fix"}, "id": {"type": "STRING", "description": "fix name or all"}},
        "required": ["mode"]},
    "handler": system_doctor,
}
