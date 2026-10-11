"""job_queue — give the assistant tasks with a time. It does them (even later, even
while you are away) and messages your Telegram when queued, finished or failed."""
from __future__ import annotations

from core.jobs import JobStore, Notifier, add_from_text, announce, describe_when


def job_queue(parameters: dict, player=None) -> str:
    mode = str(parameters.get("mode", "list")).lower()
    store, n = JobStore(), Notifier.from_config()
    try:
        if mode == "add":
            text = f"{parameters.get('task', '')} {parameters.get('when', '')}".strip()
            if not text:
                return "What should I do, and when?"
            job = add_from_text(store, text)
            sent = announce(job, n)
            if player is not None and hasattr(player, "write_log"):
                player.write_log(f"SYS: 📋 Job #{job['id']} queued — see the JOBS panel.")
            rep = f", repeating {job['repeat'].replace(':', ' ')}" if job.get("repeat") else ""
            return (f"Queued job #{job['id']}: {job['title']} — {describe_when(job['run_at'])}{rep}. "
                    + ("I sent a confirmation to your Telegram." if sent else "Telegram is not set up, so no phone message."))
        if mode == "list":
            rows = store.list(limit=8)
            return "; ".join(f"#{j['id']} {j['status']} {describe_when(j['run_at'])}: {j['title'][:50]}" for j in rows) or "The queue is empty."
        jid = int(str(parameters.get("id", "")).strip().lstrip("#") or 0)
        if mode == "cancel":
            return f"Cancelled job #{jid}." if store.cancel(jid) else f"Job #{jid} cannot be cancelled."
        if mode == "run_now":
            return f"Job #{jid} will run in a moment." if store.run_now(jid) else f"Job #{jid} cannot run now."
        if mode == "result":
            j = store.get(jid)
            return (j["result"] or j["error"] or f"Job #{jid} is {j['status']}.")[:1500] if j else f"No job #{jid}."
        return "Modes: add (task, when), list, cancel (id), run_now (id), result (id)."
    except ValueError as e:
        return f"Sir, I cannot do that: {e}."
    except Exception as e:
        return f"Sir, the job queue failed: {type(e).__name__}: {str(e)[:100]}"


TOOL = {
    "name": "job_queue",
    "description": (
        "Schedule tasks for later. mode: add (task, when e.g. 'tomorrow 8am', 'every weekday 7am', 'in 2 hours') | "
        "list | cancel/run_now/result (id). Jobs write something with an AI (optionally a PDF), remind, or clip video "
        "(not trading, meetings or video uploads). The user is messaged on Telegram."),
    "parameters": {
        "type": "OBJECT",
        "properties": {
            "mode": {"type": "STRING", "description": "add|list|cancel|run_now|result"},
            "task": {"type": "STRING", "description": "what to do"},
            "when": {"type": "STRING", "description": "when to do it"},
            "id": {"type": "STRING", "description": "job number"},
        },
        "required": ["mode"],
    },
    "handler": job_queue,
}
