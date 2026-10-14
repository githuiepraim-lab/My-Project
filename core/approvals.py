"""
Approvals on your phone.

Something wants to happen that you should OK first (send these files, post to that webhook,
accept what a job produced). It asks on your Telegram with ✅ / ❌ buttons; tapping one
decides it. The approval lives in the job database, so it survives restarts and works
while the laptop is off if the worker runs elsewhere.

Safe by construction: only an allow-list of NAMED actions can run after approval (nothing
the model writes is ever executed), the answer must come from YOUR chat, each request has
its own random id and expires (default 24 h), and a request can be decided only once.
"""
from __future__ import annotations

import json
import sqlite3
import time

from core.jobs import DB_PATH, Notifier, JobStore

_SCHEMA = """CREATE TABLE IF NOT EXISTS approvals(
  id INTEGER PRIMARY KEY AUTOINCREMENT, title TEXT NOT NULL, detail TEXT, action TEXT NOT NULL, payload TEXT NOT NULL DEFAULT '{}',
  status TEXT NOT NULL DEFAULT 'pending', created REAL NOT NULL, expires REAL NOT NULL, decided REAL, result TEXT, message_id INTEGER)"""


def _act_note(payload: dict) -> str:
    return payload.get("message", "Approved.")


def _act_integration(payload: dict) -> str:
    from core.integrations import Integrations
    return Integrations().call(payload["name"], payload["action"], payload.get("params") or {}, approved=True)


def _act_send_files(payload: dict) -> str:
    from actions.report_sender import _deliver
    from core import reports
    files = reports.check_files(payload["files"])
    return _deliver(payload.get("channel", "telegram"), files, payload.get("caption", ""), payload.get("to", ""), False)


ACTIONS = {"note": _act_note, "integration": _act_integration, "send_files": _act_send_files}


class Approvals:
    def __init__(self, path=None, notifier: Notifier | None = None, actions: dict | None = None) -> None:
        self.path = path or DB_PATH
        JobStore(self.path)                                     # make sure the file and folder exist
        self.n = notifier or Notifier()
        self.actions = actions if actions is not None else ACTIONS
        with self._c() as c:
            c.execute(_SCHEMA)

    def _c(self) -> sqlite3.Connection:
        c = sqlite3.connect(self.path, timeout=30, isolation_level=None)
        c.row_factory = sqlite3.Row
        return c

    def get(self, aid: int) -> dict | None:
        with self._c() as c:
            r = c.execute("SELECT * FROM approvals WHERE id=?", (int(aid),)).fetchone()
        return dict(r) if r else None

    def pending(self) -> list[dict]:
        with self._c() as c:
            return [dict(r) for r in c.execute("SELECT * FROM approvals WHERE status='pending' AND expires>? ORDER BY id", (time.time(),))]

    def request(self, title: str, detail: str, action: str = "note", payload: dict | None = None, ttl_hours: float = 24) -> dict:
        if action not in self.actions:
            raise ValueError(f"'{action}' is not an approvable action (allowed: {', '.join(self.actions)})")
        now = time.time()
        with self._c() as c:
            aid = c.execute("INSERT INTO approvals(title,detail,action,payload,created,expires) VALUES(?,?,?,?,?,?)",
                            (title[:200], detail[:1500], action, json.dumps(payload or {}), now, now + ttl_hours * 3600)).lastrowid
        mid = self.n.send_buttons(f"🔐 Approval needed (#{aid})\n{title}\n\n{detail[:900]}\n\nExpires in {ttl_hours:g} h.",
                                  [[("✅ Approve", f"ap:{aid}:y"), ("❌ Reject", f"ap:{aid}:n")]])
        if mid:
            with self._c() as c:
                c.execute("UPDATE approvals SET message_id=? WHERE id=?", (mid, aid))
        return self.get(aid)

    def decide(self, aid: int, yes: bool) -> str:
        """Approve or reject, once. Returns what happened, in a sentence."""
        now = time.time()
        with self._c() as c:
            c.execute("BEGIN IMMEDIATE")
            r = c.execute("SELECT * FROM approvals WHERE id=?", (int(aid),)).fetchone()
            if r is None:
                c.execute("COMMIT"); return f"No approval #{aid}."
            if r["status"] != "pending":
                c.execute("COMMIT"); return f"#{aid} was already {r['status']}."
            if r["expires"] < now:
                c.execute("UPDATE approvals SET status='expired', decided=? WHERE id=?", (now, aid)); c.execute("COMMIT")
                return f"#{aid} expired, so nothing was done."
            c.execute("UPDATE approvals SET status=?, decided=? WHERE id=?", ("approved" if yes else "rejected", now, aid))
            c.execute("COMMIT")
        if not yes:
            return f"Rejected #{aid}: {r['title']}. Nothing was done."
        try:
            out = str(self.actions[r["action"]](json.loads(r["payload"] or "{}")))[:1500]
            status = "done"
        except Exception as e:
            from core.ai.redact import redact
            out, status = f"{type(e).__name__}: {redact(e)}"[:300], "failed"
        with self._c() as c:
            c.execute("UPDATE approvals SET status=?, result=? WHERE id=?", (status, out, aid))
        return (f"Approved #{aid}: {r['title']}. {out}" if status == "done" else f"Approved #{aid} but it failed: {out}")
