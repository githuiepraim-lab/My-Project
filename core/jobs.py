"""
Job queue: give the assistant tasks with a time, and it does them and tells you.

    "summarise this week's forex lessons as a PDF tomorrow 8am"
    "remind me to submit the report at 5pm"
    "every weekday at 7am write my study plan"
    "clip demo.mp4 from 00:01:00 to 00:01:30 in 2 hours"

A job is stored in a small SQLite file, run by a JobRunner when due, and reported to
your Telegram when assigned, finished or failed. Past-due jobs (the computer was
off) run as soon as the runner starts. To have jobs run while your laptop is OFF,
run the same runner on an always-on machine:  python -m core.worker --telegram

What a job can do (honest scope): write something with an AI (optionally as a PDF),
remind you, or cut a video clip with ffmpeg. It CANNOT trade, attend a meeting,
generate or upload videos, or operate your desktop; those are not in this queue.
"""
from __future__ import annotations

import json
import re
import shutil
import sqlite3
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent.parent / "memory" / "jobs.db"
MAX_ATTEMPTS = 3
RETRY_DELAY_S = 300
JOB_TIMEOUT_S = 900

# ── time phrases ────────────────────────────────────────────────────────────
_WD = {"monday": 0, "mon": 0, "tuesday": 1, "tues": 1, "tue": 1, "wednesday": 2, "wed": 2,
       "thursday": 3, "thurs": 3, "thur": 3, "thu": 3, "friday": 4, "fri": 4, "saturday": 5,
       "sat": 5, "sunday": 6, "sun": 6}
_WD_RE = "|".join(sorted(_WD, key=len, reverse=True))
_UNIT = {"m": 1, "min": 1, "mins": 1, "minute": 1, "minutes": 1, "h": 60, "hr": 60, "hrs": 60, "hour": 60,
         "hours": 60, "d": 1440, "day": 1440, "days": 1440, "w": 10080, "week": 10080, "weeks": 10080}
_DAYPART = {"morning": (9, 0), "noon": (12, 0), "afternoon": (15, 0), "evening": (18, 0),
            "tonight": (20, 0), "night": (21, 0), "midnight": (0, 0)}


@dataclass
class When:
    run_at: datetime
    repeat: str | None = None


def _time_of(s: str) -> tuple[tuple[int, int] | None, tuple[int, int] | None]:
    """((hour, minute), span) for the first time-of-day phrase in s, else (None, None)."""
    m = re.search(r"\b(?:at\s+)?(\d{1,2})(?::(\d{2}))?\s*(am|pm)\b", s)
    if m:
        h, mi = int(m.group(1)), int(m.group(2) or 0)
        if h == 12:
            h = 0
        return ((h + (12 if m.group(3) == "pm" else 0)) % 24, mi), m.span()
    m = re.search(r"\b(?:at\s+)?(\d{1,2}):(\d{2})\b", s) or re.search(r"\bat\s+(\d{1,2})\b", s)
    if m:
        return (int(m.group(1)) % 24, int(m.group(2)) if m.lastindex and m.lastindex >= 2 else 0), m.span()
    m = re.search(r"\b(?:in the\s+|this\s+)?(morning|noon|afternoon|evening|tonight|night|midnight)\b", s)
    if m:
        return _DAYPART[m.group(1)], m.span()
    return None, None


def parse_when(text: str, now: datetime | None = None) -> tuple[When | None, str]:
    """Find a time phrase in `text`. Returns (When or None, the text without that phrase)."""
    now = (now or datetime.now()).replace(second=0, microsecond=0)
    s = text.lower()
    spans: list[tuple[int, int]] = []
    repeat = None
    base_day = None
    out_time = None

    def take(m):
        spans.append(m.span())

    if (m := re.search(r"\bevery\s+(\d+)\s*(minutes?|mins?|hours?|hrs?|days?)\b", s)):
        n = int(m.group(1)) * _UNIT[m.group(2)]
        repeat = f"every:{n}"
        take(m)
        return When(now + timedelta(minutes=n), repeat), _strip(text, spans)
    if (m := re.search(r"\b(hourly|every hour)\b", s)):
        take(m)
        return When(now + timedelta(hours=1), "every:60"), _strip(text, spans)
    if (m := re.search(r"\b(every\s+weekday|weekdays)\b", s)):
        repeat = "weekdays"
        take(m)
    elif (m := re.search(rf"\bevery\s+({_WD_RE})\b", s)):
        repeat = f"weekly:{_WD[m.group(1)]}"
        base_day = _WD[m.group(1)]
        take(m)
    elif (m := re.search(r"\b(every\s*day|daily|each day)\b", s)):
        repeat = "daily"
        take(m)

    if (m := re.search(r"\bin\s+(\d+)\s*(minutes?|mins?|hours?|hrs?|days?|weeks?|[mhdw])\b", s)):
        take(m)
        return When(now + timedelta(minutes=int(m.group(1)) * _UNIT[m.group(2)]), repeat), _strip(text, spans)
    if (m := re.search(r"\b(right now|now|asap|immediately)\b", s)) and not repeat:
        take(m)
        return When(now, None), _strip(text, spans)

    day = None
    if (m := re.search(r"\bday after tomorrow\b", s)):
        day, _ = now.date() + timedelta(days=2), take(m)
    elif (m := re.search(r"\btomorrow\b", s)):
        day, _ = now.date() + timedelta(days=1), take(m)
    elif (m := re.search(r"\btoday\b", s)):
        day, _ = now.date(), take(m)
    elif (m := re.search(r"\b(\d{4})-(\d{2})-(\d{2})\b", s)):
        try:
            day = datetime(int(m.group(1)), int(m.group(2)), int(m.group(3))).date()
            take(m)
        except ValueError:
            pass
    elif base_day is None and (m := re.search(rf"\b(?:on\s+|next\s+|this\s+)?({_WD_RE})\b", s)):
        base_day = _WD[m.group(1)]
        take(m)

    tm, span = _time_of(s)
    if span:
        spans.append(span)
        out_time = tm
    if day is None and base_day is None and repeat is None and out_time is None:
        return None, text
    if out_time is None:
        out_time = (9, 0)
    h, mi = out_time
    if day is None and base_day is not None:
        ahead = (base_day - now.weekday()) % 7
        cand = datetime.combine(now.date() + timedelta(days=ahead), datetime.min.time()).replace(hour=h, minute=mi)
        day = (cand.date() if cand > now else cand.date() + timedelta(days=7))
    if day is None:
        cand = datetime.combine(now.date(), datetime.min.time()).replace(hour=h, minute=mi)
        day = cand.date() if cand > now else cand.date() + timedelta(days=1)
    run = datetime.combine(day, datetime.min.time()).replace(hour=h, minute=mi)
    if repeat == "weekdays":
        while run.weekday() > 4:
            run += timedelta(days=1)
    return When(run, repeat), _strip(text, spans)


def _strip(text: str, spans: list[tuple[int, int]]) -> str:
    out = text
    for a, b in sorted(spans, reverse=True):
        out = out[:a] + " " + out[b:]
    out = re.sub(r"\b(?:at|on|by|for|next|this)\s*$", "", out.strip(), flags=re.I)
    out = re.sub(r"^\s*(?:at|on|by|for)\s+", "", out, flags=re.I)
    return re.sub(r"\s{2,}", " ", out).strip(" ,.-")


def next_run(repeat: str, last: datetime, now: datetime) -> datetime:
    """The next time a repeating job is due: never in the past, never several catch-ups."""
    kind, _, arg = repeat.partition(":")
    if kind == "every":
        nxt = last + timedelta(minutes=int(arg))
        return nxt if nxt > now else now + timedelta(minutes=int(arg))
    step = {"daily": 1, "weekdays": 1, "weekly": 7}[kind]
    nxt = last + timedelta(days=step)
    while nxt <= now or (kind == "weekdays" and nxt.weekday() > 4):
        nxt += timedelta(days=step)
    return nxt


def describe_when(ts: float, now: datetime | None = None) -> str:
    now = now or datetime.now()
    d = datetime.fromtimestamp(ts)
    if d.date() == now.date():
        return f"today {d:%H:%M}"
    if d.date() == (now + timedelta(days=1)).date():
        return f"tomorrow {d:%H:%M}"
    return f"{d:%a %d %b %H:%M}"


# ── storage ─────────────────────────────────────────────────────────────────
_SCHEMA = """CREATE TABLE IF NOT EXISTS jobs(
  id INTEGER PRIMARY KEY AUTOINCREMENT, title TEXT NOT NULL, kind TEXT NOT NULL DEFAULT 'ai',
  payload TEXT NOT NULL DEFAULT '{}', run_at REAL NOT NULL, repeat TEXT, status TEXT NOT NULL DEFAULT 'queued',
  created REAL NOT NULL, started REAL, finished REAL, result TEXT, file TEXT, error TEXT, attempts INTEGER NOT NULL DEFAULT 0)"""


class JobStore:
    def __init__(self, path: Path | str | None = None) -> None:
        self.path = Path(path) if path else DB_PATH
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._c() as c:
            c.execute("PRAGMA journal_mode=WAL")
            c.execute(_SCHEMA)

    def _c(self) -> sqlite3.Connection:
        c = sqlite3.connect(self.path, timeout=30, isolation_level=None)
        c.row_factory = sqlite3.Row
        return c

    @staticmethod
    def _row(r) -> dict | None:
        if r is None:
            return None
        d = dict(r)
        d["payload"] = json.loads(d["payload"] or "{}")
        return d

    def add(self, title: str, kind: str = "ai", payload: dict | None = None, run_at: float | None = None,
            repeat: str | None = None) -> dict:
        title = (title or "").strip()
        if not title:
            raise ValueError("a job needs a description")
        if kind not in HANDLERS:
            raise ValueError(f"unknown job kind '{kind}' (use: {', '.join(HANDLERS)})")
        now = time.time()
        with self._c() as c:
            cur = c.execute("INSERT INTO jobs(title,kind,payload,run_at,repeat,created) VALUES(?,?,?,?,?,?)",
                            (title[:300], kind, json.dumps(payload or {}), run_at or now, repeat, now))
            return self.get(cur.lastrowid)

    def get(self, jid: int) -> dict | None:
        with self._c() as c:
            return self._row(c.execute("SELECT * FROM jobs WHERE id=?", (int(jid),)).fetchone())

    def list(self, status: str | None = None, limit: int = 100) -> list[dict]:
        q, args = "SELECT * FROM jobs", []
        if status:
            q += " WHERE status=?"
            args.append(status)
        q += " ORDER BY CASE status WHEN 'running' THEN 0 WHEN 'queued' THEN 1 ELSE 2 END, " \
             "CASE WHEN status IN ('queued','running') THEN run_at ELSE -finished END LIMIT ?"
        with self._c() as c:
            return [self._row(r) for r in c.execute(q, (*args, limit)).fetchall()]

    def claim_due(self, now: float | None = None) -> dict | None:
        """Atomically take the oldest due job (two runners can never get the same one)."""
        now = now or time.time()
        c = self._c()
        try:
            c.execute("BEGIN IMMEDIATE")
            r = c.execute("SELECT id FROM jobs WHERE status='queued' AND run_at<=? ORDER BY run_at LIMIT 1", (now,)).fetchone()
            if r is None:
                c.execute("COMMIT")
                return None
            c.execute("UPDATE jobs SET status='running', started=?, attempts=attempts+1 WHERE id=?", (now, r["id"]))
            c.execute("COMMIT")
            return self.get(r["id"])
        except Exception:
            try:
                c.execute("ROLLBACK")
            except Exception:
                pass
            raise
        finally:
            c.close()

    def finish(self, jid: int, result: str = "", file: str | None = None) -> None:
        with self._c() as c:
            c.execute("UPDATE jobs SET status='done', finished=?, result=?, file=?, error=NULL WHERE id=? AND status='running'",
                      (time.time(), result[:20000], file, jid))

    def fail(self, jid: int, error: str, retry_at: float | None = None) -> None:
        with self._c() as c:
            if retry_at:
                c.execute("UPDATE jobs SET status='queued', run_at=?, error=? WHERE id=?", (retry_at, error[:500], jid))
            else:
                c.execute("UPDATE jobs SET status='failed', finished=?, error=? WHERE id=?", (time.time(), error[:500], jid))

    def reschedule(self, jid: int, run_at: float) -> None:
        with self._c() as c:
            c.execute("UPDATE jobs SET status='queued', run_at=?, attempts=0 WHERE id=?", (run_at, jid))

    def cancel(self, jid: int) -> bool:
        with self._c() as c:
            return c.execute("UPDATE jobs SET status='cancelled', finished=? WHERE id=? AND status IN ('queued','running')",
                             (time.time(), int(jid))).rowcount > 0

    def run_now(self, jid: int) -> bool:
        with self._c() as c:
            return c.execute("UPDATE jobs SET run_at=?, status='queued', attempts=0 WHERE id=? AND status IN ('queued','failed','cancelled','done')",
                             (time.time(), int(jid))).rowcount > 0

    def delete_finished(self) -> int:
        with self._c() as c:
            return c.execute("DELETE FROM jobs WHERE status IN ('done','cancelled','failed')").rowcount

    def recover(self) -> int:
        """Jobs left 'running' by a crash or a shutdown: put them back (or fail them after 3 tries)."""
        with self._c() as c:
            c.execute("UPDATE jobs SET status='failed', finished=?, error='interrupted too many times' WHERE status='running' AND attempts>=?",
                      (time.time(), MAX_ATTEMPTS))
            return c.execute("UPDATE jobs SET status='queued' WHERE status='running'").rowcount


# ── notifications ───────────────────────────────────────────────────────────
class Notifier:
    """Telegram messages. Silent no-op (returns False) when Telegram is not set up."""

    def __init__(self, token: str = "", chat_id: str = "") -> None:
        self.token, self.chat_id = token, str(chat_id or "")

    @classmethod
    def from_config(cls) -> "Notifier":
        try:
            from core.ai import config as C
            cfg = json.loads(C.LEGACY_FILE.read_text(encoding="utf-8")) if C.LEGACY_FILE.exists() else {}
            return cls(C.get_api_key("telegram"), str(cfg.get("telegram_chat_id", "")))
        except Exception:
            return cls()

    @property
    def ready(self) -> bool:
        return bool(self.token and self.chat_id)

    def send(self, text: str, files: list[str] | None = None) -> bool:
        if not self.ready:
            return False
        try:
            import requests
            r = requests.post(f"https://api.telegram.org/bot{self.token}/sendMessage",
                              data={"chat_id": self.chat_id, "text": text[:3900]}, timeout=20)
            ok = r.status_code == 200
            if ok and files:
                from core import reports
                reports.send_telegram(self.token, self.chat_id, reports.check_files(files))
            return ok
        except Exception as e:
            from core.ai.redact import redact
            print(f"[Jobs] Telegram notify failed: {redact(e)[:120]}")
            return False


    def _api(self, method: str, **data) -> dict:
        import requests
        r = requests.post(f"https://api.telegram.org/bot{self.token}/{method}", json=data, timeout=20)
        return r.json() if r.status_code == 200 else {}

    def send_buttons(self, text: str, rows: list[list[tuple[str, str]]]) -> int | None:
        """Message with tap buttons. rows = [[(label, callback_data), ...]]. Returns the message id."""
        if not self.ready:
            return None
        try:
            r = self._api("sendMessage", chat_id=self.chat_id, text=text[:3900],
                          reply_markup={"inline_keyboard": [[{"text": a, "callback_data": b} for a, b in row] for row in rows]})
            return (r.get("result") or {}).get("message_id")
        except Exception as e:
            from core.ai.redact import redact
            print(f"[Jobs] Telegram buttons failed: {redact(e)[:120]}")
            return None

    def answer_callback(self, callback_id: str, text: str = "") -> None:
        try:
            self._api("answerCallbackQuery", callback_query_id=callback_id, text=text[:180])
        except Exception:
            pass

    def edit_text(self, message_id: int, text: str) -> None:
        try:
            self._api("editMessageText", chat_id=self.chat_id, message_id=message_id, text=text[:3900])
        except Exception:
            pass


# ── handlers: what a job can actually do ────────────────────────────────────
_AI_SYSTEM = ("You are completing a scheduled task for the user while they are away. Produce the finished work "
              "directly: no questions, no preamble. Be concrete and well organised. You have NO internet access and "
              "cannot act outside this text, so when something needs live data (prices, news, schedules) say so "
              "and base the work on general knowledge, clearly labelled. Never claim to have done anything else.")


def _h_ai(job: dict, ctx: "Context") -> tuple[str, str | None]:
    from core.ai.types import Request
    p = job["payload"]
    prompt = str(p.get("prompt") or job["title"])
    r = ctx.hub().ask(Request(messages=[{"role": "user", "content": prompt}], system=_AI_SYSTEM,
                              task="plan" if len(prompt) > 200 else "chat", min_tier=1,
                              max_output_tokens=int(p.get("max_tokens", 1800)), timeout=120, cacheable=False))
    text = r.text.strip()
    if p.get("pdf") or len(text) > 2500:
        from core import reports
        return text, reports.make_pdf(job["title"][:80], text)
    return text, None


def _h_remind(job: dict, ctx: "Context") -> tuple[str, str | None]:
    return str(job["payload"].get("text") or job["title"]), None


def _ts(v) -> str:
    v = str(v).strip()
    if not re.fullmatch(r"\d{1,2}(:\d{2}){0,2}(\.\d+)?|\d+(\.\d+)?", v):
        raise ValueError(f"'{v}' is not a time like 00:01:30")
    return v


def _h_clip(job: dict, ctx: "Context") -> tuple[str, str | None]:
    p = job["payload"]
    ff = shutil.which("ffmpeg")
    if not ff:
        raise RuntimeError("ffmpeg is not installed on this machine (needed for video clips)")
    src = Path(str(p.get("input", ""))).expanduser()
    if not src.is_file():
        raise ValueError(f"I cannot find the video: {src.name or p.get('input')}")
    start, end = _ts(p.get("start", "0")), _ts(p["end"])
    out_dir = Path.home() / "Documents" / "Ephraim Reports" / "clips"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{src.stem}_{start.replace(':', '-')}_{end.replace(':', '-')}.mp4"
    r = subprocess.run([ff, "-y", "-ss", start, "-to", end, "-i", str(src), "-c:v", "libx264", "-c:a", "aac", str(out)],
                       capture_output=True, text=True, timeout=JOB_TIMEOUT_S)
    if r.returncode != 0 or not out.exists():
        raise RuntimeError("ffmpeg failed: " + (r.stderr.strip().splitlines() or ["unknown error"])[-1][:160])
    return f"Clip saved: {out.name}", str(out)


def _h_video(job: dict, ctx: "Context") -> tuple[str, str | None]:
    from core import video_maker
    p = job["payload"]
    return video_maker.make_video(ctx.hub(), str(p.get("topic") or job["title"]), int(p.get("seconds", 45)), str(p.get("style", "")))


HANDLERS = {"ai": _h_ai, "remind": _h_remind, "clip": _h_clip, "video": _h_video}


@dataclass
class Context:
    hub: callable


# ── the runner ──────────────────────────────────────────────────────────────
class JobRunner:
    def __init__(self, store: JobStore, hub=None, notifier: Notifier | None = None, on_event=None,
                 interval: float = 5.0, workers: int = 2, clock=time.time) -> None:
        self.store, self.notifier = store, notifier or Notifier()
        self._hub = hub
        self.on_event = on_event or (lambda kind, job: None)
        self.interval, self._clock = interval, clock
        self._stop = threading.Event()
        self._pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="job")
        self._thread: threading.Thread | None = None
        self._ctx = Context(hub=self._get_hub)

    def _get_hub(self):
        if self._hub is None:
            from core.ai import get_hub
            self._hub = get_hub()
        return self._hub

    def start(self) -> None:
        self.store.recover()
        self._thread = threading.Thread(target=self._loop, daemon=True, name="job-runner")
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._pool.shutdown(wait=False, cancel_futures=True)

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                if not self.tick():
                    self._stop.wait(self.interval)
            except Exception as e:
                print(f"[Jobs] runner error: {type(e).__name__}: {e}")
                self._stop.wait(self.interval)

    def tick(self) -> bool:
        """Claim and start one due job. Returns True if there was one (so the loop can look again)."""
        job = self.store.claim_due(self._clock())
        if job is None:
            return False
        self._pool.submit(self._run, job)
        return True

    def run_job(self, job: dict) -> None:           # synchronous; used by tests and --once
        self._run(job)

    def _run(self, job: dict) -> None:
        late = (self._clock() - job["run_at"]) > 120
        self.on_event("started", job)
        try:
            text, file = HANDLERS[job["kind"]](job, self._ctx)
        except Exception as e:
            from core.ai.redact import redact
            err = f"{type(e).__name__}: {redact(e)}"[:300]
            # Bad input or a missing tool (ValueError / RuntimeError) will not fix itself on retry;
            # everything else (a provider outage, a timeout) gets up to MAX_ATTEMPTS tries, 5 min apart.
            transient = job["attempts"] < MAX_ATTEMPTS and not isinstance(e, (ValueError, RuntimeError))
            if transient:
                self.store.fail(job["id"], err, retry_at=self._clock() + RETRY_DELAY_S)
                self.on_event("retry", self.store.get(job["id"]))
                return
            self.store.fail(job["id"], err)
            j = self.store.get(job["id"])
            self.on_event("failed", j)
            self.notifier.send(f"❌ Job #{job['id']} failed: {job['title']}\n{err}")
            return
        self.store.finish(job["id"], text, file)
        j = self.store.get(job["id"])
        if job.get("repeat"):
            nxt = next_run(job["repeat"], datetime.fromtimestamp(job["run_at"]), datetime.fromtimestamp(self._clock()))
            clone = self.store.add(job["title"], job["kind"], job["payload"], nxt.timestamp(), job["repeat"])
            j["next_id"] = clone["id"]
        self.on_event("done", j)
        head = f"✅ Job #{job['id']} done{' (ran late — the computer was off)' if late else ''}: {job['title']}"
        if job["kind"] == "remind":
            self.notifier.send_buttons(f"⏰ Reminder: {text}" + (" (sent late — the computer was off)" if late else ""), [[("✅ Done", f"rm:{job['id']}:d"), ("💤 10 min", f"rm:{job['id']}:s10"),
                                                              ("💤 1 hour", f"rm:{job['id']}:s60")]])
        else:
            self.notifier.send(f"{head}\n\n{text[:2500]}", [file] if file else None)
        if job["payload"].get("approve"):                         # "…and ask me to approve it"
            try:
                from core.approvals import Approvals
                Approvals(self.store.path, self.notifier).request(f"Accept the result of job #{job['id']}?", f"{job['title']}\n\n{text[:700]}")
            except Exception as e:
                print(f"[Jobs] approval request failed: {type(e).__name__}")


# ── turning words into a job ────────────────────────────────────────────────
def add_from_text(store: JobStore, text: str, now: datetime | None = None) -> dict:
    """"remind me to X at 5pm" / "write a study plan every weekday 7am" -> a stored job."""
    when, rest = parse_when(text, now)
    task = rest or text.strip()
    kind, payload = "ai", {"prompt": task}
    m = re.match(r"^(?:please\s+)?remind(?:\s+me)?(?:\s+to)?\s+(.+)$", task, re.I)
    v = re.match(r"^(?:please\s+)?(?:make|create|generate|produce)\s+(?:me\s+)?(?:an?\s+)?(?:ai\s+)?(?:\d{2,3}[- ]?(?:seconds?|secs?)\s+)?(?:short\s+|explainer\s+)?video(?:\s+(?:about|on|of|explaining))?\s+(.+)$", task, re.I)
    sec = re.search(r"\b(\d{2,3})[- ]?(?:second|sec|s)\b", task, re.I)
    if m:
        kind, payload = "remind", {"text": m.group(1).strip()}
    elif v:
        kind, payload = "video", {"topic": re.sub(r"\b\d{2,3}[- ]?(?:second|sec|s)\b", "", v.group(1), flags=re.I).strip(" ,.-"),
                                  "seconds": int(sec.group(1)) if sec else 45}
    elif re.search(r"\b(as a |into a |in a )?(pdf|report)\b", task, re.I):
        payload["pdf"] = True
    if re.search(r"\b(for (my )?approval|ask me to approve|needs? my approval|let me approve)\b", task, re.I):
        payload["approve"] = True
        task = re.sub(r"\s*,?\s*\b(and )?(for (my )?approval|ask me to approve( it)?|needs? my approval|let me approve( it)?)\b", "", task, flags=re.I).strip() or task
        payload["prompt"] = payload.get("prompt", task) if kind == "ai" else payload.get("prompt")
        if kind == "ai":
            payload["prompt"] = task
    ts = (when.run_at.timestamp() if when else time.time())
    return store.add(task, kind, payload, ts, when.repeat if when else None)


def announce(job: dict, notifier: Notifier) -> bool:
    """Tell the user on Telegram that a job was queued."""
    rep = f" (repeats {job['repeat'].replace(':', ' ')})" if job.get("repeat") else ""
    return notifier.send(f"📋 Job #{job['id']} queued: {job['title']}\n⏰ {describe_when(job['run_at'])}{rep}")


# ── Telegram inbox (phone -> queue, works while the laptop is off if this runs elsewhere) ──
class TelegramInbox:
    HELP = ("Send me a task and when, e.g. 'write a study plan for Python tomorrow 7am'.\n"
            "/jobs  list   ·  /result N  ·  /cancel N  ·  /run N  ·  /help")

    def __init__(self, store: JobStore, notifier: Notifier, http=None) -> None:
        self.store, self.n = store, notifier
        self.offset = 0
        self._http = http

    def handle(self, text: str) -> str:
        t = (text or "").strip()
        if not t or t.startswith("/start") or t.startswith("/help"):
            return self.HELP
        m = re.match(r"^/(jobs|result|cancel|run)\s*(\d+)?", t)
        if m:
            cmd, n = m.group(1), m.group(2)
            if cmd == "jobs":
                rows = self.store.list(limit=8)
                return "\n".join(f"#{j['id']} [{j['status']}] {describe_when(j['run_at'])} — {j['title'][:60]}" for j in rows) or "No jobs."
            if not n:
                return f"Usage: /{cmd} <job number>"
            j = self.store.get(int(n))
            if j is None:
                return f"No job #{n}."
            if cmd == "result":
                return (j["result"] or j["error"] or f"Job #{n} is {j['status']}.")[:3500]
            if cmd == "cancel":
                return f"Cancelled #{n}." if self.store.cancel(int(n)) else f"#{n} cannot be cancelled ({j['status']})."
            return f"Running #{n} now." if self.store.run_now(int(n)) else f"#{n} cannot run now ({j['status']})."
        task = re.sub(r"^/task\s*", "", t)
        job = add_from_text(self.store, task)
        rep = f" (repeats {job['repeat'].replace(':', ' ')})" if job.get("repeat") else ""
        return f"📋 Queued #{job['id']}: {job['title']}\n⏰ {describe_when(job['run_at'])}{rep}"

    def _tap(self, cq: dict) -> None:
        """A button on one of our messages was pressed."""
        kind, _, rest = str(cq.get("data", "")).partition(":")
        a, _, b = rest.partition(":")
        msg_id = (cq.get("message") or {}).get("message_id")
        try:
            if kind == "ap" and a.isdigit():                              # approval buttons
                from core.approvals import Approvals
                say = Approvals(self.store.path, self.n).decide(int(a), b == "y")
            elif kind == "rm" and a.isdigit():                            # reminder buttons
                j = self.store.get(int(a))
                if j is None:
                    say = "That reminder is gone."
                elif b == "d":
                    say = "Marked done."
                else:
                    mins = 10 if b == "s10" else 60
                    self.store.add(j["title"], "remind", j["payload"], time.time() + mins * 60)
                    say = f"Snoozed for {mins} min."
            elif kind == "cf" and a:                                      # the on-screen confirmation gate
                from core import confirm
                say = ("Confirmed." if b == "y" else "Cancelled.") if confirm.resolve_remote(a, b == "y") \
                    else "That request has expired or was already answered."
            else:
                say = "Unknown button."
        except Exception as e:
            say = f"Could not do that ({type(e).__name__})."
        self.n.answer_callback(cq.get("id", ""), say[:150])
        if msg_id:
            self.n.edit_text(msg_id, f"{(cq.get('message') or {}).get('text', '')[:700]}\n\n➡ {say}")

    def poll_once(self, timeout: int = 25) -> int:
        """One long-poll. ONLY messages from the configured chat are obeyed."""
        import requests
        http = self._http or requests
        r = http.get(f"https://api.telegram.org/bot{self.n.token}/getUpdates",
                     params={"offset": self.offset, "timeout": timeout}, timeout=timeout + 10)
        handled = 0
        for u in (r.json().get("result") or []):
            self.offset = max(self.offset, int(u["update_id"]) + 1)
            cq = u.get("callback_query")
            if cq:
                if str(((cq.get("message") or {}).get("chat") or {}).get("id", "")) == self.n.chat_id:   # only YOUR chat
                    self._tap(cq)
                    handled += 1
                continue
            msg = u.get("message") or {}
            if str((msg.get("chat") or {}).get("id", "")) != self.n.chat_id or not msg.get("text"):
                continue                                  # a stranger messaging your bot: ignored
            self.n.send(self.handle(msg["text"]))
            handled += 1
        return handled


# ── who polls Telegram: the worker if one is running, otherwise the app ──────────────────
LOCK_PATH = DB_PATH.parent / "worker.lock"


def worker_alive(path: Path | None = None, now: float | None = None) -> bool:
    """True if a headless worker has checked in within the last minute (it then owns Telegram,
    because two programs polling one bot make Telegram reject both)."""
    try:
        return (now or time.time()) - float((path or LOCK_PATH).read_text().strip()) < 60
    except Exception:
        return False


def touch_worker_lock(path: Path | None = None) -> None:
    try:
        (path or LOCK_PATH).write_text(str(time.time()))
    except OSError:
        pass


class TelegramPoller(threading.Thread):
    """Background Telegram listener for the desktop app (button taps and tasks from your phone)."""

    def __init__(self, store: JobStore, notifier: Notifier, lock_path: Path | None = None, http=None) -> None:
        super().__init__(daemon=True, name="telegram-poller")
        self.inbox = TelegramInbox(store, notifier, http=http)
        self.lock_path = lock_path
        self._stop_ev = threading.Event()

    def stop(self) -> None:
        self._stop_ev.set()

    def run(self) -> None:
        while not self._stop_ev.is_set():
            if worker_alive(self.lock_path):
                self._stop_ev.wait(15)
                continue
            try:
                self.inbox.poll_once(timeout=20)
            except Exception:
                self._stop_ev.wait(10)
