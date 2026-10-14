import json
import os
import threading
import time
import types
from datetime import datetime, timedelta

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import pytest
from core import jobs

try:                                                    # one real QApplication for every Qt test in this file
    from PyQt6.QtWidgets import QApplication
    _APP = QApplication.instance() or QApplication([])
except ImportError:
    _APP = None
from core.jobs import (JobRunner, JobStore, Notifier, TelegramInbox, add_from_text, next_run, parse_when)

SUN = datetime(2026, 10, 11, 10, 0)        # a Sunday


@pytest.fixture
def store(tmp_path):
    return JobStore(tmp_path / "jobs.db")


class Rec(Notifier):
    def __init__(self):
        super().__init__("tok", "42"); self.sent = []; self.buttons = []
    def send(self, text, files=None):
        self.sent.append((text, files)); return True
    def send_buttons(self, text, rows):
        self.buttons.append((text, rows)); return 1


# ── time phrases ───────────────────────────────────────────────────────────
@pytest.mark.parametrize("text,when,repeat,rest", [
    ("summarise forex lessons tomorrow 8am", "2026-10-12 08:00", None, "summarise forex lessons"),
    ("remind me at 5pm to submit report", "2026-10-11 17:00", None, "remind me to submit report"),
    ("remind me at 9am to call", "2026-10-12 09:00", None, "remind me to call"),          # 9am already passed today
    ("every weekday at 7am write my study plan", "2026-10-12 07:00", "weekdays", "write my study plan"),
    ("in 2 hours clip demo", "2026-10-11 12:00", None, "clip demo"),
    ("in 45 min stretch", "2026-10-11 10:45", None, "stretch"),
    ("on friday at 6:30pm review code", "2026-10-16 18:30", None, "review code"),
    ("every monday 7pm gym plan", "2026-10-12 19:00", "weekly:0", "gym plan"),
    ("2026-10-20 14:00 mock exam", "2026-10-20 14:00", None, "mock exam"),
    ("do homework tonight", "2026-10-11 20:00", None, "do homework"),
    ("every 30 minutes check list", "2026-10-11 10:30", "every:30", "check list"),
    ("daily at 6am read news", "2026-10-12 06:00", "daily", "read news"),
    ("12am backup", "2026-10-12 00:00", None, "backup"), ("12pm lunch", "2026-10-11 12:00", None, "lunch"),
])
def test_time_phrases(text, when, repeat, rest):
    w, r = parse_when(text, SUN)
    assert w is not None and w.run_at.strftime("%Y-%m-%d %H:%M") == when and w.repeat == repeat and r == rest


@pytest.mark.parametrize("text", ["just do it", "top 5 stocks to study", "learn python in depth", "chapter 3 summary"])
def test_no_time_means_no_time(text):
    w, r = parse_when(text, SUN)
    assert w is None and r == text


def test_repeat_never_catches_up_or_lands_in_the_past():
    now = datetime(2026, 10, 14, 12, 0)
    assert next_run("daily", datetime(2026, 10, 11, 7), now) == datetime(2026, 10, 15, 7)        # laptop was off 3 days: ONE next run
    assert next_run("weekdays", datetime(2026, 10, 16, 7), datetime(2026, 10, 16, 8)) == datetime(2026, 10, 19, 7)   # Fri -> Mon
    assert next_run("weekly:0", datetime(2026, 10, 12, 19), datetime(2026, 10, 12, 20)) == datetime(2026, 10, 19, 19)
    assert next_run("every:30", datetime(2026, 10, 11, 10), datetime(2026, 10, 11, 12)) == datetime(2026, 10, 11, 12, 30)


# ── store ──────────────────────────────────────────────────────────────────
def test_two_runners_can_never_claim_the_same_job(store):
    for i in range(30):
        store.add(f"job {i}", "remind", {"text": "x"}, time.time() - 1)
    claimed, lock = [], threading.Lock()
    def grab():
        while (j := store.claim_due()) is not None:
            with lock: claimed.append(j["id"])
    ts = [threading.Thread(target=grab) for _ in range(6)]
    [t.start() for t in ts]; [t.join() for t in ts]
    assert len(claimed) == 30 and len(set(claimed)) == 30


def test_future_jobs_wait_and_order_is_by_time(store):
    now = time.time()
    store.add("later", "remind", {}, now + 3600); a = store.add("first", "remind", {}, now - 50); b = store.add("second", "remind", {}, now - 10)
    assert store.claim_due()["id"] == a["id"] and store.claim_due()["id"] == b["id"] and store.claim_due() is None


def test_unknown_kind_and_empty_title_are_refused(store):
    with pytest.raises(ValueError): store.add("x", "trade_forex")
    with pytest.raises(ValueError): store.add("   ")


def test_crash_recovery_requeues_then_gives_up(store):
    j = store.add("x", "remind", {}, time.time() - 1)
    store.claim_due()
    assert store.recover() == 1 and store.get(j["id"])["status"] == "queued"
    for _ in range(3):
        store.claim_due(); store.recover()
    assert store.get(j["id"])["status"] == "failed" and "interrupted" in store.get(j["id"])["error"]


def test_cancel_run_now_and_clear(store):
    a = store.add("a", "remind", {}, time.time() + 999)
    assert store.cancel(a["id"]) and not store.cancel(a["id"]) and store.get(a["id"])["status"] == "cancelled"
    assert store.run_now(a["id"]) and store.claim_due()["id"] == a["id"]
    store.finish(a["id"], "ok"); assert store.delete_finished() == 1 and store.list() == []


# ── runner ─────────────────────────────────────────────────────────────────
def runner(store, hub=None, n=None, clock=time.time):
    events = []
    return JobRunner(store, hub=hub, notifier=n or Rec(), on_event=lambda k, j: events.append((k, j["id"])), clock=clock), events


def fake_hub(text="Here is your plan.", fail=0):
    calls = {"n": 0}
    class H:
        def ask(self, req):
            calls["n"] += 1
            if calls["n"] <= fail:
                from core.ai.types import AIError; raise AIError("unavailable", "down")
            return types.SimpleNamespace(text=text)
    h = H(); h.calls = calls
    return h


def test_ai_job_runs_stores_result_and_notifies(store):
    n = Rec()
    r, ev = runner(store, fake_hub("Day 1: variables."), n)
    j = store.add("write a python study plan", "ai", {"prompt": "write a python study plan"}, time.time() - 1)
    r.tick(); r._pool.shutdown(wait=True)
    d = store.get(j["id"])
    assert d["status"] == "done" and d["result"] == "Day 1: variables." and ("done", j["id"]) in ev
    assert "✅ Job #1 done" in n.sent[0][0] and "Day 1: variables." in n.sent[0][0]


def test_long_ai_result_becomes_a_pdf_attachment(store, tmp_path):
    pytest.importorskip("PyQt6")
    n = Rec()
    r, _ = runner(store, fake_hub("Paragraph. " * 400), n)
    j = store.add("big report", "ai", {"prompt": "big report"}, time.time() - 1)
    r.run_job(store.claim_due())
    d = store.get(j["id"])
    assert d["file"] and d["file"].endswith(".pdf") and open(d["file"], "rb").read(4) == b"%PDF"
    assert n.sent[0][1] == [d["file"]]


def test_provider_outage_retries_then_succeeds(store):
    t = {"now": 1000.0}
    r, ev = runner(store, fake_hub("ok", fail=1), clock=lambda: t["now"])
    j = store.add("x", "ai", {"prompt": "x"}, 900)
    r.run_job(store.claim_due(t["now"]))
    d = store.get(j["id"]); assert d["status"] == "queued" and d["run_at"] == 1300 and "retry" in [k for k, _ in ev]
    t["now"] = 1301
    r.run_job(store.claim_due(t["now"]))
    assert store.get(j["id"])["status"] == "done"


def test_permanent_failure_after_three_tries_notifies(store):
    t = {"now": 1000.0}; n = Rec()
    r, _ = runner(store, fake_hub(fail=99), n, clock=lambda: t["now"])
    j = store.add("x", "ai", {"prompt": "x"}, 900)
    for _ in range(3):
        t["now"] += 400; r.run_job(store.claim_due(t["now"]))
    d = store.get(j["id"])
    assert d["status"] == "failed" and d["attempts"] == 3 and any("❌" in s[0] for s in n.sent)


def test_bad_input_fails_at_once_without_retrying(store):
    n = Rec()
    r, _ = runner(store, None, n)
    j = store.add("clip", "clip", {"input": "/nope.mp4", "end": "00:00:05"}, time.time() - 1)
    r.run_job(store.claim_due())
    assert store.get(j["id"])["status"] == "failed" and store.get(j["id"])["attempts"] == 1


def test_repeating_job_queues_exactly_one_next_run(store):
    base = datetime(2026, 10, 11, 7, 0).timestamp()
    t = {"now": datetime(2026, 10, 14, 12, 0).timestamp()}          # 3 days late: the computer was off
    n = Rec()
    r, _ = runner(store, None, n, clock=lambda: t["now"])
    j = store.add("daily plan", "remind", {"text": "plan"}, base, "daily")
    r.run_job(store.claim_due(t["now"]))
    rows = store.list()
    queued = [x for x in rows if x["status"] == "queued"]
    assert len(queued) == 1 and datetime.fromtimestamp(queued[0]["run_at"]) == datetime(2026, 10, 15, 7, 0)
    assert "sent late" in n.buttons[0][0]                                   # a reminder says it was late


def test_runner_thread_picks_up_due_jobs_and_stops(store):
    r, ev = runner(store, None)
    r.interval = 0.05
    j = store.add("ping", "remind", {"text": "hi"}, time.time())
    r.start()
    for _ in range(60):
        if store.get(j["id"])["status"] == "done": break
        time.sleep(0.05)
    r.stop()
    assert store.get(j["id"])["status"] == "done"


def test_clip_uses_ffmpeg_without_a_shell_and_validates_times(store, tmp_path, monkeypatch):
    v = tmp_path / "demo.mp4"; v.write_bytes(b"x")
    calls = []
    monkeypatch.setattr(jobs.shutil, "which", lambda n: "/usr/bin/ffmpeg")
    monkeypatch.setattr(jobs.Path, "home", staticmethod(lambda: tmp_path))
    def fake_run(cmd, **k):
        calls.append(cmd); out = tmp_path / "Documents" / "Ephraim Reports" / "clips" / cmd[-1].rsplit("/", 1)[-1]; out.write_bytes(b"clip")
        return types.SimpleNamespace(returncode=0, stderr="")
    monkeypatch.setattr(jobs.subprocess, "run", fake_run)
    r, _ = runner(store)
    j = store.add("clip", "clip", {"input": str(v), "start": "00:01:00", "end": "00:01:30"}, time.time() - 1)
    r.run_job(store.claim_due())
    assert store.get(j["id"])["status"] == "done" and isinstance(calls[0], list) and "-ss" in calls[0] and "shell" not in str(calls)
    evil = store.add("clip", "clip", {"input": str(v), "end": "5; rm -rf /"}, time.time() - 1)
    r.run_job(store.claim_due())
    assert store.get(evil["id"])["status"] == "failed" and len(calls) == 1


# ── text -> job, notifications, Telegram inbox ─────────────────────────────
def test_add_from_text_picks_kind_and_pdf(store):
    a = add_from_text(store, "remind me to submit the report at 5pm", SUN)
    assert a["kind"] == "remind" and a["payload"]["text"] == "submit the report"
    b = add_from_text(store, "summarise my notes as a PDF tomorrow 8am", SUN)
    assert b["kind"] == "ai" and b["payload"]["pdf"] is True
    c = add_from_text(store, "learn about candlestick charts", SUN)
    assert c["kind"] == "ai" and abs(c["run_at"] - time.time()) < 5          # no time -> now


def test_assignment_message_is_sent_to_telegram(store):
    n = Rec()
    j = add_from_text(store, "write study plan every weekday 7am", SUN)
    assert jobs.announce(j, n) and "📋 Job #1 queued" in n.sent[0][0] and "repeats weekdays" in n.sent[0][0]


def test_notifier_is_silent_when_telegram_is_not_set_up():
    assert Notifier().send("hi") is False and not Notifier("t", "").ready


def test_telegram_send_uses_the_bot_api_and_attaches_files(tmp_path, monkeypatch):
    import requests
    calls = []
    monkeypatch.setattr(requests, "post", lambda url, **k: calls.append(url.rsplit("/", 1)[1]) or types.SimpleNamespace(status_code=200, headers={}))
    f = tmp_path / "r.pdf"; f.write_bytes(b"%PDF")
    assert Notifier("TOKEN", "42").send("done", [str(f)]) and calls == ["sendMessage", "sendDocument"]


class FakeHTTP:
    def __init__(self, updates): self.updates = updates; self.params = []
    def get(self, url, params=None, timeout=None):
        self.params.append(params); return types.SimpleNamespace(json=lambda: {"result": self.updates})


def upd(i, chat, text): return {"update_id": i, "message": {"chat": {"id": chat}, "text": text}}


def test_inbox_only_obeys_your_own_chat(store):
    n = Rec()
    http = FakeHTTP([upd(1, 999, "delete everything every minute"), upd(2, 42, "write a study plan tomorrow 7am"), upd(3, 42, "/jobs")])
    inbox = TelegramInbox(store, n, http=http)
    assert inbox.poll_once(1) == 2
    assert [j["title"] for j in store.list()] == ["write a study plan"]          # the stranger queued nothing
    assert inbox.offset == 4 and "Queued #1" in n.sent[0][0] and "#1 [queued]" in n.sent[1][0]
    assert all("everything" not in s[0] for s in n.sent)


def test_inbox_commands(store):
    n = Rec(); inbox = TelegramInbox(store, n)
    j = add_from_text(store, "plan tomorrow 7am", SUN)
    assert "Cancelled" in inbox.handle(f"/cancel {j['id']}") and "cannot be cancelled" in inbox.handle(f"/cancel {j['id']}")
    assert "No job #99" in inbox.handle("/result 99") and "Usage" in inbox.handle("/run")
    assert "Send me a task" in inbox.handle("/help")


def test_worker_once_runs_what_is_due_and_exits(tmp_path, monkeypatch, capsys):
    from core import worker
    st = JobStore(tmp_path / "w.db")
    st.add("ping", "remind", {"text": "hi"}, time.time() - 1)
    monkeypatch.setattr(jobs, "DB_PATH", tmp_path / "w.db")
    monkeypatch.setattr(jobs.Notifier, "from_config", classmethod(lambda cls: Rec()))
    assert worker.main(["--once"]) == 0 and "ran 1 job" in capsys.readouterr().out
    assert JobStore(tmp_path / "w.db").list()[0]["status"] == "done"


# ── tool and panel ─────────────────────────────────────────────────────────
def test_tool_adds_lists_cancels_and_confirms_on_telegram(store, monkeypatch):
    import actions.job_queue as jq
    n = Rec()
    monkeypatch.setattr(jq, "JobStore", lambda: store); monkeypatch.setattr(jq.Notifier, "from_config", classmethod(lambda c: n))
    out = jq.job_queue({"mode": "add", "task": "write my study plan", "when": "tomorrow 7am"})
    assert "Queued job #1" in out and "tomorrow 07:00" in out and "Telegram" in out and n.sent
    assert "#1 queued" in jq.job_queue({"mode": "list"}) and "Cancelled job #1" in jq.job_queue({"mode": "cancel", "id": "1"})
    assert "What should I do" in jq.job_queue({"mode": "add"})


def test_tool_registers_cleanly():
    from pathlib import Path
    from core.action_loader import discover_actions
    d = {x["name"]: x for x in discover_actions(Path("actions"), logger=lambda s: None).get_tool_declarations()}
    assert set(d["job_queue"]) <= {"name", "description", "parameters"} and "not trading" in d["job_queue"]["description"]


def test_panel_adds_previews_and_shows_status(store):
    pytest.importorskip("PyQt6")
    from PyQt6.QtWidgets import QApplication
    QApplication.instance() or QApplication([])
    from core.jobs_panel import JobsPanel
    n, msgs = Rec(), []
    p = JobsPanel(store, n)
    p.status.connect(msgs.append)
    p.input.setText("write my study plan tomorrow 7am")
    assert "tomorrow 07:00" in p.when.text()
    job = p.add()
    assert job and p.table.rowCount() == 1 and "queued" in p.table.item(0, 3).text() and n.sent and "Telegram confirmation sent" in msgs[0]
    p.table.selectRow(0); p.run_now()
    claimed = store.claim_due(); store.finish(claimed["id"], "the plan")
    p.refresh()
    assert "done" in p.table.item(0, 3).text() and "the plan" in p.table.item(0, 4).text()
    p.clear(); assert p.table.rowCount() == 0
    p.input.setText("something with no time"); assert "run right away" in p.when.text()


def test_ui_wiring_exists_and_events_reach_the_chat(tmp_path):
    ui = pytest.importorskip("ui")
    st = JobStore(tmp_path / "u.db"); j = st.add("plan", "remind", {}, time.time()); st.claim_due(); st.finish(j["id"], "ok", None)
    log, files = [], []
    panel = types.SimpleNamespace(refresh=lambda: None)
    fake = types.SimpleNamespace(_job_store=st, _log=types.SimpleNamespace(append_log=log.append),
                                 _on_file=lambda p, n: files.append(p), _ensure_jobs_panel=lambda: panel)
    ui.MainWindow._on_job_event(fake, "done", j["id"])
    assert "✅ Job #1 finished" in log[0]
    src = open("main.py", encoding="utf-8").read()
    assert "ui.start_jobs()" in src


def test_a_headless_process_can_make_a_pdf_without_a_window(tmp_path):
    """Regression: with no Qt application object, Qt ABORTS the process. This runs the
    PDF writer in a fresh interpreter that has never created one, as the worker does."""
    import subprocess, sys
    out = tmp_path / "h.pdf"
    r = subprocess.run([sys.executable, "-c",
                        f"import sys; sys.path.insert(0, '.'); from core import reports; print(reports.make_pdf('Headless', 'It works.', out={str(out)!r}))"],
                       capture_output=True, text=True, timeout=60, env={**os.environ, "QT_QPA_PLATFORM": "offscreen"})
    assert r.returncode == 0, r.stderr[-300:]
    assert out.read_bytes().startswith(b"%PDF")
