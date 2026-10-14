import json
import time
import types

import pytest
from core import approvals as A
from core import confirm, jobs
from core.approvals import Approvals
from core.jobs import JobRunner, JobStore, Notifier, TelegramInbox, TelegramPoller, add_from_text, worker_alive


class Rec(Notifier):
    def __init__(self):
        super().__init__("tok", "42"); self.sent, self.buttons, self.answers, self.edits = [], [], [], []
    def send(self, text, files=None): self.sent.append((text, files)); return True
    def send_buttons(self, text, rows): self.buttons.append((text, rows)); return 900 + len(self.buttons)
    def answer_callback(self, cid, text=""): self.answers.append((cid, text))
    def edit_text(self, mid, text): self.edits.append((mid, text))


@pytest.fixture
def db(tmp_path):
    return tmp_path / "j.db"


def tap(data, chat=42, uid="cb1", text="orig"):
    return {"update_id": 1, "callback_query": {"id": uid, "data": data, "message": {"chat": {"id": chat}, "message_id": 7, "text": text}}}


class HTTP:
    def __init__(self, updates): self.u = updates
    def get(self, url, params=None, timeout=None): return types.SimpleNamespace(json=lambda: {"result": self.u})


def inbox(db, n, updates):
    return TelegramInbox(JobStore(db), n, http=HTTP(updates))


def test_request_asks_on_telegram_with_two_buttons(db):
    n = Rec(); ap = Approvals(db, n)
    a = ap.request("Send the report to Amina", "plan.pdf", "note")
    text, rows = n.buttons[0]
    assert "Approval needed" in text and [b[1] for b in rows[0]] == [f"ap:{a['id']}:y", f"ap:{a['id']}:n"] and a["message_id"] == 901


def test_only_named_actions_can_be_approved(db):
    with pytest.raises(ValueError, match="not an approvable action"):
        Approvals(db, Rec()).request("x", "y", "os.system('rm -rf /')")


def test_approve_runs_the_named_action_once(db):
    ran = []
    ap = Approvals(db, Rec(), actions={"note": lambda p: ran.append(p) or "did it"})
    a = ap.request("t", "d", "note", {"k": 1})
    assert "Approved" in ap.decide(a["id"], True) and "did it" in ap.get(a["id"])["result"] and ran == [{"k": 1}]
    assert "already done" in ap.decide(a["id"], True) and ran == [{"k": 1}]               # a second tap does nothing


def test_reject_does_nothing_and_cannot_be_reversed(db):
    ran = []
    ap = Approvals(db, Rec(), actions={"note": lambda p: ran.append(1) or "x"})
    a = ap.request("t", "d")
    assert "Nothing was done" in ap.decide(a["id"], False) and "already rejected" in ap.decide(a["id"], True) and ran == []


def test_an_expired_request_is_refused(db):
    ran = []
    ap = Approvals(db, Rec(), actions={"note": lambda p: ran.append(1) or "x"})
    a = ap.request("t", "d", ttl_hours=-1)
    assert "expired" in ap.decide(a["id"], True) and ran == [] and ap.pending() == []


def test_a_failing_action_is_reported_not_raised(db):
    def boom(p): raise RuntimeError("disk full")
    ap = Approvals(db, Rec(), actions={"note": boom})
    a = ap.request("t", "d")
    assert "failed" in ap.decide(a["id"], True) and ap.get(a["id"])["status"] == "failed"


def test_telegram_button_decides_and_the_message_is_updated(db):
    n = Rec(); ap = Approvals(db, n, actions={"note": lambda p: "sent!"})
    a = ap.request("Send it", "d")
    ib = inbox(db, n, [tap(f"ap:{a['id']}:y")])
    assert ib.poll_once(1) == 1
    assert ap.get(a["id"])["status"] == "done" and n.answers[0][0] == "cb1" and "Approved" in n.answers[0][1]
    assert n.edits[0][0] == 7 and "orig" in n.edits[0][1]


def test_a_stranger_cannot_tap_your_buttons(db):
    n = Rec(); ap = Approvals(db, n, actions={"note": lambda p: "sent!"})
    a = ap.request("Send it", "d")
    assert inbox(db, n, [tap(f"ap:{a['id']}:y", chat=999)]).poll_once(1) == 0
    assert ap.get(a["id"])["status"] == "pending" and n.answers == []


def test_garbage_button_data_is_harmless(db):
    n = Rec()
    for data in ("", "zz", "ap:abc:y", "ap::y", "rm:9999:d", "cf:nope:y", "ap:1:y:extra"):
        inbox(db, n, [tap(data)]).poll_once(1)
    assert len(n.answers) == 7


def test_reminder_buttons_snooze_and_complete(db):
    n = Rec(); st = JobStore(db)
    j = st.add("call the bank", "remind", {"text": "call the bank"}, time.time() - 1)
    r = JobRunner(st, notifier=n); r.run_job(st.claim_due())
    text, rows = n.buttons[0]
    assert "call the bank" in text and [b[1] for b in rows[0]] == [f"rm:{j['id']}:d", f"rm:{j['id']}:s10", f"rm:{j['id']}:s60"]
    TelegramInbox(st, n, http=HTTP([tap(f"rm:{j['id']}:s10")])).poll_once(1)
    nxt = [x for x in st.list() if x["status"] == "queued"]
    assert len(nxt) == 1 and 540 < nxt[0]["run_at"] - time.time() < 660 and "Snoozed" in n.answers[0][1]


def test_jobs_can_ask_for_your_approval_when_they_finish(db):
    n = Rec(); st = JobStore(db)
    job = add_from_text(st, "draft a cover letter for the Safaricom internship and ask me to approve it")
    assert job["payload"]["approve"] is True and "approve" not in job["title"].lower() and job["payload"]["prompt"].startswith("draft a cover letter")
    hub = types.SimpleNamespace(ask=lambda req: types.SimpleNamespace(text="Dear hiring manager..."))
    r = JobRunner(st, hub=hub, notifier=n); r.run_job(st.claim_due())
    assert any("Approval needed" in b[0] and "Dear hiring manager" in b[0] for b in n.buttons)
    assert Approvals(db, n).pending()[0]["title"].startswith("Accept the result of job #1")


def test_the_onscreen_gate_can_also_be_answered_from_the_phone():
    ran, shown, hidden, remote = [], [], [], []
    confirm.bind(lambda t, d: shown.append(t), lambda: hidden.append(1), lambda m: None)
    confirm.bind_remote(lambda t, d, tok: remote.append(tok))
    confirm.request("danger", "Delete things", "details", lambda: ran.append(1) or "done")
    for _ in range(50):
        if remote: break
        time.sleep(0.02)
    tok = remote[0]
    assert shown == ["Delete things"] and not confirm.resolve_remote("wrong-token", True) and ran == []
    assert confirm.resolve_remote(tok, True)
    for _ in range(50):
        if ran: break
        time.sleep(0.02)
    assert ran == [1] and hidden                                           # it ran, and the on-screen banner was dismissed
    assert not confirm.resolve_remote(tok, True)                           # the same button cannot approve twice
    confirm.bind_remote(None)


def test_a_remote_cancel_and_an_expired_request_do_nothing():
    ran, remote = [], []
    confirm.bind(lambda t, d: None, lambda: None, lambda m: None); confirm.bind_remote(lambda t, d, tok: remote.append(tok))
    confirm.request("a", "b", "c", lambda: ran.append(1) or "x")
    time.sleep(0.1); assert confirm.resolve_remote(remote[0], False); time.sleep(0.1)
    assert ran == []
    remote.clear(); confirm.request("a", "b", "c", lambda: ran.append(1) or "x"); time.sleep(0.1)
    confirm._pending.at -= 200                                               # older than the 90 s limit
    assert not confirm.resolve_remote(remote[0], True) and ran == []
    confirm.resolve(False); confirm.bind_remote(None)


def test_the_phone_button_for_the_gate_goes_through_the_inbox(db):
    n = Rec(); ran, remote = [], []
    confirm.bind(lambda t, d: None, lambda: None, lambda m: None); confirm.bind_remote(lambda t, d, tok: remote.append(tok))
    confirm.request("x", "y", "z", lambda: ran.append(1) or "ok"); time.sleep(0.1)
    inbox(db, n, [tap(f"cf:{remote[0]}:y")]).poll_once(1)
    time.sleep(0.2)
    assert ran == [1] and "Confirmed" in n.answers[0][1]
    inbox(db, n, [tap(f"cf:{remote[0]}:y", uid="cb2")]).poll_once(1)
    assert "expired or was already answered" in n.answers[1][1]; confirm.bind_remote(None)


def test_worker_owns_telegram_while_it_is_alive(tmp_path):
    lock = tmp_path / "w.lock"
    assert not worker_alive(lock)
    jobs.touch_worker_lock(lock); assert worker_alive(lock)
    lock.write_text(str(time.time() - 120)); assert not worker_alive(lock)
    polled = []
    class H:
        def get(self, *a, **k): polled.append(1); time.sleep(0.05); return types.SimpleNamespace(json=lambda: {"result": []})
    jobs.touch_worker_lock(lock)
    p = TelegramPoller(JobStore(tmp_path / "p.db"), Rec(), lock, http=H()); p.start(); time.sleep(0.4); p.stop()
    assert polled == []                                                    # the worker holds the lock: the app stays quiet
    lock.write_text(str(time.time() - 500))
    p2 = TelegramPoller(JobStore(tmp_path / "p.db"), Rec(), lock, http=H()); p2.start(); time.sleep(0.5); p2.stop()
    assert polled
