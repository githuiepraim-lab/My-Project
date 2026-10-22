import os
import types

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import pytest
from core.telegram_setup import SetupError, TelegramSetup, clean_token

try:                                                        # one QApplication for the whole module (if it is collected, Qt aborts)
    from PyQt6.QtWidgets import QApplication
    _APP = QApplication.instance() or QApplication([])
except ImportError:
    _APP = None

TOKEN = "123456789:" + "A" * 35


class Resp:
    def __init__(self, data, status=200): self._d, self.status_code = data, status
    def json(self): return self._d


class Http:
    """Scripted Telegram. calls = [(method, params)]"""
    def __init__(self, updates=None, me=None, errors=None):
        self.calls, self.updates, self.errors = [], list(updates or []), errors or {}
        self.me = me or {"ok": True, "result": {"id": 1, "is_bot": True, "first_name": "Jarvis", "username": "my_jarvis_bot"}}
    def post(self, url, json=None, timeout=None):
        method = url.rsplit("/", 1)[1]
        self.calls.append((method, json))
        if method in self.errors: return Resp(self.errors[method], self.errors[method].get("error_code", 400))
        if method == "getMe": return Resp(self.me)
        if method == "getUpdates":
            ups, self.updates = self.updates, []
            return Resp({"ok": True, "result": ups})
        return Resp({"ok": True, "result": True})


def upd(i, text, chat_id=42, kind="private", first="Ephraim"):
    return {"update_id": i, "message": {"text": text, "chat": {"id": chat_id, "type": kind, "first_name": first}}}


@pytest.mark.parametrize("raw", [TOKEN, f"  {TOKEN}\n", f'"{TOKEN}"', f"bot{TOKEN}", f"https://api.telegram.org/bot{TOKEN}", TOKEN[:5] + " " + TOKEN[5:]])
def test_pasted_tokens_are_cleaned(raw):
    assert clean_token(raw) == TOKEN


def test_verify_accepts_a_real_bot_and_explains_every_failure():
    me = TelegramSetup(Http()).verify(f" {TOKEN}\n")
    assert me["username"] == "my_jarvis_bot" and me["token"] == TOKEN
    for bad in ("hello", "12345:short", "", "ghp_notatoken"):
        with pytest.raises(SetupError, match="does not look like a bot token"):
            TelegramSetup(Http()).verify(bad)
    with pytest.raises(SetupError, match="rejected that token"):
        TelegramSetup(Http(errors={"getMe": {"ok": False, "error_code": 401}})).verify(TOKEN)
    with pytest.raises(SetupError, match="does not belong to a bot"):
        TelegramSetup(Http(me={"ok": True, "result": {"id": 1, "is_bot": False}})).verify(TOKEN)

    class Down:
        def post(self, *a, **k): raise OSError("no route")
    with pytest.raises(SetupError, match="Could not reach Telegram"):
        TelegramSetup(Down()).verify(TOKEN)


def test_codes_are_unambiguous_and_unique():
    codes = {TelegramSetup.new_code() for _ in range(200)}
    assert len(codes) > 190 and all(len(c) == 6 and not set(c) & set("0O1IL") for c in codes)


def test_pairing_binds_only_the_chat_that_sent_the_code():
    h = Http([upd(1, "hello bot", chat_id=999, first="Stranger"), upd(2, "ABC234", chat_id=998, first="Eve", kind="group"),
              upd(3, "my code is abc 234 ok", chat_id=42, first="Ephraim")])
    got = TelegramSetup(h, sleep=lambda s: None).pair(TOKEN, "ABC234", timeout=30)
    assert got == {"chat_id": "42", "name": "Ephraim"}                  # not the stranger, not the group, case/spacing tolerant
    assert h.calls[0][0] == "deleteWebhook" and h.calls[0][1]["drop_pending_updates"] is True


def test_a_stranger_who_messages_first_cannot_take_over():
    h = Http([upd(1, "/start", chat_id=666, first="Mallory"), upd(2, "WRONG1", chat_id=666, first="Mallory")])
    clock = iter([0.0, 0.0, 1.0, 2.0, 99.0, 99.0, 99.0, 99.0])
    s = TelegramSetup(h, clock=lambda: next(clock, 99.0), sleep=lambda s: None)
    with pytest.raises(SetupError, match="No message containing the code ABC234"):
        s.pair(TOKEN, "ABC234", timeout=5)


def test_pairing_reports_progress_and_a_competing_listener():
    waits = []
    h = Http([upd(1, "noise")]); ticks = {"n": 0}
    def clock():
        ticks["n"] += 1; return 0.0 if ticks["n"] < 4 else 100.0
    with pytest.raises(SetupError):
        TelegramSetup(h, clock=clock, sleep=lambda s: None).pair(TOKEN, "ABC234", timeout=10, on_wait=lambda left, ign: waits.append(ign))
    assert waits and waits[0] == 1                                        # one foreign message was ignored, and we said so
    with pytest.raises(SetupError, match="Another program is already listening"):
        TelegramSetup(Http(errors={"getUpdates": {"ok": False, "error_code": 409}}), sleep=lambda s: None).pair(TOKEN, "ABC234", timeout=5)


def test_save_keeps_the_token_secret_and_the_chat_id_in_config():
    keys, flags = {}, {}
    TelegramSetup.save(TOKEN, 42, set_key=lambda n, k: keys.__setitem__(n, k), save_flag=lambda k, v: flags.__setitem__(k, v))
    assert keys == {"telegram": TOKEN} and flags == {"telegram_chat_id": "42"}
    TelegramSetup.disconnect(delete_key=lambda n: keys.pop(n), save_flag=lambda k, v: flags.__setitem__(k, v))
    assert keys == {} and flags["telegram_chat_id"] == ""


def test_the_test_message_goes_to_the_paired_chat():
    h = Http()
    TelegramSetup(h).send_test(TOKEN, "42")
    m, p = h.calls[-1]
    assert m == "sendMessage" and p["chat_id"] == "42" and "connected" in p["text"]


def test_status_reflects_the_stored_connection(monkeypatch):
    from core import jobs
    monkeypatch.setattr(jobs.Notifier, "from_config", classmethod(lambda c: jobs.Notifier(TOKEN, "42")))
    assert TelegramSetup.status() == {"connected": True, "chat_id": "42"}
    monkeypatch.setattr(jobs.Notifier, "from_config", classmethod(lambda c: jobs.Notifier("", "")))
    assert TelegramSetup.status() == {"connected": False, "chat_id": ""}


# ── the window ─────────────────────────────────────────────────────────────
class FakeSetup(TelegramSetup):
    def __init__(self, fail=None): super().__init__(); self.saved, self.sent, self.fail, self.removed = [], [], fail, False
    def verify(self, raw):
        if self.fail: raise SetupError(self.fail)
        return {"token": TOKEN, "id": 1, "username": "my_jarvis_bot", "name": "Jarvis"}
    def pair(self, token, code, timeout=0, on_wait=None):
        on_wait and on_wait(120, 2)
        return {"chat_id": "42", "name": "Ephraim"}
    def save(self, token, chat_id, **k): self.saved.append((token, chat_id))
    def send_test(self, token, chat_id): self.sent.append(chat_id)
    def disconnect(self, **k): self.removed = True
    def status(self): return {"connected": bool(self.saved) and not self.removed}


def wait_for(app, cond, secs=3):
    import time
    t = time.time()
    while time.time() - t < secs:
        app.processEvents()
        if cond(): return True
        time.sleep(0.02)
    return False


def test_dialog_walks_through_verify_code_pair_and_save():
    app = _APP
    from core.telegram_dialog import TelegramDialog
    fs = FakeSetup(); d = TelegramDialog(setup=fs); done = []
    d.connected.connect(lambda: done.append(1))
    d.token.setText(f" {TOKEN}\n"); d.verify()
    assert wait_for(app, lambda: bool(d.code.text()) and d.code.text() != "✓" or fs.saved)
    assert wait_for(app, lambda: done and "Connected to Ephraim" in d.status.text())
    assert fs.saved == [(TOKEN, "42")] and fs.sent == ["42"] and d.token.text() == ""        # the token is wiped from the box


def test_dialog_shows_a_plain_error_and_lets_you_retry():
    app = _APP
    from core.telegram_dialog import TelegramDialog
    d = TelegramDialog(setup=FakeSetup(fail="Telegram rejected that token."))
    d.token.setText("x"); d.verify()
    assert wait_for(app, lambda: "rejected" in d.status.text()) and d.go.isEnabled() and not d.code.text()


def test_dialog_can_disconnect():
    from core.telegram_dialog import TelegramDialog
    fs = FakeSetup(); fs.saved.append(("t", "1"))
    d = TelegramDialog(setup=fs); gone = []; d.connected.connect(lambda: gone.append(1))
    d.disconnect()
    assert fs.removed and gone and "removed" in d.status.text()


# ── controlling it from your phone ─────────────────────────────────────────
def test_inbox_ask_and_status_and_help(tmp_path, monkeypatch):
    from core import jobs
    st = jobs.JobStore(tmp_path / "j.db"); st.add("x", "remind", {}, 0)
    ib = jobs.TelegramInbox(st, jobs.Notifier(TOKEN, "42"))
    hub = types.SimpleNamespace(ask=lambda req: types.SimpleNamespace(text="Nairobi is the capital of Kenya."))
    monkeypatch.setattr("core.ai.get_hub", lambda: hub)
    assert "Nairobi" in ib.handle("/ask what is the capital of kenya") and "Usage" in ib.handle("/ask")
    out = ib.handle("/status")
    assert "JARVIS is running" in out and "1 queued" in out
    assert "/ask" in ib.handle("/help") and "/status" in ib.handle("/start")
    monkeypatch.setattr("core.ai.get_hub", lambda: types.SimpleNamespace(ask=lambda r: (_ for _ in ()).throw(RuntimeError("down"))))
    assert "could not reach an AI" in ib.handle("/ask hello")
