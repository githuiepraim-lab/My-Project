import os, time, types

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")
from core import people as P
from core.cctv import CCTV, MotionDetector, analyze_video
from core.jobs import JobStore, Notifier, TelegramInbox
from core.people import People, SpeakerTracker, voiceprint


class Faces:                                    # a frame filled with 40*k (k=1..3) contains face k (a patch average survives video compression)
    def detect(self, f):
        k = int(round(float(f[:8, :8].mean()) / 40.0))
        return [((20, 20, 40, 40), np.eye(8)[k] + 0.02)] if k in (1, 2, 3) else []


def fr(k, moved=0):
    f = np.full((120, 160, 3), k * 40, np.uint8)
    if moved: f[30:90, 40:120] = 255 if k * 40 < 128 else 0
    return f


class Rec(Notifier):
    def __init__(self): super().__init__("t", "1"); self.sent, self.buttons = [], []
    def send(self, text, files=None): self.sent.append((text, files)); return True
    def send_buttons(self, text, rows): self.buttons.append((text, rows)); return 1


def mk(tmp, n=None):
    p = People(tmp / "p", Faces()); p.enable()
    c = CCTV(p, n, tmp / "c", clock=lambda: c.now); c.now = 1000.0
    return c, p


def test_nothing_is_recognised_until_switched_on_and_strangers_wait_for_the_owner(tmp_path):
    assert People(tmp_path / "x", Faces()).see(fr(1)) == []
    p = People(tmp_path, Faces()); p.enable(); p.enroll("Ephraim", np.eye(8)[1], owner=True)
    assert p.see(fr(1))[0]["name"] == "Ephraim"
    r = p.see(fr(2))[0]; again = p.see(fr(2))[0]
    assert r["name"] is None and r["new"] and not again["new"] and p.pending()["1"]["seen"] == 2
    assert p.resolve(1, "Kamau") == "Saved as Kamau." and p.see(fr(2))[0]["name"] == "Kamau" and "No unknown" in p.resolve(1, "x")


def test_forget_and_bad_names(tmp_path):
    p = People(tmp_path); p.enable(); p.enroll("A", np.eye(8)[0]); p.enroll("B", np.eye(8)[1], thumb=b"j")
    assert p.forget("A") == 1 and p.forget() == 1 and p.names() == [] and not list(tmp_path.glob("*.jpg"))
    with pytest.raises(ValueError): p.enroll(" ", np.eye(8)[0])


def voice(f0, formants, seed, sr=16000):
    r = np.random.RandomState(seed); t = np.arange(sr * 3) / sr
    x = sum(np.sin(2 * np.pi * f0 * (1 + 0.02 * np.sin(6 * np.pi * t)) * k * t) / k for k in range(1, 30))
    fr_ = np.fft.rfftfreq(len(x), 1 / sr); g = sum(np.exp(-0.5 * ((fr_ - f) / 150) ** 2) for f in formants) + 0.02
    y = np.fft.irfft(np.fft.rfft(x) * g, len(x)) + r.normal(0, 0.01, len(x)); return y / np.abs(y).max() * 0.3


A, B = (110, (500, 1500, 2500)), (210, (800, 1200, 3000))


def test_voices_are_told_apart_and_a_new_voice_is_parked_for_the_owner(tmp_path):
    p = People(tmp_path); p.enable(); p.enroll("Ephraim", voiceprint(voice(*A, 1)), "voice")
    assert p.identify(voiceprint(voice(*A, 7)), "voice")[0] == "Ephraim" and p.identify(voiceprint(voice(*B, 8)), "voice")[0] is None
    t = SpeakerTracker(p)
    ev = [t.feed(voice(*A, 3)), t.feed(voice(*A, 4)), t.feed(voice(*B, 5)), t.feed(voice(*B, 6)), t.feed(np.zeros(48000))]
    assert ev[0]["name"] == "Ephraim" and ev[1] is None and ev[2]["name"] is None and ev[2]["pending_id"] == 1 and ev[3] is None and ev[4] is None


def test_motion_known_face_and_stranger_alert_once(tmp_path):
    n = Rec(); c, p = mk(tmp_path, n); p.enroll("Ephraim", np.eye(8)[1])
    assert not MotionDetector().update(fr(2)) and c.process("d", fr(1)) == []
    assert c.process("d", fr(1, 1))[0]["who"] == "Ephraim" and not n.sent
    c.process("d", fr(2)); c.process("d", fr(2, 1))
    assert "don't recognise" in n.sent[0][0] and n.sent[0][1][0].endswith(".jpg") and [b[1] for b in n.buttons[0][1][0]] == ["nm:1", "ig:1"]
    c.process("d", fr(2)); c.process("d", fr(2, 1)); assert len(n.sent) == 1


def test_owner_names_a_stranger_from_telegram_and_strangers_cannot(tmp_path, monkeypatch):
    n = Rec(); c, p = mk(tmp_path, n); c.process("d", fr(3)); c.process("d", fr(3, 1))
    monkeypatch.setattr(P, "PEOPLE_DIR", tmp_path / "p")
    ib = TelegramInbox(JobStore(tmp_path / "j.db"), n); n.answer_callback = n.edit_text = lambda *a, **k: None
    tap = lambda chat: {"update_id": 1, "callback_query": {"id": "x", "data": "nm:1", "message": {"chat": {"id": chat}, "message_id": 5, "text": "?"}}}
    ib._http = types.SimpleNamespace(get=lambda *a, **k: types.SimpleNamespace(json=lambda: {"result": [tap(999)]})); ib.poll_once(1)
    assert ib.awaiting_name is None
    ib._http = types.SimpleNamespace(get=lambda *a, **k: types.SimpleNamespace(json=lambda: {"result": [tap(1)]})); ib.poll_once(1)
    assert ib.awaiting_name == 1 and "Saved as Kamau" in ib.handle("Kamau") and p.identify(np.eye(8)[3])[0] == "Kamau"


def test_old_snapshots_are_deleted(tmp_path):
    c, _ = mk(tmp_path); c.process("d", fr(2)); c.process("d", fr(2, 1))
    s = list((tmp_path / "c").rglob("*.jpg")); os.utime(s[0], (1, 1)); c._clock = time.time
    assert c.cleanup() == 1


def test_footage_analysis_gives_a_timeline(tmp_path):
    path = tmp_path / "v.mp4"; w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (160, 120))
    for i in range(60): w.write(fr(1, i % 2) if 20 <= i < 40 else fr(0))
    w.release()
    p = People(tmp_path / "p", Faces()); p.enable(); p.enroll("Ephraim", np.eye(8)[1])
    r = analyze_video(str(path), p, step_s=0.5)
    assert "Ephraim" in r["people"] and 1.5 <= r["people"]["Ephraim"][0] <= 2.6 and "first seen" in r["summary"]
    with pytest.raises(RuntimeError, match="cannot open"): analyze_video(str(tmp_path / "no.mp4"), p)


def test_model_download_is_checked_and_sensitive_tool_steps_are_gated(tmp_path, monkeypatch):
    f = P.OpenCVFaces(tmp_path, get=lambda u, **k: types.SimpleNamespace(status_code=200, content=b"x" * 200_000)); assert f._file(P.YUNET).endswith(".onnx")
    with pytest.raises(RuntimeError, match="could not download"):
        P.OpenCVFaces(tmp_path / "b", get=lambda u, **k: types.SimpleNamespace(status_code=302, content=b""))._file(P.SFACE)
    import actions.people as A
    got = {}; monkeypatch.setattr(A.confirm, "request", lambda k, t, d, run: got.update(k=k, run=run) or "pending"); monkeypatch.setattr(A, "People", lambda: People(tmp_path / "p2"))
    assert A.people({"mode": "enable"}) == "pending" and not People(tmp_path / "p2").enabled
    got["run"](); assert People(tmp_path / "p2").enabled
    assert A.people({"mode": "forget", "name": "X"}) == "pending" and got["k"] == "people_forget"
    from pathlib import Path
    from core.action_loader import discover_actions
    d = {x["name"] for x in discover_actions(Path("actions"), logger=lambda s: None).get_tool_declarations()}
    assert {"people", "cctv"} <= d
