import json
import math
import os
import re
import struct
import subprocess
import types
import wave
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import pytest

pytest.importorskip("PyQt6")
from PyQt6.QtWidgets import QApplication

app = QApplication.instance() or QApplication([])
from core import video_maker as V
from core.jobs import JobRunner, JobStore, Notifier, add_from_text

needs_ffmpeg = pytest.mark.skipif(V.ffmpeg_path() is None, reason="no ffmpeg available")

SCRIPT = {"scenes": [{"narration": "Gravity pulls objects toward each other.", "image": "an apple falling from a tree"},
                     {"narration": "On Earth it makes things fall at 9.8 metres per second squared.", "image": "a ball dropping"},
                     {"narration": "That is why we stay on the ground.", "image": "people walking on Earth from space"}]}


def hub(text=None):
    return types.SimpleNamespace(ask=lambda req: types.SimpleNamespace(text=text or json.dumps(SCRIPT)))


def png(color=(200, 80, 40)):
    from PyQt6.QtCore import QBuffer, QByteArray, QIODeviceBase
    from PyQt6.QtGui import QColor, QImage
    im = QImage(640, 360, QImage.Format.Format_RGB32); im.fill(QColor(*color))
    ba = QByteArray(); b = QBuffer(ba); b.open(QIODeviceBase.OpenModeFlag.WriteOnly); im.save(b, "PNG")
    return bytes(ba) + b"\0" * 5000


def fake_speak(text, out):                          # a real WAV file: a tone as long as the narration deserves
    secs = max(1.0, len(text.split()) / 3.0)
    path = out.with_suffix(".wav")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(22050)
        w.writeframes(b"".join(struct.pack("<h", int(8000 * math.sin(i / 20))) for i in range(int(22050 * secs))))
    return secs


def probe(mp4):
    r = subprocess.run([V.ffmpeg_path(), "-i", str(mp4)], capture_output=True, text=True)
    dur = re.search(r"Duration: (\d+):(\d+):(\d+\.\d+)", r.stderr)
    return {"seconds": int(dur.group(1)) * 3600 + int(dur.group(2)) * 60 + float(dur.group(3)), "video": "Video:" in r.stderr,
            "audio": "Audio:" in r.stderr, "size": re.search(r"(\d{3,4})x(\d{3,4})", r.stderr).groups()}


def test_script_parsing_accepts_what_models_really_return():
    assert len(V.parse_script(json.dumps(SCRIPT))) == 3
    fenced = "Sure!\n```json\n" + json.dumps(SCRIPT) + "\n```\nHope it helps"
    assert V.parse_script(fenced)[0].image_prompt == "an apple falling from a tree"
    assert len(V.parse_script(json.dumps({"scenes": [{"narration": f"s{i}", "image_prompt": "x"} for i in range(20)]}), 5)) == 5
    for bad in ("no json here", "{}", json.dumps({"scenes": [{"narration": ""}]})):
        with pytest.raises(ValueError):
            V.parse_script(bad)


def test_image_service_failures_return_none_not_exceptions():
    ok = types.SimpleNamespace(status_code=200, headers={"content-type": "image/jpeg"}, content=b"x" * 9000)
    assert V.fetch_image("a cat", 1, get=lambda *a, **k: ok) == b"x" * 9000
    html = types.SimpleNamespace(status_code=200, headers={"content-type": "text/html"}, content=b"x" * 9000)
    tiny = types.SimpleNamespace(status_code=200, headers={"content-type": "image/png"}, content=b"x" * 10)
    err = types.SimpleNamespace(status_code=500, headers={}, content=b"")
    for r in (html, tiny, err):
        assert V.fetch_image("a cat", 1, get=lambda *a, **k: r) is None
    def boom(*a, **k): raise OSError("offline")
    assert V.fetch_image("a cat", 1, get=boom) is None
    seen = {}
    V.fetch_image("a red & blue \"fox\"", 7, get=lambda url, **k: seen.update(u=url) or ok)
    assert "seed=7" in seen["u"] and " " not in seen["u"] and "image.pollinations.ai" in seen["u"]


def test_frames_are_drawn_with_or_without_a_picture(tmp_path):
    from PyQt6.QtGui import QImage
    a, b = tmp_path / "a.png", tmp_path / "b.png"
    V.render_frame(png(), "A caption that is long enough to need wrapping across more than one line of the frame " * 2, a, 0)
    V.render_frame(None, "Offline card", b, 1, title="Gravity")
    for f in (a, b):
        im = QImage(str(f))
        assert (im.width(), im.height()) == (1280, 720)
    assert QImage(str(a)).pixelColor(640, 100).red() > 150                       # the picture shows through
    assert QImage(str(b)).pixelColor(640, 20) != QImage(str(b)).pixelColor(1270, 700)   # a gradient, not a flat fill


@needs_ffmpeg
def test_a_real_video_with_pictures_voice_and_captions(tmp_path):
    steps = []
    msg, path = V.make_video(hub(), "How gravity works", 30, out=tmp_path / "g.mp4",
                             fetch=lambda prompt, seed: png((seed * 60, 90, 40)), speak=fake_speak, progress=steps.append)
    info = probe(path)
    assert Path(path).exists() and info["video"] and info["audio"] and info["size"] == ("1280", "720")
    assert 8 < info["seconds"] < 30 and "3 scenes" in msg and "Note" not in msg
    assert steps[0] == "writing the script" and steps[-1] == "assembling the video"
    frame = tmp_path / "f.png"
    subprocess.run([V.ffmpeg_path(), "-y", "-ss", "1.5", "-i", path, "-frames:v", "1", str(frame)], capture_output=True)
    from PyQt6.QtGui import QImage
    im = QImage(str(frame)); assert not im.isNull() and im.pixelColor(300, 150).red() > 40          # not a black frame


@needs_ffmpeg
def test_it_still_makes_a_video_when_every_free_service_is_down(tmp_path):
    msg, path = V.make_video(hub(), "Offline test", 30, out=tmp_path / "o.mp4", fetch=lambda p, s: None, speak=lambda t, o: None)
    info = probe(path)
    assert info["video"] and info["audio"] and "title card" in msg and "captions only" in msg      # honest about what it fell back to
    assert info["seconds"] >= 9                                                                  # timed from the word count


def test_bad_requests_and_missing_ffmpeg_are_clear_errors(tmp_path, monkeypatch):
    with pytest.raises(ValueError):
        V.make_video(hub(), "   ")
    monkeypatch.setattr(V, "ffmpeg_path", lambda: None)
    with pytest.raises(RuntimeError, match="imageio-ffmpeg"):
        V.make_video(hub(), "x")


@needs_ffmpeg
def test_length_is_clamped(tmp_path):
    seen = {}
    h = types.SimpleNamespace(ask=lambda req: seen.update(p=req.system, c=req.messages[0]["content"]) or types.SimpleNamespace(text=json.dumps(SCRIPT)))
    V.make_video(h, "x", 9999, out=tmp_path / "c.mp4", fetch=lambda p, s: None, speak=lambda t, o: None)
    assert "180 seconds" in seen["c"] and "8 scenes" in seen["p"]


# ── as a job ───────────────────────────────────────────────────────────────
def test_video_requests_become_video_jobs(tmp_path):
    st = JobStore(tmp_path / "j.db")
    j = add_from_text(st, "make a 60 second video about how vaccines work tomorrow 8am")
    assert j["kind"] == "video" and j["payload"] == {"topic": "how vaccines work", "seconds": 60}
    assert add_from_text(st, "create an AI video explaining photosynthesis")["payload"]["topic"] == "photosynthesis"
    assert add_from_text(st, "write a study plan")["kind"] == "ai"


class Rec(Notifier):
    def __init__(self): super().__init__("t", "1"); self.sent = []
    def send(self, text, files=None): self.sent.append((text, files)); return True
    def send_buttons(self, text, rows): self.sent.append((text, None)); return 1


@needs_ffmpeg
def test_the_job_makes_the_video_and_sends_it_to_telegram(tmp_path, monkeypatch):
    monkeypatch.setattr(V, "OUT_DIR", tmp_path / "videos")
    monkeypatch.setattr(V, "fetch_image", lambda p, s, get=None: png())
    monkeypatch.setattr(V, "narrate", fake_speak)
    st = JobStore(tmp_path / "j.db"); n = Rec()
    job = add_from_text(st, "make a video about gravity and ask me to approve it")
    assert job["kind"] == "video" and job["payload"]["approve"] is True
    r = JobRunner(st, hub=hub(), notifier=n)
    r.run_job(st.claim_due())
    d = st.get(job["id"])
    assert d["status"] == "done" and d["file"].endswith(".mp4") and Path(d["file"]).exists() and "Video ready" in d["result"]
    assert n.sent[0][1] == [d["file"]]                                       # the MP4 itself goes to Telegram
    assert any("Approval needed" in t for t, _ in n.sent)                    # and it asks before you publish anything


def test_the_tool_queues_a_job_without_blocking_the_conversation(tmp_path, monkeypatch):
    import actions.ai_video as A
    st = JobStore(tmp_path / "j.db"); n = Rec()
    monkeypatch.setattr(A, "JobStore", lambda: st); monkeypatch.setattr(A.Notifier, "from_config", classmethod(lambda c: n))
    monkeypatch.setattr(V, "ffmpeg_path", lambda: "/bin/ffmpeg")
    log = []
    out = A.ai_video({"topic": "How the internet works", "seconds": 40, "when": "tomorrow 8am"}, types.SimpleNamespace(write_log=log.append))
    j = st.list()[0]
    assert "Started video job #1" in out and j["kind"] == "video" and j["payload"]["seconds"] == 40 and "tomorrow" in out
    assert n.sent and "queued" in n.sent[0][0] and "Video job #1 queued" in log[0]
    assert "about?" in A.ai_video({})
    monkeypatch.setattr(V, "ffmpeg_path", lambda: None)
    assert "imageio-ffmpeg" in A.ai_video({"topic": "x"})


def test_tool_registers_cleanly():
    from core.action_loader import discover_actions
    d = {x["name"]: x for x in discover_actions(Path("actions"), logger=lambda s: None).get_tool_declarations()}
    assert set(d["ai_video"]) <= {"name", "description", "parameters"}
