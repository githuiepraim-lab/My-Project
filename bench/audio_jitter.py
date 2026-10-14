"""
Does the face make speech stutter?

Speech is played by a background thread that must wake on schedule (every ~200 ms) and take
Python's interpreter lock for a moment. If the interface thread is busy painting, the wake-up
is late and the speaker runs dry: an audible break. This measures how late that thread wakes
while the GUI paints the avatar at the HUD's frame rate.

    python bench/audio_jitter.py
"""
import os, sys, threading, time
os.environ["QT_QPA_PLATFORM"] = "offscreen"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from PyQt6.QtCore import QTimer
from PyQt6.QtGui import QColor, QImage, QPainter
from PyQt6.QtWidgets import QApplication

app = QApplication([])
from core.avatar import HoloAvatar


def run(style, interval_ms, seconds=6.0, switch=0.005, quality=None, speaking=True):
    sys.setswitchinterval(switch)
    av = HoloAvatar(); av.set_look(style=style)
    if quality is not None and hasattr(av, "set_quality"):
        av.set_quality(quality)
    img = QImage(520, 520, QImage.Format.Format_RGB32)
    late, stop = [], threading.Event()

    def writer():                                    # stands in for asyncio.to_thread(stream.write, ...)
        while not stop.is_set():
            due = time.perf_counter() + 0.19
            time.sleep(0.19)                         # PortAudio's blocking write: no GIL while waiting
            _ = [i * i for i in range(300)]          # the little Python work done after it returns
            late.append((time.perf_counter() - due) * 1000)
    t = threading.Thread(target=writer, daemon=True); t.start()
    n = {"i": 0}

    def tick():
        n["i"] += 1
        av.step(interval_ms / 1000, 0.5 if speaking else 0.0, speaking=speaking, muted=False)
        img.fill(QColor(3, 10, 16)); p = QPainter(img); p.setRenderHint(QPainter.RenderHint.Antialiasing)
        av.paint(p, 260, 260, 220, QColor(0, 220, 255), QColor(255, 160, 40), QColor(3, 10, 16)); p.end()
    tm = QTimer(); tm.timeout.connect(tick); tm.start(interval_ms)
    QTimer.singleShot(int(seconds * 1000), app.quit)
    app.exec(); stop.set(); tm.stop(); t.join(1)
    late = sorted(late[2:])
    return {"frames": n["i"], "p50": late[len(late) // 2], "p95": late[int(len(late) * .95)], "max": late[-1]}


if __name__ == "__main__":
    rows = [("hologram, 60 fps (original)", dict(style="holo", interval_ms=16)),
            ("realistic, 60 fps (as shipped)", dict(style="real", interval_ms=16)),
            ("realistic, 30 fps", dict(style="real", interval_ms=33)),
            ("realistic, 30 fps + 1 ms GIL switch", dict(style="real", interval_ms=33, switch=0.001))]
    extra = [("realistic, adaptive (reduced) 30 fps + 1 ms switch", dict(style="real", interval_ms=33, switch=0.001, quality=1))]
    print(f"{'GUI load while speaking':52} {'frames':>6} {'wake delay p50':>15} {'p95':>8} {'worst':>8}")
    for name, kw in rows + (extra if hasattr(HoloAvatar, "set_quality") else []):
        r = run(**kw)
        print(f"{name:52} {r['frames']:6d} {r['p50']:12.1f} ms {r['p95']:5.1f} ms {r['max']:5.1f} ms")
