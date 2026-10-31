"""
CCTV: watch cameras (IP/RTSP, USB or a file), notice movement, identify people (core/people.py), alert you, and turn recorded
footage into a timeline. All local. Camera addresses (they often hold a password) live in the credential store. An unknown
person triggers ONE alert with a snapshot and buttons to name or ignore them. Old snapshots are deleted. Tell people they are filmed.
"""
from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import numpy as np

CCTV_DIR = Path(__file__).resolve().parent.parent / "memory" / "cctv"


class MotionDetector:
    def __init__(self, delta: int = 25, fraction: float = 0.01) -> None:
        self.delta, self.fraction, self.prev = delta, fraction, None

    def update(self, frame) -> bool:
        g = np.asarray(frame); g = (g.mean(2) if g.ndim == 3 else g.astype(float))[::4, ::4]
        moved = self.prev is not None and float((np.abs(g - self.prev) > self.delta).mean()) > self.fraction
        self.prev = g
        return moved


class CCTV:
    def __init__(self, people, notifier=None, out_dir: Path | None = None, keep_days: int = 7, cooldown: float = 60.0, clock=time.time) -> None:
        self.people, self.notifier, self.keep_days, self.cooldown, self._clock = people, notifier, keep_days, cooldown, clock
        self.dir = Path(out_dir or CCTV_DIR); self.dir.mkdir(parents=True, exist_ok=True)
        self.motion: dict[str, MotionDetector] = {}; self._alerted: dict[int, float] = {}; self._threads: dict = {}

    def _snapshot(self, cam: str, frame) -> Path | None:
        try:
            import cv2
            lt = time.localtime(self._clock()); d = self.dir / cam / time.strftime("%Y-%m-%d", lt); d.mkdir(parents=True, exist_ok=True)
            p = d / f"{time.strftime('%H%M%S', lt)}.jpg"; cv2.imwrite(str(p), frame); return p
        except Exception:
            return None

    def events(self, limit: int = 20) -> list[dict]:
        try: return [json.loads(x) for x in (self.dir / "events.jsonl").read_text(encoding="utf-8").splitlines()[-limit:]]
        except OSError: return []

    def process(self, cam: str, frame) -> list[dict]:
        if not self.motion.setdefault(cam, MotionDetector()).update(frame): return []
        seen, snap, out = self.people.see(frame, cam), None, []
        snap = self._snapshot(cam, frame)
        for r in seen or [{"name": None, "pending_id": None, "new": False}]:
            ev = {"t": self._clock(), "camera": cam, "snapshot": str(snap or ""), "pending_id": r["pending_id"],
                  "who": r["name"] or ("unknown person" if r["pending_id"] else "movement")}
            out.append(ev)
            with (self.dir / "events.jsonl").open("a", encoding="utf-8") as f: f.write(json.dumps(ev) + "\n")
            pid = r["pending_id"]
            if pid and self.notifier is not None and self._clock() - self._alerted.get(pid, -1e9) > self.cooldown:
                self._alerted[pid] = self._clock()
                self.notifier.send(f"📹 {cam}: someone I don't recognise (unknown #{pid}).", [str(snap)] if snap else None)
                self.notifier.send_buttons("Who is this?", [[("🏷 Name them", f"nm:{pid}"), ("🙈 Ignore", f"ig:{pid}")]])
        return out

    def cleanup(self) -> int:
        n, cutoff = 0, self._clock() - self.keep_days * 86400
        for f in self.dir.rglob("*.jpg"):
            if f.stat().st_mtime < cutoff: f.unlink(missing_ok=True); n += 1
        return n

    def start(self, cam: str, url, fps: float = 4.0) -> None:
        if cam in self._threads: return
        import cv2
        cap = cv2.VideoCapture(int(url) if str(url).isdigit() else str(url))
        if not cap.isOpened(): raise RuntimeError(f"cannot open camera '{cam}' (check the address, network and the camera's own login)")
        stop = threading.Event()
        def loop():
            try:
                while not stop.is_set():
                    ok, frame = cap.read()
                    if not ok: break
                    self.process(cam, frame); time.sleep(1.0 / fps)
            finally:
                cap.release(); self._threads.pop(cam, None)
        t = threading.Thread(target=loop, daemon=True, name=f"cctv-{cam}"); self._threads[cam] = (t, stop); t.start()

    def stop(self, cam: str | None = None) -> None:
        for n in ([cam] if cam else list(self._threads)):
            if n in self._threads: self._threads[n][1].set()

    def running(self) -> list[str]: return sorted(self._threads)


def analyze_video(path: str, people, step_s: float = 1.0, max_samples: int = 3600) -> dict:
    """A recorded file -> {timeline, summary, people}: who appeared when."""
    import cv2
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened(): raise RuntimeError(f"cannot open {Path(path).name}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0; total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    step, md, timeline, i, seen = max(1, int(round(fps * step_s))), MotionDetector(), [], 0, {}
    while len(timeline) < max_samples:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i); ok, frame = cap.read()
        if not ok: break
        t, moved, who = i / fps, False, []
        moved = md.update(frame)
        if moved: who = [r["name"] or f"unknown #{r['pending_id']}" for r in people.see(frame, "file")]
        for w in who:
            a = seen.setdefault(w, [t, t, 0]); a[1] = t; a[2] += 1
        timeline.append({"t": round(t, 1), "motion": moved, "who": who}); i += step
        if total and i >= total: break
    cap.release()
    moves = [x for x in timeline if x["motion"]]
    lines = [f"{Path(path).name}: {total / fps:.0f} s analysed, movement in {len(moves)} of {len(timeline)} samples."]
    lines += [f"- {w}: first seen {a:.0f}s, last {b:.0f}s ({c} sightings)" for w, (a, b, c) in sorted(seen.items(), key=lambda kv: kv[1][0])]
    if not seen and moves: lines.append("- movement but no faces were seen")
    return {"timeline": timeline, "summary": "\n".join(lines), "people": {k: v[:2] for k, v in seen.items()}}
