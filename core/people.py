"""
People recognition (faces and voices), kept ON THIS COMPUTER.
* Off until you switch it on. Nothing is sent anywhere. Faces are stored as numeric fingerprints + a small thumbnail
  in memory/people/ (delete one person or everyone any time).
* Only YOU add people. An unrecognised face/voice is parked as "unknown" and you are asked who it is; naming and
  deleting need your confirmation. Nobody can enrol themselves.
* Faces: OpenCV YuNet + SFace (free, downloaded once to models/). Voices: a simple MFCC voiceprint, good for telling a
  few household voices apart, NOT secure (a recording can fool it).
Legal note (Kenya Data Protection Act 2019 and similar): if you film or identify customers/staff (e.g. the arcade),
post a notice and register with the regulator as required.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
PEOPLE_DIR, MODELS = ROOT / "memory" / "people", ROOT / "models"
YUNET = ("face_detection_yunet_2023mar.onnx", "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx")
SFACE = ("face_recognition_sface_2021dec.onnx", "https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx")
FACE_THRESHOLD, VOICE_THRESHOLD, PENDING_DAYS = 0.363, 0.93, 7


def cosine(a, b) -> float:
    a, b = np.asarray(a, float).ravel(), np.asarray(b, float).ravel()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


class OpenCVFaces:
    """detect(frame_bgr) -> [(box, embedding)]. Models download on first use."""
    def __init__(self, models=MODELS, get=None) -> None:
        import cv2
        self.cv2, self.models, self._get, self._det, self._rec = cv2, Path(models), get, None, None

    def _file(self, spec) -> str:
        name, url = spec
        p = self.models / name
        if not (p.exists() and p.stat().st_size > 100_000):
            import requests
            self.models.mkdir(parents=True, exist_ok=True)
            r = (self._get or requests.get)(url, timeout=120)
            if r.status_code != 200 or len(r.content) < 100_000:
                raise RuntimeError(f"could not download {name} (HTTP {r.status_code}); save it from {url} into {self.models}")
            p.write_bytes(r.content)
        return str(p)

    def detect(self, frame):
        cv2 = self.cv2
        if self._det is None:
            self._det = cv2.FaceDetectorYN.create(self._file(YUNET), "", (320, 320), 0.8, 0.3, 5000)
            self._rec = cv2.FaceRecognizerSF.create(self._file(SFACE), "")
        h, w = frame.shape[:2]
        self._det.setInputSize((w, h))
        _, faces = self._det.detect(frame)
        return [(tuple(int(v) for v in f[:4]), np.asarray(self._rec.feature(self._rec.alignCrop(frame, f)), float).ravel())
                for f in (faces if faces is not None else [])]


def mfcc(x, sr=16000, n_mfcc=20, n_fft=512, hop=160, n_mels=40) -> np.ndarray:
    x = np.asarray(x, float); x = np.append(x[0], x[1:] - 0.97 * x[:-1])
    n = max(1, 1 + (len(x) - n_fft) // hop)
    fr = np.stack([x[i * hop:i * hop + n_fft] * np.hanning(n_fft) for i in range(n) if len(x[i * hop:i * hop + n_fft]) == n_fft]) if len(x) >= n_fft else np.zeros((1, n_fft))
    pw = np.abs(np.fft.rfft(fr, n_fft)) ** 2 / n_fft
    mel = lambda f: 2595 * np.log10(1 + f / 700)
    pts = 700 * (10 ** (np.linspace(mel(0), mel(sr / 2), n_mels + 2) / 2595) - 1)
    bins = np.floor((n_fft + 1) * pts / sr).astype(int); fb = np.zeros((n_mels, n_fft // 2 + 1))
    for m in range(1, n_mels + 1):
        l, c, r = bins[m - 1], bins[m], bins[m + 1]
        for k in range(l, c): fb[m - 1, k] = (k - l) / max(1, c - l)
        for k in range(c, r): fb[m - 1, k] = (r - k) / max(1, r - c)
    e = np.log(np.maximum(pw @ fb.T, 1e-10))
    return e @ np.cos(np.arange(n_mfcc)[:, None] * (2 * np.arange(n_mels) + 1)[None] * np.pi / (2 * n_mels)).T


def voiceprint(x, sr=16000) -> np.ndarray:
    m = mfcc(x, sr)[:, 1:]
    return np.concatenate([m.mean(0), m.std(0)])


class People:
    def __init__(self, root: Path | None = None, backend=None) -> None:
        self.root, self.backend = Path(root or PEOPLE_DIR), backend
        self.root.mkdir(parents=True, exist_ok=True); self.db = self.root / "people.json"

    def _load(self) -> dict:
        try: d = json.loads(self.db.read_text(encoding="utf-8"))
        except Exception: d = {}
        return {"enabled": bool(d.get("enabled")), "people": d.get("people", {}), "pending": d.get("pending", {}), "next": int(d.get("next", 1))}

    def _save(self, d: dict) -> None: self.db.write_text(json.dumps(d), encoding="utf-8")
    @property
    def enabled(self) -> bool: return self._load()["enabled"]
    def enable(self, on: bool = True) -> None:
        d = self._load(); d["enabled"] = on; self._save(d)
    def names(self) -> list[str]: return sorted(self._load()["people"])
    def pending(self) -> dict: return self._load()["pending"]

    def identify(self, emb, kind: str = "face"):
        d, thr, key = self._load(), (FACE_THRESHOLD if kind == "face" else VOICE_THRESHOLD), ("faces" if kind == "face" else "voices")
        best = (None, -1.0)
        for name, p in d["people"].items():
            for ref in p.get(key, []):
                s = cosine(emb, ref)
                if s > best[1]: best = (name, s)
        return (best[0], best[1]) if best[0] and best[1] >= thr else (None, best[1])

    def enroll(self, name: str, emb, kind: str = "face", thumb: bytes | None = None, owner: bool = False) -> None:
        name = " ".join(str(name).split())[:40]
        if not name: raise ValueError("a person needs a name")
        d = self._load(); p = d["people"].setdefault(name, {"faces": [], "voices": [], "owner": False, "added": time.time()})
        key = "faces" if kind == "face" else "voices"
        p[key] = (p[key] + [list(map(float, np.asarray(emb).ravel()))])[-10:]; p["owner"] = p["owner"] or owner
        if thumb: (self.root / f"{abs(hash(name)) % 10**8}.jpg").write_bytes(thumb)
        self._save(d)

    def forget(self, name: str | None = None) -> int:
        d = self._load(); n = len(d["people"]) if name is None else int(name in d["people"])
        d["people"] = {} if name is None else {k: v for k, v in d["people"].items() if k != name}; self._save(d)
        if name is None:
            for f in self.root.glob("*.jpg"): f.unlink(missing_ok=True)
        return n

    def add_unknown(self, emb, kind: str = "face", thumb: bytes | None = None, source: str = "") -> tuple[int, bool]:
        """(pending id, is_new): the same stranger seen again is counted, not re-announced."""
        d, now = self._load(), time.time()
        d["pending"] = {k: v for k, v in d["pending"].items() if now - v["t"] < PENDING_DAYS * 86400}
        for k, v in d["pending"].items():
            if v["kind"] == kind and cosine(emb, v["emb"]) >= (FACE_THRESHOLD if kind == "face" else VOICE_THRESHOLD):
                v["seen"] += 1; self._save(d); return int(k), False
        pid = d["next"]; d["next"] += 1
        d["pending"][str(pid)] = {"emb": list(map(float, np.asarray(emb).ravel())), "kind": kind, "t": now, "seen": 1, "source": source}
        if thumb: (self.root / f"pending_{pid}.jpg").write_bytes(thumb)
        self._save(d); return pid, True

    def resolve(self, pid: int, name: str, owner: bool = False) -> str:
        v = self._load()["pending"].get(str(pid))
        if not v: return f"No unknown person #{pid} (it may have expired)."
        th = self.root / f"pending_{pid}.jpg"
        self.enroll(name, v["emb"], v["kind"], th.read_bytes() if th.exists() else None, owner)
        d = self._load(); d["pending"].pop(str(pid), None); self._save(d); th.unlink(missing_ok=True)
        return f"Saved as {name}."

    def ignore(self, pid: int) -> None:
        d = self._load(); d["pending"].pop(str(pid), None); self._save(d); (self.root / f"pending_{pid}.jpg").unlink(missing_ok=True)

    def see(self, frame, source: str = "") -> list[dict]:
        """Faces in a frame -> [{box, name|None, score, pending_id, new}]; unknown faces are parked for the owner."""
        if not self.enabled or self.backend is None: return []
        out = []
        for box, emb in self.backend.detect(frame):
            name, score = self.identify(emb)
            r = {"box": box, "name": name, "score": score, "pending_id": None, "new": False}
            if name is None: r["pending_id"], r["new"] = self.add_unknown(emb, "face", _thumb(frame, box), source)
            out.append(r)
        return out


def _thumb(frame, box) -> bytes | None:
    try:
        import cv2
        x, y, w, h = box
        ok, buf = cv2.imencode(".jpg", cv2.resize(frame[max(0, y - h // 4):y + h + h // 4, max(0, x - w // 4):x + w + w // 4], (160, 160)))
        return bytes(buf) if ok else None
    except Exception:
        return None


class SpeakerTracker:
    """Feed microphone audio (16 kHz float); it reports when the speaker changes or is unknown."""
    def __init__(self, people: People, window_s: float = 3.0, sr: int = 16000) -> None:
        self.people, self.win, self.sr, self.buf, self.current = people, int(window_s * sr), sr, np.zeros(0), "__none__"

    def feed(self, pcm) -> dict | None:
        self.buf = np.concatenate([self.buf, np.asarray(pcm, float)])[-self.win * 2:]
        if len(self.buf) < self.win or not self.people.enabled: return None
        seg, self.buf = self.buf[-self.win:], np.zeros(0)
        if float(np.sqrt(np.mean(seg ** 2))) < 0.005: return None
        vp = voiceprint(seg, self.sr); name, score = self.people.identify(vp, "voice")
        if name == self.current: return None
        self.current = name
        return {"name": name, "score": score, "pending_id": None if name else self.people.add_unknown(vp, "voice", source="microphone")[0]}
