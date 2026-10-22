"""
Hand-gesture control: your webcam sees your hand, and gestures drive the computer.

    pinch (fingers close)   -> close the tab            spread (fingers open)  -> new tab
    swipe right / left      -> next / previous tab      swipe up / down        -> scroll down / up
    fist (hold)             -> play / pause             thumbs up / down       -> volume up / down
    peace sign              -> screenshot               OK sign                -> mute the assistant's mic
    three fingers           -> show the JOBS window     open palm (hold)       -> ARM (see below)

Safe by design
* OFF until you switch it on, and a visible preview shows exactly what the camera sees.
* ARMING: by default gestures do nothing until you hold an open palm for a second (it then
  stays armed for 10 s after your last gesture). Waving, typing or scratching your nose
  cannot close a tab.
* A gesture can only trigger a name from a FIXED list of actions (tab, scroll, media keys,
  screenshot, and a few app windows). It can never run a command or open a program.
* Video frames are processed on this computer and never stored or sent anywhere.

Needs:  pip install mediapipe  (the hand model, ~8 MB, downloads once on first use).
"""
from __future__ import annotations

import math
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "hand_landmarker.task"
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"

# ── what a gesture may do: a fixed list, never a free-form command ─────────────────────────
MOD = "command" if sys.platform == "darwin" else "ctrl"
KEYS = {
    "new_tab": [MOD, "t"], "close_tab": [MOD, "w"], "next_tab": ["ctrl", "tab"], "prev_tab": ["ctrl", "shift", "tab"],
    "play_pause": ["playpause"], "volume_up": ["volumeup"], "volume_down": ["volumedown"],
    "screenshot": (["command", "shift", "4"] if sys.platform == "darwin" else ["win", "shift", "s"]),
}
SCROLL = {"scroll_down": -500, "scroll_up": 500}
APP_ACTIONS = ("show_files", "show_jobs", "show_code", "toggle_mute", "toggle_gestures_arm")
ACTIONS = tuple(KEYS) + tuple(SCROLL) + APP_ACTIONS + ("none",)

DEFAULT_MAP = {"pinch_in": "close_tab", "spread": "new_tab", "swipe_right": "next_tab", "swipe_left": "prev_tab",
               "swipe_up": "scroll_down", "swipe_down": "scroll_up", "fist_hold": "play_pause", "thumbs_up": "volume_up",
               "thumbs_down": "volume_down", "peace": "screenshot", "ok": "toggle_mute", "three": "show_jobs"}
GESTURES = tuple(DEFAULT_MAP) + ("open_palm_hold",)

# MediaPipe hand landmark indices
WRIST, THUMB_CMC, THUMB_MCP, THUMB_IP, THUMB_TIP = 0, 1, 2, 3, 4
INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP = 5, 6, 7, 8
MIDDLE_MCP, MIDDLE_PIP, MIDDLE_TIP = 9, 10, 12
RING_MCP, RING_PIP, RING_TIP = 13, 14, 16
PINKY_MCP, PINKY_PIP, PINKY_TIP = 17, 18, 20
CONNECTIONS = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10), (10, 11), (11, 12), (9, 13),
               (13, 14), (14, 15), (15, 16), (13, 17), (17, 18), (18, 19), (19, 20), (0, 17)]


@dataclass
class Hand:
    landmarks: list[tuple[float, float, float]]        # 21 points, x/y in 0..1 of the (mirrored) frame
    label: str = ""
    score: float = 1.0


def _d(a, b) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def palm_size(h: Hand) -> float:
    return max(1e-6, _d(h.landmarks[WRIST], h.landmarks[MIDDLE_MCP]))


def pinch_ratio(h: Hand) -> float:
    """Thumb-tip to index-tip distance in palm lengths: ~0.1 pinched, 1+ spread."""
    return _d(h.landmarks[THUMB_TIP], h.landmarks[INDEX_TIP]) / palm_size(h)


def finger_states(h: Hand) -> dict[str, bool]:
    """Which fingers are extended (rotation-independent: compares distances to the wrist)."""
    p, w = h.landmarks, h.landmarks[WRIST]
    ext = {n: _d(p[t], w) > _d(p[q], w) * 1.12 for n, t, q in (("index", INDEX_TIP, INDEX_PIP), ("middle", MIDDLE_TIP, MIDDLE_PIP),
                                                                ("ring", RING_TIP, RING_PIP), ("pinky", PINKY_TIP, PINKY_PIP))}
    ext["thumb"] = _d(p[THUMB_TIP], p[PINKY_MCP]) > _d(p[THUMB_IP], p[PINKY_MCP]) * 1.1 and _d(p[THUMB_TIP], p[INDEX_MCP]) > palm_size(h) * 0.45
    return ext


def classify_pose(h: Hand) -> str:
    """A resting hand shape: open_palm, fist, point, peace, three, thumbs_up, thumbs_down, ok, pinch, or ''."""
    f, p = finger_states(h), h.landmarks
    others = [f["index"], f["middle"], f["ring"], f["pinky"]]
    ratio = pinch_ratio(h)
    if ratio < 0.3:
        return "ok" if f["middle"] and f["ring"] and f["pinky"] else "pinch"
    if f["thumb"] and not any(others):
        up = p[WRIST][1] - p[THUMB_TIP][1]
        if up > palm_size(h) * 0.5:
            return "thumbs_up"
        if -up > palm_size(h) * 0.5:
            return "thumbs_down"
    if all(others) and f["thumb"]:
        return "open_palm"
    if all(others):
        return "open_palm"
    if not any(others) and not f["thumb"]:
        return "fist"
    if f["index"] and not f["middle"] and not f["ring"] and not f["pinky"]:
        return "point"
    if f["index"] and f["middle"] and not f["ring"] and not f["pinky"]:
        return "peace"
    if f["index"] and f["middle"] and f["ring"] and not f["pinky"]:
        return "three"
    return ""


@dataclass
class Event:
    gesture: str
    t: float
    armed: bool = True
    note: str = ""


@dataclass
class _Sample:
    t: float
    ratio: float
    cx: float
    cy: float
    pose: str


class GestureEngine:
    """Turns a stream of hands (with timestamps) into gesture events.

    One hand is followed (the first). Pinch/spread are detected from the thumb-index distance
    crossing two thresholds (hysteresis, so wobble near one threshold does nothing). Swipes
    need a fast, mostly-straight movement of an open palm. Resting poses fire after being held.
    """
    CLOSE, OPEN = 0.30, 0.75           # pinch ratio thresholds
    HOLD_S = 0.6                       # how long a resting pose must be held
    ARM_HOLD_S = 1.0
    ARM_WINDOW_S = 10.0
    SWIPE_DX, SWIPE_MAX_DY, SWIPE_MIN_S, SWIPE_MAX_S = 0.28, 0.14, 0.12, 0.7
    COOLDOWN_S = 0.9

    def __init__(self, require_arm: bool = True) -> None:
        self.require_arm = require_arm
        self.armed_until = 0.0
        self._hist: deque[_Sample] = deque(maxlen=40)
        self._pinch_state = None              # "open" | "closed" | None (unknown yet)
        self._pose, self._pose_since, self._pose_fired = "", 0.0, False
        self._last: dict[str, float] = {}
        self._last_any = -9.0

    # ── arming ──────────────────────────────────────────────────────────────
    def armed(self, now: float) -> bool:
        return not self.require_arm or now < self.armed_until

    def arm(self, now: float) -> None:
        self.armed_until = now + self.ARM_WINDOW_S

    def disarm(self) -> None:
        self.armed_until = 0.0

    # ── main entry ──────────────────────────────────────────────────────────
    def update(self, hands: list[Hand], now: float) -> list[Event]:
        if not hands:
            self._hist.clear(); self._pose, self._pinch_state, self._pose_fired = "", None, False
            return []
        h = hands[0]
        lm = h.landmarks
        cx = sum(p[0] for p in (lm[WRIST], lm[INDEX_MCP], lm[PINKY_MCP], lm[MIDDLE_MCP])) / 4
        cy = sum(p[1] for p in (lm[WRIST], lm[INDEX_MCP], lm[PINKY_MCP], lm[MIDDLE_MCP])) / 4
        pose, ratio = classify_pose(h), pinch_ratio(h)
        self._hist.append(_Sample(now, ratio, cx, cy, pose))
        out: list[Event] = []
        for name in self._pinch_events(ratio, now) + self._swipe_events(now) + self._pose_events(pose, now):
            ev = self._gate(name, now)
            if ev:
                out.append(ev)
        return out

    # ── detectors ───────────────────────────────────────────────────────────
    def _pinch_events(self, ratio: float, now: float) -> list[str]:
        prev = self._pinch_state
        if ratio <= self.CLOSE:
            self._pinch_state = "closed"
        elif ratio >= self.OPEN:
            self._pinch_state = "open"
        if prev is None or prev == self._pinch_state:
            return []
        # the move must be a quick, deliberate one: recent samples show the other extreme
        recent = [s for s in self._hist if now - s.t <= 0.9]
        still = max((abs(s.cx - recent[0].cx) + abs(s.cy - recent[0].cy)) for s in recent) < 0.18       # not a swipe
        if not still:
            return []
        if self._pinch_state == "closed" and max(s.ratio for s in recent) >= self.OPEN:
            return ["pinch_in"]
        if self._pinch_state == "open" and min(s.ratio for s in recent) <= self.CLOSE:
            return ["spread"]
        return []

    def _swipe_events(self, now: float) -> list[str]:
        win = [s for s in self._hist if now - s.t <= self.SWIPE_MAX_S]
        if len(win) < 4 or (win[-1].t - win[0].t) < self.SWIPE_MIN_S:
            return []
        if sum(s.pose == "open_palm" for s in win) < len(win) * 0.6:
            return []
        dx, dy = win[-1].cx - win[0].cx, win[-1].cy - win[0].cy
        if abs(dx) >= self.SWIPE_DX and abs(dy) <= self.SWIPE_MAX_DY:
            return ["swipe_right" if dx > 0 else "swipe_left"]
        if abs(dy) >= self.SWIPE_DX and abs(dx) <= self.SWIPE_MAX_DY:
            return ["swipe_down" if dy > 0 else "swipe_up"]
        return []

    def _pose_events(self, pose: str, now: float) -> list[str]:
        if pose != self._pose:
            self._pose, self._pose_since, self._pose_fired = pose, now, False
            return []
        if self._pose_fired or not pose:
            return []
        held = now - self._pose_since
        if pose == "open_palm" and held >= self.ARM_HOLD_S:
            self._pose_fired = True
            return ["open_palm_hold"]
        if pose in ("fist", "thumbs_up", "thumbs_down", "peace", "ok", "three") and held >= self.HOLD_S:
            self._pose_fired = True
            return ["fist_hold" if pose == "fist" else pose]
        return []

    # ── gate: arming, cooldowns ─────────────────────────────────────────────
    def _gate(self, name: str, now: float) -> Event | None:
        if name == "open_palm_hold":
            self.arm(now)
            return Event(name, now, True, "armed for %d s" % self.ARM_WINDOW_S) if self.require_arm else None
        if now - self._last.get(name, -9) < self.COOLDOWN_S or now - self._last_any < 0.35:
            return None
        if not self.armed(now):
            return Event(name, now, False, "ignored: hold an open palm to arm")
        self._last[name] = self._last_any = now
        self.armed_until = now + self.ARM_WINDOW_S if self.require_arm else 0.0           # activity keeps it armed
        return Event(name, now, True)


# ── doing something with a gesture ─────────────────────────────────────────────────────────
class Controller:
    """Maps gesture -> a NAMED action and performs it. press/scroll are injectable for tests."""

    def __init__(self, mapping: dict | None = None, app_callback=None, press=None, scroll=None) -> None:
        self.map = self.clean_map(mapping or DEFAULT_MAP)
        self.app_callback = app_callback
        self._press, self._scroll = press, scroll

    @staticmethod
    def clean_map(m: dict) -> dict:
        """Keep only known gestures mapped to known actions; anything else is dropped, not run."""
        return {g: a for g, a in m.items() if g in GESTURES and a in ACTIONS}

    @staticmethod
    def merged_map(saved: dict | None) -> dict:
        """Defaults, overridden by saved choices — but a saved value that is not a known action
        falls back to that gesture's default instead of switching the gesture off."""
        saved = saved or {}
        return {g: (saved[g] if saved.get(g) in ACTIONS else a) for g, a in DEFAULT_MAP.items()}

    def action_for(self, gesture: str) -> str:
        return self.map.get(gesture, "none")

    def perform(self, gesture: str) -> tuple[str, bool]:
        action = self.action_for(gesture)
        if action == "none":
            return action, False
        if action in APP_ACTIONS:
            if self.app_callback:
                self.app_callback(action)
                return action, True
            return action, False
        try:
            if action in KEYS:
                press = self._press or _pyautogui().hotkey
                press(*KEYS[action])
            else:
                (self._scroll or _pyautogui().scroll)(SCROLL[action])
            return action, True
        except Exception as e:
            print(f"[Gestures] {action} failed: {type(e).__name__}: {str(e)[:80]}")
            return action, False


def _pyautogui():
    import pyautogui
    pyautogui.FAILSAFE = True                       # slam the mouse into a screen corner to abort any automation
    return pyautogui


# ── seeing the hand ────────────────────────────────────────────────────────────────────────
def ensure_model(path: Path = MODEL_PATH, url: str = MODEL_URL, get=None) -> Path:
    """The hand model is one small file; download it once (to models/, which is not committed)."""
    if path.exists() and path.stat().st_size > 1_000_000:
        return path
    import requests
    path.parent.mkdir(parents=True, exist_ok=True)
    r = (get or requests.get)(url, timeout=120)
    if r.status_code != 200 or len(r.content) < 1_000_000:
        raise RuntimeError(f"could not download the hand model (HTTP {r.status_code}). Download {url} and save it as {path}")
    tmp = path.with_suffix(".part")
    tmp.write_bytes(r.content)
    tmp.replace(path)
    return path


class MediaPipeHands:
    """Hand landmarks from MediaPipe's HandLandmarker (the Tasks API; the old 'solutions' API is gone)."""

    def __init__(self, model_path: Path | None = None, num_hands: int = 1) -> None:
        try:
            import mediapipe as mp
            from mediapipe.tasks.python import BaseOptions, vision
        except ImportError as e:
            raise RuntimeError("hand tracking needs MediaPipe:  pip install mediapipe") from e
        self._mp = mp
        opts = vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(ensure_model(model_path or MODEL_PATH))),
            running_mode=vision.RunningMode.VIDEO, num_hands=num_hands,
            min_hand_detection_confidence=0.6, min_hand_presence_confidence=0.5, min_tracking_confidence=0.5)
        self._lm = vision.HandLandmarker.create_from_options(opts)

    def detect(self, rgb, ts_ms: int) -> list[Hand]:
        res = self._lm.detect_for_video(self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb), int(ts_ms))
        hands = []
        for i, pts in enumerate(res.hand_landmarks or []):
            cat = (res.handedness[i][0] if res.handedness and i < len(res.handedness) and res.handedness[i] else None)
            hands.append(Hand([(p.x, p.y, p.z) for p in pts], getattr(cat, "category_name", ""), float(getattr(cat, "score", 1.0))))
        return hands


class GestureService:
    """Camera -> hands -> gestures -> actions, on a background thread."""

    def __init__(self, camera_index: int = 0, tracker=None, engine: GestureEngine | None = None, controller: Controller | None = None,
                 on_event=None, on_frame=None, fps: float = 15.0, capture=None, mirror: bool = True) -> None:
        self.camera_index, self.tracker, self.engine = camera_index, tracker, engine or GestureEngine()
        self.controller = controller or Controller()
        self.on_event, self.on_frame, self.fps, self.mirror = on_event or (lambda e: None), on_frame, fps, mirror
        self._capture_factory = capture
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._t0 = time.monotonic()
        self.error = ""
        self.running = False

    def now(self) -> float:
        """The engine's clock (seconds since the camera started): use this, never time.monotonic(), when arming."""
        return time.monotonic() - self._t0

    def _open(self):
        if self._capture_factory:
            return self._capture_factory()
        import cv2
        cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW) if sys.platform == "win32" else cv2.VideoCapture(self.camera_index)
        if not cap.isOpened():
            raise RuntimeError(f"camera {self.camera_index} could not be opened (is another program, like the camera preview, using it?)")
        return cap

    def start(self) -> None:
        if self.running:
            return
        cap = self._open()                                  # fail early, with a clear message, on the caller's thread
        if self.tracker is None:
            self.tracker = MediaPipeHands()
        self._stop.clear()
        self._t0 = time.monotonic()
        self.running = True
        self._thread = threading.Thread(target=self._loop, args=(cap,), daemon=True, name="gestures")
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=3)
        self.running = False

    def _loop(self, cap) -> None:
        import cv2
        period = 1.0 / self.fps
        try:
            while not self._stop.is_set():
                started = time.monotonic()
                ok, frame = cap.read()
                if not ok:
                    self.error = "the camera stopped giving frames"
                    break
                if self.mirror:
                    frame = cv2.flip(frame, 1)
                now = started - self._t0
                hands = self.tracker.detect(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), int(now * 1000))
                for ev in self.engine.update(hands, now):
                    action, done = self.controller.perform(ev.gesture) if ev.armed and ev.gesture != "open_palm_hold" else ("none", False)
                    self.on_event({"gesture": ev.gesture, "armed": ev.armed, "action": action, "done": done, "note": ev.note})
                if self.on_frame:
                    self.on_frame(frame, hands, {"armed": self.engine.armed(now), "pose": classify_pose(hands[0]) if hands else ""})
                time.sleep(max(0.0, period - (time.monotonic() - started)))
        except Exception as e:
            self.error = f"{type(e).__name__}: {str(e)[:100]}"
        finally:
            try:
                cap.release()
            except Exception:
                pass
            self.running = False
