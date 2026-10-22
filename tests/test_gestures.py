import math
import threading
import time

import numpy as np
import pytest
from core import gestures as G
from core.gestures import Controller, GestureEngine, GestureService, Hand, classify_pose, pinch_ratio

# ── synthetic hands: (u across, v up) in palm lengths, wrist at v=-1, mapped into frame coordinates ──
MCP = {"index": (-0.4, -0.1), "middle": (0.0, 0.0), "ring": (0.4, -0.1), "pinky": (0.75, -0.25)}
IDX = {"index": (5, 6, 7, 8), "middle": (9, 10, 11, 12), "ring": (13, 14, 15, 16), "pinky": (17, 18, 19, 20)}


def hand(cx=0.5, cy=0.5, s=0.15, fingers=(), thumb="curl", ratio=None, rot=0.0) -> Hand:
    uv = {0: (0.0, -1.0), 1: (-0.35, -0.8), 2: (-0.6, -0.55)}
    for name, (mcp, pip, dip, tip) in IDX.items():
        u, v = MCP[name]
        uv[mcp] = (u, v)
        if name in fingers:
            uv[pip], uv[dip], uv[tip] = (u, v + 0.5), (u, v + 0.8), (u, v + 1.05)
        else:
            uv[pip], uv[dip], uv[tip] = (u, v + 0.45), (u, v + 0.3), (u, v + 0.05)
    t = {"out": [(-0.85, -0.3), (-1.15, -0.05)], "up": [(-0.6, 0.2), (-0.5, 0.9)], "down": [(-0.6, -1.0), (-0.5, -1.9)],
         "curl": [(-0.85, -0.3), (-0.2, -0.3)]}[thumb]
    uv[3], uv[4] = t
    if ratio is not None:                                    # thumb tip placed `ratio` palm-lengths from the index tip
        iu, iv = uv[8]
        uv[3], uv[4] = (iu - ratio * 0.5, iv - 0.3), (iu - ratio, iv)
    pts = []
    for i in range(21):
        u, v = uv[i]
        c, sn = math.cos(rot), math.sin(rot)
        u, v = u * c - v * sn, u * sn + v * c
        pts.append((cx + u * s, cy - v * s, 0.0))
    return Hand(pts)


OPEN = dict(fingers=("index", "middle", "ring", "pinky"), thumb="out")
POSES = {
    "open_palm": OPEN, "fist": dict(), "point": dict(fingers=("index",)), "peace": dict(fingers=("index", "middle")),
    "three": dict(fingers=("index", "middle", "ring")), "thumbs_up": dict(thumb="up"), "thumbs_down": dict(thumb="down"),
    "pinch": dict(fingers=("index",), ratio=0.1), "ok": dict(fingers=("index", "middle", "ring", "pinky"), ratio=0.1),
}


@pytest.mark.parametrize("name", POSES)
def test_every_pose_is_recognised(name):
    assert classify_pose(hand(**POSES[name])) == name


@pytest.mark.parametrize("name", POSES)
@pytest.mark.parametrize("rot", [0.6, -0.9, math.pi / 2])
def test_poses_survive_rotating_the_hand(name, rot):
    if name in ("thumbs_up", "thumbs_down") and abs(rot) > 0.7:
        pytest.skip("thumbs up/down are defined relative to the screen's up")
    assert classify_pose(hand(rot=rot, **POSES[name])) == name


@pytest.mark.parametrize("size", [0.08, 0.15, 0.3])
def test_hand_distance_from_the_camera_does_not_matter(size):
    assert classify_pose(hand(s=size, **POSES["peace"])) == "peace" and pinch_ratio(hand(s=size, ratio=0.5, fingers=("index",))) == pytest.approx(0.5, abs=0.05)


def run(engine, frames, fps=15.0, t0=0.0):
    """frames: list of Hand|None. returns [(gesture, armed)]"""
    out = []
    for i, h in enumerate(frames):
        out += [(e.gesture, e.armed) for e in engine.update([h] if h else [], t0 + i / fps)]
    return out


def pinch_sequence(start, end, n=7, **kw):
    return [hand(ratio=start + (end - start) * i / (n - 1), fingers=("index",), **kw) for i in range(n)]


def test_pinch_in_closes_and_spread_opens():
    e = GestureEngine(require_arm=False)
    seq = [hand(ratio=1.1, fingers=("index",))] * 6 + pinch_sequence(1.1, 0.1) + [hand(ratio=0.1, fingers=("index",))] * 4
    assert run(e, seq) == [("pinch_in", True)]
    assert run(e, pinch_sequence(0.1, 1.2, 8) + [hand(ratio=1.2, fingers=("index",))] * 4, t0=5.0) == [("spread", True)]


def test_wobble_near_a_threshold_does_nothing():
    e = GestureEngine(require_arm=False)
    rnd = np.random.RandomState(3)
    seq = [hand(ratio=0.5 + float(rnd.uniform(-0.12, 0.12)), fingers=("index",)) for _ in range(120)]
    assert run(e, seq) == []


def test_nothing_happens_until_armed_then_it_does():
    e = GestureEngine(require_arm=True)
    pinch = [hand(ratio=1.1, fingers=("index",))] * 6 + pinch_sequence(1.1, 0.1)
    assert [g for g, armed in run(e, pinch) if armed] == []                     # not armed: reported as ignored, never acted on
    arm = [hand(**OPEN)] * 20                                                 # 1.3 s of open palm
    assert ("open_palm_hold", True) in run(e, arm, t0=10.0)
    assert e.armed(10.0 + 20 / 15)
    again = [hand(ratio=1.1, fingers=("index",))] * 6 + pinch_sequence(1.1, 0.1)
    assert run(e, again, t0=12.0) == [("pinch_in", True)]
    assert not e.armed(12.0 + 30)                                             # and it disarms itself


def test_armed_state_lapses_after_ten_idle_seconds():
    e = GestureEngine(); e.arm(100.0)
    assert e.armed(105.0) and not e.armed(111.0)


def swipe(dx, dy, secs=0.4, pose=OPEN, n=7):
    return [hand(cx=0.5 - dx / 2 + dx * i / (n - 1), cy=0.5 - dy / 2 + dy * i / (n - 1), **pose) for i in range(n)]


@pytest.mark.parametrize("dx,dy,expect", [(0.45, 0.0, "swipe_right"), (-0.45, 0.02, "swipe_left"), (0.0, 0.45, "swipe_down"), (0.01, -0.45, "swipe_up")])
def test_swipes_in_four_directions(dx, dy, expect):
    e = GestureEngine(require_arm=False)
    assert run(e, swipe(dx, dy), fps=17.0) == [(expect, True)]


@pytest.mark.parametrize("name,frames", [
    ("too slow", swipe(0.45, 0.0, n=40)), ("too short", swipe(0.12, 0.0)), ("diagonal", swipe(0.4, 0.35)),
    ("a fist moving (not an open palm)", swipe(0.45, 0.0, pose=dict()))])
def test_things_that_are_not_swipes(name, frames):
    e = GestureEngine(require_arm=False)
    fps = 17.0 if len(frames) < 20 else 9.0                                   # the slow one spans ~4 s
    assert run(e, frames, fps=fps) == [], name


def test_resting_poses_fire_once_after_a_hold_and_not_before():
    e = GestureEngine(require_arm=False)
    assert run(e, [hand(**POSES["fist"])] * 4) == []                           # 0.27 s: not held long enough
    e = GestureEngine(require_arm=False)
    assert run(e, [hand(**POSES["fist"])] * 30) == [("fist_hold", True)]       # held for 2 s: once, not repeatedly
    for pose, gesture in (("thumbs_up", "thumbs_up"), ("peace", "peace"), ("ok", "ok"), ("three", "three"), ("thumbs_down", "thumbs_down")):
        e = GestureEngine(require_arm=False)
        assert run(e, [hand(**POSES[pose])] * 15) == [(gesture, True)], pose


def test_cooldown_stops_double_triggers():
    e = GestureEngine(require_arm=False)
    seq = ([hand(ratio=1.1, fingers=("index",))] * 4 + pinch_sequence(1.1, 0.1, 5)) * 2         # two pinches inside 2 s but < cooldown apart? ~1.3 s
    got = [g for g, _ in run(e, seq, fps=30.0)]
    assert got.count("pinch_in") == 1


def test_the_hand_leaving_the_frame_resets_everything():
    e = GestureEngine(require_arm=False)
    run(e, [hand(**POSES["fist"])] * 8)
    run(e, [None] * 5, t0=1.0)
    assert run(e, [hand(**POSES["fist"])] * 4, t0=2.0) == []                   # the hold timer started over


# ── actions: a fixed list, never a command ──────────────────────────────────────────────────
def test_the_map_keeps_only_known_gestures_and_known_actions():
    c = Controller({"pinch_in": "close_tab", "spread": "rm -rf /", "swipe_up": "os.system('x')", "made_up": "new_tab", "peace": "none"})
    assert c.map == {"pinch_in": "close_tab", "peace": "none"}


def test_defaults_are_all_valid_and_the_action_list_is_closed():
    assert all(a in G.ACTIONS for a in G.DEFAULT_MAP.values()) and all(g in G.GESTURES for g in G.DEFAULT_MAP)
    allowed_keys = {"ctrl", "command", "shift", "win", "tab", "t", "w", "4", "playpause", "volumeup", "volumedown", "s"}
    assert all(k in allowed_keys for keys in G.KEYS.values() for k in keys)


def test_gestures_perform_the_named_actions():
    pressed, scrolled, app = [], [], []
    c = Controller(G.DEFAULT_MAP, app_callback=app.append, press=lambda *k: pressed.append(k), scroll=scrolled.append)
    assert c.perform("pinch_in") == ("close_tab", True) and pressed[-1] == tuple(G.KEYS["close_tab"])
    assert c.perform("spread")[0] == "new_tab" and pressed[-1] == tuple(G.KEYS["new_tab"])
    assert c.perform("swipe_right")[0] == "next_tab" and pressed[-1] == ("ctrl", "tab")
    assert c.perform("swipe_left")[0] == "prev_tab" and pressed[-1] == ("ctrl", "shift", "tab")
    assert c.perform("swipe_up") == ("scroll_down", True) and scrolled == [-500]
    assert c.perform("three") == ("show_jobs", True) and app == ["show_jobs"]
    assert c.perform("open_palm_hold") == ("none", False)


def test_a_failing_keypress_is_reported_not_raised():
    def boom(*k): raise RuntimeError("no display")
    assert Controller(press=boom).perform("pinch_in") == ("close_tab", False)


def test_the_failsafe_corner_abort_is_always_on(monkeypatch):
    import types, sys
    fake = types.SimpleNamespace(FAILSAFE=False)
    monkeypatch.setitem(sys.modules, "pyautogui", fake)
    G._pyautogui()
    assert fake.FAILSAFE is True


# ── the camera service ──────────────────────────────────────────────────────────────────────
class Cam:
    def __init__(self, n=40): self.n, self.released, self.i = n, False, 0
    def isOpened(self): return True
    def read(self):
        self.i += 1
        return (self.i <= self.n), np.zeros((120, 160, 3), np.uint8)
    def release(self): self.released = True


class Script:
    def __init__(self, frames): self.frames, self.k = frames, 0
    def detect(self, rgb, ts):
        assert rgb.shape == (120, 160, 3)
        h = self.frames[min(self.k, len(self.frames) - 1)]; self.k += 1
        return [h] if h else []


def test_the_service_turns_camera_frames_into_actions_and_releases_the_camera():
    seq = [hand(ratio=1.1, fingers=("index",))] * 6 + pinch_sequence(1.1, 0.1) + [hand(ratio=0.1, fingers=("index",))] * 4
    pressed, events, frames = [], [], []
    cam = Cam(len(seq))
    svc = GestureService(tracker=Script(seq), engine=GestureEngine(require_arm=False), controller=Controller(press=lambda *k: pressed.append(k)),
                         on_event=events.append, on_frame=lambda f, h, st: frames.append(st), fps=400.0, capture=lambda: cam)
    svc.start()
    for _ in range(100):
        if not svc.running: break
        time.sleep(0.02)
    svc.stop()
    assert [e["gesture"] for e in events] == ["pinch_in"] and events[0]["action"] == "close_tab" and events[0]["done"]
    assert pressed == [tuple(G.KEYS["close_tab"])] and cam.released and len(frames) >= len(seq) - 1


def test_while_unarmed_the_service_reports_but_does_not_act():
    seq = [hand(ratio=1.1, fingers=("index",))] * 6 + pinch_sequence(1.1, 0.1) + [hand(ratio=0.1, fingers=("index",))] * 4
    pressed, events = [], []
    svc = GestureService(tracker=Script(seq), engine=GestureEngine(require_arm=True), controller=Controller(press=lambda *k: pressed.append(k)),
                         on_event=events.append, fps=400.0, capture=lambda: Cam(len(seq)))
    svc.start()
    for _ in range(100):
        if not svc.running: break
        time.sleep(0.02)
    assert pressed == [] and events and not events[0]["armed"] and "arm" in events[0]["note"]


def test_a_busy_or_missing_camera_gives_a_useful_error(monkeypatch):
    import cv2
    monkeypatch.setattr(cv2, "VideoCapture", lambda *a, **k: types_cap())
    with pytest.raises(RuntimeError, match="could not be opened"):
        GestureService(tracker=Script([None])).start()


def types_cap():
    class C:
        def isOpened(self): return False
    return C()


def test_a_camera_that_stops_is_reported_and_cleaned_up():
    cam = Cam(3)
    svc = GestureService(tracker=Script([None]), fps=400.0, capture=lambda: cam)
    svc.start()
    for _ in range(100):
        if not svc.running: break
        time.sleep(0.02)
    assert "stopped" in svc.error and cam.released


def test_the_hand_model_downloads_once_and_is_checked(tmp_path):
    import types
    path = tmp_path / "models" / "hand.task"
    calls = []
    ok = types.SimpleNamespace(status_code=200, content=b"x" * 2_000_000)
    assert G.ensure_model(path, "http://m", get=lambda u, **k: calls.append(u) or ok) == path and path.stat().st_size == 2_000_000
    G.ensure_model(path, "http://m", get=lambda u, **k: calls.append(u) or ok)
    assert len(calls) == 1                                                     # second time: already there
    bad = types.SimpleNamespace(status_code=200, content=b"<html>error</html>")
    with pytest.raises(RuntimeError, match="could not download the hand model"):
        G.ensure_model(tmp_path / "other.task", "http://m", get=lambda u, **k: bad)
    assert not (tmp_path / "other.task").exists()


# ── panel, tool and wiring ──────────────────────────────────────────────────────────────────
import os
import types
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


_QAPP = None


def qapp():
    """One QApplication kept alive for the whole run (if it is garbage-collected, Qt aborts the process)."""
    global _QAPP
    pytest.importorskip("PyQt6")
    from PyQt6.QtWidgets import QApplication
    _QAPP = QApplication.instance() or QApplication([])
    return _QAPP


class FakeService:
    def __init__(self, camera_index=0, engine=None, controller=None, on_event=None, on_frame=None, **k):
        self.engine, self.controller, self.on_event, self.on_frame, self.running, self.error = engine, controller, on_event, on_frame, False, ""
        self._t0 = time.monotonic()
    def start(self): self.running = True
    def stop(self): self.running = False
    def now(self): return time.monotonic() - self._t0


def test_the_service_has_one_clock_so_arming_cannot_last_forever():
    svc = GestureService(tracker=Script([None]), capture=lambda: Cam(1))
    svc._t0 = time.monotonic() - 1000                                        # the camera has been on for ~17 minutes
    assert 999 < svc.now() < 1001
    e = GestureEngine(); e.arm(svc.now())
    assert e.armed(svc.now() + 5) and not e.armed(svc.now() + 11)           # still exactly 10 s, whatever the uptime


def test_panel_start_stop_preview_events_and_arming():
    qapp()
    from core.gesture_panel import GesturePanel
    saved = {}
    p = GesturePanel({}, save=saved.__setitem__, service_factory=FakeService)
    msgs = []; p.status.connect(msgs.append)
    assert p.state.text() == "OFF" and not p.is_on()
    assert p.start() and p.is_on() and "ON" in p.state.text() and "hold an open palm" in p.state.text() and "ON (camera in use)" in msgs[0]
    img = np.zeros((120, 160, 3), np.uint8)
    p._on_frame(img, True, "fist")
    assert "ARMED" in p.state.text() and "fist" in p.state.text() and p.preview.pixmap() is not None
    p._on_event({"gesture": "pinch_in", "armed": True, "action": "close_tab", "done": True, "note": ""})
    assert "close tab" in p.last.text() and any("close tab" in m for m in msgs)
    p._on_event({"gesture": "spread", "armed": False, "action": "none", "done": False, "note": ""})
    assert "ignored" in p.last.text()
    p.arming.setChecked(False)
    assert saved["gesture_arming"] is False and p.service.engine.require_arm is False
    p.stop()
    assert not p.is_on() and p.state.text() == "OFF" and "camera released" in msgs[-1]


def test_panel_reports_why_it_could_not_start():
    qapp()
    from core.gesture_panel import GesturePanel
    def broken(**k): raise RuntimeError("camera 0 could not be opened (is another program using it?)")
    p = GesturePanel({}, service_factory=broken)
    msgs = []; p.status.connect(msgs.append)
    assert not p.start() and not p.is_on() and "could not start" in p.state.text() and "another program" in p.last.text() and msgs


def test_the_gesture_map_is_editable_saved_and_validated():
    qapp()
    from core.gesture_panel import GesturePanel
    saved = {}
    p = GesturePanel({"gesture_map": {"peace": "none", "spread": "rm -rf /"}}, save=saved.__setitem__, service_factory=FakeService)
    assert p.mapping()["peace"] == "none" and p.mapping()["spread"] == "new_tab"             # a bad saved value falls back to the default
    p.start()
    assert p.set_mapping("pinch_in", "volume_up") and saved["gesture_map"]["pinch_in"] == "volume_up"
    assert p.service.controller.map["pinch_in"] == "volume_up"                               # takes effect immediately
    assert not p.set_mapping("pinch_in", "delete_everything") and not p.set_mapping("nonsense", "new_tab")


def test_app_gestures_reach_the_main_window_and_arm_toggles():
    qapp()
    from core.gesture_panel import GesturePanel
    p = GesturePanel({}, service_factory=FakeService)
    got = []; p.appAction.connect(got.append)
    p.start(); p._app_action("show_jobs")
    assert got == ["show_jobs"]
    e, now = p.service.engine, p.service.now()
    p._app_action("toggle_gestures_arm"); assert e.armed(p.service.now())
    p._app_action("toggle_gestures_arm"); assert not e.armed(p.service.now())


def test_the_hand_is_drawn_over_the_preview():
    pytest.importorskip("cv2")
    from core.gesture_panel import draw_hands
    frame = np.zeros((240, 320, 3), np.uint8)
    out = draw_hands(frame, [hand(cx=0.5, cy=0.5, **POSES["open_palm"])])
    assert out.sum() > 0 and frame.sum() == 0                                                # drawn on a copy


def test_voice_tool_enable_needs_confirmation_and_set_is_validated(monkeypatch):
    import actions.gestures as A
    cmds, got = [], {}
    player = types.SimpleNamespace(gesture_command=cmds.append, gesture_status=lambda: "Gesture control is off.")
    monkeypatch.setattr(A.confirm, "request", lambda k, t, d, run: got.update(t=t, d=d, run=run) or "pending")
    assert A.gestures({"mode": "enable"}, player) == "pending" and cmds == [] and "camera" in got["d"]
    got["run"](); assert cmds == ["on"]
    assert "off" in A.gestures({"mode": "disable"}, player) and cmds[-1] == "off"
    assert A.gestures({"mode": "status"}, player) == "Gesture control is off."
    assert "Gestures:" in A.gestures({"mode": "set", "gesture": "pinch_in", "action": "format_disk"}, player)
    assert "needs the app" in A.gestures({"mode": "enable"}, None)


def test_voice_tool_set_saves_a_clean_map(monkeypatch):
    import actions.gestures as A
    import memory.config_manager as cm
    saved = {}
    monkeypatch.setattr(cm, "_save_flag", lambda k, v: saved.update({k: v})); monkeypatch.setattr(cm, "load_api_keys", lambda: {})
    out = A.gestures({"mode": "set", "gesture": "pinch in", "action": "close tab"}, types.SimpleNamespace(gesture_command=lambda c: None))
    assert "pinch in now does: close tab" in out and saved["gesture_map"]["pinch_in"] == "close_tab"
    assert "pinch in → close tab" in A.gestures({"mode": "list"}, None) or "pinch" in A.gestures({"mode": "list"}, None)


def test_ui_wiring_and_tool_registration():
    ui = pytest.importorskip("ui")
    got = []
    j = types.SimpleNamespace(_win=types.SimpleNamespace(_gesture_cmd_sig=types.SimpleNamespace(emit=got.append), _gesture_panel=None))
    ui.JarvisUI.gesture_command(j, "on")
    assert got == ["on"] and ui.JarvisUI.gesture_status(j) == "Gesture control is off."
    from pathlib import Path
    from core.action_loader import discover_actions
    d = {x["name"]: x for x in discover_actions(Path("actions"), logger=lambda s: None).get_tool_declarations()}
    assert set(d["gestures"]) <= {"name", "description", "parameters"}
