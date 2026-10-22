"""The GESTURES window: what the camera sees (with your hand drawn over it), whether control is
on / armed, the last gesture, and the gesture -> action map you can change."""
from __future__ import annotations

import time

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont, QImage, QPixmap
from PyQt6.QtWidgets import (QCheckBox, QComboBox, QHBoxLayout, QLabel, QPushButton, QTableWidget, QTableWidgetItem,
                             QVBoxLayout, QWidget, QHeaderView, QAbstractItemView)

from core import gestures as G

_QSS = """
QWidget { background:#020d14; color:#9fe9f5; font-family:'Courier New'; font-size:11px; }
QLabel { background:transparent; } QLabel#status { font-size:13px; font-weight:bold; }
QLabel#hint { color:#5a8f9c; }
QPushButton { background:#00091a; color:#00d4ff; border:1px solid #0b5b73; border-radius:3px; padding:6px 12px; }
QPushButton:hover { border-color:#00d4ff; } QPushButton:checked { background:#0b5b73; color:white; }
QTableWidget { background:#00091a; gridline-color:#0b3a4a; border:1px solid #0b3a4a; }
QHeaderView::section { background:#04202c; color:#5fd4e6; border:0; padding:4px; }
QComboBox { background:#00091a; border:1px solid #0b3a4a; padding:2px; color:#c8f6ff; }
"""
LABEL = {"pinch_in": "🤏 Pinch (fingers close)", "spread": "🖐 Spread (fingers open)", "swipe_right": "👉 Swipe right", "swipe_left": "👈 Swipe left",
         "swipe_up": "☝ Swipe up", "swipe_down": "👇 Swipe down", "fist_hold": "✊ Fist (hold)", "thumbs_up": "👍 Thumbs up",
         "thumbs_down": "👎 Thumbs down", "peace": "✌ Peace sign", "ok": "👌 OK sign", "three": "3 fingers (hold)"}
ACTION_TEXT = {"none": "do nothing", "new_tab": "new tab", "close_tab": "close tab", "next_tab": "next tab", "prev_tab": "previous tab",
               "scroll_down": "scroll down", "scroll_up": "scroll up", "play_pause": "play / pause", "volume_up": "volume up",
               "volume_down": "volume down", "screenshot": "screenshot", "show_files": "show FILES", "show_jobs": "show JOBS",
               "show_code": "show CODE preview", "toggle_mute": "mute / unmute assistant", "toggle_gestures_arm": "arm / disarm"}


def draw_hands(frame, hands):
    """Draw the tracked hand on a copy of the frame so you can see what the computer sees."""
    import cv2
    img = frame.copy()
    h, w = img.shape[:2]
    for hand in hands:
        pts = [(int(x * w), int(y * h)) for x, y, _ in hand.landmarks]
        for a, b in G.CONNECTIONS:
            cv2.line(img, pts[a], pts[b], (255, 212, 0), 2)
        for p in pts:
            cv2.circle(img, p, 3, (0, 220, 120), -1)
    return img


class GesturePanel(QWidget):
    appAction = pyqtSignal(str)                  # show_files / show_jobs / toggle_mute ... for the main window
    status = pyqtSignal(str)                     # a line for the chat log
    _event_sig = pyqtSignal(dict)
    _frame_sig = pyqtSignal(object, bool, str)

    def __init__(self, config: dict | None = None, save=None, service_factory=None, parent=None) -> None:
        super().__init__(parent)
        self.cfg = dict(config or {})
        self._save = save or (lambda k, v: None)
        self._factory = service_factory
        self.service: G.GestureService | None = None
        self.setStyleSheet(_QSS)
        self.setMinimumWidth(420)
        lay = QVBoxLayout(self)
        head = QLabel("🖐  GESTURE CONTROL"); head.setFont(QFont("Courier New", 11, QFont.Weight.Bold)); lay.addWidget(head)
        self.preview = QLabel("Camera is off"); self.preview.setFixedSize(400, 300); self.preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview.setStyleSheet("background:#000; border:1px solid #0b3a4a;"); lay.addWidget(self.preview)
        self.state = QLabel("OFF"); self.state.setObjectName("status"); lay.addWidget(self.state)
        self.last = QLabel(""); self.last.setObjectName("hint"); self.last.setWordWrap(True); lay.addWidget(self.last)
        row = QHBoxLayout()
        self.toggle = QPushButton("Turn on"); self.toggle.setCheckable(True); self.toggle.clicked.connect(self._toggle); row.addWidget(self.toggle)
        self.arming = QCheckBox("Require open palm to arm"); self.arming.setChecked(bool(self.cfg.get("gesture_arming", True)))
        self.arming.toggled.connect(self._arming_changed); row.addWidget(self.arming)
        lay.addLayout(row)
        tip = QLabel("Hold an open palm for 1 s to ARM; gestures then work for 10 s after your last one. "
                     "Video stays on this computer and is never saved."); tip.setObjectName("hint"); tip.setWordWrap(True); lay.addWidget(tip)
        self.table = QTableWidget(len(LABEL), 2); self.table.setHorizontalHeaderLabels(["Gesture", "Does"])
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.table.verticalHeader().hide(); self.table.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        self._combos: dict[str, QComboBox] = {}
        current = G.Controller.merged_map(self.cfg.get("gesture_map"))
        for r, (g, text) in enumerate(LABEL.items()):
            self.table.setItem(r, 0, QTableWidgetItem(text))
            cb = QComboBox()
            for a in G.ACTIONS:
                cb.addItem(ACTION_TEXT.get(a, a), a)
            cb.setCurrentIndex(max(0, cb.findData(current.get(g, "none"))))
            cb.currentIndexChanged.connect(lambda _i, g=g, cb=cb: self._map_changed(g, cb.currentData()))
            self.table.setCellWidget(r, 1, cb); self._combos[g] = cb
        lay.addWidget(self.table, 1)
        self._event_sig.connect(self._on_event); self._frame_sig.connect(self._on_frame)
        self._last_frame = 0.0

    # ── config ──────────────────────────────────────────────────────────────
    def mapping(self) -> dict:
        return G.Controller.clean_map({g: cb.currentData() for g, cb in self._combos.items()})

    def _map_changed(self, g: str, action: str) -> None:
        m = self.mapping()
        self._save("gesture_map", m)
        if self.service:
            self.service.controller.map = m

    def set_mapping(self, gesture: str, action: str) -> bool:
        if gesture not in self._combos or action not in G.ACTIONS:
            return False
        cb = self._combos[gesture]; cb.setCurrentIndex(cb.findData(action))
        return True

    def _arming_changed(self, on: bool) -> None:
        self._save("gesture_arming", bool(on))
        if self.service:
            self.service.engine.require_arm = on
            self.service.engine.disarm()

    # ── on / off ────────────────────────────────────────────────────────────
    def is_on(self) -> bool:
        return self.service is not None and self.service.running

    def _toggle(self, checked: bool) -> None:
        self.start() if checked else self.stop()

    def start(self) -> bool:
        if self.is_on():
            return True
        try:
            engine = G.GestureEngine(require_arm=self.arming.isChecked())
            ctl = G.Controller(self.mapping(), app_callback=self._app_action)
            svc = (self._factory or G.GestureService)(camera_index=int(self.cfg.get("gesture_camera", self.cfg.get("camera_index", 0))),
                                                      engine=engine, controller=ctl, on_event=self._event_sig.emit, on_frame=self._frame_cb)
            svc.start()
        except Exception as e:
            self.service = None
            self.toggle.setChecked(False); self.toggle.setText("Turn on")
            self.state.setText("OFF — could not start")
            self.last.setText(f"{e}")
            self.status.emit(f"SYS: Gesture control could not start ({str(e)[:120]})")
            return False
        self.service = svc
        self.toggle.setChecked(True); self.toggle.setText("Turn off")
        self._set_state(False)
        self.status.emit("SYS: 🖐 Gesture control is ON (camera in use). Hold an open palm to arm it.")
        return True

    def stop(self) -> None:
        if self.service:
            self.service.stop()
        self.service = None
        self.toggle.setChecked(False); self.toggle.setText("Turn on")
        self.state.setText("OFF"); self.preview.setPixmap(QPixmap()); self.preview.setText("Camera is off")
        self.status.emit("SYS: 🖐 Gesture control is OFF (camera released).")

    def _app_action(self, action: str) -> None:
        if action == "toggle_gestures_arm" and self.service:
            e, now = self.service.engine, self.service.now()
            e.disarm() if e.armed(now) else e.arm(now)
            return
        self.appAction.emit(action)

    # ── events from the camera thread (arrive via signals, so they run on the Qt thread) ──
    def _frame_cb(self, frame, hands, st) -> None:
        now = time.monotonic()
        if now - self._last_frame < 0.1:                 # ~10 fps preview is plenty
            return
        self._last_frame = now
        self._frame_sig.emit(draw_hands(frame, hands), bool(st.get("armed")), str(st.get("pose", "")))

    def _on_frame(self, img, armed: bool, pose: str) -> None:
        h, w = img.shape[:2]
        qi = QImage(img.data, w, h, w * 3, QImage.Format.Format_BGR888).copy()
        self.preview.setPixmap(QPixmap.fromImage(qi).scaled(400, 300, Qt.AspectRatioMode.KeepAspectRatio))
        self._set_state(armed, pose)
        if self.service and self.service.error and not self.service.running:
            self.stop(); self.last.setText(self.service.error)

    def _set_state(self, armed: bool, pose: str = "") -> None:
        if not self.arming.isChecked():
            self.state.setText("ON — gestures active" + (f"   (seeing: {pose})" if pose else ""))
        else:
            self.state.setText(("ARMED ✅ — gestures active" if armed else "ON — hold an open palm to arm") + (f"   (seeing: {pose})" if pose else ""))

    def _on_event(self, ev: dict) -> None:
        g = LABEL.get(ev["gesture"], ev["gesture"].replace("_", " "))
        if ev["gesture"] == "open_palm_hold":
            self.last.setText("Armed ✅")
        elif not ev["armed"]:
            self.last.setText(f"{g}: ignored (not armed — hold an open palm first)")
        else:
            self.last.setText(f"{g} → {ACTION_TEXT.get(ev['action'], ev['action'])}" + ("" if ev["done"] else "  (could not do it)"))
            self.status.emit(f"SYS: 🖐 {g} → {ACTION_TEXT.get(ev['action'], ev['action'])}")
