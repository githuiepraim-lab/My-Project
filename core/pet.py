"""
Desktop pet: a small animated animal that lives on top of your windows.

It walks along the bottom of the screen, sits, naps and jumps; its eyes follow
your cursor; you can drag it; and when the assistant speaks it stops, turns to
you and talks with a moving mouth and a speech bubble. Drawn entirely with
QPainter, so there are no image files to ship.
"""
from __future__ import annotations

import math
import random
import time

from PyQt6.QtCore import QPointF, QRect, QRectF, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import (QBrush, QColor, QCursor, QFont, QFontMetrics, QPainter, QPainterPath,
                         QPen, QRegion)
from PyQt6.QtWidgets import QApplication, QMenu, QWidget

ANIMALS = {
    "cat":  {"body": (236, 160, 72), "dark": (176, 106, 40), "belly": (252, 228, 196), "ear": "point", "tail": "long", "whisk": True},
    "dog":  {"body": (178, 128, 84), "dark": (118, 78, 46), "belly": (242, 224, 198), "ear": "flop", "tail": "short", "whisk": False},
    "fox":  {"body": (226, 104, 44), "dark": (150, 60, 26), "belly": (252, 242, 228), "ear": "point", "tail": "fluffy", "whisk": True},
    "panda": {"body": (246, 246, 246), "dark": (34, 34, 38), "belly": (255, 255, 255), "ear": "round", "tail": "short", "whisk": False},
}
W, H = 340, 300
S = 1.25                       # animal scale inside the window


def _c(rgb, a=255):
    return QColor(rgb[0], rgb[1], rgb[2], a)


class DesktopPet(QWidget):
    clicked = pyqtSignal()
    closed = pyqtSignal()

    def __init__(self, kind: str = "cat", speaking_probe=None, parent=None) -> None:
        super().__init__(parent, Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint
                         | Qt.WindowType.Tool | Qt.WindowType.NoDropShadowWindowHint)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
        self.setFixedSize(W, H)
        self.kind = kind if kind in ANIMALS else "cat"
        self._probe = speaking_probe or (lambda: False)
        self._rng = random.Random()
        self._t = 0.0
        self._last = time.monotonic()
        self._x = 200.0
        self._dir = 1
        self._state = "sit"
        self._state_until = self._t + 2.0
        self._jump_t = -1.0
        self._blink_at = 2.0
        self._blink = 0.0
        self._bubble = ""
        self._bubble_until = 0.0
        self._talk_until = 0.0
        self._mouth = 0.0
        self._drag = None
        self._dy = 0.0                  # vertical offset (jump / being held)
        self._fall = 0.0
        self._place()
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._timer.start(33)

    # ── geometry ────────────────────────────────────────────────────────────
    def _screen(self) -> QRect:
        s = self.screen() or QApplication.primaryScreen()
        return s.availableGeometry()

    def _place(self) -> None:
        g = self._screen()
        self._x = max(g.left(), min(self._x, g.right() - W))
        self.move(int(self._x), g.bottom() - H + 6 + int(self._dy))

    # ── public API ──────────────────────────────────────────────────────────
    def say(self, text: str, seconds: float | None = None) -> None:
        """Show a speech bubble and talk for about as long as it takes to read."""
        text = " ".join(str(text).split())
        if not text:
            return
        self._bubble = text if len(text) <= 240 else text[:237] + "…"
        dur = seconds if seconds else max(3.0, min(14.0, len(text) / 13.0))
        self._bubble_until = self._t + dur
        self._talk_until = self._t + dur
        if self._state == "sleep":
            self._state = "sit"

    def set_kind(self, kind: str) -> None:
        if kind in ANIMALS:
            self.kind = kind
            self.update()

    @property
    def talking(self) -> bool:
        return self._t < self._talk_until or bool(self._probe())

    # ── behaviour ───────────────────────────────────────────────────────────
    def _tick(self) -> None:
        now = time.monotonic()
        dt = min(0.1, now - self._last)
        self._last = now
        self._t += dt
        g = self._screen()
        talking = self.talking
        if self._drag is None:
            if self._fall:
                self._dy = min(0.0, self._dy + self._fall * dt)
                self._fall += 900 * dt
                if self._dy >= 0:
                    self._dy, self._fall = 0.0, 0.0
            if talking:
                if self._state in ("walk", "sleep", "jump"):
                    self._state, self._state_until = "sit", self._t + 1.0
            elif self._t >= self._state_until:
                self._pick_state()
            if self._state == "walk":
                self._x += self._dir * 70 * dt
                if self._x <= g.left() or self._x >= g.right() - W:
                    self._dir *= -1
                    self._x = max(g.left(), min(self._x, g.right() - W))
            if self._state == "jump":
                u = (self._t - self._jump_t) / 0.7
                self._dy = -90 * 4 * u * (1 - u) if 0 <= u <= 1 else 0.0
                if u > 1:
                    self._state, self._dy, self._state_until = "sit", 0.0, self._t + 1.0
        if self._t >= self._blink_at:
            self._blink = 0.16
            self._blink_at = self._t + self._rng.uniform(2.0, 5.5)
        self._blink = max(0.0, self._blink - dt)
        target = (0.35 + 0.65 * abs(math.sin(self._t * 11))) if talking else 0.0
        self._mouth += (target - self._mouth) * min(1.0, dt * 18)
        if self._bubble and self._t > self._bubble_until:
            self._bubble = ""
        self._place()
        self._update_mask()
        self.update()

    def _pick_state(self) -> None:
        r = self._rng.random()
        if r < 0.45:
            self._state, self._dir = "walk", self._rng.choice((-1, 1))
            self._state_until = self._t + self._rng.uniform(3, 8)
        elif r < 0.65:
            self._state, self._state_until = "sit", self._t + self._rng.uniform(2, 5)
        elif r < 0.80:
            self._state, self._jump_t = "jump", self._t
            self._state_until = self._t + 0.8
        else:
            self._state, self._state_until = "sleep", self._t + self._rng.uniform(6, 14)

    def _update_mask(self) -> None:
        body = QRegion(QRect(int(W / 2 - 110), H - 190, 220, 190))
        if self._bubble:
            body = body.united(QRegion(self._bubble_rect().toAlignedRect()))
        self.setMask(body)

    # ── input ───────────────────────────────────────────────────────────────
    def mousePressEvent(self, e) -> None:
        if e.button() == Qt.MouseButton.LeftButton:
            self._drag = e.globalPosition().toPoint() - self.frameGeometry().topLeft()
            self._moved = False
            if self._state == "sleep":
                self._state, self._state_until = "sit", self._t + 2.0

    def mouseMoveEvent(self, e) -> None:
        if self._drag is not None:
            p = e.globalPosition().toPoint() - self._drag
            g = self._screen()
            self._x = float(max(g.left(), min(p.x(), g.right() - W)))
            self._dy = float(min(0, p.y() - (g.bottom() - H + 6)))
            self._moved = True

    def mouseReleaseEvent(self, e) -> None:
        if e.button() == Qt.MouseButton.LeftButton and self._drag is not None:
            self._drag = None
            if self._moved:
                self._fall = 60.0                         # let go: it drops back down
            else:
                self.clicked.emit()

    def contextMenuEvent(self, e) -> None:
        m = QMenu(self)
        for k in ANIMALS:
            m.addAction(f"Be a {k}", lambda k=k: self.set_kind(k))
        m.addSeparator()
        m.addAction("Say hello", lambda: self.say("Hello! I'm here whenever you need me."))
        m.addAction("Hide pet", self.hide)
        m.addAction("Quit pet", lambda: (self.closed.emit(), self.close()))
        m.exec(e.globalPos())

    # ── drawing ─────────────────────────────────────────────────────────────
    def _bubble_rect(self) -> QRectF:
        f = QFont("Segoe UI", 10)
        fm = QFontMetrics(f)
        r = fm.boundingRect(QRect(0, 0, 250, 400), int(Qt.TextFlag.TextWordWrap), self._bubble or " ")
        w, h = min(270, r.width() + 24), r.height() + 18
        return QRectF(W / 2 - w / 2, max(4, H - 200 - h), w, h)

    def paintEvent(self, _e) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        cx, ground = W / 2, H - 8
        self._draw_animal(p, cx, ground)
        if self._bubble:
            self._draw_bubble(p)
        p.end()

    def _draw_bubble(self, p: QPainter) -> None:
        r = self._bubble_rect()
        path = QPainterPath()
        path.addRoundedRect(r, 12, 12)
        tail = QPainterPath(QPointF(W / 2 - 8, r.bottom() - 1))
        tail.lineTo(QPointF(W / 2 + 2, r.bottom() + 16))
        tail.lineTo(QPointF(W / 2 + 12, r.bottom() - 1))
        path = path.united(tail)
        p.setPen(QPen(QColor(40, 60, 80, 220), 1.5))
        p.setBrush(QColor(255, 255, 255, 245))
        p.drawPath(path)
        p.setPen(QColor(25, 35, 50))
        p.setFont(QFont("Segoe UI", 10))
        p.drawText(r.adjusted(12, 8, -12, -8), int(Qt.TextFlag.TextWordWrap | Qt.AlignmentFlag.AlignLeft), self._bubble)

    def _draw_animal(self, p: QPainter, cx: float, ground: float) -> None:
        a = ANIMALS[self.kind]
        t = self._t
        walking = self._state == "walk" and self._drag is None
        asleep = self._state == "sleep"
        talking = self.talking
        held = self._drag is not None
        s = S
        facing = self._dir if (walking or held) else (1 if talking else self._dir)
        p.save()
        p.translate(cx, 0)
        p.scale(facing, 1)
        p.translate(-cx, 0)
        bob = math.sin(t * 16) * 2.5 if walking else (math.sin(t * 2.2) * 1.2 if not asleep else math.sin(t * 1.4) * 1.5)
        sit = 0.0 if walking else 10.0
        by = ground - 46 * s + bob + sit * 0.6 + (14 if asleep else 0)       # body centre y
        body, dark, belly = _c(a["body"]), _c(a["dark"]), _c(a["belly"])
        p.setPen(Qt.PenStyle.NoPen)

        # shadow
        p.setBrush(QColor(0, 0, 0, 55))
        p.drawEllipse(QPointF(cx, ground + 2), 54 * s, 7 * s)

        # tail
        sway = math.sin(t * (7 if walking else 2.6)) * (22 if not asleep else 4)
        p.setBrush(Qt.BrushStyle.NoBrush)
        pen = QPen(body, 11 * s if a["tail"] == "fluffy" else 8 * s, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap)
        p.setPen(pen)
        tl = {"long": 46, "fluffy": 40, "short": 16}[a["tail"]] * s
        tail = QPainterPath(QPointF(cx - 38 * s, by - 2 * s))
        tail.cubicTo(QPointF(cx - 38 * s - tl * .6, by - 10 * s), QPointF(cx - 38 * s - tl, by - 26 * s + sway * .3),
                     QPointF(cx - 38 * s - tl * .7 + sway * .4, by - 48 * s - (tl * .2)))
        p.drawPath(tail)
        if a["tail"] == "fluffy":
            p.setPen(QPen(_c((250, 245, 235)), 6 * s, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
            p.drawPoint(tail.pointAtPercent(0.99).toPoint())
        p.setPen(Qt.PenStyle.NoPen)

        # legs (back pair slightly darker)
        for i, (lx, col) in enumerate(((-26, dark), (-12, body), (14, dark), (28, body))):
            swing = math.sin(t * 9 + i * math.pi / 2) * (13 if walking else 0)
            leg_h = (20 if not (asleep or not walking) else 12) * s
            p.setBrush(col)
            p.drawRoundedRect(QRectF(cx + lx * s - 6 * s + swing * 0.35, by + 16 * s, 12 * s, ground - by - 16 * s + 3 - abs(swing) * 0.1),
                              6 * s, 6 * s)

        # body + belly
        p.setBrush(body)
        p.drawEllipse(QPointF(cx, by), 44 * s, 27 * s)
        p.setBrush(belly)
        p.drawEllipse(QPointF(cx + 2 * s, by + 11 * s), 30 * s, 14 * s)

        # head
        hx, hy = cx + 36 * s, by - 34 * s + (16 if asleep else 0) + (3 if talking else 0) * math.sin(t * 11)
        r = 25 * s
        e = a["ear"]
        p.setBrush(body)
        if e == "point":
            for sx in (-1, 1):
                ear = QPainterPath(QPointF(hx + sx * 6 * s, hy - 18 * s))
                ear.lineTo(QPointF(hx + sx * 20 * s, hy - 40 * s))
                ear.lineTo(QPointF(hx + sx * 24 * s, hy - 8 * s))
                p.setBrush(body)
                p.drawPath(ear)
                inner = QPainterPath(QPointF(hx + sx * 11 * s, hy - 19 * s))
                inner.lineTo(QPointF(hx + sx * 19 * s, hy - 33 * s))
                inner.lineTo(QPointF(hx + sx * 21 * s, hy - 14 * s))
                p.setBrush(_c((244, 160, 160)))
                p.drawPath(inner)
        elif e == "flop":
            p.setBrush(dark)
            for sx in (-1, 1):
                p.drawEllipse(QPointF(hx + sx * 23 * s, hy - 6 * s), 9 * s, 17 * s)
        else:                                                               # round (panda)
            p.setBrush(dark)
            for sx in (-1, 1):
                p.drawEllipse(QPointF(hx + sx * 19 * s, hy - 22 * s), 10 * s, 10 * s)
        p.setBrush(body)
        p.drawEllipse(QPointF(hx, hy), r, r * 0.94)
        p.setBrush(belly)
        p.drawEllipse(QPointF(hx + 5 * s, hy + 8 * s), 15 * s, 11 * s)        # muzzle

        # eyes follow the cursor
        cur = self.mapFromGlobal(QCursor.pos())
        ang = math.atan2(cur.y() - hy, (cur.x() - cx) * facing - (hx - cx))
        dist = min(1.0, math.hypot(cur.x() - hx, cur.y() - hy) / 200.0)
        px, py = math.cos(ang) * 3 * s * dist, math.sin(ang) * 3 * s * dist
        closed = asleep or self._blink > 0
        for ex in (-9, 11):
            ex_, ey_ = hx + ex * s + 3 * s, hy - 4 * s
            if self.kind == "panda":
                p.setBrush(dark)
                p.drawEllipse(QPointF(ex_, ey_), 8 * s, 9 * s)
            if closed:
                p.setPen(QPen(QColor(30, 25, 25), 2.2 * s, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
                p.setBrush(Qt.BrushStyle.NoBrush)
                p.drawArc(QRectF(ex_ - 5 * s, ey_ - 3 * s, 10 * s, 7 * s), 0, -180 * 16)
                p.setPen(Qt.PenStyle.NoPen)
            else:
                p.setBrush(QColor(255, 255, 255))
                p.drawEllipse(QPointF(ex_, ey_), 6 * s, 6.5 * s)
                p.setBrush(QColor(30, 25, 30))
                p.drawEllipse(QPointF(ex_ + px, ey_ + py), 3.4 * s, 3.8 * s)
                p.setBrush(QColor(255, 255, 255))
                p.drawEllipse(QPointF(ex_ + px - 1.2 * s, ey_ + py - 1.4 * s), 1.2 * s, 1.2 * s)

        # nose, mouth, whiskers
        p.setBrush(_c((70, 40, 40)) if self.kind != "cat" else _c((240, 130, 140)))
        p.drawEllipse(QPointF(hx + 13 * s, hy + 5 * s), 3.6 * s, 2.8 * s)
        mo = self._mouth
        if mo > 0.05:
            p.setBrush(QColor(120, 30, 40))
            p.drawEllipse(QPointF(hx + 11 * s, hy + 13 * s + mo * 2 * s), 6 * s, (2 + 7 * mo) * s)
            p.setBrush(_c((235, 120, 130)))
            p.drawEllipse(QPointF(hx + 11 * s, hy + 15 * s + mo * 4 * s), 3.5 * s, 2.2 * s * mo)
        else:
            p.setPen(QPen(QColor(60, 40, 40), 1.6 * s))
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawArc(QRectF(hx + 5 * s, hy + 8 * s, 8 * s, 7 * s), 200 * 16, 140 * 16)
            p.drawArc(QRectF(hx + 12 * s, hy + 8 * s, 8 * s, 7 * s), 200 * 16, 140 * 16)
            p.setPen(Qt.PenStyle.NoPen)
        if a["whisk"]:
            p.setPen(QPen(QColor(255, 255, 255, 210), 1.1 * s))
            for dy in (-3, 3):
                p.drawLine(QPointF(hx + 16 * s, hy + 8 * s + dy * s), QPointF(hx + 38 * s, hy + 5 * s + dy * 2.2 * s))
                p.drawLine(QPointF(hx + 6 * s, hy + 8 * s + dy * s), QPointF(hx - 16 * s, hy + 5 * s + dy * 2.2 * s))
            p.setPen(Qt.PenStyle.NoPen)
        p.restore()
        if asleep:                                                           # zzz, never mirrored
            p.setPen(QColor(120, 160, 220, 230))
            p.setFont(QFont("Segoe UI", 14, QFont.Weight.Bold))
            for i in range(3):
                off = (t * 1.2 + i * 0.7) % 2.1
                p.drawText(QPointF(cx + 40 + off * 14, ground - 110 - off * 22), "z")
