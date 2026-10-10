"""
Holographic AI head for the HUD centre — the thing that used to be a ring stack
with the assistant's name in the middle.

Design notes
------------
* **The face is real human geometry.** `core.avatar_mesh` builds the head around
  MediaPipe's canonical face model, so eyelids, nostrils, lips and cheekbones
  are measured anatomy rather than fitted curves. This renderer's whole job is
  to light it, pose it and animate it.
* **Software rendered, on purpose.** Everything is QPainter, so there is no
  OpenGL context, no shader compile, no GPU driver to disagree with us and no
  new pip dependency. It looks the same on a gaming rig, a 2013 laptop, a VM
  and a remote desktop session.
* **Lip-sync comes from the audio pipeline, not from the avatar.** `main.py`
  already computes a real RMS level off the PCM (`_pcm_level`) for both the mic
  and JARVIS's own output. The avatar just consumes that number, so there is no
  second audio path to fall out of sync. The mouth only tracks the level while
  JARVIS is *speaking* — during listening the same level drives the aura, so the
  head never lip-syncs to the user's voice.

The renderer is theme-agnostic: `paint()` takes its colours as arguments, which
is what lets the HueWheel accent picker retint the avatar for free.
"""

from __future__ import annotations

import math
import random

import numpy as np
from PyQt6.QtCore import QLineF, QPointF, QRectF, Qt
from PyQt6.QtGui import (QBrush, QColor, QImage, QLinearGradient, QPainter, QPainterPath, QPen,
                         QPolygonF, QRadialGradient)

from core.avatar_mesh import JAW_MAX, JAW_PIVOT, get_head_mesh

# Perspective camera distance in head-half-heights. Large enough that the nose
# does not balloon, small enough to keep a sense of depth.
_CAM_D = 4.6

# Wireframe opacity buckets, so the whole lattice draws in a handful of batched
# drawLines() calls instead of one call per line.
_BUCKETS = 4
_MIN_ALPHA = 0.05

# Resolution of the surface-shading colour ramp. Banding across a filled facet
# is far more visible than banding in line alpha, so this is fine-grained — and
# being a lookup it costs nothing per face.
_LUT_N = 192

# How far the brows travel at full lift, in head-half-heights. Derived, not
# tuned: the brow-to-eye gap is 0.198 and a real raise covers about a third of
# it, then the drawn landmarks only carry half the rig weight.
_BROW_LIFT = 0.14

# Mouth timing, as time constants in seconds rather than per-frame fractions.
# A fixed per-frame lerp silently changes speed with the frame rate: the HUD
# runs at 60 Hz here, throttles its paint to 30, and drops to 20 when idle, so
# the same constant meant three different mouths. These do not.
#
# Shutting is the fastest of the three, and it is measured rather than chosen.
# A short closure occupies a single 20 ms schedule frame, so the mouth has one
# step to reach it: at 20 ms the jaw got a third of the way and the closure
# vanished, at 12 ms it arrives, and going below that changes nothing because
# the analysis window is then the limit, not the smoothing. Halving it doubled
# the closures the mouth visibly makes across a test paragraph, 5 of 21 to 10,
# with no loss of opening on the vowels. Only the return to rest, once talking
# has actually stopped, is leisurely.
_TAU_OPEN = 0.022     # jaw dropping toward a vowel
_TAU_SHUT = 0.012     # lips closing on a consonant, mid-word
_TAU_REST = 0.055     # settling back to rest after speech ends
_TAU_SHAPE = 0.018    # viseme openness following the schedule

# Only the microphone path needs a level floor: it has one coarse RMS and no way
# to tell speech from room tone. JARVIS's own voice arrives as a per-20 ms
# schedule whose silences are already silent, so it needs no floor and must not
# have one — a floor there swallows the gaps between words.
_MIC_FLOOR = 0.14

# How far below this voice's own loud level counts as a closure: -20 dB, which
# is what a stop consonant actually drops to. Expressed as a ratio so it holds
# at any speaker volume.
_CLOSE_FRAC = 0.10


def _rate(dt: float, tau: float) -> float:
    """Per-frame lerp factor for an exponential approach with time constant
    `tau`. Frame-rate independent: the motion takes the same wall-clock time at
    20, 30 or 60 fps, and a long frame catches up instead of stalling."""
    return 1.0 - math.exp(-dt / tau)


def _c(col: QColor, a: float) -> QColor:
    """Copy of `col` at alpha `a` (0-255, clamped)."""
    q = QColor(col)
    q.setAlpha(int(max(0.0, min(255.0, a))))
    return q


def _blend(bg: QColor, col: QColor, a: float) -> QColor:
    """`col` at alpha `a` pre-mixed onto `bg`, returned fully **opaque**.

    Qt's raster engine has a fast path for opaque antialiased lines and a much
    slower blended path for everything else — measured at 1.0 ms versus 2.9 ms
    for the same 780 lines. The HUD paints a flat background behind the avatar,
    so mixing the alpha in by hand is visually equivalent and three times cheaper.
    """
    f = max(0.0, min(1.0, a / 255.0))
    return QColor(int(bg.red() + (col.red() - bg.red()) * f),
                  int(bg.green() + (col.green() - bg.green()) * f),
                  int(bg.blue() + (col.blue() - bg.blue()) * f))


class HoloAvatar:
    """Animated holographic head. One instance per HUD canvas.

    Lifecycle:
        av = HoloAvatar()
        av.step(dt, amp, speaking=..., muted=...)          # once per tick
        av.paint(painter, cx, cy, r, primary, accent, bg)  # once per frame
    """

    # Look: True paints a lit, solid head with a wireframe over it; False is a
    # see-through glass wireframe. Flip here, or per instance.
    shaded = True

    # "real": a lit human face with real eyes, hair, brows and lips.
    # "holo": the original see-through hologram. Chosen in ⚙ CONTROLS and kept
    # in config/api_keys.json; both paths share the same animation state, so
    # lip-sync, gaze, blinking and status behaviour are identical in either.
    style = "real"
    skin = (201, 146, 108)
    iris = (92, 64, 38)
    hair = (30, 22, 18)

    # Facial expressions. Each is a set of target weights that the face eases
    # into (and back out of), so a reaction looks like a person's, not a switch:
    # smile (negative = frown), brow (raise), worry (inner brows up), widen /
    # squint (eyes), jaw (mouth drop), yaw / pitch (head turn, radians) and
    # gx / gy (where the eyes go).
    EMOTIONS = {
        "neutral":   {},
        "happy":     {"smile": 0.85, "brow": 0.15, "squint": 0.35, "pitch": -0.03},
        "laugh":     {"smile": 1.0, "brow": 0.2, "squint": 0.6, "jaw": 0.35, "pitch": -0.06},
        "surprised": {"brow": 1.0, "widen": 0.7, "jaw": 0.40, "pitch": -0.05},
        "thinking":  {"brow": 0.35, "gx": -0.7, "gy": -0.7, "yaw": -0.22, "pitch": 0.05},
        "concerned": {"worry": 0.9, "smile": -0.35, "brow": 0.1, "pitch": 0.03},
        "curious":   {"brow": 0.45, "yaw": 0.20, "pitch": -0.02, "widen": 0.15},
        "serious":   {"brow": -0.25, "squint": 0.2, "smile": -0.1},
        "agree":     {"smile": 0.35, "pitch": 0.10},
    }
    _STATE_EMOTION = {"listening": "curious", "thinking": "thinking", "processing": "thinking"}

    def react(self, emotion: str, hold: float = 2.5) -> None:
        """Show an emotion for `hold` seconds, then ease back to neutral."""
        if emotion in self.EMOTIONS:
            self._emo, self._emo_until = emotion, self._t + max(0.3, hold)

    def react_to_text(self, text: str) -> None:
        """Pick a fitting reaction from what was just said. Deliberately simple."""
        t = (text or "").lower()
        if not t.strip():
            return
        for words, emo in ((("sorry", "unfortunately", "can't", "cannot", "unable", "problem"), "concerned"),
                           (("haha", "funny", "lol"), "laugh"),
                           (("wow", "incredible", "unbelievable", "no way"), "surprised"),
                           (("congrat", "great", "awesome", "excellent", "glad", "nice", "well done", "done"), "happy"),
                           (("let me think", "hmm", "perhaps", "maybe", "not sure"), "thinking"),
                           (("yes", "sure", "of course", "right away"), "agree")):
            if any(w in t for w in words):
                return self.react(emo, 2.2)
        if t.rstrip().endswith("?"):
            self.react("curious", 2.0)

    def _ease_emotion(self, dt: float, state: str) -> None:
        if not hasattr(self, "_ev"):
            self._ev = {k: 0.0 for k in ("smile", "brow", "worry", "widen", "squint", "jaw",
                                         "yaw", "pitch", "gx", "gy")}
            self._emo, self._emo_until = "neutral", 0.0
        name = self._emo if self._t < self._emo_until else \
            self._STATE_EMOTION.get((state or "").lower(), "neutral")
        target = self.EMOTIONS.get(name, {})
        k = 1.0 - math.exp(-dt / 0.22)
        for key in self._ev:
            self._ev[key] += (target.get(key, 0.0) - self._ev[key]) * k

    def set_look(self, style: str | None = None, skin=None, iris=None) -> None:
        if style in ("real", "holo"):
            self.style = style
        if skin:
            self.skin = tuple(int(c) for c in skin)
        if iris:
            self.iris = tuple(int(c) for c in iris)
        self._rv_key = None
        self._skin_cache = None            # a new look must never be drawn from the old buffer

    def __init__(self) -> None:
        mesh = get_head_mesh()
        self._v0 = mesh["verts"]
        self._n0 = mesh["normals"]
        self._jaw = mesh["jaw"]
        self._brow_w = mesh["brow"]
        self._lips_w = mesh["lips"]
        self._lip_c = mesh["lip_centre"]
        self._fade = mesh["fade"]
        self._f = mesh["faces"]
        self._fgroup = mesh["face_group"]
        self._fa, self._fb, self._fc = (self._f[:, i] for i in range(3))
        self._e0 = mesh["edges"][:, 0]
        self._e1 = mesh["edges"][:, 1]
        self._lm = mesh["landmarks"]
        # The inner-lip ring runs lower-lip left→right, then upper-lip back.
        # Splitting it lets the upper arc anchor a strip of teeth, which is what
        # keeps an open mouth from reading as a hole punched in the face.
        lips_in = mesh["landmarks"]["lips_in"]
        self._lip_up = np.concatenate([lips_in[10:], lips_in[:1]])

        # Crown (+1.0) down to the bottom of the neck, in head-half-heights.
        # Callers size the head to the room they have with this.
        self.SPAN = mesh["span"][0] - mesh["span"][1]

        self._lut_cache: list = []
        self._lut_key = None
        self._rv_key = None
        self._load_look()

        n = self._v0.shape[0]
        self._v = np.empty((n, 3), dtype=np.float32)

        self._t = 0.0
        self._sway = 0.0           # integrated sway phase — see step()
        self._yaw = 0.0
        self._pitch = 0.0
        self._mouth = 0.0          # 0..1 smoothed jaw opening
        self._glow = 0.0           # 0..1 smoothed overall energy
        self._scan = -1.6          # vertical position of the energy sweep
        self._blink = 0.0          # 0 = open, 1 = shut
        self._blink_at = 3.0

        # ── expression ──────────────────────────────────────────────────────
        # Speech is not just a moving jaw. Brows ride the loudness envelope,
        # eyes widen with the brows, and the gaze flicks between fixation
        # points — those three are what make it read as talking rather than as
        # a puppet chewing.
        self._amp_slow = 0.0
        self._expr = 0.0
        self._expr_tgt = 0.0
        self._expr_at = 0.0
        self._brow = 0.0           # smoothed brow lift, -0.4 .. 1.2
        self._emph = 0.0           # syllable emphasis, drives the head nod
        self._gaze = [0.0, 0.0]
        self._gaze_tgt = [0.0, 0.0]
        self._gaze_at = 0.0

        # ── state expression ────────────────────────────────────────────────
        # The face is the fastest status indicator in the app: you read a gaze
        # before you read a word. Saccades orbit a bias that the assistant's
        # state moves — eyes off to the side while it thinks, back on you while
        # it listens, lids low while it sleeps.
        self._gaze_bias = [0.0, 0.0]
        self._bias_tgt = [0.0, 0.0]
        self._bias_at = 0.0
        self._lids = 1.0           # 1 = wide, 0 = shut; low while asleep
        self._brow_bias = 0.0      # concentration pulls the brows down
        self._glance = None        # (dx, dy, until_t) — a deliberate look

        # ── viseme ──────────────────────────────────────────────────────────
        # Loudness alone only answers "how far open", which is why an RMS-driven
        # mouth flaps rather than speaks. These two carry the *shape*: how open
        # the jaw is for this sound, and whether the lips are spread (/i/) or
        # rounded (/u/). They come from a formant read of the audio actually
        # being played — see `_pcm_visemes` in main.py.
        self._v_open = 1.0
        self._v_wide = 0.0
        self._wide = 0.0           # smoothed lip spread, -1 round .. +1 spread
        self._v_peak = 0.18        # running estimate of this voice's loud level

    # ── animation ───────────────────────────────────────────────────────────

    def _mouth_step(self, dt: float, amp: float, live: bool,
                    v_open: float | None, v_level: float | None) -> None:
        """One increment of the jaw. Called once per viseme frame while JARVIS
        speaks, once per rendered frame otherwise."""
        if v_open is None:
            shape = 1.0
        else:
            self._v_open += (v_open - self._v_open) * _rate(dt, _TAU_SHAPE)
            shape = self._v_open

        if v_level is None:
            gated = max(0.0, (amp - _MIC_FLOOR) / (1.0 - _MIC_FLOOR))
            drive = (gated ** 0.6) * (shape ** 0.75)
        else:
            # Speech RMS spends most of its time well below full scale, so the
            # raw value alone would only ever half-open the jaw. Normalise it
            # against a running estimate of this voice's own loud level rather
            # than a constant: it then reads the same whether the user has the
            # volume low or the model happens to be speaking softly.
            self._v_peak = max(v_level, self._v_peak - dt * 0.55)
            ref = max(0.18, self._v_peak)
            # The floor is a fraction of this voice's own loud level, not a
            # fixed number, so it means the same thing at any volume and in any
            # language. A stop consonant drops 20 dB or more below the vowels
            # around it, which is this ratio — so a real closure lands at
            # exactly zero rather than at some small positive value the curve
            # would otherwise lift back up. That lift is what kept the mouth
            # from ever quite shutting between words.
            q = (v_level - _CLOSE_FRAC * ref) / (ref * (1.0 - _CLOSE_FRAC))
            drive = max(0.0, min(1.0, q)) ** 0.85 * (shape ** 0.75)

        target = min(1.0, drive) if live else 0.0
        if target > self._mouth:
            tau = _TAU_OPEN
        elif live:
            tau = _TAU_SHUT      # mid-word: a consonant, and it must shut now
        else:
            tau = _TAU_REST      # speech is over; settle, don't snap
        self._mouth += (target - self._mouth) * _rate(dt, tau)
        if self._mouth < 0.002:
            self._mouth = 0.0

    def step(self, dt: float, amp: float, speaking: bool = False,
             muted: bool = False, state: str = "",
             v_open: float | None = None, v_wide: float = 0.0,
             v_level: float | None = None,
             v_seq: list | None = None, v_hop: float = 0.02) -> None:
        """Advance the animation.

        `amp` is the 0..1 display audio level. `v_open` / `v_wide` / `v_level`
        are the viseme schedule's shape and true level for this instant; passing
        None falls back to loudness-only articulation, which is what the
        microphone path uses. `v_seq` is every schedule frame the last rendered
        frame spanned, so no closure is lost when the paint rate drops.
        """
        dt = max(0.001, min(0.10, float(dt)))
        self._t += dt
        t = self._t
        amp = max(0.0, min(1.0, float(amp)))
        live = speaking and not muted

        # Idle sway. The phase is *integrated* rather than taken as
        # sin(t * rate * speed): multiplying absolute time by a speed that
        # changes when JARVIS starts or stops talking jumps the phase by
        # t * rate * delta, which after a minute of uptime is several radians
        # and visibly teleports the head the instant a sentence ends.
        speed = (1.0 if not muted else 0.55) * (1.25 if live else 1.0)
        self._sway += dt * speed
        s = self._sway
        self._yaw = 0.26 * math.sin(s * 0.31) + 0.09 * math.sin(s * 0.73 + 1.3)
        self._pitch = (0.060 * math.sin(s * 0.23 + 0.7)
                       + 0.024 * math.sin(s * 0.61))
        self._ease_emotion(dt, state)
        self._yaw += self._ev["yaw"]
        self._pitch += self._ev["pitch"]

        # Mouth. Which level is driving it matters more than any rate here.
        #
        # `v_level` is this 20 ms frame's own RMS, taken from the very audio
        # about to be heard, so its silences are real silences. `amp` is the
        # waveform display's level, and that one is a *peak hold*: it keeps the
        # loudest value it has seen and decays gently, on purpose, so the bars
        # do not stutter between audio chunks. Driving a mouth from a peak hold
        # is why the gaps between words never closed — the hold spans exactly
        # the consonant it was supposed to reveal. So the schedule drives the
        # jaw whenever there is one, and `amp` is left to the microphone path,
        # which has nothing better.
        # Advance the mouth once per *schedule* frame rather than once per
        # rendered frame. A bilabial closure lasts around 40 ms — two frames of
        # a 50 Hz schedule — and the HUD throttles its paint to 30 fps and to 20
        # when idle. Point-sampling at 20 fps steps 50 ms at a time, so a whole
        # closure can fall between two samples and simply never be seen; that is
        # information loss no smoothing constant can recover. Sub-stepping costs
        # a few float operations per frame and makes the mouth identical at 20,
        # 30 and 60 fps.
        # An empty list is meaningful and is not the same as None: it says a
        # schedule is playing but this tick landed inside a frame already
        # spoken. The mouth's clock is the schedule's, so the right thing then
        # is to do nothing. Re-stepping the same frame — which is what a
        # truthiness test here would do — advances the jaw twice for one 20 ms
        # of audio, and at 60 fps that alone made the mouth behave differently
        # than at 20.
        if v_seq is not None:
            for lv, op, _wd in v_seq:
                self._mouth_step(v_hop, amp, live, op, lv)
        else:
            self._mouth_step(dt, amp, live, v_open, v_level)

        if self._ev["jaw"] > self._mouth:
            self._mouth = self._ev["jaw"]            # a gasp or a laugh opens the mouth

        # Syllable emphasis. Applied unconditionally: `_emph` decays to zero on
        # its own once the mouth closes, whereas gating it on `live` deleted the
        # whole offset in a single frame and snapped the head at sentence end.
        self._emph += (self._mouth - self._emph) * _rate(
            dt, 0.055 if self._mouth > self._emph else 0.32)
        self._pitch -= self._emph * 0.028
        self._yaw += 0.018 * math.sin(t * 1.7) * self._emph

        # Loudness envelope, deliberately lazier than the mouth: brows track the
        # shape of a phrase, not individual syllables.
        env = amp if live else 0.0
        self._amp_slow += (env - self._amp_slow) * _rate(
            dt, 0.16 if env > self._amp_slow else 0.36)

        if live:
            if t >= self._expr_at:
                self._expr_tgt = random.uniform(-0.35, 1.0)
                self._expr_at = t + 1.1 + 2.0 * random.random()
        else:
            self._expr_tgt = 0.0
            self._expr_at = t + 0.8
        self._expr += (self._expr_tgt - self._expr) * 0.075

        brow_t = 0.55 * self._amp_slow + 0.60 * self._expr + self._brow_bias
        self._brow += (max(-0.4, min(1.2, brow_t)) - self._brow) * 0.20

        # ── what the state does to the face ─────────────────────────────────
        st = (state or "").upper()
        thinking = st in ("THINKING", "PROCESSING")
        asleep = st in ("SLEEPING", "STANDBY", "OFFLINE")

        if thinking:
            # People look away to think, and hold it. The direction re-rolls
            # slowly so it reads as thought rather than as scanning.
            if t >= self._bias_at:
                self._bias_tgt = [random.choice((-1.0, 1.0)) * random.uniform(0.45, 0.8),
                                  random.uniform(0.25, 0.55)]
                self._bias_at = t + 1.4 + 1.6 * random.random()
            brow_bias, lid_tgt = -0.28, 0.94
        elif asleep:
            self._bias_tgt = [0.0, -0.25]
            brow_bias, lid_tgt = -0.05, 0.22
        else:
            # LISTENING / idle / speaking: eyes come back to the user.
            self._bias_tgt = [0.0, 0.0]
            self._bias_at = 0.0
            brow_bias = 0.10 if st == "LISTENING" else 0.0
            lid_tgt = 1.0

        for i in (0, 1):
            self._gaze_bias[i] += (self._bias_tgt[i] - self._gaze_bias[i]) * 0.06
        self._lids += (lid_tgt - self._lids) * 0.08
        self._brow_bias += (brow_bias - self._brow_bias) * 0.06

        # Gaze: saccades are near-instant jumps between fixations, and they get
        # more frequent when there is something to say. While thinking they slow
        # right down — a darting eye reads as nervous, not thoughtful.
        if t >= self._gaze_at:
            reach = 0.9 if live else (0.35 if thinking else 0.55)
            self._gaze_tgt = [random.uniform(-1.0, 1.0) * reach,
                              random.uniform(-1.0, 1.0) * reach * 0.55]
            if live:
                self._gaze_at = t + 0.55 + 1.7 * random.random()
            elif thinking:
                self._gaze_at = t + 1.8 + 2.4 * random.random()
            else:
                self._gaze_at = t + 1.3 + 2.8 * random.random()

        # A deliberate glance (something appeared on screen) overrides the
        # wandering for a moment, then hands control back.
        if self._glance is not None:
            gx, gy, until = self._glance
            if t < until:
                self._gaze_tgt = [gx, gy]
            else:
                self._glance = None

        for i, b in enumerate(self._gaze_bias):
            tgt = max(-1.0, min(1.0, self._gaze_tgt[i] + b))
            self._gaze[i] += (tgt - self._gaze[i]) * 0.30

        # Lips lead the jaw slightly in real speech, so they track a touch
        # faster; they also relax to neutral the moment the voice stops.
        wide_t = v_wide if (live and v_open is not None) else 0.0
        self._wide += (max(-1.0, min(1.0, wide_t)) - self._wide) * _rate(dt, 0.030)

        self._glow += ((0.0 if muted else amp) - self._glow) * (
            0.35 if (0.0 if muted else amp) > self._glow else 0.10)

        self._scan += dt * (0.55 + 1.5 * self._glow)
        if self._scan > 1.35:
            self._scan = -1.75

        if self._blink > 0.0:
            self._blink = max(0.0, self._blink - dt * 8.5)
        elif t >= self._blink_at:
            # Concentration suppresses blinking; a sleeping face has no need of
            # it at all, since the lids are already down.
            if asleep:
                self._blink_at = t + 6.0
            else:
                self._blink = 1.0
                gap = 5.5 if thinking else 3.4
                self._blink_at = t + gap + 3.1 * random.random()

    def glance(self, dx: float, dy: float, hold: float = 1.1) -> None:
        """Look deliberately somewhere for `hold` seconds, then wander again.

        Used when something appears on screen: a face that looks at what just
        showed up tells the user it landed, without a word being spoken.
        """
        self._glance = (max(-1.0, min(1.0, float(dx))),
                        max(-1.0, min(1.0, float(dy))),
                        self._t + max(0.1, float(hold)))

    # ── posing ──────────────────────────────────────────────────────────────

    def _pose(self):
        """Jaw drop, brow lift and head rotation, applied to the real geometry."""
        v = self._v
        np.copyto(v, self._v0)

        if self._brow > 0.004 or self._brow < -0.004:
            # The brow-to-eye gap is 0.198 head-half-heights and a real raise
            # moves a third of it. The old 0.045 — halved again by the landmark
            # weights, which average 0.5 — worked out to six pixels on a 250 px
            # head, which is to say invisible.
            v[:, 1] += self._brow_w * (self._brow * _BROW_LIFT)

        if abs(self._wide) > 0.01 and self._mouth > 0.0:
            # Spread pulls the corners out and flattens the lips back; rounding
            # draws them in and pushes them forward into a purse.
            k = self._lips_w * (self._wide * self._mouth)
            v[:, 0] += k * (v[:, 0] - self._lip_c[0]) * 0.55
            v[:, 1] += k * (v[:, 1] - self._lip_c[1]) * 0.30
            v[:, 2] -= k * 0.055

        if self._mouth > 0.004:
            px, py, pz = JAW_PIVOT
            ang = self._jaw * (self._mouth * JAW_MAX)
            ca, sa = np.cos(ang), np.sin(ang)
            dy = v[:, 1] - py
            dz = v[:, 2] - pz
            v[:, 1] = py + dy * ca - dz * sa
            v[:, 2] = pz + dy * sa + dz * ca

        cy, sy = math.cos(self._yaw), math.sin(self._yaw)
        cp, sp = math.cos(self._pitch), math.sin(self._pitch)
        m = np.array([
            [cy, 0.0, sy],
            [sp * sy, cp, -sp * cy],
            [-cp * sy, sp, cp * cy],
        ], dtype=np.float32)

        return v @ m.T, self._n0 @ m.T

    # ── rendering ───────────────────────────────────────────────────────────

    def _lut(self, bg: QColor, primary: QColor):
        """Cached ramp of opaque surface brushes from `bg` to `primary`."""
        key = (bg.rgb(), primary.rgb())
        if self._lut_key != key:
            self._lut_cache = [QBrush(_blend(bg, primary, 255.0 * (i + 0.5) / _LUT_N))
                               for i in range(_LUT_N)]
            self._lut_key = key
        return self._lut_cache

    def paint(self, p: QPainter, cx: float, cy: float, r: float,
              primary: QColor, accent: QColor, bg: QColor | None = None) -> None:
        """Draw the avatar with its head centre at (cx, cy).

        `r` is the head's half-height in pixels — the caller owns the layout, so
        the HUD can fit the head to whatever room the status line leaves it.
        """
        if bg is None:
            bg = QColor(0, 0, 0)
        amp = self._glow
        verts, norms = self._pose()

        # ── aura ────────────────────────────────────────────────────────────
        ar = r * 1.95
        grad = QRadialGradient(cx, cy, ar)
        grad.setColorAt(0.00, _c(primary, 34 + 66 * amp))
        grad.setColorAt(0.38, _c(primary, 20 + 40 * amp))
        grad.setColorAt(1.00, _c(primary, 0))
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QBrush(grad))
        p.drawEllipse(QRectF(cx - ar, cy - ar, ar * 2, ar * 2))

        # ── project ─────────────────────────────────────────────────────────
        w = _CAM_D - verts[:, 2]
        np.maximum(w, 0.35, out=w)
        k = (_CAM_D / w) * r
        xs = cx + verts[:, 0] * k
        ys = cy - verts[:, 1] * k

        if self.style == "real":
            self._paint_real(p, xs, ys, norms, verts, r, primary, bg, amp)
            return
        if self.shaded:
            self._paint_surface(p, xs, ys, norms, verts, primary, bg, amp)
        self._paint_wire(p, xs, ys, norms, verts, primary, bg, amp)
        self._paint_features(p, xs, ys, norms, r, primary, accent, bg, amp)

    def _paint_surface(self, p: QPainter, xs, ys, norms, verts,
                       primary: QColor, bg: QColor, amp: float) -> None:
        """Fill the camera-facing triangles so the head reads as a lit volume."""
        a, b, c = self._fa, self._fb, self._fc

        # Flat normals, taken from each triangle's own posed geometry — NOT the
        # averaged vertex normals. Averaging smears the nose, lips and brow
        # relief into their neighbours and renders the face as a blank egg;
        # per-facet normals are exactly what makes the anatomy visible.
        fn = np.cross(verts[b] - verts[a], verts[c] - verts[a])
        fn /= np.maximum(np.linalg.norm(fn, axis=1, keepdims=True), 1e-9)

        # Point them outwards by agreeing with the vertex normals, which were
        # oriented at build time. Flipping on the sign of n_z instead would
        # negate x and y as well and scramble the lighting into moiré.
        ref = norms[a] + norms[b] + norms[c]
        fn *= np.sign((fn * ref).sum(1))[:, None]

        nz = fn[:, 2]
        area = np.abs((xs[b] - xs[a]) * (ys[c] - ys[a])
                      - (xs[c] - xs[a]) * (ys[b] - ys[a]))
        vis = np.flatnonzero((nz > 0.015) & (area > 3.0))
        if vis.size == 0:
            return
        fn = fn[vis]
        nz = nz[vis]

        ax, ay = xs[a][vis], ys[a][vis]
        bx, by = xs[b][vis], ys[b][vis]
        cxx, cyy = xs[c][vis], ys[c][vis]

        # A rim term for the glass edge plus a key light high on the left. The
        # light leans off-axis on purpose: weight it towards the camera and
        # every front-facing facet returns the same value, which is a flat mask.
        fres = np.clip(1.0 - nz, 0.0, 2.0) ** 1.7
        lam = np.clip(fn[:, 0] * -0.55 + fn[:, 1] * 0.50 + nz * 0.52, 0.0, 1.0)
        bright = 0.26 + 0.20 * fres + 0.66 * lam ** 1.05
        bright *= (self._fade[a][vis] + self._fade[b][vis] + self._fade[c][vis]) / 3.0
        bright *= 0.88 + 0.24 * amp

        idx = np.clip((bright * _LUT_N).astype(np.int32), 0, _LUT_N - 1)

        # Far facets first: the neck passes behind the jaw and the head is not
        # convex around the chin.
        # Sort far-to-near, but group first: neck facets all draw before head
        # facets, because the two meshes interpenetrate and a pure depth sort
        # interleaves them into a torn seam.
        fz = (verts[a, 2][vis] + verts[b, 2][vis] + verts[c, 2][vis]) * (1.0 / 3.0)
        order = np.argsort(self._fgroup[vis] * 1000.0 + fz, kind="stable")
        tris = np.stack([ax, ay, bx, by, cxx, cyy], axis=1)[order].tolist()
        shade = idx[order].tolist()
        lut = self._lut(bg, primary)

        # Aliased fills: adjacent antialiased polygons leave hairline seams, and
        # the interior of a tiled surface has no silhouette worth smoothing —
        # the antialiased wireframe drawn afterwards covers the outline.
        p.setRenderHint(QPainter.RenderHint.Antialiasing, False)
        p.setPen(Qt.PenStyle.NoPen)
        for q, sh in zip(tris, shade):
            p.setBrush(lut[sh])
            p.drawPolygon(QPolygonF([QPointF(q[0], q[1]), QPointF(q[2], q[3]),
                                     QPointF(q[4], q[5])]))
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)

    def _paint_wire(self, p: QPainter, xs, ys, norms, verts,
                    primary: QColor, bg: QColor, amp: float) -> None:
        nz = norms[:, 2]
        fres = np.abs(1.0 - np.abs(nz)) ** 1.5
        if self.shaded:
            # The lit surface underneath is opaque, so back-facing edges would
            # float on top of the face — cull them and let the wire read as
            # structure lines over skin.
            front = nz > -0.05
            va = np.where(front, 0.10 + 0.42 * fres, 0.0)
            va += 0.30 * np.exp(-((verts[:, 1] - self._scan) / 0.13) ** 2) * front
        else:
            va = np.where(nz < 0.0, 0.13 + 0.26 * fres, 0.28 + 0.72 * fres)
            va += 0.42 * np.exp(-((verts[:, 1] - self._scan) / 0.13) ** 2)
        va *= self._fade * (0.80 + 0.45 * amp)

        ea = 0.5 * (va[self._e0] + va[self._e1])
        keep = np.flatnonzero(ea > _MIN_ALPHA)
        if keep.size == 0:
            return

        # Sort by opacity bucket once so each bucket is a contiguous *slice* of
        # one QLineF list; masking and rebuilding per bucket cost more than the
        # drawing itself.
        bucket = np.clip((ea[keep] * _BUCKETS).astype(np.int32), 0, _BUCKETS - 1)
        order = np.argsort(bucket, kind="stable")
        keep = keep[order]
        bounds = np.searchsorted(bucket[order], np.arange(_BUCKETS + 1))

        e0, e1 = self._e0[keep], self._e1[keep]
        quad = np.stack([xs[e0], ys[e0], xs[e1], ys[e1]], axis=1).tolist()
        lines = [QLineF(q[0], q[1], q[2], q[3]) for q in quad]

        skin = _blend(bg, primary, 132) if self.shaded else bg
        p.setBrush(Qt.BrushStyle.NoBrush)
        for b in range(_BUCKETS):
            lo, hi = int(bounds[b]), int(bounds[b + 1])
            if hi <= lo:
                continue
            seg = lines[lo:hi]
            a = 255.0 * min(1.0, (b + 0.5) / _BUCKETS)
            if self.shaded:
                # These sit on lit skin, so pre-mix against a representative
                # *skin* tone rather than the background — same fast opaque
                # path, and the lines still read as highlights over the face.
                p.setPen(QPen(_blend(skin, primary, a * 0.75), 1.0))
            else:
                p.setPen(QPen(_blend(bg, primary, a), 1.0))
            p.drawLines(seg)


    # ── realistic face ──────────────────────────────────────────────────────

    def _load_look(self) -> None:
        """Read the saved look (style / skin / eye colour). Never raises."""
        try:
            from memory.config_manager import load_api_keys
            cfg = load_api_keys()

            def hexrgb(v):
                v = str(v or "").lstrip("#")
                return tuple(int(v[i:i + 2], 16) for i in (0, 2, 4)) if len(v) == 6 else None
            self.set_look(style=str(cfg.get("avatar_style", "real")).lower(),
                          skin=hexrgb(cfg.get("avatar_skin")),
                          iris=hexrgb(cfg.get("avatar_eye")))
        except Exception:
            pass

    @staticmethod
    def _smooth(pts: np.ndarray, closed: bool, iters: int = 2) -> np.ndarray:
        """Chaikin corner cutting: a 16-point eye ring becomes a smooth almond."""
        for _ in range(iters):
            nxt = np.roll(pts, -1, axis=0) if closed else pts[1:]
            cur = pts if closed else pts[:-1]
            q = 0.75 * cur + 0.25 * nxt
            r = 0.25 * cur + 0.75 * nxt
            out = np.empty((len(cur) * 2, 2), dtype=pts.dtype)
            out[0::2], out[1::2] = q, r
            pts = out if closed else np.vstack([pts[:1], out, pts[-1:]])
        return pts

    @staticmethod
    def _path(pts: np.ndarray, closed: bool = True) -> QPainterPath:
        path = QPainterPath(QPointF(float(pts[0, 0]), float(pts[0, 1])))
        for x, y in pts[1:]:
            path.lineTo(float(x), float(y))
        if closed:
            path.closeSubpath()
        return path

    def _vertex_colours(self, norms, verts, primary: QColor, bg: QColor, amp: float):
        """Skin + hair colour per vertex: wrapped diffuse for a soft terminator,
        a warm subsurface band where light turns into shadow, a small specular
        and a rim light that picks up the HUD's accent colour."""
        n = norms
        L = np.array([-0.46, 0.52, 0.72], dtype=np.float32)
        L /= np.linalg.norm(L)
        F = np.array([0.62, -0.05, 0.55], dtype=np.float32)
        F /= np.linalg.norm(F)
        H = L + np.array([0, 0, 1.0], dtype=np.float32)
        H /= np.linalg.norm(H)
        ndl = n @ L
        wrap = np.clip((ndl + 0.30) / 1.30, 0.0, 1.0)
        fill = np.clip(n @ F, 0.0, 1.0) * 0.20
        spec = np.clip(n @ H, 0.0, 1.0) ** 26 * 0.12
        ss = np.clip(0.30 - np.abs(ndl - 0.06), 0.0, 0.30) / 0.30
        rim = np.clip(1.0 - n[:, 2], 0.0, 1.0) ** 3 * np.clip(n[:, 0] * 0.8 + 0.3, 0.0, 1.0)

        skin = np.array(self.skin, dtype=np.float32)
        lit = np.clip(0.20 + 0.88 * wrap ** 1.15 + fill, 0.0, 1.25)[:, None]
        col = skin * lit
        col += ss[:, None] * np.array([30.0, 5.0, 2.0], dtype=np.float32)      # blood under skin
        col += (spec * 255.0)[:, None]
        pc = np.array([primary.red(), primary.green(), primary.blue()], dtype=np.float32)
        col += rim[:, None] * pc * (0.30 + 0.25 * amp)

        # Living skin is not one colour: blood shows in the cheeks, nose tip and
        # ears, and the brow ridge and under-chin fall into shade.
        vx, vy = verts[:, 0], verts[:, 1]
        def _g(cx, cy, rad):
            return np.exp(-(((vx - cx) ** 2 + (vy - cy) ** 2) / (rad * rad)))[:, None]
        col += (_g(-0.52, -0.22, 0.24) + _g(0.52, -0.22, 0.24)) * np.array([24.0, -5.0, -9.0], dtype=np.float32)
        col += _g(0.0, -0.34, 0.10) * np.array([18.0, -3.0, -6.0], dtype=np.float32)
        col -= (_g(-0.30, 0.20, 0.16) + _g(0.30, 0.20, 0.16)) * 14.0        # brow ridge
        col -= np.clip((-0.78 - vy) / 0.30, 0.0, 1.0)[:, None] * 26.0       # under the chin

        # Fade the back of the head and the neck into the background.
        f = (0.30 + 0.70 * self._fade)[:, None]
        bgc = np.array([bg.red(), bg.green(), bg.blue()], dtype=np.float32)
        col = bgc + (col - bgc) * f
        # One pass of neighbour averaging: a 468-point mesh shades in visible
        # steps; sharing colour across each vertex's neighbours turns the steps
        # into a gradient without adding geometry.
        if getattr(self, "_nb", None) is None:
            f = self._f
            i = np.concatenate([f[:, 0], f[:, 1], f[:, 2], f[:, 1], f[:, 2], f[:, 0]])
            j = np.concatenate([f[:, 1], f[:, 2], f[:, 0], f[:, 0], f[:, 1], f[:, 2]])
            self._nb = (i, j, np.maximum(np.bincount(i, minlength=len(col)), 1)[:, None])
        i, j, cnt = self._nb
        acc = np.zeros_like(col)
        np.add.at(acc, i, col[j])
        col = 0.45 * col + 0.55 * (acc / cnt)
        return np.clip(col, 0, 255)

    def _paint_shoulders(self, p: QPainter, xs, ys, primary: QColor) -> None:
        """A collar and shoulders under the neck, with a slow breath, so the
        head sits on a body instead of floating."""
        cx, w, yb = float(xs.mean()), float(np.ptp(xs)), float(ys.max())
        breath = math.sin(self._t * 1.15) * w * 0.006
        y0 = yb - w * 0.16 + breath
        path = QPainterPath(QPointF(cx - w * 0.20, y0))
        path.quadTo(QPointF(cx - w * 0.45, y0 + w * 0.05), QPointF(cx - w * 0.95, y0 + w * 0.26))
        path.quadTo(QPointF(cx - w * 1.25, y0 + w * 0.42), QPointF(cx - w * 1.30, y0 + w * 1.4))
        path.lineTo(QPointF(cx + w * 1.30, y0 + w * 1.4))
        path.quadTo(QPointF(cx + w * 1.25, y0 + w * 0.42), QPointF(cx + w * 0.95, y0 + w * 0.26))
        path.quadTo(QPointF(cx + w * 0.45, y0 + w * 0.05), QPointF(cx + w * 0.20, y0))
        path.closeSubpath()
        g = QLinearGradient(0, y0, 0, y0 + w * 0.9)
        g.setColorAt(0.0, QColor(58, 70, 90))
        g.setColorAt(1.0, QColor(18, 24, 34))
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QBrush(g))
        p.drawPath(path)
        rim = QColor(primary)                       # the HUD colour catches the shoulder edge
        rim.setAlpha(70)
        p.setPen(QPen(rim, max(1.0, w * 0.006)))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawPath(path)

    def _hair_anchors(self, verts, norms):
        """Vertices tracing the hairline across the forehead and down the
        temples, picked once from the mesh so the hair follows the skull."""
        if getattr(self, "_hair_idx", None) is not None:
            return self._hair_idx
        base = self._v0
        idx = []
        for x in np.linspace(-0.98, 0.98, 21):
            yh = 0.60 - 0.62 * max(0.0, abs(x) - 0.45) / 0.55
            d = (base[:, 0] - x) ** 2 + (base[:, 1] - yh) ** 2 + np.where(base[:, 2] > -0.2, 0.0, 9.0)
            idx.append(int(np.argmin(d)))
        self._hair_idx = np.array(idx)
        return self._hair_idx

    def _paint_hair(self, bp: QPainter, xs, ys, verts, norms, vcol_light) -> None:
        idx = self._hair_anchors(verts, norms)
        line = np.stack([xs[idx], ys[idx]], axis=1)
        line = line[np.argsort(line[:, 0])]
        line = self._smooth(line, False, 3)
        top_y = float(ys.min()) - 400.0
        poly = np.vstack([[line[0, 0] - 220.0, top_y], [line[0, 0] - 40.0, line[0, 1] + 6.0],
                          line, [line[-1, 0] + 40.0, line[-1, 1] + 6.0],
                          [line[-1, 0] + 220.0, top_y]])
        bp.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceAtop)
        h = self.hair
        sheen = float(vcol_light)
        bp.setPen(Qt.PenStyle.NoPen)
        # feathered edge: three passes, each a little lower and a little fainter
        for dy, al in ((6.0, 90), (0.0, 255)):
            g = QLinearGradient(0, float(line[:, 1].min()) - 70, 0, float(line[:, 1].max()) + 10)
            g.setColorAt(0.0, QColor(min(255, int(h[0] * (0.7 + sheen))),
                                     min(255, int(h[1] * (0.7 + sheen))),
                                     min(255, int(h[2] * (0.7 + sheen))), al))
            g.setColorAt(1.0, QColor(h[0], h[1], h[2], al))
            bp.setBrush(QBrush(g))
            bp.drawPath(self._path(poly + np.array([0.0, dy])))
        # soft sheen where the key light (upper left) catches the crown
        cxm, top = float(xs.mean()), float(ys.min())
        sg = QRadialGradient(cxm - (float(np.ptp(xs)) * 0.16), top + 38.0, float(np.ptp(xs)) * 0.34)
        sg.setColorAt(0.0, QColor(min(255, h[0] + 95), min(255, h[1] + 80), min(255, h[2] + 70), 120))
        sg.setColorAt(1.0, QColor(h[0], h[1], h[2], 0))
        bp.setPen(Qt.PenStyle.NoPen)
        bp.setBrush(QBrush(sg))
        bp.drawEllipse(QPointF(cxm - float(np.ptp(xs)) * 0.16, top + 38.0),
                       float(np.ptp(xs)) * 0.34, 60.0)
        rng = random.Random(5)
        for _ in range(90):                                    # strands, swept back from the hairline
            k = rng.randrange(0, len(line))
            x0, y0 = line[k]
            ln = rng.uniform(14.0, 58.0)
            lift = rng.choice((0, 0, 1))
            bp.setPen(QPen(QColor(min(255, h[0] + (70 if lift else 30)),
                                  min(255, h[1] + (60 if lift else 26)),
                                  min(255, h[2] + (52 if lift else 22)),
                                  rng.randint(35, 95)), 1.0))
            bp.drawLine(QPointF(float(x0), float(y0) - 2.0),
                        QPointF(float(x0) + rng.uniform(-14, 14), float(y0) - ln))
        bp.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceOver)

    def _raster_skin(self, order, sx, sy, vcol, iw: int, ih: int) -> np.ndarray:
        """Gouraud-shade triangles `order` (back to front) into a BGRA array.

        Triangles are batched by bounding-box size and filled K x K pixels at a
        time, so a whole batch is a handful of numpy operations. Later
        triangles overwrite earlier ones, which is the painter's algorithm."""
        out = np.zeros((ih, iw, 4), dtype=np.uint8)
        a, b, c = self._fa[order], self._fb[order], self._fc[order]
        X = np.stack([sx[a], sx[b], sx[c]], axis=1).astype(np.float32)
        Y = np.stack([sy[a], sy[b], sy[c]], axis=1).astype(np.float32)
        minx, miny = np.floor(X.min(1)), np.floor(Y.min(1))
        bwid = np.ceil(X.max(1)) - minx + 1
        bhei = np.ceil(Y.max(1)) - miny + 1
        size = np.maximum(bwid, bhei)
        den = ((Y[:, 1] - Y[:, 2]) * (X[:, 0] - X[:, 2]) + (X[:, 2] - X[:, 1]) * (Y[:, 0] - Y[:, 2]))
        good = (np.abs(den) > 1e-4) & (size > 0)
        C = np.stack([vcol[a], vcol[b], vcol[c]], axis=1).astype(np.float32)    # N,3,3
        rank = np.full((ih, iw), -1, dtype=np.int32)       # draw order already at each pixel
        taken = np.zeros(len(order), dtype=bool)
        prev_hi = None
        for K, lo in ((96, 40), (40, 28), (28, 20), (20, 15), (15, 11), (11, 8), (8, 6), (6, 0)):
            sel = good & ~taken & (size <= K if K != 96 else size > 40) & (size > lo if K != 96 else True)
            idx = np.flatnonzero(sel)
            taken = taken | sel
            if idx.size == 0:
                continue
            ar = np.arange(K, dtype=np.float32)
            px = minx[idx][:, None, None] + ar[None, None, :] + 0.5            # N,1,K
            py = miny[idx][:, None, None] + ar[None, :, None] + 0.5            # N,K,1
            x0_, x1_, x2_ = (X[idx, i][:, None, None] for i in range(3))
            y0_, y1_, y2_ = (Y[idx, i][:, None, None] for i in range(3))
            d = den[idx][:, None, None]
            l0 = ((y1_ - y2_) * (px - x2_) + (x2_ - x1_) * (py - y2_)) / d
            l1 = ((y2_ - y0_) * (px - x2_) + (x0_ - x2_) * (py - y2_)) / d
            l2 = 1.0 - l0 - l1
            inside = (l0 >= -0.04) & (l1 >= -0.04) & (l2 >= -0.04)            # slight overlap: no cracks
            n, yy, xx = np.nonzero(inside)
            if n.size == 0:
                continue
            gx = (minx[idx][n] + xx).astype(np.int32)
            gy = (miny[idx][n] + yy).astype(np.int32)
            ok = (gx >= 0) & (gx < iw) & (gy >= 0) & (gy < ih)
            n, yy, xx, gx, gy = n[ok], yy[ok], xx[ok], gx[ok], gy[ok]
            rk = idx[n]                                    # position in the back-to-front order
            win = rk > rank[gy, gx]                        # only draw over what is BEHIND us
            n, yy, xx, gx, gy, rk = n[win], yy[win], xx[win], gx[win], gy[win], rk[win]
            if n.size == 0:
                continue
            rank[gy, gx] = rk
            w0 = np.clip(l0[n, yy, xx], 0.0, 1.0)[:, None]
            w1 = np.clip(l1[n, yy, xx], 0.0, 1.0)[:, None]
            w2 = np.clip(l2[n, yy, xx], 0.0, 1.0)[:, None]
            tot = np.maximum(w0 + w1 + w2, 1e-6)
            col = (C[idx][n, 0] * w0 + C[idx][n, 1] * w1 + C[idx][n, 2] * w2) / tot
            px8 = np.clip(col, 0, 255).astype(np.uint8)
            out[gy, gx, 0] = px8[:, 2]       # B
            out[gy, gx, 1] = px8[:, 1]       # G
            out[gy, gx, 2] = px8[:, 0]       # R
            out[gy, gx, 3] = 255
        return self._soften(out)

    @staticmethod
    def _soften(a: np.ndarray) -> np.ndarray:
        """1-2-1 blur on all four channels. The buffer is premultiplied and
        empty pixels are zero, so this feathers the silhouette (no stair-steps)
        without any colour fringing."""
        t = a.astype(np.uint16)
        t[:, 1:-1] = (t[:, :-2] + 2 * t[:, 1:-1] + t[:, 2:]) >> 2
        t[1:-1] = (t[:-2] + 2 * t[1:-1] + t[2:]) >> 2
        return t.astype(np.uint8)

    def _paint_real(self, p: QPainter, xs, ys, norms, verts, r: float,
                    primary: QColor, bg: QColor, amp: float) -> None:
        a, b, c = self._fa, self._fb, self._fc
        vcol = self._vertex_colours(norms, verts, primary, bg, amp)
        self._primary = primary

        fn = np.cross(verts[b] - verts[a], verts[c] - verts[a])
        fn /= np.maximum(np.linalg.norm(fn, axis=1, keepdims=True), 1e-9)
        ref = norms[a] + norms[b] + norms[c]
        fn *= np.sign((fn * ref).sum(1))[:, None]
        area = np.abs((xs[b] - xs[a]) * (ys[c] - ys[a]) - (xs[c] - xs[a]) * (ys[b] - ys[a]))
        vis = np.flatnonzero((fn[:, 2] > 0.0) & (area > 1.5))
        if vis.size:
            fz = (verts[a, 2][vis] + verts[b, 2][vis] + verts[c, 2][vis]) * (1.0 / 3.0)
            order = vis[np.argsort(self._fgroup[vis] * 1000.0 + fz, kind="stable")]
            # The skin is shaded per PIXEL (Gouraud) in a half-size buffer, then
            # scaled back up with smoothing. Vectorised: the old per-triangle
            # Qt loop spent its time creating ~1000 Python objects a frame.
            sc = 0.45
            x0, y0 = float(xs.min()) - 6.0, float(ys.min()) - 6.0
            bw = int(float(xs.max()) - x0 + 8.0)
            bh = int(float(ys.max()) - y0 + 8.0)
            iw, ih = max(8, int(bw * sc)), max(8, int(bh * sc))
            # While the mouth is still the skin changes only by the slow head
            # sway, so re-shading it every other frame is invisible and halves
            # the idle cost. Speaking re-shades every frame (the jaw moves).
            import time as _t
            now = _t.monotonic()
            cache = getattr(self, "_skin_cache", None)
            if (cache and self._mouth < 0.04 and now - cache[0] < 0.030
                    and abs(cache[1][0] - iw) <= 1 and abs(cache[1][1] - ih) <= 1
                    and abs(cache[2] - x0) < 3.0 and abs(cache[3] - y0) < 3.0):
                p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
                p.drawImage(QRectF(cache[2], cache[3], cache[1][2], cache[1][3]), cache[4])
                self._paint_real_features(p, xs, ys, r)
                return
            arr = self._raster_skin(order, (xs - x0) * (iw / bw), (ys - y0) * (ih / bh),
                                    vcol, iw, ih)
            buf = QImage(arr.data, iw, ih, iw * 4, QImage.Format.Format_ARGB32_Premultiplied)
            bp = QPainter(buf)
            bp.setRenderHint(QPainter.RenderHint.Antialiasing, True)
            bp.scale(iw / bw, ih / bh)
            bp.translate(-x0, -y0)
            self._paint_hair(bp, xs, ys, verts, norms, 0.55 + 0.5 * max(0.0, float(norms[:, 1].mean()) + 0.5))
            bp.end()
            p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
            p.drawImage(QRectF(x0, y0, bw, bh), buf)
            self._arr = arr          # the QImage borrows this memory; keep it alive
            self._skin_cache = (now, (iw, ih, bw, bh), x0, y0, buf.copy())

        self._paint_real_features(p, xs, ys, r)

    def _paint_real_features(self, p: QPainter, xs, ys, r: float) -> None:
        self._paint_shoulders(p, xs, ys, getattr(self, "_primary", QColor(0, 220, 255)))
        face = max(0.0, math.cos(self._yaw) * math.cos(self._pitch)) ** 2
        if face < 0.02:
            return
        lm = self._lm
        for key in ("brow_l", "brow_r"):
            self._brow_real(p, xs, ys, lm[key], r, face)
        self._mouth_real(p, xs, ys, lm, r, face)
        vis_eye = 1.0 - self._blink
        for key in ("eye_l", "eye_r"):
            self._eye_real(p, xs[lm[key]], ys[lm[key]], r, face, vis_eye, key == "eye_l")

    # eyes ------------------------------------------------------------------
    def _eye_real(self, p: QPainter, ex, ey, r: float, face: float, vis: float,
                  left: bool) -> None:
        cx, cy = float(ex.mean()), float(ey.mean())
        w = float(ex.max() - ex.min())
        # The mesh eye is a narrow closed slit; open it to a natural almond,
        # then close it by the blink and by how awake the face is.
        ev = getattr(self, "_ev", None) or {}
        k = 1.85 * max(0.03, vis) * (0.40 + 0.60 * self._lids)
        k *= 1.0 + 0.35 * ev.get("widen", 0.0) - 0.30 * ev.get("squint", 0.0)
        pts = np.stack([ex, cy + (ey - cy) * k], axis=1)
        ring = self._smooth(pts, True, 2)
        path = self._path(ring)
        h = float(pts[:, 1].max() - pts[:, 1].min())
        upper = pts[[0, 15, 14, 13, 12, 11, 10, 9, 8]]
        lower = pts[[0, 1, 2, 3, 4, 5, 6, 7, 8]]
        inner = (cx + w * 0.5) if left else (cx - w * 0.5)     # nose side
        sgn = 1.0 if left else -1.0

        p.setPen(Qt.PenStyle.NoPen)
        # socket shadow, so the eye sits IN the face
        g = QRadialGradient(cx, cy + h * 0.1, w * 0.95)
        g.setColorAt(0.0, QColor(70, 30, 24, int(95 * face)))
        g.setColorAt(1.0, QColor(70, 30, 24, 0))
        p.setBrush(QBrush(g))
        p.drawEllipse(QPointF(cx, cy + h * 0.1), w * 0.95, w * 0.62)

        p.save()
        p.setClipPath(path)
        # sclera: never pure white — shaded under the lid, pink at the corners
        sg = QLinearGradient(cx, cy - h * 0.55, cx, cy + h * 0.55)
        sg.setColorAt(0.0, QColor(168, 152, 146))
        sg.setColorAt(0.30, QColor(236, 230, 224))
        sg.setColorAt(1.0, QColor(226, 214, 208))
        p.setBrush(QBrush(sg))
        p.drawRect(QRectF(cx - w, cy - h, 2 * w, 2 * h))
        cg = QRadialGradient(inner, cy, w * 0.28)
        cg.setColorAt(0.0, QColor(206, 110, 108, 190))
        cg.setColorAt(1.0, QColor(206, 110, 108, 0))
        p.setBrush(QBrush(cg))
        p.drawRect(QRectF(cx - w, cy - h, 2 * w, 2 * h))

        # eyes keep looking at the viewer as the head turns: counter-rotate
        gx = max(-1.0, min(1.0, self._gaze[0] - self._yaw * 2.2 + ev.get("gx", 0.0)))
        gy = max(-1.0, min(1.0, self._gaze[1] - self._pitch * 2.0 + ev.get("gy", 0.0)))
        R = w * 0.235
        ix = cx + gx * w * 0.17
        iy = cy + gy * max(h, w * 0.18) * 0.16
        # a near-synonym of attention: pupils open a touch when listening
        pr = R * (0.36 + 0.08 * (1.0 - self._lids) + 0.05 * (1.0 - face))
        ir = self.iris
        ig = QRadialGradient(ix, iy, R)
        ig.setColorAt(0.00, QColor(min(255, ir[0] + 70), min(255, ir[1] + 55), min(255, ir[2] + 40)))
        ig.setColorAt(0.55, QColor(*ir))
        ig.setColorAt(0.88, QColor(int(ir[0] * 0.55), int(ir[1] * 0.55), int(ir[2] * 0.55)))
        ig.setColorAt(1.00, QColor(18, 12, 10))                # limbal ring
        p.setBrush(QBrush(ig))
        p.drawEllipse(QPointF(ix, iy), R, R)
        # radial fibres
        rng = random.Random(7 if left else 11)
        for _ in range(16):
            ang = rng.uniform(0, math.tau)
            r0, r1 = pr * 1.15, R * rng.uniform(0.78, 0.97)
            light = rng.random() < 0.5
            p.setPen(QPen(QColor(255, 225, 180, 46) if light else QColor(20, 10, 4, 70), 0.8))
            p.drawLine(QPointF(ix + math.cos(ang) * r0, iy + math.sin(ang) * r0),
                       QPointF(ix + math.cos(ang) * r1, iy + math.sin(ang) * r1))
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor(4, 3, 3))
        p.drawEllipse(QPointF(ix, iy), pr, pr)                 # pupil
        # wet highlights: fixed relative to the light, not to the gaze
        p.setBrush(QColor(255, 255, 255, 235))
        p.drawEllipse(QPointF(ix - R * 0.34, iy - R * 0.36), R * 0.20, R * 0.17)
        p.setBrush(QColor(255, 255, 255, 90))
        p.drawEllipse(QPointF(ix + R * 0.30, iy + R * 0.30), R * 0.10, R * 0.08)
        # the upper lid casts a shadow on the eyeball
        lg = QLinearGradient(cx, cy - h * 0.5, cx, cy + h * 0.1)
        lg.setColorAt(0.0, QColor(20, 8, 6, 150))
        lg.setColorAt(1.0, QColor(20, 8, 6, 0))
        p.setBrush(QBrush(lg))
        p.drawRect(QRectF(cx - w, cy - h, 2 * w, h * 1.1))
        p.restore()

        # lids, lashes and crease
        up = self._smooth(upper, False, 2)
        lo = self._smooth(lower, False, 2)
        p.setBrush(Qt.BrushStyle.NoBrush)
        crease = up + np.array([0.0, -max(h, w * 0.2) * 0.62])
        p.setPen(QPen(QColor(110, 60, 44, int(110 * face)), max(1.0, w * 0.018)))
        p.drawPath(self._path(crease, False))
        p.setPen(QPen(QColor(30, 18, 14, 235), max(1.5, w * 0.034), Qt.PenStyle.SolidLine,
                      Qt.PenCapStyle.RoundCap))
        p.drawPath(self._path(up, False))
        p.setPen(QPen(QColor(120, 70, 58, 120), max(1.0, w * 0.016)))
        p.drawPath(self._path(lo, False))
        if vis > 0.4:
            p.setPen(QPen(QColor(18, 12, 10, 215), max(1.0, w * 0.014), Qt.PenStyle.SolidLine,
                          Qt.PenCapStyle.RoundCap))
            n = len(up)
            for i in range(3, n - 2, 3):
                x0, y0 = up[i]
                t = i / (n - 1)
                outward = -sgn * (1.0 - t) if left else sgn * t   # lashes sweep to the outer corner
                outward = (t - 0.5) * (-1.0 if left else 1.0)
                ln = w * (0.085 + 0.04 * math.sin(math.pi * t))
                p.drawLine(QPointF(x0, y0),
                           QPointF(x0 + outward * ln * 1.2, y0 - ln))

    # brows -----------------------------------------------------------------
    def _brow_real(self, p: QPainter, xs, ys, idx, r: float, face: float) -> None:
        pts = np.stack([xs[idx], ys[idx]], axis=1)
        ev = getattr(self, "_ev", None)
        if ev:
            inner = np.linspace(0.0, 1.0, len(pts))
            if pts[0, 0] < pts[-1, 0]:          # left-to-right: nose end is the last point
                pass
            else:
                inner = inner[::-1]
            pts = pts.copy()
            pts[:, 1] -= r * (0.062 * ev["brow"] + 0.075 * ev["worry"] * inner ** 2
                              + 0.012 * ev["widen"])
        line = self._smooth(pts, False, 3)
        d = np.gradient(line, axis=0)
        nrm = np.stack([-d[:, 1], d[:, 0]], axis=1)
        nrm /= np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-6)
        t = np.linspace(0.0, 1.0, len(line))
        left_side = pts[0, 0] < pts[-1, 0]
        # thick at the nose end, thin at the tail
        taper = (1.0 - t) if (not left_side) else t
        taper = 0.28 + 0.72 * np.clip(taper, 0, 1) ** 0.8
        th = r * 0.028 * taper[:, None]
        top = line + nrm * th
        bot = line - nrm * th * 0.8
        poly = np.vstack([top, bot[::-1]])
        p.setPen(Qt.PenStyle.NoPen)
        hair = self.hair
        p.setBrush(QColor(hair[0] + 8, hair[1] + 6, hair[2] + 4, int(215 * face)))
        p.drawPath(self._path(poly))
        rng = random.Random(3)
        p.setPen(QPen(QColor(hair[0] // 2, hair[1] // 2, hair[2] // 2, int(150 * face)), 0.9))
        for _ in range(26):
            i = rng.randrange(0, len(line) - 1)
            o = rng.uniform(-0.8, 0.8)
            a0 = line[i] + nrm[i] * th[i] * o
            a1 = line[i + 1] + nrm[i + 1] * th[i + 1] * (o + rng.uniform(-0.2, 0.2))
            p.drawLine(QPointF(float(a0[0]), float(a0[1])), QPointF(float(a1[0]), float(a1[1])))

    # mouth -----------------------------------------------------------------
    def _mouth_real(self, p: QPainter, xs, ys, lm, r: float, face: float) -> None:
        outer = np.stack([xs[lm["lips_out"]], ys[lm["lips_out"]]], axis=1)
        inner = np.stack([xs[lm["lips_in"]], ys[lm["lips_in"]]], axis=1)
        sm = (getattr(self, "_ev", None) or {}).get("smile", 0.0)
        if abs(sm) > 0.01:
            mx0 = float(outer[:, 0].mean())
            half = max(1.0, float(np.ptp(outer[:, 0])) / 2.0)
            for arr in (outer, inner):
                u = (arr[:, 0] - mx0) / half                   # -1 .. 1 across the mouth
                arr[:, 1] -= sm * r * 0.058 * np.abs(u) ** 2.2  # corners rise (or fall)
                arr[:, 0] = mx0 + (arr[:, 0] - mx0) * (1.0 + 0.07 * sm)
        o_path = self._path(self._smooth(outer, True, 2))
        top, bot = float(outer[:, 1].min()), float(outer[:, 1].max())
        lg = QLinearGradient(0, top, 0, bot)
        lg.setColorAt(0.0, QColor(150, 74, 76, int(235 * face)))
        lg.setColorAt(0.55, QColor(172, 84, 88, int(235 * face)))
        lg.setColorAt(1.0, QColor(196, 104, 108, int(235 * face)))
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QBrush(lg))
        p.drawPath(o_path)

        i_poly = self._path(self._smooth(inner, True, 1))
        open_h = float(inner[:, 1].max() - inner[:, 1].min())
        if self._mouth > 0.02 and open_h > 1.0:
            p.setBrush(QColor(52, 16, 20, int(250 * face)))
            p.drawPath(i_poly)
            p.save()
            p.setClipPath(i_poly)
            up = np.stack([xs[self._lip_up], ys[self._lip_up]], axis=1)
            th = open_h * 0.34
            teeth = np.vstack([up, (up + np.array([0.0, th]))[::-1]])
            p.setBrush(QColor(238, 232, 222, int(245 * face)))
            p.drawPath(self._path(teeth))
            if self._mouth > 0.30:                         # tongue, only when wide
                mx = float(inner[:, 0].mean())
                p.setBrush(QColor(176, 78, 86, int(225 * face)))
                p.drawEllipse(QPointF(mx, float(inner[:, 1].max()) - open_h * 0.12),
                              float(np.ptp(inner[:, 0])) * 0.26, open_h * 0.26)
            p.restore()
        else:
            p.setPen(QPen(QColor(96, 40, 44, int(210 * face)), max(1.0, r * 0.008)))
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawPath(i_poly)
        # soft highlight on the lower lip
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor(255, 220, 215, int(58 * face)))
        mx = float(outer[:, 0].mean())
        p.drawEllipse(QPointF(mx, bot - (bot - top) * 0.22),
                      float(np.ptp(outer[:, 0])) * 0.14, (bot - top) * 0.07)

    # ── face ────────────────────────────────────────────────────────────────

    def _ring(self, xs, ys, idx) -> QPolygonF:
        return QPolygonF([QPointF(float(x), float(y))
                          for x, y in zip(xs[idx], ys[idx])])

    def _paint_features(self, p: QPainter, xs, ys, norms, r: float,
                        primary: QColor, accent: QColor, bg: QColor,
                        amp: float) -> None:
        """Eyes, brows and the mouth cavity, drawn from the real landmark rings.

        The canonical model's eyes and lips are closed skin — the geometry gives
        the *shape* of the lids and mouth but no opening, so the openings are
        painted here, exactly on the landmarks that bound them.
        """
        face = max(0.0, math.cos(self._yaw) * math.cos(self._pitch)) ** 2
        if face < 0.02:
            return

        lm = self._lm
        vis = 1.0 - self._blink

        # ── eyes ────────────────────────────────────────────────────────────
        for key in ("eye_l", "eye_r"):
            idx = lm[key]
            ex, ey = xs[idx], ys[idx]
            mid_y = float(ey.mean())
            if vis < 0.999:
                ey = mid_y + (ey - mid_y) * max(0.04, vis)
            poly = QPolygonF([QPointF(float(a), float(b)) for a, b in zip(ex, ey)])

            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QBrush(_blend(bg, primary, 22)))       # socket shadow
            p.drawPolygon(poly)
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.setPen(QPen(_c(primary, 210 * face), 1.3))      # lid line
            p.drawPolygon(poly)

            if vis > 0.35:
                br = poly.boundingRect()
                gx = br.center().x() + self._gaze[0] * br.width() * 0.16
                gy = br.center().y() + self._gaze[1] * br.height() * 0.20
                cpt = QPointF(gx, gy)
                rad = min(br.height() * 0.62, br.width() * 0.20)
                p.setPen(Qt.PenStyle.NoPen)
                p.setBrush(QBrush(_c(accent, (70 + 60 * amp) * face * vis)))
                p.drawEllipse(cpt, rad, rad * vis)            # iris
                p.setBrush(QBrush(_c(accent, 245 * face * vis)))
                p.drawEllipse(cpt, rad * 0.42, rad * 0.42 * vis)   # pupil

        # ── brows ───────────────────────────────────────────────────────────
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.setPen(QPen(_c(primary, 150 * face), 1.7))
        for key in ("brow_l", "brow_r"):
            idx = lm[key]
            p.drawPolyline(self._ring(xs, ys, idx))

        # ── mouth ───────────────────────────────────────────────────────────
        inner = self._ring(xs, ys, lm["lips_in"])
        open_h = inner.boundingRect().height()

        p.setPen(Qt.PenStyle.NoPen)
        if self._mouth > 0.02:
            # The cavity is dark but never pure black — a black oval on a glowing
            # head reads as a hole, not a mouth. Tinting it with the theme keeps
            # it part of the hologram.
            p.setBrush(QBrush(_blend(bg, primary, 16 + 26 * self._mouth)))
            p.drawPolygon(inner)

            # Upper teeth: a bright strip hanging from the upper lip. It is the
            # single cheapest thing that makes an open mouth look like speech.
            ux, uy = xs[self._lip_up], ys[self._lip_up]
            th = open_h * 0.30
            pts = [QPointF(float(x), float(y)) for x, y in zip(ux, uy)]
            pts += [QPointF(float(x), float(y) + th)
                    for x, y in zip(ux[::-1], uy[::-1])]
            p.setBrush(QBrush(_blend(bg, primary, 150 + 60 * self._mouth)))
            p.drawPolygon(QPolygonF(pts))

            # A warm pool at the back of the throat, strongest when wide open.
            p.setBrush(QBrush(_c(accent, 40 * self._mouth * face)))
            p.drawPolygon(inner)

        p.setBrush(Qt.BrushStyle.NoBrush)
        p.setPen(QPen(_c(primary, (150 + 70 * self._mouth) * face), 1.3))
        p.drawPolygon(inner)                       # lip edge
        p.setPen(QPen(_c(primary, 110 * face), 1.1))
        p.drawPolygon(self._ring(xs, ys, lm["lips_out"]))
