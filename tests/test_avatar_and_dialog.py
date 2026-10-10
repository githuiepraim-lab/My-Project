import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import numpy as np
import pytest

pytest.importorskip("PyQt6")
from PyQt6.QtGui import QColor, QImage, QPainter
from PyQt6.QtWidgets import QApplication

app = QApplication.instance() or QApplication([])

from core.avatar import HoloAvatar


def render(av, size=520, r=220):
    img = QImage(size, size, QImage.Format.Format_RGB32)
    img.fill(QColor(3, 10, 16))
    p = QPainter(img)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    av.paint(p, size // 2, size // 2, r, QColor(0, 220, 255), QColor(255, 160, 40), QColor(3, 10, 16))
    p.end()
    ptr = img.constBits()
    ptr.setsize(img.sizeInBytes())
    return np.frombuffer(ptr, np.uint8).reshape(size, img.bytesPerLine() // 4, 4)[:, :size, :3].astype(int)


def settle(av, n=40, **kw):
    for _ in range(n):
        av.step(1 / 30, kw.get("amp", 0.0), speaking=kw.get("speaking", False), muted=False)


def skin_pixels(a):
    b, g, r = a[..., 0], a[..., 1], a[..., 2]
    return int(((r > 150) & (r > g + 20) & (g > b) & (r < 255)).sum())


def sclera_pixels(a):
    b, g, r = a[..., 0], a[..., 1], a[..., 2]
    return int(((r > 215) & (g > 205) & (b > 195)).sum())


def test_realistic_face_has_skin_and_visible_eyes():
    av = HoloAvatar()
    av.set_look(style="real")
    settle(av)
    a = render(av)
    assert skin_pixels(a) > 20_000
    assert sclera_pixels(a) > 150          # eye whites are actually drawn


def test_hologram_is_still_available_and_different():
    av = HoloAvatar()
    settle(av)
    av.set_look(style="holo")
    holo = render(av)
    av.set_look(style="real")
    real = render(av)
    assert skin_pixels(holo) < 2_000 < skin_pixels(real)


def test_blink_hides_the_eyes():
    av = HoloAvatar()
    av.set_look(style="real")
    settle(av)
    av._blink = 0.0
    open_px = sclera_pixels(render(av))
    av._blink = 0.95
    shut_px = sclera_pixels(render(av))
    assert shut_px < open_px * 0.4


def test_every_state_renders_without_error():
    av = HoloAvatar()
    av.set_look(style="real")
    for kw in ({}, {"speaking": True, "amp": 0.9}):
        settle(av, **kw)
        render(av)
    av._mouth = 1.0
    av._yaw, av._pitch = 0.5, -0.3
    render(av)
    av._yaw = 1.6                                    # turned fully away: no crash
    render(av)
    render(av, size=260, r=100)                      # small HUD size
    render(av, size=900, r=380)


def test_eye_contact_counter_rotates_with_head_turn():
    av = HoloAvatar()
    av.set_look(style="real")
    settle(av)
    av._yaw = 0.0
    a = render(av)
    av._yaw = 0.3
    b = render(av)
    assert (a != b).any()


def test_skin_tone_is_configurable():
    av = HoloAvatar()
    settle(av)
    av.set_look(style="real", skin=(110, 70, 48))
    dark = render(av).reshape(-1, 3)
    av.set_look(style="real", skin=(236, 190, 160))
    light = render(av).reshape(-1, 3)
    assert light.mean() > dark.mean()


def test_settings_dialog_roundtrip_without_leaking_keys(tmp_path, monkeypatch):
    from core.ai import config as C
    monkeypatch.setattr(C, "CONFIG_DIR", tmp_path)
    monkeypatch.setattr(C, "PROVIDERS_FILE", tmp_path / "providers.json")
    monkeypatch.setattr(C, "SECRETS_FILE", tmp_path / "secrets.json")
    monkeypatch.setattr(C, "LEGACY_FILE", tmp_path / "api_keys.json")
    monkeypatch.setattr(C, "_keyring", lambda: None)
    monkeypatch.setattr(C, "_cache", None)
    from core.ai.settings_dialog import AISettingsDialog
    C.set_api_key("groq", "gsk_SECRETSECRETSECRET12345")
    dlg = AISettingsDialog()
    names = [dlg.table.item(r, 1).text() for r in range(dlg.table.rowCount())]
    r = names.index("groq")
    dlg.table.cellWidget(r, 0).setChecked(True)
    dlg.table.cellWidget(r, 2).setValue(3)
    dlg.table.item(r, 5).setText("llama-3.1-8b-instant:0, llama-3.3-70b-versatile:2")
    dlg.strategy.setCurrentText("priority")
    dlg.max_out.setValue(300)
    dlg._save()
    saved = C.load(force=True)
    g = saved["providers"]["groq"]
    assert g["enabled"] and g["priority"] == 3
    assert [(m["id"], m["tier"]) for m in g["models"]] == [("llama-3.1-8b-instant", 0),
                                                            ("llama-3.3-70b-versatile", 2)]
    assert saved["settings"]["strategy"] == "priority" and saved["settings"]["max_output_tokens"] == 300
    assert "SECRETSECRET" not in (tmp_path / "providers.json").read_text()
    assert "✓" in dlg.table.item(r, 3).text()


# ── facial reactions ───────────────────────────────────────────────────────
def _expr(av, name, steps=40):
    av.react(name, 10)
    settle(av, steps)
    return render(av)


def test_smile_raises_mouth_corners_and_changes_the_face():
    av = HoloAvatar(); av.set_look(style="real"); settle(av)
    base = render(av)
    happy = _expr(av, "happy")
    assert (base != happy).sum() > 3000 and av._ev["smile"] > 0.6


def test_surprise_opens_eyes_and_mouth_and_lifts_brows():
    av = HoloAvatar(); av.set_look(style="real"); settle(av)
    sclera0 = sclera_pixels(render(av))
    _expr(av, "surprised", 60)
    assert av._ev["brow"] > 0.8 and av._mouth > 0.2
    assert sclera_pixels(render(av)) > sclera0


def test_head_turns_toward_a_reaction_and_returns_to_neutral():
    av = HoloAvatar(); av.set_look(style="real"); settle(av, 10)
    av.react("thinking", 3.0)
    settle(av, 25)
    assert av._ev["yaw"] < -0.1                       # head turned away to think
    settle(av, 200)                                   # hold expired
    assert abs(av._ev["yaw"]) < 0.02 and av._ev["brow"] < 0.02


def test_reacts_to_what_was_said():
    av = HoloAvatar(); settle(av, 5)
    av.react_to_text("Sorry, I can't do that."); assert av._emo == "concerned"
    av.react_to_text("Congratulations, great news!"); assert av._emo == "happy"
    av.react_to_text("Should we begin?"); assert av._emo == "curious"


def test_every_emotion_renders():
    av = HoloAvatar(); av.set_look(style="real")
    for name in av.EMOTIONS:
        _expr(av, name, 25)
