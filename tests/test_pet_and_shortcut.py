import os
import types
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import pytest

pytest.importorskip("PyQt6")
from PyQt6.QtWidgets import QApplication

app = QApplication.instance() or QApplication([])

from core import pet as petmod
from core import shortcut
from core.pet import ANIMALS, DesktopPet


def make(kind="cat", probe=None):
    p = DesktopPet(kind, speaking_probe=probe)
    p._timer.stop()
    return p


def step(p, n=30, dt=0.033):
    for _ in range(n):
        p._last -= dt                      # simulate elapsed time deterministically
        p._tick()


def test_every_animal_paints_in_every_state():
    from PyQt6.QtGui import QImage
    for kind in ANIMALS:
        p = make(kind)
        for state in ("walk", "sit", "sleep", "jump"):
            p._state = state
            img = QImage(petmod.W, petmod.H, QImage.Format.Format_ARGB32); img.fill(0)
            p.render(img)
            assert any(img.pixelColor(x, y).alpha() for x in range(0, petmod.W, 12) for y in range(0, petmod.H, 12))


def test_pet_talks_with_a_bubble_and_moving_mouth_then_stops():
    p = make()
    p.say("Your report is ready.")
    assert p._bubble and p.talking
    step(p, 15)
    assert p._mouth > 0.2                  # mouth is moving
    step(p, 400)
    assert p._bubble == "" and not p.talking and p._mouth < 0.05


def test_pet_talks_while_the_assistant_speaks():
    flag = {"v": True}
    p = make(probe=lambda: flag["v"])
    step(p, 20)
    assert p.talking and p._mouth > 0.2 and p._state == "sit"
    flag["v"] = False
    step(p, 40)
    assert not p.talking


def test_pet_wakes_up_to_talk():
    p = make()
    p._state = "sleep"
    p.say("Wake up!")
    assert p._state == "sit"


def test_pet_stays_on_screen_while_walking():
    p = make()
    g = p._screen()
    p._state, p._dir, p._state_until = "walk", 1, 1e9
    for _ in range(600):
        step(p, 5, 0.05)
        assert g.left() - 1 <= p.x() <= g.right() - petmod.W + 1
    assert p._dir == -1 or p._x < g.right()      # it turned round at the edge


def test_long_text_is_trimmed():
    p = make()
    p.say("word " * 200)
    assert len(p._bubble) <= 240


def test_switching_animal():
    p = make()
    p.set_kind("fox"); assert p.kind == "fox"
    p.set_kind("dragon"); assert p.kind == "fox"      # unknown ignored


def test_linux_shortcut_launches_with_pet_flag(tmp_path, monkeypatch):
    monkeypatch.setattr(shortcut.os, "name", "posix")
    monkeypatch.setattr(shortcut.sys, "platform", "linux")
    f = shortcut.create_shortcut(folder=tmp_path)
    txt = f.read_text()
    assert f.suffix == ".desktop" and "--pet" in txt and "main.py" in txt and os.access(f, os.X_OK)
    g = shortcut.create_shortcut(pet_only=True, folder=tmp_path / "x")
    assert "--pet-only" in g.read_text()


def test_windows_shortcut_script_is_quoted_and_uses_pythonw():
    script = shortcut.windows_script(Path(r"C:\Users\O'Neil\Desktop\Ephraim AI.lnk"),
                                     r"C:\Python\pythonw.exe", [r"C:\My App\main.py", "--pet"], None)
    assert "O''Neil" in script and '"C:\\My App\\main.py" "--pet"' in script
    assert "CreateShortcut" in script and script.endswith("$s.Save()")


def test_windows_command_prefers_pythonw(monkeypatch):
    monkeypatch.setattr(shortcut.os, "name", "nt")
    monkeypatch.setattr(shortcut.sys, "executable", r"C:\Python\python.exe")
    exe, args = shortcut._command("--pet")
    assert exe.endswith("pythonw.exe") and args[-1] == "--pet"


# ── the signal path that the first version of the face hook got wrong ──────────
def test_main_uses_the_thread_safe_say_not_a_missing_attribute():
    src = Path("main.py").read_text(encoding="utf-8")
    assert "self.ui.say(full_out)" in src and "self.ui.hud._avatar" not in src


def test_on_say_makes_the_face_react_and_the_pet_talk():
    ui = pytest.importorskip("ui")
    face = types.SimpleNamespace(got=None, react_to_text=lambda t: setattr(face, "got", t))
    said = []
    fake = types.SimpleNamespace(hud=types.SimpleNamespace(_avatar=face),
                                 _pet=types.SimpleNamespace(isVisible=lambda: True, say=said.append))
    ui.MainWindow._on_say(fake, "Sorry, that failed.")
    assert face.got == "Sorry, that failed." and said == ["Sorry, that failed."]


def test_jarvisui_say_emits_the_signal():
    ui = pytest.importorskip("ui")
    got = []
    fake = types.SimpleNamespace(_win=types.SimpleNamespace(_say_sig=types.SimpleNamespace(emit=got.append)))
    ui.JarvisUI.say(fake, "hello")
    assert got == ["hello"]


def test_pet_starts_by_default_and_only_stays_off_if_you_turned_it_off(monkeypatch):
    ui = pytest.importorskip("ui")
    import memory.config_manager as cm
    started = []
    fake = types.SimpleNamespace(_win=types.SimpleNamespace(start_pet=lambda only: started.append(only),
                                                            _refresh_pet_btn=lambda: None, _log_sig=types.SimpleNamespace(emit=lambda m: None)))
    monkeypatch.setattr(cm, "load_api_keys", lambda: {})                       # never configured -> ON
    ui.JarvisUI.start_pet(fake, force=False)
    monkeypatch.setattr(cm, "load_api_keys", lambda: {"pet_enabled": False})   # the user switched it off -> stays off
    ui.JarvisUI.start_pet(fake, force=False)
    assert started == [False]


def test_a_pet_failure_is_shown_in_the_chat_not_swallowed():
    ui = pytest.importorskip("ui")
    log = []
    def boom(only): raise RuntimeError("no display")
    fake = types.SimpleNamespace(_win=types.SimpleNamespace(start_pet=boom, _refresh_pet_btn=lambda: None,
                                                            _log_sig=types.SimpleNamespace(emit=log.append)))
    ui.JarvisUI.start_pet(fake)
    assert "desktop pet could not start" in log[0] and "no display" in log[0]
