import os
import types
import zipfile
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import pytest

pytest.importorskip("PyQt6")
from PyQt6.QtWidgets import QApplication

app = QApplication.instance() or QApplication([])

from core import files_panel as fp
from core.files_panel import FilesPanel, PreviewDialog, Registry, is_sensitive, kind_of


@pytest.fixture
def reg(tmp_path):
    return Registry(tmp_path / "Shared")


def mk(tmp_path, name, data=b"x" * 2048):
    p = tmp_path / name
    p.write_bytes(data)
    return p


def test_kinds_and_sizes():
    assert [kind_of(n) for n in ("a.PDF", "b.png", "c.mp4", "d.zip", "e.txt", "f.bin")] == \
        ["pdf", "image", "video", "zip", "text", "file"]
    assert fp.human_size(500) == "500 B" and fp.human_size(2048) == "2.0 KB" and fp.human_size(5 * 1024 ** 2) == "5.0 MB"


def test_add_copies_into_shared_and_survives_the_original_moving(reg, tmp_path):
    src = mk(tmp_path, "report.pdf")
    row = reg.add(src, "weekly")
    assert Path(row["path"]).parent == reg.dir and row["kind"] == "pdf"
    src.unlink()
    assert [r["name"] for r in reg.list()] == ["report.pdf"]            # still there, still listed


def test_same_file_twice_is_one_card_and_name_clash_is_kept_apart(reg, tmp_path):
    a = mk(tmp_path, "r.pdf", b"a" * 10)
    reg.add(a); reg.add(a)
    assert len(reg.list()) == 1
    b = tmp_path / "other"; b.mkdir()
    reg.add(mk(b, "r.pdf", b"b" * 99))
    assert sorted(r["name"] for r in reg.list()) == ["r (1).pdf", "r.pdf"]


@pytest.mark.parametrize("name", ["secrets.json", "api_keys.json", ".env", "my_token.txt", "id_rsa", "cert.pem"])
def test_credential_files_are_never_shared(reg, tmp_path, name):
    with pytest.raises(ValueError, match="credentials"):
        reg.add(mk(tmp_path, name))


def test_files_in_config_or_ssh_dirs_are_refused(tmp_path):
    for d in ("config", ".ssh"):
        (tmp_path / d).mkdir()
        assert is_sensitive(mk(tmp_path / d, "notes.txt"))


def test_missing_file_is_an_error(reg, tmp_path):
    with pytest.raises(ValueError):
        reg.add(tmp_path / "nope.pdf")


def test_panel_shows_a_card_and_counts(reg, tmp_path):
    p = FilesPanel(reg)
    seen = []
    p.countChanged.connect(seen.append)
    row = p.add(mk(tmp_path, "plan.pdf"), "my plan")
    assert p.count() == 1 and seen == [1] and not p._empty.isVisibleTo(p)
    assert any("plan.pdf" in w.text() for w in p.findChildren(type(p._empty)))
    p.add(tmp_path / "plan.pdf")                                         # duplicate: no second card
    assert p.count() == 1


def test_panel_reloads_earlier_files(reg, tmp_path):
    FilesPanel(reg).add(mk(tmp_path, "one.zip"))
    assert FilesPanel(reg).count() == 1


def test_download_saves_a_copy_where_you_choose(reg, tmp_path, monkeypatch):
    p = FilesPanel(reg)
    row = p.add(mk(tmp_path, "movie.mp4", b"video-bytes"))
    dest = tmp_path / "Downloads" / "mine.mp4"; dest.parent.mkdir()
    monkeypatch.setattr(fp.QFileDialog, "getSaveFileName", staticmethod(lambda *a, **k: (str(dest), "")))
    msgs = []
    p.status.connect(msgs.append)
    assert p.download(row) == str(dest) and dest.read_bytes() == b"video-bytes"
    assert Path(row["path"]).exists() and "Saved movie.mp4" in msgs[0]


def test_cancelled_download_changes_nothing(reg, tmp_path, monkeypatch):
    p = FilesPanel(reg)
    row = p.add(mk(tmp_path, "a.pdf"))
    monkeypatch.setattr(fp.QFileDialog, "getSaveFileName", staticmethod(lambda *a, **k: ("", "")))
    assert p.download(row) is None


def test_remove_drops_the_card_but_keeps_the_file(reg, tmp_path):
    p = FilesPanel(reg)
    row = p.add(mk(tmp_path, "a.pdf"))
    p.remove(row, p._card_widgets()[0])
    assert p.count() == 0 and reg.list() == []              # card and index entry gone
    assert Path(row["path"]).exists()                       # the file itself is kept
    assert FilesPanel(reg).count() == 0


def test_previews_for_every_kind(reg, tmp_path):
    from PyQt6.QtGui import QColor, QImage
    from core import reports
    pdf = reports.make_pdf("Preview test", "Hello.\n\nWorld.", out=str(tmp_path / "t.pdf"))
    img = tmp_path / "i.png"
    im = QImage(50, 40, QImage.Format.Format_RGB32); im.fill(QColor(10, 200, 90)); im.save(str(img))
    z = tmp_path / "z.zip"
    with zipfile.ZipFile(z, "w") as zf:
        zf.writestr("inside.txt", "hi")
    txt = mk(tmp_path, "n.txt", b"plain text")
    for src in (pdf, img, z, txt):
        row = reg.add(src)
        d = PreviewDialog(row)
        assert d.view is not None and "not available" not in getattr(d.view, "text", lambda: "")()
    zrow = reg.add(z)
    assert "inside.txt" in PreviewDialog(zrow).view.item(0).text()


def test_unpreviewable_and_broken_files_degrade_gracefully(reg, tmp_path):
    row = reg.add(mk(tmp_path, "blob.bin"))
    assert PreviewDialog(row).view is not None
    bad = reg.add(mk(tmp_path, "broken.zip", b"not a zip"))
    assert "not available" in PreviewDialog(bad).view.text()               # message, not a crash


def test_video_preview_builds_or_falls_back(reg, tmp_path):
    row = reg.add(mk(tmp_path, "v.mp4", b"\x00" * 64))
    d = PreviewDialog(row)
    assert d.view is not None
    d.done(0)                                                              # stops the player cleanly


# ── wiring ─────────────────────────────────────────────────────────────────
def test_jarvisui_share_file_emits_the_signal():
    ui = pytest.importorskip("ui")
    got = []
    fake = types.SimpleNamespace(_win=types.SimpleNamespace(_file_sig=types.SimpleNamespace(emit=lambda p, n: got.append((p, n)))))
    ui.JarvisUI.share_file(fake, "/tmp/x.pdf", "note")
    assert got == [("/tmp/x.pdf", "note")]


def test_on_file_adds_card_opens_panel_and_announces_in_chat(reg, tmp_path):
    ui = pytest.importorskip("ui")
    panel = FilesPanel(reg)
    log = []
    dock = types.SimpleNamespace(shown=False, show=lambda: setattr(dock, "shown", True), raise_=lambda: None)
    fake = types.SimpleNamespace(_ensure_files_panel=lambda: panel, _files_dock=dock,
                                 _log=types.SimpleNamespace(append_log=log.append))
    ui.MainWindow._on_file(fake, str(mk(tmp_path, "report.pdf")), "weekly")
    assert panel.count() == 1 and dock.shown and "report.pdf" in log[0] and "FILES panel" in log[0]


def test_on_file_reports_a_refused_file_in_chat(reg, tmp_path):
    ui = pytest.importorskip("ui")
    panel, log = FilesPanel(reg), []
    fake = types.SimpleNamespace(_ensure_files_panel=lambda: panel, _files_dock=None,
                                 _log=types.SimpleNamespace(append_log=log.append))
    ui.MainWindow._on_file(fake, str(mk(tmp_path, "secrets.json")), "")
    assert "Could not share" in log[0] and panel.count() == 0


def test_report_tool_puts_the_pdf_in_the_panel(tmp_path):
    import actions.report_sender as rs
    shared = []
    player = types.SimpleNamespace(share_file=lambda p, n="": shared.append(p))
    out = rs.report_sender({"mode": "make_pdf", "title": "Q3 report", "body": "Numbers.\n\nMore."}, player)
    assert shared and shared[0].endswith(".pdf") and "FILES panel" in out and Path(shared[0]).read_bytes().startswith(b"%PDF")


def test_share_mode_and_credentials_guard(tmp_path):
    import actions.report_sender as rs
    shared = []
    player = types.SimpleNamespace(share_file=lambda p, n="": shared.append(p))
    v, z = mk(tmp_path, "clip.mp4"), mk(tmp_path, "bundle.zip")
    assert "Shared 2" in rs.report_sender({"mode": "share", "files": f"{v}, {z}"}, player)
    assert shared == [str(v), str(z)]
    assert "credential" in rs.report_sender({"mode": "share", "files": str(mk(tmp_path, "secrets.json"))}, player)
    assert "not available" in rs.report_sender({"mode": "share", "files": str(v)}, None)
