import json
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import pytest
from core import reports


@pytest.fixture
def qapp():
    pytest.importorskip("PyQt6")
    from PyQt6.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


def test_pdf_is_a_real_pdf_with_text_and_image(qapp, tmp_path):
    from PyQt6.QtGui import QImage, QColor
    img = tmp_path / "p.png"
    im = QImage(60, 40, QImage.Format.Format_RGB32); im.fill(QColor(200, 50, 50)); im.save(str(img))
    out = reports.make_pdf("Weekly <report>", "First paragraph.\n\nSecond one & more.", [str(img)],
                           out=str(tmp_path / "r.pdf"))
    data = open(out, "rb").read()
    assert data.startswith(b"%PDF") and len(data) > 1500


def test_file_checks(tmp_path):
    f = tmp_path / "a.txt"; f.write_text("x")
    assert reports.check_files([str(f)]) == [f]
    with pytest.raises(ValueError): reports.check_files([])
    with pytest.raises(ValueError): reports.check_files([str(tmp_path / "missing.txt")])
    with pytest.raises(ValueError): reports.check_files([str(f)] * 11)


def test_zip_bundles_files(tmp_path):
    a, b = tmp_path / "a.txt", tmp_path / "b.txt"; a.write_text("1"); b.write_text("2")
    z = reports.zip_files([a, b], "bundle")
    import zipfile
    assert sorted(zipfile.ZipFile(z).namelist()) == ["a.txt", "b.txt"]


def test_telegram_uses_the_right_method_per_type(tmp_path, monkeypatch):
    calls = []
    class R: status_code = 200; headers = {}
    import requests
    monkeypatch.setattr(requests, "post", lambda url, **k: calls.append((url.rsplit("/", 1)[1], list(k["files"]))) or R())
    fs = []
    for n in ("a.pdf", "b.jpg", "c.mp4", "d.zip"):
        p = tmp_path / n; p.write_bytes(b"x"); fs.append(p)
    reports.send_telegram("TOKEN", "1", fs, "cap")
    assert calls == [("sendDocument", ["document"]), ("sendPhoto", ["photo"]),
                     ("sendVideo", ["video"]), ("sendDocument", ["document"])]


def test_telegram_error_names_the_file_without_leaking_token(tmp_path, monkeypatch):
    import requests
    class R: status_code = 401; headers = {"content-type": "application/json"}; json = lambda s: {"description": "Unauthorized"}
    monkeypatch.setattr(requests, "post", lambda *a, **k: R())
    f = tmp_path / "a.pdf"; f.write_bytes(b"x")
    with pytest.raises(RuntimeError) as e:
        reports.send_telegram("SECRETTOKEN123", "1", [f])
    assert "SECRETTOKEN123" not in str(e.value) and "a.pdf" in str(e.value)


def test_email_attaches_files(tmp_path, monkeypatch):
    sent = {}
    class S:
        def __init__(self, *a, **k): pass
        def __enter__(self): return self
        def __exit__(self, *a): pass
        def starttls(self, **k): pass
        def login(self, u, p): sent["login"] = (u, p)
        def send_message(self, m): sent["m"] = m
    monkeypatch.setattr(reports.smtplib, "SMTP", S)
    f = tmp_path / "r.pdf"; f.write_bytes(b"%PDF")
    reports.send_email({"host": "h", "user": "me@x.com"}, "pw", "you@x.com", "Hi", "body", [f])
    assert [p.get_filename() for p in sent["m"].iter_attachments()] == ["r.pdf"]


def test_call_links_are_unique_and_https():
    a, b = reports.call_link(), reports.call_link()
    assert a != b and a.startswith("https://meet.jit.si/")


def test_send_waits_for_confirmation_and_names_the_files(tmp_path, monkeypatch):
    import actions.report_sender as rs
    got = {}
    monkeypatch.setattr(rs.confirm, "request", lambda k, t, d, run: got.update(t=t, d=d, run=run) or "waiting")
    f = tmp_path / "plan.pdf"; f.write_bytes(b"%PDF")
    out = rs.report_sender({"mode": "send", "files": str(f), "channel": "email"})
    assert out == "waiting" and "plan.pdf" in got["d"] and "email" in got["t"]


def test_unconfigured_channel_explains_how(tmp_path, monkeypatch):
    import actions.report_sender as rs
    from core.ai import config as C
    monkeypatch.setattr(C, "LEGACY_FILE", tmp_path / "none.json")
    monkeypatch.setattr(rs.C, "get_api_key", lambda n: "")
    f = tmp_path / "x.pdf"; f.write_bytes(b"x")
    assert "BotFather" in rs._deliver("telegram", [f], "", "", False)
    assert "smtp_host" in rs._deliver("email", [f], "", "", False)


def test_call_opens_browser_and_is_honest(monkeypatch):
    import actions.report_sender as rs
    opened = []
    monkeypatch.setattr(rs.webbrowser, "open", lambda u: opened.append(u))
    monkeypatch.setattr(rs, "_cfg", lambda: {})
    out = rs.report_sender({"mode": "call"})
    assert opened and "cannot join the call" in out
