import types
from pathlib import Path

import pytest


def test_server_tells_the_app_when_an_upload_finishes(tmp_path):
    pytest.importorskip("fastapi")
    from dashboard.server import DashboardServer
    srv = DashboardServer.__new__(DashboardServer)              # no network, no TLS: just the hook
    got = []
    srv.set_file_callback(got.append)
    srv._notify_file(tmp_path / "plan.pdf")
    assert got == [str(tmp_path / "plan.pdf")]


def test_a_failing_callback_never_breaks_the_upload(tmp_path, capsys):
    pytest.importorskip("fastapi")
    from dashboard.server import DashboardServer
    srv = DashboardServer.__new__(DashboardServer)
    srv.set_file_callback(lambda p: 1 / 0)
    srv._notify_file(tmp_path / "x.pdf")                        # must not raise
    assert "file callback failed" in capsys.readouterr().out


def test_no_callback_is_fine(tmp_path):
    pytest.importorskip("fastapi")
    from dashboard.server import DashboardServer
    DashboardServer.__new__(DashboardServer)._notify_file(tmp_path / "x.pdf")


def test_main_hands_the_upload_to_the_ui_and_wires_the_callback():
    src = Path("main.py").read_text(encoding="utf-8")
    assert "set_file_callback(self._on_phone_file)" in src and "self.ui.receive_remote_file(path)" in src


def test_jarvisui_emits_the_remote_file_signal():
    ui = pytest.importorskip("ui")
    got = []
    fake = types.SimpleNamespace(_win=types.SimpleNamespace(_remote_file_sig=types.SimpleNamespace(emit=got.append)))
    ui.JarvisUI.receive_remote_file(fake, "/uploads/a.pdf")
    assert got == ["/uploads/a.pdf"]


def test_phone_upload_is_loaded_like_a_desktop_drop_and_shown_in_files(tmp_path):
    ui = pytest.importorskip("ui")
    f = tmp_path / "notes.pdf"; f.write_bytes(b"%PDF")
    seen, log, shared = [], [], []
    fake = types.SimpleNamespace(_drop_zone=types.SimpleNamespace(_set_file=seen.append),
                                 _log=types.SimpleNamespace(append_log=log.append),
                                 _on_file=lambda p, n: shared.append((p, n)))
    ui.MainWindow._on_remote_file(fake, str(f))
    assert seen == [str(f)] and shared == [(str(f), "Uploaded from your phone")] and "Received from your phone" in log[0]


def test_a_bad_upload_is_reported_not_raised(tmp_path):
    ui = pytest.importorskip("ui")
    log = []
    def boom(p): raise OSError("gone")
    fake = types.SimpleNamespace(_drop_zone=types.SimpleNamespace(_set_file=boom), _log=types.SimpleNamespace(append_log=log.append),
                                 _on_file=lambda p, n: None)
    ui.MainWindow._on_remote_file(fake, "/nope.pdf")
    assert any("Could not open the phone upload" in m for m in log)


def test_real_upload_through_the_fastapi_app_reaches_the_app_callback(tmp_path):
    """The actual HTTP route a phone uses, not a stand-in: auth, multipart, saved file, callback."""
    pytest.importorskip("fastapi"); pytest.importorskip("httpx"); pytest.importorskip("multipart")
    from fastapi.testclient import TestClient
    from dashboard.server import DashboardServer
    srv = DashboardServer()
    srv._uploads_dir = tmp_path
    got = []
    srv.set_file_callback(got.append)
    srv._tokens.add("tok123")
    client = TestClient(srv._build_app())
    r = client.post("/api/upload", files={"file": ("my report.pdf", b"%PDF-1.4 data", "application/pdf")},
                    headers={"Authorization": "Bearer tok123"})
    assert r.status_code == 200 and r.json()["ok"], r.text
    saved = tmp_path / r.json()["name"]
    assert saved.read_bytes() == b"%PDF-1.4 data" and got == [str(saved)]          # the app was told
    assert [f["name"] for f in client.get("/api/files", headers={"Authorization": "Bearer tok123"}).json()["files"]] == [saved.name]
    # a second file with the same name does not overwrite the first, and is announced too
    r2 = client.post("/api/upload", files={"file": ("my report.pdf", b"second", "application/pdf")}, headers={"Authorization": "Bearer tok123"})
    assert r2.json()["name"] != r.json()["name"] and len(got) == 2
    # no token, no upload, no callback
    assert client.post("/api/upload", files={"file": ("x.txt", b"x")}).status_code == 401 and len(got) == 2
