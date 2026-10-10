import os
import subprocess

import pytest

import actions.open_app as oa


@pytest.fixture
def spy(monkeypatch):
    calls = {"popen": [], "startfile": []}
    monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: calls["popen"].append((a, k)))
    monkeypatch.setattr(os, "startfile", lambda t: calls["startfile"].append(t), raising=False)
    monkeypatch.setattr(oa.time, "sleep", lambda s: None)
    return calls


def test_executable_launch_never_uses_a_shell(spy, monkeypatch):
    monkeypatch.setattr(oa.shutil, "which", lambda n: r"C:\Windows\notepad.exe" if n == "notepad" else None)
    assert oa._launch_windows("notepad")
    assert spy["startfile"] == [r"C:\Windows\notepad.exe"]
    assert not any(k.get("shell") for _a, k in spy["popen"])


@pytest.mark.parametrize("evil", ["calc & del /q *", "x:foo && shutdown /s", "a:b|c", "x: y",
                                  'spotify:"; rm -rf', "calc.exe; whoami"])
def test_injection_attempts_are_not_launched(spy, monkeypatch, evil):
    monkeypatch.setattr(oa.shutil, "which", lambda n: None)
    monkeypatch.setitem(__import__("sys").modules, "pyautogui", None)       # no Start-menu fallback
    oa._launch_windows(evil)
    assert spy["startfile"] == [] and spy["popen"] == []


@pytest.mark.parametrize("uri", ["spotify:", "ms-settings:display", "steam://open/main"])
def test_real_uri_schemes_still_work(spy, monkeypatch, uri):
    monkeypatch.setattr(oa.shutil, "which", lambda n: None)
    assert oa._launch_windows(uri)
    assert spy["startfile"] == [uri]
