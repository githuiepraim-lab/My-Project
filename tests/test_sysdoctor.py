import importlib.metadata as md
import json
import sys
import time
import types
from pathlib import Path

import pytest
from core import sysdoctor as S


def ps(mem=40, cpu=10):
    return types.SimpleNamespace(virtual_memory=lambda: types.SimpleNamespace(percent=mem), cpu_percent=lambda interval=0: cpu,
                                 process_iter=lambda attrs: [types.SimpleNamespace(info={"name": "chrome.exe", "memory_percent": 22.0})])


def run(tmp_path, **kw):
    d = dict(psutil_mod=ps(), net=lambda: (True, ""), audio=lambda: (True, ""), disk_path=tmp_path)
    d.update(kw)
    return {f.id: f for f in S.diagnose(tmp_path, **d)}


def test_healthy_machine_is_all_clear(tmp_path):
    (tmp_path / "requirements.txt").write_text("pytest\n")
    f = run(tmp_path)
    assert all(x.severity in ("ok", "info") for x in f.values()) and f["internet"].severity == "ok"


def test_requirements_parsing_skips_comments_markers_and_options(tmp_path):
    r = tmp_path / "r.txt"
    r.write_text("# c\nnumpy>=1.2,<3  # fast\n-r other.txt\npywin32>=300 ; sys_platform == 'win32'\ngoogle-genai>=2\n\nPyQt6[extra]==6.5\n")
    names = S.requirement_names(r)
    assert names == ["numpy", "google-genai", "PyQt6"] or (sys.platform == "win32" and "pywin32" in names)


def test_missing_packages_are_found_and_fixable(tmp_path):
    (tmp_path / "requirements.txt").write_text("pytest\nthis-package-does-not-exist-xyz\n")
    f = run(tmp_path)
    assert f["packages"].severity == "bad" and f["packages"].fix == "install_missing" and f["packages"].data["missing"] == ["this-package-does-not-exist-xyz"]
    calls = []
    out = S.fix_install_missing(tmp_path, run=lambda cmd, **k: calls.append(cmd) or types.SimpleNamespace(returncode=0, stderr=""))
    assert calls[0][:4] == [sys.executable, "-m", "pip", "install"] and calls[0][4:] == ["this-package-does-not-exist-xyz"] and "Installed" in out
    bad = S.fix_install_missing(tmp_path, run=lambda cmd, **k: types.SimpleNamespace(returncode=1, stderr="ERROR: no match\n"))
    assert "could not install" in bad and "no match" in bad


@pytest.mark.parametrize("mem,cpu,sev_mem,sev_cpu", [(50, 20, "ok", "ok"), (85, 95, "warn", "warn"), (96, 20, "bad", "ok")])
def test_memory_and_cpu_thresholds(tmp_path, mem, cpu, sev_mem, sev_cpu):
    f = run(tmp_path, psutil_mod=ps(mem, cpu))
    assert f["memory"].severity == sev_mem and f["cpu"].severity == sev_cpu
    if mem > 80:
        assert "chrome.exe" in f["memory"].detail


def test_problems_are_named_not_hidden(tmp_path, monkeypatch):
    monkeypatch.setattr(S.shutil, "disk_usage", lambda p: types.SimpleNamespace(free=int(0.5 * S.GB)))
    f = run(tmp_path, net=lambda: (False, "cannot resolve names (DNS)"), audio=lambda: (False, "no usable microphone"))
    assert f["disk"].severity == "bad" and f["disk"].fix == "clear_cache"
    assert f["internet"].severity == "bad" and "DNS" in f["internet"].detail
    assert f["audio"].severity == "bad" and "microphone" in f["audio"].detail
    ordered = [x.severity for x in S.diagnose(tmp_path, psutil_mod=ps(), net=lambda: (False, "x"), audio=lambda: (True, ""), disk_path=tmp_path)]
    assert ordered == sorted(ordered, key=lambda s: -S.SEV[s])                         # worst first


def test_corrupt_keys_file_and_broken_provider_config_are_reported(tmp_path):
    (tmp_path / "config").mkdir()
    (tmp_path / "config" / "api_keys.json").write_text("{not json")
    (tmp_path / "config" / "providers.json.bad").write_text("x")
    f = run(tmp_path)
    assert f["keysfile"].severity == "bad" and f["provcfg"].fix == "repair_config"


def test_a_failing_ai_provider_is_spotted_from_the_usage_log(tmp_path):
    (tmp_path / "logs").mkdir()
    rows = [{"provider": "groq", "status": "auth"}] * 8 + [{"provider": "gemini", "status": "ok"}] * 8
    (tmp_path / "logs" / "ai_usage.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    f = run(tmp_path)
    assert f["ai_groq"].fix == "reset_ai" and "python -m core.ai check groq" in f["ai_groq"].detail and "ai_gemini" not in f


def test_the_gemini_key_is_checked_when_given(tmp_path, monkeypatch):
    import core.ai.doctor as D
    monkeypatch.setattr(D, "diagnose", lambda key: ("bad_key", "Google rejected the key"))
    f = run(tmp_path, key="AIza" + "x" * 35)
    assert f["gemini"].severity == "bad" and "rejected" in f["gemini"].detail


def test_stuck_and_failed_jobs_are_reported_and_recoverable(tmp_path):
    from core.jobs import JobStore
    (tmp_path / "memory").mkdir()
    st = JobStore(tmp_path / "memory" / "jobs.db")
    a = st.add("a", "remind", {}, time.time() - 9000); st.claim_due()
    with st._c() as c:
        c.execute("UPDATE jobs SET started=? WHERE id=?", (time.time() - 7200, a["id"]))
    b = st.add("b", "remind", {}, time.time() - 1); st.claim_due(); st.fail(b["id"], "boom")
    f = run(tmp_path)
    assert f["jobs_stuck"].fix == "recover_jobs" and "boom" in f["jobs_failed"].detail
    assert "1 stuck" in S.fix_recover_jobs(tmp_path)
    assert st.get(a["id"])["status"] == "queued"


def test_clear_cache_only_removes_caches_and_old_logs(tmp_path):
    (tmp_path / "pkg" / "__pycache__").mkdir(parents=True); (tmp_path / "pkg" / "__pycache__" / "x.pyc").write_bytes(b"x" * 1000)
    (tmp_path / "pkg" / "keep.py").write_text("print(1)")
    (tmp_path / "logs").mkdir(); (tmp_path / "logs" / "ai_usage.1.jsonl").write_text("old"); (tmp_path / "logs" / "ai_usage.jsonl").write_text("current")
    (tmp_path / "proposals" / "old").mkdir(parents=True); (tmp_path / "proposals" / "new").mkdir()
    import os
    os.utime(tmp_path / "proposals" / "old", (1, 1))
    (tmp_path / "memory").mkdir(); (tmp_path / "memory" / "long_term.json").write_text("{}")
    out = S.fix_clear_cache(tmp_path)
    assert "Cleared" in out and not (tmp_path / "pkg" / "__pycache__").exists() and (tmp_path / "pkg" / "keep.py").exists()
    assert not (tmp_path / "logs" / "ai_usage.1.jsonl").exists() and (tmp_path / "logs" / "ai_usage.jsonl").exists()
    assert not (tmp_path / "proposals" / "old").exists() and (tmp_path / "proposals" / "new").exists() and (tmp_path / "memory" / "long_term.json").exists()


def test_flush_dns_uses_a_fixed_command(monkeypatch):
    calls = []
    monkeypatch.setattr(S.shutil, "which", lambda n: "/x/" + n)
    S.fix_flush_dns(run=lambda cmd, **k: calls.append(cmd) or types.SimpleNamespace(returncode=0))
    assert calls[0][0] in ("resolvectl", "ipconfig", "dscacheutil")


def test_every_fix_is_a_named_function_not_a_command_string():
    assert set(S.FIXES) == set(S.FIX_DOC) and all(callable(f) for f in S.FIXES.values())


def test_the_tool_diagnoses_then_fixes_only_through_the_gate(tmp_path, monkeypatch):
    import actions.system_doctor as T
    monkeypatch.setattr(S, "diagnose", lambda *a, **k: [S.Finding("disk", "warn", "Disk space: 2.0 GB free", "low", "clear_cache"),
                                                          S.Finding("internet", "ok", "Internet: reachable")])
    out = T.system_doctor({"mode": "diagnose"})
    assert "1 problem" in out and "clear_cache" in out and "fix all" in out
    got = {}
    monkeypatch.setattr(T.confirm, "request", lambda k, t, d, run: got.update(t=t, d=d, run=run) or "pending")
    ran = []
    monkeypatch.setitem(S.FIXES, "clear_cache", lambda: ran.append(1) or "freed")
    assert T.system_doctor({"mode": "fix", "id": "all"}) == "pending" and ran == [] and "clear caches" in got["t"]
    assert "freed" in got["run"]() and ran == [1]
    assert "Fixes I know" in T.system_doctor({"mode": "fix", "id": "rm -rf /"}) and ran == [1]


def test_tool_registers_cleanly():
    from core.action_loader import discover_actions
    d = {x["name"]: x for x in discover_actions(Path("actions"), logger=lambda s: None).get_tool_declarations()}
    assert set(d["system_doctor"]) <= {"name", "description", "parameters"}
