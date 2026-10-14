"""
System doctor: find what is wrong with this computer's setup for the assistant — and fix it
when you say so.

DIAGNOSE reads and measures only; it changes nothing. FIX runs one of a fixed list of named
repairs (never anything the model composes), each shown to you for confirmation first.
"""
from __future__ import annotations

import importlib.metadata as md
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
GB = 1024 ** 3
SEV = {"ok": 0, "info": 1, "warn": 2, "bad": 3}


@dataclass
class Finding:
    id: str
    severity: str
    title: str
    detail: str = ""
    fix: str | None = None
    data: dict = field(default_factory=dict)

    def line(self) -> str:
        icon = {"ok": "✓", "info": "ℹ", "warn": "⚠", "bad": "✗"}[self.severity]
        return f"{icon} {self.title}" + (f" — {self.detail}" if self.detail else "") + (f"  [fix: {self.fix}]" if self.fix else "")


def requirement_names(req_file: Path) -> list[str]:
    out = []
    for line in req_file.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.split("#")[0].strip()
        if not line or line.startswith("-"):
            continue
        m = re.match(r"^([A-Za-z0-9][A-Za-z0-9._-]*)", line)
        if not m:
            continue
        marker = line.partition(";")[2].strip()
        if marker and "win32" in marker and sys.platform != "win32":
            continue
        out.append(m.group(1))
    return out


def missing_packages(req_file: Path, version=md.version) -> list[str]:
    miss = []
    for name in requirement_names(req_file):
        try:
            version(name)
        except md.PackageNotFoundError:
            miss.append(name)
    return miss


def _internet(host="www.google.com") -> tuple[bool, str]:
    try:
        socket.getaddrinfo(host, 443)
    except OSError:
        return False, "cannot resolve names (DNS) — no internet, or a VPN/DNS problem"
    try:
        import requests
        requests.get("https://www.google.com/generate_204", timeout=6)
        return True, ""
    except Exception as e:
        return False, f"names resolve but connections fail ({type(e).__name__}) — firewall, proxy or weak signal"


def _audio() -> tuple[bool, str]:
    try:
        import sounddevice as sd
        d = sd.default.device
        sd.query_devices(d[0]); sd.query_devices(d[1])
        return True, ""
    except Exception as e:
        return False, f"no usable microphone/speaker found ({type(e).__name__}); check Windows sound settings and that the mic is allowed"


def diagnose(root: Path = ROOT, *, net=_internet, audio=_audio, psutil_mod="auto", key: str | None = None, disk_path=None) -> list[Finding]:
    root = Path(root)
    F: list[Finding] = []
    add = lambda *a, **k: F.append(Finding(*a, **k))

    free = shutil.disk_usage(disk_path or Path.home()).free / GB
    add("disk", "bad" if free < 1 else "warn" if free < 5 else "ok", f"Disk space: {free:.1f} GB free",
        "Videos, reports and logs need room; free some space." if free < 5 else "", "clear_cache" if free < 5 else None)

    ps = psutil_mod
    if ps == "auto":
        try:
            import psutil as ps
        except ImportError:
            ps = None
    if ps is not None:
        mem = ps.virtual_memory().percent
        cpu = ps.cpu_percent(interval=0.3)
        top = sorted(((p.info["memory_percent"] or 0, p.info["name"]) for p in ps.process_iter(["name", "memory_percent"])), reverse=True)[:3]
        add("memory", "bad" if mem > 92 else "warn" if mem > 80 else "ok", f"Memory: {mem:.0f}% used",
            "Biggest users: " + ", ".join(f"{n} ({v:.0f}%)" for v, n in top) if mem > 80 else "")
        add("cpu", "warn" if cpu > 90 else "ok", f"CPU: {cpu:.0f}% busy", "The assistant's voice and face stutter when the CPU is saturated." if cpu > 90 else "")
    else:
        add("psutil", "info", "CPU/memory not measured", "pip install psutil to include them")

    ok, why = net()
    add("internet", "ok" if ok else "bad", "Internet: reachable" if ok else "Internet: NOT reachable", why)
    if ok and key is not None:
        from core.ai.doctor import diagnose as gem
        st, msg = gem(key)
        add("gemini", "ok" if st == "ok" else "bad", "Gemini key: works" if st == "ok" else "Gemini: problem", "" if st == "ok" else msg)

    req = root / "requirements.txt"
    if req.exists():
        miss = missing_packages(req)
        add("packages", "bad" if miss else "ok", f"Python packages: {len(miss)} missing" if miss else "Python packages: all installed",
            ", ".join(miss[:8]) + (" …" if len(miss) > 8 else ""), "install_missing" if miss else None, {"missing": miss})
    add("python", "ok" if sys.version_info >= (3, 10) else "bad", f"Python {sys.version_info.major}.{sys.version_info.minor}",
        "" if sys.version_info >= (3, 10) else "Python 3.10 or newer is needed")

    aok, awhy = audio()
    add("audio", "ok" if aok else "bad", "Audio devices: found" if aok else "Audio devices: PROBLEM", awhy)

    keys = root / "config" / "api_keys.json"
    if keys.exists():
        try:
            json.loads(keys.read_text(encoding="utf-8"))
            add("keysfile", "ok", "api_keys.json: readable")
        except ValueError:
            add("keysfile", "bad", "api_keys.json is corrupt", "Open it and fix the JSON, or delete it and paste your key again in setup.")
    if (root / "config" / "providers.json.bad").exists():
        add("provcfg", "warn", "A hand-edited providers.json was broken and replaced by defaults", "The broken copy is kept as providers.json.bad.", "repair_config")

    usage = root / "logs" / "ai_usage.jsonl"
    if usage.exists():
        rows = []
        for line in usage.read_text(encoding="utf-8", errors="replace").splitlines()[-300:]:
            try:
                rows.append(json.loads(line))
            except ValueError:
                pass
        by: dict[str, list] = {}
        for r in rows:
            by.setdefault(r.get("provider", "?"), []).append(r.get("status") != "ok")
        for prov, errs in by.items():
            if len(errs) >= 5 and sum(errs) / len(errs) > 0.5:
                add(f"ai_{prov}", "warn", f"AI '{prov}' failed {sum(errs)} of its last {len(errs)} requests",
                    f"Run:  python -m core.ai check {prov}  — it tells you why.", "reset_ai")

    size = 0
    for sub in ("logs", "proposals"):
        for f in (root / sub).rglob("*") if (root / sub).exists() else []:
            size += f.stat().st_size if f.is_file() else 0
    pyc = sum(f.stat().st_size for f in root.rglob("__pycache__/*") if f.is_file())
    if size + pyc > 200 * 1024 ** 2:
        add("cache", "info", f"Logs and caches use {(size + pyc) / 1024 ** 2:.0f} MB", "Safe to clear.", "clear_cache")

    jdb = root / "memory" / "jobs.db"
    if jdb.exists():
        try:
            from core.jobs import JobStore
            rows = JobStore(jdb).list(limit=200)
            failed = [j for j in rows if j["status"] == "failed" and (j.get("finished") or 0) > time.time() - 86400]
            stuck = [j for j in rows if j["status"] == "running" and (j.get("started") or 0) < time.time() - 3600]
            if failed:
                add("jobs_failed", "warn", f"{len(failed)} job(s) failed in the last 24 h", "; ".join(f"#{j['id']} {j['error'][:50]}" for j in failed[:3]))
            if stuck:
                add("jobs_stuck", "warn", f"{len(stuck)} job(s) stuck 'running' for over an hour", "", "recover_jobs")
        except Exception:
            pass
    return sorted(F, key=lambda f: -SEV[f.severity])


# ── the fixes: a fixed list, no free-form commands ───────────────────────────────────────────
def fix_install_missing(root: Path = ROOT, run=subprocess.run, **_) -> str:
    miss = missing_packages(root / "requirements.txt")
    if not miss:
        return "Nothing is missing."
    r = run([sys.executable, "-m", "pip", "install", *miss], capture_output=True, text=True, timeout=900)      # names come from requirements.txt only
    if r.returncode != 0:
        return "pip could not install them: " + (r.stderr.strip().splitlines() or ["unknown error"])[-1][:160]
    return f"Installed {', '.join(miss)}. Restart the assistant to use them."


def fix_clear_cache(root: Path = ROOT, **_) -> str:
    freed = 0
    for d in list(root.rglob("__pycache__")) + list(root.rglob(".pytest_cache")):
        if ".git" in d.parts:
            continue
        freed += sum(f.stat().st_size for f in d.rglob("*") if f.is_file())
        shutil.rmtree(d, ignore_errors=True)
    for f in (root / "logs").glob("*.1.jsonl") if (root / "logs").exists() else []:
        freed += f.stat().st_size; f.unlink(missing_ok=True)
    cutoff = time.time() - 30 * 86400
    for d in (root / "proposals").glob("*") if (root / "proposals").exists() else []:
        if d.is_dir() and d.stat().st_mtime < cutoff:
            freed += sum(f.stat().st_size for f in d.rglob("*") if f.is_file()); shutil.rmtree(d, ignore_errors=True)
    return f"Cleared {freed / 1024 ** 2:.1f} MB of caches and old logs. Your files, reports and memory were not touched."


def fix_reset_ai(**_) -> str:
    from core.ai import get_hub
    from core.ai.provider_doctor import check_all
    hub = get_hub()
    hub.health.reset()
    return "AI providers reset. " + "; ".join(f"{n}: {'ok' if st in ('ok', 'fixed') else msg[:60]}" for n, st, msg in check_all(hub))


def fix_repair_config(root: Path = ROOT, **_) -> str:
    from core.ai import config as C
    bad = root / "config" / "providers.json.bad"
    if bad.exists():
        bad.replace(bad.with_name("providers.json.bad.old"))
    C.load(force=True)
    return "providers.json is valid again (defaults). Your keys were not affected; re-add any custom providers."


def fix_recover_jobs(root: Path = ROOT, **_) -> str:
    from core.jobs import JobStore
    return f"Put {JobStore(root / 'memory' / 'jobs.db').recover()} stuck job(s) back in the queue."


def fix_flush_dns(run=subprocess.run, **_) -> str:
    cmd = (["ipconfig", "/flushdns"] if sys.platform == "win32" else ["dscacheutil", "-flushcache"] if sys.platform == "darwin"
           else ["resolvectl", "flush-caches"])
    if shutil.which(cmd[0]) is None:
        return f"{cmd[0]} is not available on this system."
    r = run(cmd, capture_output=True, text=True, timeout=30)
    return "DNS cache flushed." if r.returncode == 0 else "Could not flush DNS (it may need administrator rights)."


FIXES = {"install_missing": fix_install_missing, "clear_cache": fix_clear_cache, "reset_ai": fix_reset_ai,
         "repair_config": fix_repair_config, "recover_jobs": fix_recover_jobs, "flush_dns": fix_flush_dns}
FIX_DOC = {"install_missing": "install the missing Python packages", "clear_cache": "clear caches and old logs",
           "reset_ai": "reset and re-test the AI providers", "repair_config": "restore a valid providers.json",
           "recover_jobs": "re-queue stuck jobs", "flush_dns": "flush the DNS cache"}
