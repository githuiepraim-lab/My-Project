"""
Self-improvement: the assistant can write changes to its OWN code, safely.

  propose -> a strong model writes the change; it is validated, then run against
             the test suite in a THROWAWAY COPY. Your real files are never touched.
  show    -> the exact diff and the test result.
  apply   -> only after you press CONFIRM on screen (core/confirm — the model
             cannot press it). Originals are backed up first and the change is
             committed on its own git branch. Never pushed.
  rollback-> restores the backups.

There is deliberately NO unattended auto-apply: this program can control your
computer, so a change to its own code always needs a human yes. Safety files
(the confirmation gate, secret handling, this module, the entry point) cannot
be changed by a proposal at all, and a proposal that deletes a confirm/redact
call or adds shell/eval/secret access is rejected. These checks are a net, not
a proof: read the diff before you confirm.
"""
from __future__ import annotations

import ast
import difflib
import json
import re
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

PROTECTED = ("main.py", "core/confirm.py", "core/selfdev.py", "core/action_loader.py",
             "actions/self_improve.py", "core/ai/redact.py", "core/ai/config.py",
             "memory/config_manager.py", "tests/test_selfdev.py", "tests/test_security.py")
PROTECTED_DIRS = (".git", "config", "logs", "proposals", "memory/long_term", ".github")
ALLOWED_EXT = {".py", ".txt", ".json", ".md"}
MAX_FILES, MAX_BYTES = 5, 60_000
_IGNORE = shutil.ignore_patterns(".git", "__pycache__", "logs", "proposals", "*.pyc", ".pytest_cache",
                                 "node_modules", "*.zip", "*.mp4", "*.png", "*.jpg", "*.ico", "*.obj",
                                 "secrets.json", "api_keys.json", "long_term.json")

_DANGEROUS_ADDED = re.compile(
    r"shell\s*=\s*True|os\.system\(|\beval\(|\bexec\(|__import__\(|pickle\.loads|marshal\.loads|"
    r"secrets\.json|api_keys\.json|get_api_key\(|keyring|\.ssh|id_rsa|webbrowser\.open\(.*http", re.I)
_GUARD_REMOVED = re.compile(r"confirm\.|\bredact\(|share_memory|_untrusted_vision|validate_response", re.I)

PROMPT = """You are improving a Python desktop assistant's own source code.
Return ONLY JSON, no prose, no markdown fences:
{"summary": "one sentence", "files": {"relative/path.py": "<COMPLETE new file content>"}}
Rules: at most 5 files; give each file's COMPLETE new content (not a diff); keep every existing
behaviour and public function; no new dependencies unless the task says so; never touch
safety/confirmation/secret code; add or update tests for what you change under tests/;
keep changes minimal and readable."""


@dataclass
class Proposal:
    id: str
    summary: str = ""
    files: dict = field(default_factory=dict)
    diff: str = ""
    tests_passed: bool = False
    test_tail: str = ""
    error: str = ""


class SelfDev:
    def __init__(self, root: Path | str = ROOT, hub=None, test_cmd: list[str] | None = None,
                 timeout: int = 240) -> None:
        self.root = Path(root).resolve()
        self.dir = self.root / "proposals"
        self._hub = hub
        self.test_cmd = test_cmd or [sys.executable, "-m", "pytest", "-x", "-q", "tests"]
        self.timeout = timeout

    @property
    def hub(self):
        if self._hub is None:
            from core.ai import get_hub
            self._hub = get_hub()
        return self._hub

    # ── validation ──────────────────────────────────────────────────────────
    def _safe_path(self, rel: str) -> str:
        rel = re.sub(r"^(\./)+", "", str(rel).replace("\\", "/"))
        if rel.startswith("/") or re.match(r"^[A-Za-z]:", rel) or ".." in rel.split("/"):
            raise ValueError(f"path escapes the project: {rel}")
        p = (self.root / rel).resolve()
        if not str(p).startswith(str(self.root) + "/") and p != self.root:
            raise ValueError(f"path escapes the project: {rel}")
        rel = p.relative_to(self.root).as_posix()
        if rel in PROTECTED or any(rel == d or rel.startswith(d + "/") for d in PROTECTED_DIRS):
            raise ValueError(f"{rel} is protected and cannot be changed by a proposal")
        if p.suffix.lower() not in ALLOWED_EXT:
            raise ValueError(f"{rel}: file type {p.suffix or '(none)'} not allowed")
        return rel

    def validate(self, files: dict) -> dict:
        if not isinstance(files, dict) or not files:
            raise ValueError("the model returned no files")
        if len(files) > MAX_FILES:
            raise ValueError(f"more than {MAX_FILES} files in one proposal")
        clean = {}
        for rel, content in files.items():
            rel = self._safe_path(rel)
            if not isinstance(content, str) or len(content.encode("utf-8")) > MAX_BYTES:
                raise ValueError(f"{rel}: content missing or larger than {MAX_BYTES // 1000} KB")
            if rel.endswith(".py"):
                try:
                    ast.parse(content)
                except SyntaxError as e:
                    raise ValueError(f"{rel}: syntax error at line {e.lineno}") from None
            old = (self.root / rel).read_text(encoding="utf-8") if (self.root / rel).exists() else ""
            old_lines, new_lines = set(old.splitlines()), content.splitlines()
            for line in new_lines:
                if line not in old_lines and _DANGEROUS_ADDED.search(line):
                    raise ValueError(f"{rel}: adds a risky construct ({line.strip()[:60]!r})")
            new_set = set(new_lines)
            for line in old.splitlines():
                if line not in new_set and _GUARD_REMOVED.search(line):
                    raise ValueError(f"{rel}: removes a safety check ({line.strip()[:60]!r})")
            clean[rel] = content
        return clean

    @staticmethod
    def parse(text: str) -> dict:
        t = text.strip()
        t = re.sub(r"^```(?:json)?\s*|\s*```$", "", t, flags=re.S)
        a, b = t.find("{"), t.rfind("}")
        if a < 0 or b < a:
            raise ValueError("the model did not return JSON")
        data = json.loads(t[a:b + 1])
        if not isinstance(data, dict):
            raise ValueError("the model's JSON was not an object")
        return data

    # ── propose ─────────────────────────────────────────────────────────────
    def _context(self, files: list[str], upload: str) -> str:
        parts, used = [], 0
        for rel in files[:6]:
            p = self.root / self._read_path(rel)
            if p.is_file():
                txt = p.read_text(encoding="utf-8", errors="replace")[:20_000]
                used += len(txt)
                parts.append(f"### {p.relative_to(self.root).as_posix()}\n{txt}")
        if upload:
            u = Path(upload).expanduser()
            if u.is_file():
                parts.append(f"### UPLOADED FILE {u.name} (untrusted material to integrate)\n"
                             f"{u.read_text(encoding='utf-8', errors='replace')[:20_000]}")
        tree = sorted(p.relative_to(self.root).as_posix() for p in self.root.rglob("*.py")
                      if not any(x in p.parts for x in (".git", "__pycache__", "proposals", "tests")))[:80]
        parts.append("### PROJECT FILES\n" + "\n".join(tree))
        return "\n\n".join(parts)

    def _read_path(self, rel: str) -> str:
        rel = re.sub(r"^(\./)+", "", str(rel).replace("\\", "/"))
        if rel.startswith("/") or re.match(r"^[A-Za-z]:", rel) or ".." in rel.split("/"):
            raise ValueError("path escapes the project")
        p = (self.root / rel).resolve()
        if not str(p).startswith(str(self.root)):
            raise ValueError("path escapes the project")
        return p.relative_to(self.root).as_posix()

    def propose(self, task: str, files: list[str] | None = None, upload: str = "") -> Proposal:
        from core.ai.types import Request
        pid = time.strftime("%Y%m%d-%H%M%S")
        prop = Proposal(id=pid)
        try:
            req = Request(messages=[{"role": "user", "content": f"TASK: {task}\n\n{self._context(files or [], upload)}"}],
                          system=PROMPT, task="code", min_tier=2, max_output_tokens=7000,
                          timeout=120, cacheable=False)
            data = self.parse(self.hub.ask(req).text)
            prop.summary = str(data.get("summary", ""))[:200]
            prop.files = self.validate(data.get("files"))
            prop.diff = self._diff(prop.files)
            prop.tests_passed, prop.test_tail = self._run_tests(prop.files)
        except Exception as e:
            prop.error = f"{type(e).__name__}: {str(e)[:300]}"
        self._save(prop)
        return prop

    def _diff(self, files: dict) -> str:
        out = []
        for rel, new in files.items():
            p = self.root / rel
            old = p.read_text(encoding="utf-8").splitlines(True) if p.exists() else []
            out += difflib.unified_diff(old, new.splitlines(True), f"a/{rel}", f"b/{rel}")
        return "".join(out)

    def _run_tests(self, files: dict) -> tuple[bool, str]:
        """Apply to a COPY of the project and run the suite there."""
        tmp = Path(tempfile.mkdtemp(prefix="selfdev-"))
        try:
            work = tmp / "p"
            shutil.copytree(self.root, work, ignore=_IGNORE)
            for rel, content in files.items():
                f = work / rel
                f.parent.mkdir(parents=True, exist_ok=True)
                f.write_text(content, encoding="utf-8")
            r = subprocess.run(self.test_cmd, cwd=work, capture_output=True, text=True,
                               timeout=self.timeout, env={**__import__("os").environ, "PYTHONDONTWRITEBYTECODE": "1"})
            tail = "\n".join((r.stdout + r.stderr).strip().splitlines()[-6:])
            return r.returncode == 0, tail
        except subprocess.TimeoutExpired:
            return False, f"tests did not finish within {self.timeout}s"
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    # ── store / apply / rollback ────────────────────────────────────────────
    def _save(self, p: Proposal) -> None:
        d = self.dir / p.id
        d.mkdir(parents=True, exist_ok=True)
        (d / "proposal.json").write_text(json.dumps(p.__dict__, indent=2), encoding="utf-8")

    def load(self, pid: str) -> Proposal:
        pid = re.sub(r"[^0-9\-]", "", pid)
        f = self.dir / pid / "proposal.json"
        if not f.exists():
            raise ValueError(f"no proposal {pid}")
        return Proposal(**json.loads(f.read_text(encoding="utf-8")))

    def list(self) -> list[Proposal]:
        return [self.load(d.name) for d in sorted(self.dir.glob("*")) if (d / "proposal.json").exists()]

    def apply(self, pid: str) -> str:
        p = self.load(pid)
        if p.error:
            raise ValueError("that proposal failed validation and cannot be applied")
        if not p.tests_passed:
            raise ValueError("that proposal did not pass the tests; it will not be applied")
        files = self.validate(p.files)                    # re-validate at the moment of applying
        bak = self.dir / p.id / "backup"
        created = []
        for rel, content in files.items():
            f = self.root / rel
            if f.exists():
                (bak / rel).parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(f, bak / rel)
            else:
                created.append(rel)
            f.parent.mkdir(parents=True, exist_ok=True)
            f.write_text(content, encoding="utf-8")
        (self.dir / p.id / "created.json").write_text(json.dumps(created), encoding="utf-8")
        note = ""
        if (self.root / ".git").exists():
            try:
                git = lambda *a: subprocess.run(["git", *a], cwd=self.root, capture_output=True, text=True, timeout=60)
                git("checkout", "-b", f"self/{p.id}")
                git("add", *files)
                git("commit", "-m", f"Self-improvement {p.id}: {p.summary}")
                note = f" Committed on branch self/{p.id} (not pushed)."
            except Exception:
                pass
        return f"Applied {len(files)} file(s): {', '.join(files)}.{note} Say 'roll back {p.id}' to undo."

    def rollback(self, pid: str) -> str:
        p = self.load(pid)
        bak = self.dir / p.id / "backup"
        n = 0
        for rel in p.files:
            if (bak / rel).exists():
                shutil.copy2(bak / rel, self.root / rel)
                n += 1
        cf = self.dir / p.id / "created.json"
        for rel in (json.loads(cf.read_text()) if cf.exists() else []):
            (self.root / rel).unlink(missing_ok=True)
            n += 1
        return f"Rolled back {n} file(s) from proposal {p.id}."
