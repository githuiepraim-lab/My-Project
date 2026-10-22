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
MAX_FILES, MAX_BYTES = 5, 60_000        # MAX_BYTES: what the model may SEND (edits + new files); finished files may be bigger
MAX_FILE_BYTES = 2_000_000
_SKIP_DIRS = {".git", "__pycache__", "logs", "proposals", ".pytest_cache", "node_modules", "venv", ".venv", ".mypy_cache"}
_SKIP_FILES = ("*.pyc", "secrets.json", "api_keys.json", "long_term.json", "jobs.db*", "providers.json*", "worker.lock",
               "pending.json", "coding_progress.json", "integrations.json", "*.mp4", "*.zip", ".env")
MAX_COPY_FILE = 8 * 1024 * 1024


def _ignore(directory, names):
    """What NOT to copy into the test sandbox. Deliberately NOT .obj/.png/.ico: the old version left those
    out, so the avatar tests could not find the face model and EVERY proposal failed its tests."""
    import fnmatch
    skip = set()
    for n in names:
        full = Path(directory) / n
        if n in _SKIP_DIRS or any(fnmatch.fnmatch(n, pat) for pat in _SKIP_FILES):
            skip.add(n)
        elif full.is_file() and full.stat().st_size > MAX_COPY_FILE:
            skip.add(n)
    return skip


_DANGEROUS_ADDED = re.compile(
    r"shell\s*=\s*True|os\.system\(|\beval\(|\bexec\(|__import__\(|pickle\.loads|marshal\.loads|"
    r"secrets\.json|api_keys\.json|get_api_key\(|keyring|\.ssh|id_rsa|webbrowser\.open\(.*http", re.I)
_GUARD_REMOVED = re.compile(r"confirm\.|\bredact\(|share_memory|_untrusted_vision|validate_response", re.I)

PROMPT = """You are improving a Python desktop assistant's own source code.
Return ONLY JSON, no prose, no markdown fences, in this shape:
{"summary": "one sentence",
 "edits": [{"path": "core/x.py", "find": "exact existing text, copied verbatim, that occurs ONCE", "replace": "its replacement"}],
 "files": {"path/new_file.py": "<COMPLETE content of a NEW file>"}}
Rules: use "edits" to change EXISTING files (small, exact, unique `find` snippets of a few lines; several edits per file
are fine) and "files" ONLY for brand-new files. At most 5 files touched. Keep every existing behaviour and public function.
No new dependencies unless the task says so. Never touch safety / confirmation / secret handling. Add or update a test
under tests/ for what you change. Keep it minimal and readable."""


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
                 timeout: int = 300) -> None:
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
            if not isinstance(content, str) or len(content.encode("utf-8")) > MAX_FILE_BYTES:
                raise ValueError(f"{rel}: content missing or larger than {MAX_FILE_BYTES // 1_000_000} MB")
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
        body = t[a:b + 1]
        for candidate in (body, re.sub(r",\s*([}\]])", r"\1", body)):          # trailing commas are common
            try:
                data = json.loads(candidate, strict=False)                         # strict=False: raw newlines inside strings
                if isinstance(data, dict):
                    return data
            except ValueError:
                continue
        raise ValueError("the model's JSON could not be read (it may have been cut off: ask for a smaller change)")

    def build_files(self, data: dict) -> dict:
        """Turn the model's answer (edits to existing files + new files) into complete file contents."""
        out: dict[str, str] = {}
        sent = sum(len(str(e.get("find", ""))) + len(str(e.get("replace", ""))) for e in (data.get("edits") or []) if isinstance(e, dict)) \
            + sum(len(str(c)) for c in (data.get("files") or {}).values())
        if sent > MAX_BYTES:
            raise ValueError(f"the change is larger than {MAX_BYTES // 1000} KB; ask for a smaller one")
        for ed in data.get("edits") or []:
            if not isinstance(ed, dict):
                raise ValueError("an edit was not an object")
            rel = self._safe_path(ed.get("path", ""))
            cur = out.get(rel)
            if cur is None:
                f = self.root / rel
                if not f.is_file():
                    raise ValueError(f"{rel} does not exist; use \"files\" to create a new file")
                cur = f.read_text(encoding="utf-8")
            find, rep = ed.get("find"), ed.get("replace")
            if not isinstance(find, str) or not find or not isinstance(rep, str):
                raise ValueError(f"{rel}: each edit needs non-empty 'find' and a 'replace' string")
            n = cur.count(find)
            if n != 1:
                raise ValueError(f"{rel}: the 'find' text {'was not found' if n == 0 else f'occurs {n} times (must be unique)'}: {find[:60]!r}")
            out[rel] = cur.replace(find, rep, 1)
        for rel, content in (data.get("files") or {}).items():
            if not isinstance(content, str) or len(content.encode("utf-8")) > MAX_BYTES:
                raise ValueError(f"{rel}: content missing or larger than {MAX_BYTES // 1000} KB")
            out[self._safe_path(rel)] = content
        return self.validate(out)

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

    def auto_files(self, task: str, limit: int = 3) -> list[str]:
        """No files named? Pick the project files that mention the task's keywords."""
        words = {w.lower() for w in re.findall(r"[A-Za-z_]{4,}", task)} - {"that", "this", "with", "from", "make", "should", "would", "when", "have", "into"}
        scored = []
        for f in self.root.rglob("*.py"):
            rel = f.relative_to(self.root)
            if any(p in ("tests", "proposals", "__pycache__", ".git") for p in rel.parts) or f.stat().st_size > 120_000:
                continue
            txt = (str(rel) + " " + f.read_text(encoding="utf-8", errors="ignore")).lower()
            sc = sum(txt.count(w) for w in words) + 5 * sum(w in str(rel).lower() for w in words)
            if sc:
                scored.append((sc, rel.as_posix()))
        return [r for _s, r in sorted(scored, reverse=True)[:limit]]

    def propose(self, task: str, files: list[str] | None = None, upload: str = "", progress=lambda m: None) -> Proposal:
        from core.ai.types import Request
        pid = time.strftime("%Y%m%d-%H%M%S")
        prop = Proposal(id=pid)
        try:
            ctx = self._context(files or self.auto_files(task), upload)
            feedback = ""
            for attempt in range(2):                                   # a second try, told exactly what was wrong
                progress("asking the AI for a change" if attempt == 0 else "asking again with the problem pointed out")
                ask = f"TASK: {task}\n\n{ctx}" + (f"\n\nYour previous answer was rejected: {feedback}\nFix exactly that and answer again." if feedback else "")
                req = Request(messages=[{"role": "user", "content": ask}], system=PROMPT, task="code", min_tier=2,
                              max_output_tokens=9000, timeout=180, cacheable=False)
                try:
                    data = self.parse(self.hub.ask(req).text)
                    prop.summary = str(data.get("summary", ""))[:200]
                    prop.files = self.build_files(data)
                    break
                except ValueError as e:
                    feedback = str(e)
                    if attempt == 1:
                        raise
            prop.diff = self._diff(prop.files)
            progress("testing the change in a copy")
            prop.tests_passed, prop.test_tail = self._run_tests(prop.files)
        except Exception as e:
            prop.error = f"{type(e).__name__}: {str(e)[:300]}"
        self._save(prop)
        if prop.diff:
            (self.dir / prop.id / "change.diff").write_text(prop.diff, encoding="utf-8")
        return prop

    def _diff(self, files: dict) -> str:
        out = []
        for rel, new in files.items():
            p = self.root / rel
            old = p.read_text(encoding="utf-8").splitlines(True) if p.exists() else []
            out += difflib.unified_diff(old, new.splitlines(True), f"a/{rel}", f"b/{rel}")
        return "".join(out)

    def select_tests(self, work: Path, changed: list[str]) -> list[str]:
        """The test files that exercise what changed (plus any test the proposal added). A docs-only
        change needs none. Running all ~450 tests took minutes and failed for reasons unrelated to the change."""
        mods = {Path(c).stem for c in changed if c.endswith(".py") and not c.startswith("tests/")}
        sel = {c for c in changed if c.startswith("tests/") and c.endswith(".py")}
        for t in (work / "tests").glob("test_*.py") if (work / "tests").exists() else []:
            txt = t.read_text(encoding="utf-8", errors="ignore")
            if any(re.search(rf"\b{re.escape(m)}\b", txt) for m in mods):
                sel.add(f"tests/{t.name}")
        return sorted(sel)

    @staticmethod
    def _failed(output: str) -> set[str]:
        return set(re.findall(r"^FAILED (\S+)", output, re.M)) | set(re.findall(r"^ERROR (\S+)", output, re.M))

    def _pytest(self, work: Path, tests: list[str]) -> tuple[bool, set[str], str]:
        cmd = [c for c in self.test_cmd if c != "tests"] if self.test_cmd[-1] == "tests" else list(self.test_cmd)
        cmd = cmd + ["-p", "no:cacheprovider", "-rfE", "--maxfail=15"] + tests
        r = subprocess.run(cmd, cwd=work, capture_output=True, text=True, timeout=self.timeout,
                           env={**__import__("os").environ, "PYTHONDONTWRITEBYTECODE": "1", "QT_QPA_PLATFORM": "offscreen"})
        out = r.stdout + r.stderr
        return r.returncode == 0, self._failed(out), "\n".join(out.strip().splitlines()[-6:])

    def _run_tests(self, files: dict) -> tuple[bool, str]:
        """Apply the change to a COPY and run only the relevant tests. Passes when the change introduces
        no NEW failures and its own tests pass; failures that were already there are reported, not blamed."""
        tmp = Path(tempfile.mkdtemp(prefix="selfdev-"))
        try:
            base = tmp / "base"
            shutil.copytree(self.root, base, ignore=_ignore)
            work = tmp / "p"
            shutil.copytree(base, work)
            for rel, content in files.items():
                f = work / rel
                f.parent.mkdir(parents=True, exist_ok=True)
                f.write_text(content, encoding="utf-8")
            # 1. everything must at least compile
            for rel in files:
                if rel.endswith(".py"):
                    r = subprocess.run([sys.executable, "-m", "py_compile", str(work / rel)], capture_output=True, text=True)
                    if r.returncode:
                        return False, f"{rel} does not compile: {(r.stderr.strip().splitlines() or [''])[-1][:160]}"
            sel = self.select_tests(work, list(files))
            if not sel:
                return True, "compiles; no tests cover these files (documentation or new module only)"
            ok, failed, tail = self._pytest(work, sel)
            if ok:
                return True, f"{len(sel)} test file(s) passed: " + ", ".join(Path(t).name for t in sel[:4])
            _, failed0, _ = self._pytest(base, [t for t in sel if (base / t).exists()]) if any((base / t).exists() for t in sel) else (True, set(), "")
            new = failed - failed0
            if not new and failed:
                return True, f"no NEW failures ({len(failed0)} test(s) were already failing before this change)"
            return False, "new failure(s): " + ", ".join(sorted(new)[:4]) + "\n" + tail
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
