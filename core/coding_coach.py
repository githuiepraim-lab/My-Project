"""
Coding coach: a short Python course with exercises that are CHECKED, plus help
with your own code.

How a lesson works: you get an explanation and a small example, a starter file
is created in Documents/Ephraim Code (opened in your editor, never
overwritten), you write the function, then 'check my code' runs it against
tests and tells you exactly which cases fail. Hints come in three steps, from a
nudge to nearly the answer. Progress is remembered.

Your code runs in a separate, isolated Python process in an empty temporary
folder with a time limit (and memory/CPU limits on Linux and macOS). That limits
accidents; it is not a security sandbox, so only run code you trust.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

CODE_DIR = Path.home() / "Documents" / "Ephraim Code"
PROGRESS = Path(__file__).resolve().parent.parent / "memory" / "coding_progress.json"


@dataclass(frozen=True)
class Lesson:
    id: str
    title: str
    explain: str
    example: str
    task: str
    func: str
    starter: str
    tests: tuple
    hints: tuple
    solution: str


def L(*a):
    return Lesson(*a)


LESSONS: tuple[Lesson, ...] = (
    L("variables", "Variables and arithmetic",
      "A variable is a name for a value: width = 3. Python does the maths with + - * / and ** (power). "
      "A function packages steps under a name; `return` hands back the answer.",
      "def double(x):\n    result = x * 2\n    return result\n\nprint(double(21))   # 42",
      "Write area(width, height) that returns the area of a rectangle.", "area",
      "def area(width, height):\n    # your code here\n    pass\n",
      (((3, 4), 12), ((2.5, 2), 5.0), ((0, 9), 0)),
      ("Area is width times height.", "Use the * operator.", "return width * height"),
      "def area(width, height):\n    return width * height\n"),
    L("strings", "Strings",
      "Text is a string: 'hello'. f-strings put values inside text: f'Hi {name}'. Handy methods: .strip() removes "
      "spaces at the ends, .title() capitalises Each Word, .upper(), .lower().",
      "name = '  ada '\nprint(f\"Hello, {name.strip().title()}!\")   # Hello, Ada!",
      "Write greet(name) returning 'Hello, Name!' with the name stripped of spaces and title-cased.", "greet",
      "def greet(name):\n    pass\n",
      ((("  ada ",), "Hello, Ada!"), (("grace hopper",), "Hello, Grace Hopper!"), (("LINUS",), "Hello, Linus!")),
      ("Clean the name first: strip, then title.", "Build the answer with an f-string.", "return f\"Hello, {name.strip().title()}!\""),
      "def greet(name):\n    return f\"Hello, {name.strip().title()}!\"\n"),
    L("numbers", "Numbers and lists",
      "A list holds many values: [3, 5, 8]. len(x) counts them, sum(x) adds them, round(x, 2) rounds. "
      "Dividing by zero is an error, so guard against empty input.",
      "scores = [3, 5, 8]\nprint(sum(scores) / len(scores))   # 5.333...",
      "Write average(numbers): the mean rounded to 2 decimals; 0 for an empty list.", "average",
      "def average(numbers):\n    pass\n",
      ((([2, 4, 6],), 4.0), (([1, 2],), 1.5), (([],), 0), (([1, 1, 2],), 1.33)),
      ("Check for an empty list first and return 0.", "mean = sum / count.", "return round(sum(numbers) / len(numbers), 2)"),
      "def average(numbers):\n    if not numbers:\n        return 0\n    return round(sum(numbers) / len(numbers), 2)\n"),
    L("conditions", "Decisions: if / elif / else",
      "if runs code only when something is true. elif adds another check, else catches the rest. "
      "Compare with == != < <= > >=. Python checks top to bottom and stops at the first true branch.",
      "n = 7\nif n % 2 == 0:\n    print('even')\nelse:\n    print('odd')",
      "Write grade(score): 'A' for 90+, 'B' for 80+, 'C' for 70+, otherwise 'F'.", "grade",
      "def grade(score):\n    pass\n",
      (((95,), "A"), ((90,), "A"), ((85,), "B"), ((70,), "C"), ((12,), "F")),
      ("Start with the highest grade: if score >= 90.", "Use elif for 80 and 70, else for F.", "if score >= 90: return 'A' ... elif ... else: return 'F'"),
      "def grade(score):\n    if score >= 90:\n        return 'A'\n    elif score >= 80:\n        return 'B'\n    elif score >= 70:\n        return 'C'\n    return 'F'\n"),
    L("loops", "Loops",
      "for repeats code for each item: for i in range(1, 4) gives 1, 2, 3 (the end is excluded). "
      "Keep a running total in a variable you update inside the loop.",
      "total = 0\nfor i in range(1, 4):\n    total += i\nprint(total)   # 6",
      "Write sum_evens(n): the sum of the even numbers from 1 to n inclusive.", "sum_evens",
      "def sum_evens(n):\n    pass\n",
      (((10,), 30), ((1,), 0), ((6,), 12), ((0,), 0)),
      ("Start total = 0 and loop with range(1, n + 1).", "i % 2 == 0 tells you i is even.", "total += i inside an if i % 2 == 0"),
      "def sum_evens(n):\n    total = 0\n    for i in range(1, n + 1):\n        if i % 2 == 0:\n            total += i\n    return total\n"),
    L("lists", "Working with lists",
      "Lists are ordered and you can loop over them, index them (items[0] is the first) and compare items. "
      "max(items, key=len) finds the longest, but a loop works too.",
      "words = ['hi', 'python']\nprint(words[0], len(words[1]))",
      "Write longest_word(words): the longest word (the first one on a tie), or '' for an empty list.", "longest_word",
      "def longest_word(words):\n    pass\n",
      ((([],), ""), ((["a", "abc", "ab"],), "abc"), ((["to", "be", "or"],), "to"), ((["hello"],), "hello")),
      ("Keep a variable best = '' and update it in a loop.", "Only replace best if len(word) > len(best).", "for w in words: if len(w) > len(best): best = w"),
      "def longest_word(words):\n    best = ''\n    for w in words:\n        if len(w) > len(best):\n            best = w\n    return best\n"),
    L("dicts", "Dictionaries",
      "A dictionary maps keys to values: {'a': 1}. d.get(key, 0) reads a value with a default. "
      "Great for counting: d[key] = d.get(key, 0) + 1.",
      "counts = {}\nfor ch in 'hello':\n    counts[ch] = counts.get(ch, 0) + 1\nprint(counts)",
      "Write count_words(text): a dict of how many times each lower-cased word appears (split on spaces).", "count_words",
      "def count_words(text):\n    pass\n",
      ((("a b a",), {"a": 2, "b": 1}), (("Hello hello",), {"hello": 2}), (("",), {})),
      ("Make text.lower().split() first.", "Loop over the words and use counts.get(word, 0) + 1.", "counts[w] = counts.get(w, 0) + 1"),
      "def count_words(text):\n    counts = {}\n    for w in text.lower().split():\n        counts[w] = counts.get(w, 0) + 1\n    return counts\n"),
    L("functions", "Writing good functions",
      "A function should do one job, take inputs as parameters and return a result. Break a problem into small steps: "
      "clean the input, then test it. str.isalnum() tells you if a character is a letter or digit.",
      "def is_even(n):\n    return n % 2 == 0",
      "Write is_palindrome(text): True if it reads the same backwards, ignoring case, spaces and punctuation.", "is_palindrome",
      "def is_palindrome(text):\n    pass\n",
      ((("Racecar",), True), (("A man, a plan, a canal: Panama",), True), (("hello",), False), (("",), True)),
      ("Build a cleaned string with only letters and digits, lower-case.", "Compare it with its reverse: cleaned[::-1].", "return cleaned == cleaned[::-1]"),
      "def is_palindrome(text):\n    cleaned = ''.join(c.lower() for c in text if c.isalnum())\n    return cleaned == cleaned[::-1]\n"),
    L("errors", "Handling errors",
      "Errors (exceptions) stop a program unless you catch them: try: ... except ZeroDivisionError: ... "
      "Catch the specific error you expect, not everything.",
      "try:\n    print(10 / 0)\nexcept ZeroDivisionError:\n    print('cannot divide by zero')",
      "Write safe_divide(a, b): a / b, or None if b is zero.", "safe_divide",
      "def safe_divide(a, b):\n    pass\n",
      (((10, 2), 5.0), ((1, 0), None), ((0, 5), 0.0), ((-9, 3), -3.0)),
      ("Wrap the division in try.", "except ZeroDivisionError and return None.", "try: return a / b  except ZeroDivisionError: return None"),
      "def safe_divide(a, b):\n    try:\n        return a / b\n    except ZeroDivisionError:\n        return None\n"),
    L("fizzbuzz", "Putting it together: FizzBuzz",
      "A classic interview warm-up that combines loops, conditions and lists: for each number 1..n use 'Fizz' if it is "
      "divisible by 3, 'Buzz' if by 5, 'FizzBuzz' if by both, otherwise the number as text. Check the 'both' case first!",
      "results = []\nfor i in range(1, 4):\n    results.append(str(i))\nprint(results)",
      "Write fizzbuzz(n) returning the list of strings for 1..n.", "fizzbuzz",
      "def fizzbuzz(n):\n    pass\n",
      (((5,), ["1", "2", "Fizz", "4", "Buzz"]), ((15,), ["1", "2", "Fizz", "4", "Buzz", "Fizz", "7", "8", "Fizz", "Buzz", "11", "Fizz", "13", "14", "FizzBuzz"]), ((0,), [])),
      ("Test divisibility by 15 (both) before 3 and 5.", "append str(i) in the else branch.", "if i % 15 == 0: 'FizzBuzz' elif i % 3 == 0: 'Fizz' elif i % 5 == 0: 'Buzz'"),
      "def fizzbuzz(n):\n    out = []\n    for i in range(1, n + 1):\n        if i % 15 == 0:\n            out.append('FizzBuzz')\n        elif i % 3 == 0:\n            out.append('Fizz')\n        elif i % 5 == 0:\n            out.append('Buzz')\n        else:\n            out.append(str(i))\n    return out\n"),
    L("recursion", "Algorithms: Fibonacci",
      "The Fibonacci numbers start 0, 1, 1, 2, 3, 5, 8: each is the sum of the two before. You can compute them with a "
      "loop that remembers the last two values, or with recursion (a function calling itself), which is slow for big n.",
      "a, b = 0, 1\nfor _ in range(5):\n    a, b = b, a + b\nprint(a)   # 5",
      "Write fibonacci(n): the n-th Fibonacci number, with fibonacci(0) = 0 and fibonacci(1) = 1.", "fibonacci",
      "def fibonacci(n):\n    pass\n",
      (((0,), 0), ((1,), 1), ((10,), 55), ((30,), 832040)),
      ("Keep two variables a, b = 0, 1.", "Repeat n times: a, b = b, a + b.", "return a after the loop"),
      "def fibonacci(n):\n    a, b = 0, 1\n    for _ in range(n):\n        a, b = b, a + b\n    return a\n"),
)
BY_ID = {l.id: l for l in LESSONS}

_HARNESS = r'''
import importlib.util, json, sys
path, spec_path = sys.argv[1], sys.argv[2]
spec = json.load(open(spec_path))
out = {"load_error": None, "cases": []}
try:
    s = importlib.util.spec_from_file_location("learner", path)
    mod = importlib.util.module_from_spec(s)
    s.loader.exec_module(mod)
    fn = getattr(mod, spec["func"])
except BaseException as e:
    out["load_error"] = f"{type(e).__name__}: {e}"
    print("@@RESULT@@" + json.dumps(out)); sys.exit(0)
for args, expected in spec["tests"]:
    try:
        got = fn(*args)
        out["cases"].append({"args": args, "expected": expected, "got": repr(got), "ok": got == expected})
    except BaseException as e:
        out["cases"].append({"args": args, "expected": expected, "got": f"{type(e).__name__}: {e}", "ok": False})
print("@@RESULT@@" + json.dumps(out))
'''


def _limits():                                   # POSIX only: CPU seconds and memory
    try:
        import resource
        resource.setrlimit(resource.RLIMIT_CPU, (8, 8))
        resource.setrlimit(resource.RLIMIT_AS, (512 * 1024 * 1024, 512 * 1024 * 1024))
        resource.setrlimit(resource.RLIMIT_FSIZE, (2 * 1024 * 1024, 2 * 1024 * 1024))
    except Exception:
        pass


def _isolated(cmd: list[str], timeout: float, cwd: str) -> subprocess.CompletedProcess:
    kw = {"preexec_fn": _limits} if os.name == "posix" else {}
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout, env={
        "PYTHONIOENCODING": "utf-8", "PYTHONDONTWRITEBYTECODE": "1", "PATH": os.environ.get("PATH", "")}, **kw)


class Coach:
    def __init__(self, code_dir: Path | None = None, progress: Path | None = None) -> None:
        self.dir = Path(code_dir) if code_dir else CODE_DIR
        self.progress_file = Path(progress) if progress else PROGRESS

    # ── progress ────────────────────────────────────────────────────────────
    def _load(self) -> dict:
        try:
            d = json.loads(self.progress_file.read_text(encoding="utf-8"))
            return d if isinstance(d, dict) else {}
        except Exception:
            return {}

    def _save(self, d: dict) -> None:
        self.progress_file.parent.mkdir(parents=True, exist_ok=True)
        self.progress_file.write_text(json.dumps(d, indent=2), encoding="utf-8")

    def next_lesson(self) -> Lesson | None:
        done = self._load().get("completed", {})
        return next((l for l in LESSONS if l.id not in done), None)

    def current(self) -> Lesson:
        d = self._load()
        return BY_ID.get(d.get("current", ""), None) or self.next_lesson() or LESSONS[-1]

    def path_for(self, lesson: Lesson) -> Path:
        return self.dir / f"{lesson.id}.py"

    def find(self, key: str) -> Lesson | None:
        k = (key or "").strip().lower()
        if not k:
            return None
        if k.isdigit() and 1 <= int(k) <= len(LESSONS):
            return LESSONS[int(k) - 1]
        return BY_ID.get(k) or next((l for l in LESSONS if k in l.title.lower() or k in l.id), None)

    # ── lessons ─────────────────────────────────────────────────────────────
    def start(self, key: str = "") -> tuple[Lesson, Path, bool]:
        les = self.find(key) or self.next_lesson()
        if les is None:
            les = LESSONS[-1]
        f = self.path_for(les)
        self.dir.mkdir(parents=True, exist_ok=True)
        created = not f.exists()
        if created:                                             # never overwrite your work
            f.write_text(f'"""{les.title}\n{les.task}\n"""\n\n{les.starter}', encoding="utf-8")
        d = self._load()
        d["current"] = les.id
        d.setdefault("hint_level", {})[les.id] = 0
        self._save(d)
        return les, f, created

    def describe(self, les: Lesson, path: Path | None = None) -> str:
        n = LESSONS.index(les) + 1
        return (f"Lesson {n}/{len(LESSONS)} — {les.title}\n\n{les.explain}\n\nExample:\n{les.example}\n\n"
                f"Your task: {les.task}\n" + (f"Your file: {path}\n" if path else "") +
                "When you are ready, say 'check my code'. Say 'hint' if you are stuck.")

    def hint(self) -> str:
        les = self.current()
        d = self._load()
        lv = d.setdefault("hint_level", {}).get(les.id, 0)
        d["hint_level"][les.id] = min(lv + 1, len(les.hints))
        self._save(d)
        return f"Hint {min(lv + 1, len(les.hints))}/{len(les.hints)}: {les.hints[min(lv, len(les.hints) - 1)]}"

    # ── running code ────────────────────────────────────────────────────────
    def _run_tests(self, path: Path, les: Lesson, timeout: float = 8.0) -> dict:
        with tempfile.TemporaryDirectory(prefix="coach-") as tmp:
            spec = Path(tmp) / "spec.json"
            spec.write_text(json.dumps({"func": les.func, "tests": [[list(a), e] for a, e in les.tests]}), encoding="utf-8")
            (Path(tmp) / "harness.py").write_text(_HARNESS, encoding="utf-8")
            work = Path(tmp) / "work"
            work.mkdir()
            shutil.copy2(path, work / "learner.py")
            try:
                r = _isolated([sys.executable, "-I", str(Path(tmp) / "harness.py"), str(work / "learner.py"), str(spec)],
                              timeout, str(work))
            except subprocess.TimeoutExpired:
                return {"load_error": f"Your code ran for more than {timeout:.0f} seconds — is there an endless loop?", "cases": []}
            marker = "@@RESULT@@"
            if marker in r.stdout:
                return json.loads(r.stdout.split(marker, 1)[1].strip().splitlines()[0])
            return {"load_error": (r.stderr.strip().splitlines() or ["the program crashed"])[-1][:200], "cases": []}

    def check(self, file: str = "") -> str:
        les = self.current()
        path = Path(file).expanduser() if file else self.path_for(les)
        if not path.is_file():
            return f"I cannot find {path}. Say 'start lesson {LESSONS.index(les) + 1}' and I will create it."
        res = self._run_tests(path, les)
        d = self._load()
        d.setdefault("attempts", {})[les.id] = d.get("attempts", {}).get(les.id, 0) + 1
        if res["load_error"]:
            self._save(d)
            return f"Your file has an error before any test could run: {res['load_error']}"
        bad = [c for c in res["cases"] if not c["ok"]]
        if not bad:
            d.setdefault("completed", {})[les.id] = time.strftime("%Y-%m-%d")
            self._save(d)
            nxt = self.next_lesson()
            tail = (f" Next up: lesson {LESSONS.index(nxt) + 1}, {nxt.title}. Say 'next lesson'."
                    if nxt else " You have finished the whole course — well done!")
            return f"All {len(res['cases'])} tests pass. Lesson '{les.title}' complete.{tail}"
        self._save(d)
        c = bad[0]
        args = ", ".join(repr(a) for a in c["args"])
        return (f"{len(res['cases']) - len(bad)}/{len(res['cases'])} tests pass. First failure: "
                f"{les.func}({args}) should give {c['expected']!r} but gave {c['got']}. Say 'hint' for a nudge.")

    def run(self, path: str, timeout: float = 10.0) -> str:
        p = Path(path).expanduser()
        if not p.is_file() or p.suffix != ".py":
            raise ValueError("give me the path of a .py file")
        with tempfile.TemporaryDirectory(prefix="coach-run-") as tmp:
            shutil.copy2(p, Path(tmp) / "main.py")
            try:
                r = _isolated([sys.executable, "-I", "main.py"], timeout, tmp)
            except subprocess.TimeoutExpired:
                return f"Stopped after {timeout:.0f} seconds (endless loop?)."
        out = (r.stdout + (("\n" + r.stderr) if r.stderr else "")).strip()
        return (out[-1500:] or "(no output)") + ("" if r.returncode == 0 else f"\n[exit code {r.returncode}]")

    def progress(self) -> str:
        d = self._load()
        done = d.get("completed", {})
        nxt = self.next_lesson()
        return (f"{len(done)}/{len(LESSONS)} lessons complete."
                + (f" Next: lesson {LESSONS.index(nxt) + 1}, {nxt.title}." if nxt else " Course finished!"))


def open_in_editor(path: Path) -> bool:
    """Optional external editor (config "code_external_editor": true); the app normally shows
    code in its own preview. Never opens in a way that would RUN the file (on Windows the
    default action for .py is to execute it)."""
    try:
        from memory.config_manager import load_api_keys
        if not load_api_keys().get("code_external_editor"):
            return False
    except Exception:
        return False
    try:
        if (code := shutil.which("code")):
            subprocess.Popen([code, str(path)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        elif os.name == "nt":
            subprocess.Popen(["notepad.exe", str(path)])
        elif sys.platform == "darwin":
            subprocess.Popen(["open", "-t", str(path)])
        else:
            subprocess.Popen(["xdg-open", str(path)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except Exception:
        return False
