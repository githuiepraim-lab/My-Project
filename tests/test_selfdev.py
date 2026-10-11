import json
import sys
from pathlib import Path

import pytest
from core.selfdev import SelfDev


class FakeHub:
    def __init__(self, files, summary="improve"):
        self.payload = json.dumps({"summary": summary, "files": files})
        self.seen = None
    def ask(self, req):
        self.seen = req
        return type("R", (), {"text": self.payload})()


@pytest.fixture
def repo(tmp_path):
    (tmp_path / "core").mkdir(); (tmp_path / "tests").mkdir(); (tmp_path / "actions").mkdir()
    (tmp_path / "core" / "math_utils.py").write_text("def add(a, b):\n    return a + b\n")
    (tmp_path / "core" / "confirm.py").write_text("# protected\n")
    (tmp_path / "tests" / "test_math.py").write_text(
        "from core.math_utils import add\ndef test_add():\n    assert add(1, 2) == 3\n")
    (tmp_path / "conftest.py").write_text("import sys, pathlib\nsys.path.insert(0, str(pathlib.Path(__file__).parent))\n")
    return tmp_path


def sd(repo, files):
    return SelfDev(repo, hub=FakeHub(files), test_cmd=[sys.executable, "-m", "pytest", "-x", "-q", "tests"], timeout=60)


GOOD = {"core/math_utils.py": "def add(a, b):\n    return a + b\n\ndef mul(a, b):\n    return a * b\n",
        "tests/test_mul.py": "from core.math_utils import mul\ndef test_mul():\n    assert mul(2, 3) == 6\n"}


def test_good_proposal_is_tested_in_a_copy_and_real_files_untouched(repo):
    s = sd(repo, GOOD)
    p = s.propose("add multiplication")
    assert not p.error and p.tests_passed and "mul" in p.diff
    assert "mul" not in (repo / "core" / "math_utils.py").read_text()        # real tree untouched
    assert not (repo / "tests" / "test_mul.py").exists()


def test_failing_tests_block_apply(repo):
    bad = {"core/math_utils.py": "def add(a, b):\n    return a - b\n"}
    s = sd(repo, bad)
    p = s.propose("break it")
    assert not p.tests_passed
    with pytest.raises(ValueError, match="did not pass"):
        s.apply(p.id)
    assert "a + b" in (repo / "core" / "math_utils.py").read_text()


def test_apply_then_rollback_restores_everything(repo):
    s = sd(repo, GOOD)
    p = s.propose("add multiplication")
    msg = s.apply(p.id)
    assert "Applied 2" in msg and "def mul" in (repo / "core" / "math_utils.py").read_text()
    assert (repo / "tests" / "test_mul.py").exists()
    s.rollback(p.id)
    assert "mul" not in (repo / "core" / "math_utils.py").read_text()
    assert not (repo / "tests" / "test_mul.py").exists()


@pytest.mark.parametrize("files,why", [
    ({"core/confirm.py": "x = 1\n"}, "protected"),
    ({"main.py": "x = 1\n"}, "protected"),
    ({"../evil.py": "x = 1\n"}, "escapes|protected"),
    ({"config/api_keys.json": "{}"}, "protected"),
    ({"core/a.exe": "x"}, "not allowed"),
    ({"core/x.py": "def f(:\n"}, "syntax"),
    ({"core/x.py": "import os\nos.system('rm -rf /')\n"}, "risky"),
    ({"core/x.py": "import subprocess\nsubprocess.run('x', shell=True)\n"}, "risky"),
    ({"core/x.py": "k = open('config/api_keys.json').read()\n"}, "risky"),
    ({"core/x.py": "eval(input())\n"}, "risky"),
    ({f"core/f{i}.py": "x=1\n" for i in range(6)}, "more than"),
    ({"core/big.py": "x = 1\n" + "# " + "a" * 70_000}, "larger"),
])
def test_unsafe_proposals_are_rejected(repo, files, why):
    p = sd(repo, files).propose("anything")
    assert p.error and __import__("re").search(why, p.error), p.error
    with pytest.raises(ValueError):
        sd(repo, files).apply(p.id)


def test_removing_a_safety_check_is_rejected(repo):
    (repo / "core" / "act.py").write_text("from core import confirm\ndef go():\n    return confirm.request('k','t','d',lambda: 1)\n")
    p = sd(repo, {"core/act.py": "def go():\n    return 1\n"}).propose("simplify")
    assert "removes a safety check" in p.error


def test_model_output_formats_are_parsed(repo):
    raw = "Here you go:\n```json\n" + json.dumps({"summary": "s", "files": GOOD}) + "\n```"
    s = sd(repo, {})
    s.hub.payload = raw
    assert s.propose("x").tests_passed


def test_non_json_output_is_a_clean_error(repo):
    s = sd(repo, {})
    s.hub.payload = "I think you should refactor."
    assert "JSON" in s.propose("x").error


def test_uploaded_code_is_passed_as_untrusted_material(repo, tmp_path):
    up = tmp_path / "up.py"; up.write_text("def shout(s):\n    return s.upper()\n")
    s = sd(repo, {"actions/shout.py": "def shout(s):\n    return s.upper()\n"})
    s.propose("integrate this as an action", upload=str(up))
    body = s.hub.seen.messages[0]["content"]
    assert "UPLOADED FILE up.py" in body and "untrusted" in body and s.hub.seen.min_tier == 2


def test_apply_is_revalidated_so_a_tampered_proposal_fails(repo):
    s = sd(repo, GOOD)
    p = s.propose("add multiplication")
    f = repo / "proposals" / p.id / "proposal.json"
    d = json.loads(f.read_text()); d["files"]["core/confirm.py"] = "x = 1\n"; f.write_text(json.dumps(d))
    with pytest.raises(ValueError, match="protected"):
        s.apply(p.id)


def test_action_apply_goes_through_the_onscreen_gate(repo, monkeypatch):
    import actions.self_improve as si
    s = sd(repo, GOOD)
    monkeypatch.setattr(si, "SelfDev", lambda: s)
    p = s.propose("add multiplication")
    got = {}
    monkeypatch.setattr(si.confirm, "request", lambda k, t, d, run: got.update(t=t, d=d, run=run) or "waiting")
    assert si.self_improve({"mode": "apply", "id": p.id}) == "waiting"
    assert "def mul" not in (repo / "core" / "math_utils.py").read_text()   # nothing applied yet
    got["run"]()
    assert "def mul" in (repo / "core" / "math_utils.py").read_text()


def test_tool_registers_without_live_only_fields():
    from core.action_loader import discover_actions
    decls = {d["name"]: d for d in discover_actions(Path("actions"), logger=lambda s: None).get_tool_declarations()}
    assert "self_improve" in decls and "report_sender" in decls and set(decls["self_improve"]) <= {"name", "description", "parameters"}


@pytest.mark.parametrize("path", ["../evil.py", "core/../../evil.py", "/etc/passwd.py", "C:/x.py", ".github/workflows/x.py",
                                  "./config/x.json", "proposals/x.json"])
def test_path_tricks_are_rejected(repo, path):
    p = sd(repo, {path: "x = 1\n"}).propose("x")
    assert p.error and ("escapes" in p.error or "protected" in p.error), p.error
