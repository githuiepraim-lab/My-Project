import json
import os
import subprocess
import time
import types
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import pytest

pytest.importorskip("PyQt6")
from PyQt6.QtWidgets import QApplication

app = QApplication.instance() or QApplication([])

from core import code_view as cv
from core.code_view import CodePreview, CodeViewer

PY = 'import math\n# note\ndef area(r):\n    return math.pi * r ** 2\nprint("hi", 42)\n'
HTML = "<html><head><style>h1{color:#036}</style></head><body><h1>My first website</h1><p>Built by the assistant.</p></body></html>"


def wait(cond, secs=3.0):
    t0 = time.time()
    while time.time() - t0 < secs:
        app.processEvents()
        if cond():
            return True
        time.sleep(0.02)
    return False


def colours(prev, line):
    blk = prev.code.document().findBlockByLineNumber(line)
    return {(f.start, f.length): f.format.foreground().color().name() for f in blk.layout().formats()}


def test_python_is_highlighted_with_line_numbers_and_never_editable(tmp_path):
    f = tmp_path / "a.py"; f.write_text(PY)
    p = CodePreview(str(f)); p.hl.rehighlight()
    assert p.lang == "python" and p.code.isReadOnly() and p.code.gutter_width() > 14 and "5 lines" in p.info.text() or "6 lines" in p.info.text()
    kw = colours(p, 2)                                   # "def area(r):"
    assert any(c == "#c792ea" for c in kw.values())      # the keyword
    com = colours(p, 1); assert "#6c7a89" in com.values() # the comment
    assert "#c3e88d" in colours(p, 4).values()            # the string
    assert p.mode() == "code"


@pytest.mark.parametrize("name,text,lang", [("x.js", "const a = 1; // hi\nfunction f() { return 'x' }", "js"), ("x.css", "h1 { color: red; }", "css"),
                                            ("x.json", '{"a": [1, true]}', "json"), ("x.sql", "SELECT * FROM t WHERE a = 1 -- c", "sql"),
                                            ("x.sh", "if [ -f a ]; then echo hi; fi # c", "shell"), ("x.c", "int main() { return 0; }", "c")])
def test_other_languages_are_recognised_and_coloured(tmp_path, name, text, lang):
    f = tmp_path / name; f.write_text(text)
    p = CodePreview(str(f)); p.hl.rehighlight()
    assert p.lang == lang and len(colours(p, 0)) > 0


def test_html_has_a_website_mode_and_a_code_mode(tmp_path):
    f = tmp_path / "index.html"; f.write_text(HTML)
    p = CodePreview(str(f))
    assert set(p.modes) == {"code", "web"} and p.mode() == "web"                      # opens as the website
    assert "My first website" in p.web.toPlainText() or cv.QWebEngineView is not None
    p.set_mode("code"); assert p.mode() == "code" and "<h1>" in p.code.toPlainText()
    p.set_mode("web"); assert p.mode() == "web"


def test_markdown_is_rendered_and_source_is_available(tmp_path):
    f = tmp_path / "r.md"; f.write_text("# Title\n\nSome **bold** text.\n\n- one\n- two\n")
    p = CodePreview(str(f))
    assert p.mode() == "md" and "Title" in p.md.toPlainText() and "**" not in p.md.toPlainText()
    p.set_mode("code"); assert "# Title" in p.code.toPlainText()


def test_diff_colours_additions_and_removals():
    p = CodePreview(text="--- a/x.py\n+++ b/x.py\n@@ -1 +1 @@\n-old\n+new\n context\n", title="proposal", lang="diff"); p.hl.rehighlight()
    assert "#66ff99" in colours(p, 4).values() and "#ff6b6b" in colours(p, 3).values() and "#00d4ff" in colours(p, 2).values()
    assert not p.modes or list(p.modes) == ["code"]


def test_the_preview_follows_the_file_while_it_is_being_written(tmp_path):
    f = tmp_path / "live.py"; f.write_text("x = 1\n")
    p = CodePreview(str(f))
    assert p.code.toPlainText() == "x = 1\n"
    f.write_text("x = 1\ny = 2\nprint(x + y)\n")
    assert wait(lambda: "print(x + y)" in p.code.toPlainText()), "preview did not refresh"
    f.write_text("x = 1\ny = 2\nprint(x + y)\nz = 3\n")                                  # a second change keeps being tracked
    assert wait(lambda: "z = 3" in p.code.toPlainText())


def test_missing_and_huge_files_degrade_cleanly(tmp_path):
    p = CodePreview(str(tmp_path / "gone.py"))
    assert "cannot read" in p.code.toPlainText()
    big = tmp_path / "big.py"; big.write_text("x=1\n" * 300_000)
    assert "not previewed" in CodePreview(str(big)).code.toPlainText()


def test_viewer_reuses_tabs_and_replaces_text_tabs(tmp_path):
    a, b = tmp_path / "a.py", tmp_path / "b.html"; a.write_text(PY); b.write_text(HTML)
    v = CodeViewer()
    v.open_file(str(a)); v.open_file(str(b)); v.open_file(str(a))
    assert v.tabs.count() == 2 and v.open_file(str(tmp_path / "nope.py")) is None
    v.open_text("proposal 1", "+a\n", "diff"); v.open_text("proposal 1", "+b\n", "diff")
    assert v.tabs.count() == 3 and "+b" in v.tabs.widget(2).code.toPlainText()


def test_files_panel_opens_code_and_websites_in_the_preview(tmp_path):
    from core import files_panel as fp
    assert [fp.kind_of(n) for n in ("a.py", "b.html", "c.md", "d.txt", "e.json")] == ["code", "web", "markdown", "text", "code"]
    f = tmp_path / "s.html"; f.write_text(HTML)
    reg = fp.Registry(tmp_path / "Shared")
    d = fp.PreviewDialog(reg.add(f))
    assert isinstance(d.view, CodePreview) and d.view.lang == "html"


# ── nothing opens in VS Code any more ──────────────────────────────────────────────────────
@pytest.fixture
def no_external(monkeypatch):
    def boom(*a, **k): raise AssertionError("an external program was launched")
    monkeypatch.setattr(subprocess, "Popen", boom)
    monkeypatch.setattr(subprocess, "run", boom)


class Player:
    def __init__(self): self.shown, self.texts = [], []
    def show_code(self, paths): self.shown.append(list(paths))
    def show_code_text(self, title, text, lang="text"): self.texts.append((title, text, lang))


def test_dev_agent_shows_the_project_in_the_app_not_in_vscode(tmp_path, no_external):
    import actions.dev_agent as da
    (tmp_path / "main.py").write_text("print(1)"); (tmp_path / "util.py").write_text("x=1"); (tmp_path / "index.html").write_text(HTML)
    (tmp_path / "node_modules").mkdir(); (tmp_path / "node_modules" / "junk.js").write_text("x")
    (tmp_path / "logo.png").write_bytes(b"x")
    pl = Player()
    assert da._show_project(tmp_path, pl)
    names = [Path(p).name for p in pl.shown[0]]
    assert names[0] in ("main.py", "index.html") and "junk.js" not in names and "logo.png" not in names and len(names) == 3


def test_the_external_editor_is_opt_in_and_has_no_shell(tmp_path, monkeypatch):
    import actions.dev_agent as da
    import memory.config_manager as cm
    called = []
    monkeypatch.setattr(da.subprocess, "Popen", lambda cmd, **k: called.append((cmd, k)))
    monkeypatch.setattr(da.shutil, "which", lambda n: "/usr/bin/code")
    monkeypatch.setattr(cm, "load_api_keys", lambda: {})
    assert da._open_vscode(tmp_path) is False and called == []                     # default: never
    monkeypatch.setattr(cm, "load_api_keys", lambda: {"code_external_editor": True})
    evil = tmp_path / "x & del"; evil.mkdir()
    assert da._open_vscode(evil) and called[0][0] == ["/usr/bin/code", str(evil)] and "shell" not in called[0][1]


def test_code_helper_previews_what_it_saves(tmp_path, no_external):
    import actions.code_helper as ch
    pl = Player(); ch._player_ref = pl
    assert "Saved to" in ch._save_file(tmp_path / "out" / "tool.py", "print(1)\n")
    assert pl.shown == [[str(tmp_path / "out" / "tool.py")]]
    ch._player_ref = None
    assert "Saved to" in ch._save_file(tmp_path / "again.py", "x=1")                 # no UI: still just saves


def test_lessons_open_in_the_app_not_an_editor(tmp_path, no_external, monkeypatch):
    import actions.coding_coach as ac
    from core.coding_coach import Coach
    c = Coach(tmp_path / "code", tmp_path / "p.json"); monkeypatch.setattr(ac, "Coach", lambda: c)
    pl = Player()
    out = ac.coding_coach({"mode": "lesson", "lesson": "1"}, pl)
    assert "code preview" in out and pl.shown and pl.shown[0][0].endswith("variables.py")
    assert "Lesson 1/11" in ac.coding_coach({"mode": "lesson", "lesson": "1"}, None)       # no UI: still works


def test_self_improvement_diffs_are_reviewed_in_the_app(tmp_path, monkeypatch):
    import actions.self_improve as si
    from core.selfdev import Proposal
    sd = types.SimpleNamespace(load=lambda pid: Proposal(id="20261011-1", summary="s", files={"a.py": "x"}, diff="+new\n", tests_passed=True),
                               propose=lambda *a, **k: Proposal(id="20261011-1", summary="s", files={"a.py": "x"}, diff="+new\n", tests_passed=True))
    monkeypatch.setattr(si, "SelfDev", lambda: sd)
    pl = Player()
    assert "Showing proposal" in si.self_improve({"mode": "show", "id": "20261011-1"}, pl)
    assert pl.texts[0][2] == "diff"
    si.self_improve({"mode": "propose", "task": "x"}, pl)
    assert len(pl.texts) == 2


def test_show_code_tool(tmp_path):
    import actions.show_code as sc
    f = tmp_path / "a.py"; f.write_text(PY)
    pl = Player()
    assert "Showing 1 file" in sc.show_code({"file": str(f)}, pl) and pl.shown == [[str(f)]]
    assert "code preview" in sc.show_code({"code": "<h1>x</h1>", "language": "html", "title": "t"}, pl) and pl.texts[0][2] == "html"
    assert "cannot find" in sc.show_code({"file": "/nope.py"}, pl) and "not available" in sc.show_code({"file": str(f)}, None)
    from core.action_loader import discover_actions
    d = {x["name"]: x for x in discover_actions(Path("actions"), logger=lambda s: None).get_tool_declarations()}
    assert set(d["show_code"]) <= {"name", "description", "parameters"}


def test_ui_wiring(tmp_path):
    ui = pytest.importorskip("ui")
    got = []
    j = types.SimpleNamespace(_win=types.SimpleNamespace(_code_sig=types.SimpleNamespace(emit=got.append)))
    ui.JarvisUI.show_code(j, [tmp_path / "a.py"]); ui.JarvisUI.show_code_text(j, "d", "+x", "diff")
    assert json.loads(got[0]) == {"paths": [str(tmp_path / "a.py")]} and json.loads(got[1])["lang"] == "diff"
    f = tmp_path / "a.py"; f.write_text(PY)
    fake = types.SimpleNamespace(_log=types.SimpleNamespace(append_log=lambda m: None))
    ui.MainWindow._on_show_code(fake, json.dumps({"paths": [str(f)]}))
    ui.MainWindow._on_show_code(fake, json.dumps({"title": "d", "text": "+x\n", "lang": "diff"}))
    assert fake._code_viewer.tabs.count() == 2
