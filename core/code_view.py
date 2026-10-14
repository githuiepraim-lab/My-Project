"""
In-app code preview. Nothing here launches VS Code or any external editor.

    Code      syntax-highlighted, line numbers, read-only (copy / save a copy)
    Website   HTML rendered right in the window (full browser engine if PyQt6-WebEngine is
              installed, otherwise Qt's built-in basic HTML view) + "Open in browser"
    Rendered  Markdown as formatted text
    Diff      colour +/- changes (what a self-improvement proposal would change)

A viewer watches the file, so while the assistant builds or fixes it the preview updates
itself — a live view of its work.
"""
from __future__ import annotations

import html as _html
import re
from pathlib import Path

from PyQt6.QtCore import QFileSystemWatcher, QRect, QSize, Qt, QTimer, QUrl
from PyQt6.QtGui import (QColor, QDesktopServices, QFont, QPainter, QSyntaxHighlighter, QTextCharFormat)
from PyQt6.QtWidgets import (QApplication, QFileDialog, QHBoxLayout, QLabel, QMainWindow, QPlainTextEdit,
                             QPushButton, QStackedWidget, QTabWidget, QTextBrowser, QVBoxLayout, QWidget)

try:                                                    # optional: a full browser engine
    from PyQt6.QtWebEngineWidgets import QWebEngineView
except Exception:                                       # pragma: no cover - depends on the install
    QWebEngineView = None

MAX_BYTES = 800_000
LANG = {".py": "python", ".pyw": "python", ".js": "js", ".mjs": "js", ".jsx": "js", ".ts": "js", ".tsx": "js",
        ".html": "html", ".htm": "html", ".xml": "html", ".svg": "html", ".css": "css", ".scss": "css",
        ".json": "json", ".sh": "shell", ".bash": "shell", ".bat": "shell", ".ps1": "shell",
        ".c": "c", ".h": "c", ".cpp": "c", ".hpp": "c", ".cs": "c", ".java": "c", ".go": "c", ".rs": "c",
        ".kt": "c", ".swift": "c", ".php": "c", ".sql": "sql", ".yml": "yaml", ".yaml": "yaml", ".toml": "yaml",
        ".md": "markdown", ".markdown": "markdown", ".diff": "diff", ".patch": "diff", ".ino": "c", ".lua": "python"}
CODE_EXT = set(LANG)

_KW = {
    "python": "False None True and as assert async await break class continue def del elif else except finally for from "
              "global if import in is lambda nonlocal not or pass raise return try while with yield self print",
    "js": "break case catch class const continue debugger default delete do else export extends false finally for "
          "function if import in instanceof let new null of return super switch this throw true try typeof var void "
          "while with yield async await undefined interface type enum implements public private readonly",
    "c": "abstract auto bool break case catch char class const continue default delete do double else enum extern "
         "false final finally float for func fn goto if impl import int interface let long mod mut namespace new null "
         "nil override package private protected public pub return short signed sizeof static struct super switch "
         "template this throw true try typedef union unsigned use using var virtual void volatile while",
    "shell": "if then else elif fi for while do done case esac function in echo export set cd exit return",
    "sql": "SELECT FROM WHERE INSERT INTO VALUES UPDATE SET DELETE CREATE TABLE DROP ALTER JOIN LEFT RIGHT INNER OUTER "
           "ON GROUP BY ORDER LIMIT AND OR NOT NULL AS DISTINCT HAVING UNION PRIMARY KEY select from where insert into "
           "values update set delete create table join on group by order limit and or not null as",
    "yaml": "true false null yes no", "css": "", "json": "true false null", "html": "",
}
_C = {"kw": "#c792ea", "str": "#c3e88d", "com": "#6c7a89", "num": "#f78c6c", "fn": "#82aaff", "tag": "#ff6b8b",
      "attr": "#ffcb6b", "head": "#82aaff", "add": "#66ff99", "del": "#ff6b6b", "hunk": "#00d4ff"}


def _fmt(color: str, bold=False, italic=False) -> QTextCharFormat:
    f = QTextCharFormat(); f.setForeground(QColor(color))
    if bold: f.setFontWeight(700)
    if italic: f.setFontItalic(True)
    return f


class Highlighter(QSyntaxHighlighter):
    def __init__(self, doc, lang: str) -> None:
        super().__init__(doc)
        self.lang = lang
        kw = _KW.get(lang, "")
        self.rules: list[tuple[re.Pattern, QTextCharFormat]] = []
        add = lambda pat, c, **k: self.rules.append((re.compile(pat), _fmt(_C[c], **k)))
        if lang == "diff":
            return
        if lang == "markdown":
            add(r"^#{1,6} .*$", "head", bold=True); add(r"`[^`]+`", "str"); add(r"\*\*[^*]+\*\*", "attr", bold=True)
            add(r"^\s*[-*+] ", "kw", bold=True); add(r"\[[^\]]+\]\([^)]+\)", "fn")
            return
        if kw:
            add(r"\b(" + "|".join(kw.split()) + r")\b", "kw", bold=True)
        add(r"\b\d+(\.\d+)?\b", "num")
        if lang in ("python", "js", "c"):
            add(r"\b([A-Za-z_]\w*)(?=\()", "fn")
        if lang == "html":
            add(r"</?[A-Za-z][\w:-]*", "tag", bold=True); add(r"/?>", "tag", bold=True); add(r"\b[\w:-]+(?==)", "attr")
        if lang == "css":
            add(r"[.#]?[A-Za-z][\w-]*(?=\s*[{,])", "tag"); add(r"[\w-]+(?=\s*:)", "attr")
        if lang == "json":
            add(r'"[^"\\]*"(?=\s*:)', "attr")
        add(r"\"([^\"\\]|\\.)*\"|'([^'\\]|\\.)*'|`([^`\\]|\\.)*`", "str")
        com = {"python": r"#.*$", "shell": r"#.*$", "yaml": r"#.*$", "sql": r"--.*$", "html": r"<!--.*?-->",
               "css": r"/\*.*?\*/"}.get(lang, r"//.*$")
        add(com, "com", italic=True)

    def highlightBlock(self, text: str) -> None:
        if self.lang == "diff":
            c = ("add" if text.startswith("+") and not text.startswith("+++") else
                 "del" if text.startswith("-") and not text.startswith("---") else
                 "hunk" if text.startswith("@@") else "head" if text.startswith(("---", "+++", "diff")) else None)
            if c:
                self.setFormat(0, len(text), _fmt(_C[c]))
            return
        for pat, f in self.rules:
            for m in pat.finditer(text):
                self.setFormat(m.start(), m.end() - m.start(), f)


class _Gutter(QWidget):
    def __init__(self, ed) -> None:
        super().__init__(ed); self.ed = ed
    def sizeHint(self): return QSize(self.ed.gutter_width(), 0)
    def paintEvent(self, e): self.ed.paint_gutter(e)


class CodeEdit(QPlainTextEdit):
    """Read-only code view with line numbers."""
    def __init__(self) -> None:
        super().__init__()
        self.setReadOnly(True); self.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        f = QFont("Consolas"); f.setStyleHint(QFont.StyleHint.Monospace); f.setPointSize(10); self.setFont(f)
        self.setStyleSheet("QPlainTextEdit { background:#0b1520; color:#d6deeb; border:1px solid #0b3a4a; }")
        self.setTabStopDistance(self.fontMetrics().horizontalAdvance(" ") * 4)
        self._gutter = _Gutter(self)
        self.blockCountChanged.connect(lambda _: self.setViewportMargins(self.gutter_width(), 0, 0, 0))
        self.updateRequest.connect(lambda r, dy: self._gutter.scroll(0, dy) if dy else self._gutter.update(0, r.y(), self._gutter.width(), r.height()))
        self.setViewportMargins(self.gutter_width(), 0, 0, 0)

    def gutter_width(self) -> int:
        return 14 + self.fontMetrics().horizontalAdvance("9") * len(str(max(1, self.blockCount())))

    def resizeEvent(self, e):
        super().resizeEvent(e)
        cr = self.contentsRect()
        self._gutter.setGeometry(QRect(cr.left(), cr.top(), self.gutter_width(), cr.height()))

    def paint_gutter(self, ev) -> None:
        p = QPainter(self._gutter); p.fillRect(ev.rect(), QColor("#08111b")); p.setPen(QColor("#4a6a7a"))
        b = self.firstVisibleBlock(); n = b.blockNumber()
        top = round(self.blockBoundingGeometry(b).translated(self.contentOffset()).top()); bot = top + round(self.blockBoundingRect(b).height())
        while b.isValid() and top <= ev.rect().bottom():
            if b.isVisible() and bot >= ev.rect().top():
                p.drawText(0, top, self._gutter.width() - 6, self.fontMetrics().height(), Qt.AlignmentFlag.AlignRight, str(n + 1))
            b = b.next(); top = bot; bot = top + round(self.blockBoundingRect(b).height()); n += 1


class CodePreview(QWidget):
    """One file (or one block of text) with the right modes for its type."""
    def __init__(self, path: str | None = None, text: str | None = None, title: str = "", lang: str | None = None,
                 parent=None) -> None:
        super().__init__(parent)
        self.path = Path(path) if path else None
        self.title = title or (self.path.name if self.path else "code")
        self.lang = lang or (LANG.get(self.path.suffix.lower(), "text") if self.path else "text")
        self._static = text
        self.setStyleSheet("QWidget { background:#020d14; color:#9fe9f5; font-family:'Courier New'; }"
                           "QPushButton { background:#00091a; color:#00d4ff; border:1px solid #0b5b73; border-radius:3px; padding:4px 10px; }"
                           "QPushButton:checked { background:#0b5b73; color:#fff; } QPushButton:hover { border-color:#00d4ff; }"
                           "QLabel { background:transparent; color:#5a8f9c; }")
        lay = QVBoxLayout(self); lay.setContentsMargins(0, 0, 0, 0)
        bar = QHBoxLayout()
        self.modes: dict[str, QPushButton] = {}
        order = [("code", "Code")] + ([("web", "Website")] if self.lang == "html" else []) + \
                ([("md", "Rendered")] if self.lang == "markdown" else [])
        for key, label in order:
            b = QPushButton(label); b.setCheckable(True); b.clicked.connect(lambda _=False, k=key: self.set_mode(k))
            bar.addWidget(b); self.modes[key] = b
        self.info = QLabel(""); bar.addWidget(self.info, 1)
        for text_, fn in (("Open in browser", self._browser), ("Copy", self._copy), ("Save a copy", self._save)):
            if text_ == "Open in browser" and self.lang != "html":
                continue
            b = QPushButton(text_); b.clicked.connect(fn); bar.addWidget(b)
        lay.addLayout(bar)
        self.stack = QStackedWidget(); lay.addWidget(self.stack, 1)
        self.code = CodeEdit(); self.hl = Highlighter(self.code.document(), self.lang); self.stack.addWidget(self.code)
        self.web = None
        if self.lang == "html":
            self.web = QWebEngineView() if QWebEngineView is not None else QTextBrowser()
            if QWebEngineView is None:
                self.web.setStyleSheet("QTextBrowser { background:#ffffff; color:#111111; font-family:sans-serif; }")
            self.stack.addWidget(self.web)
        self.md = None
        if self.lang == "markdown":
            self.md = QTextBrowser(); self.md.setOpenExternalLinks(True); self.stack.addWidget(self.md)
        self.note = QLabel("" if (self.lang != "html" or QWebEngineView is not None) else
                           "Basic preview (no scripts). For a full browser view use “Open in browser”, "
                           "or  pip install PyQt6-WebEngine.")
        lay.addWidget(self.note)
        self._timer = QTimer(self); self._timer.setSingleShot(True); self._timer.timeout.connect(self.reload)
        self._watch = None
        if self.path:
            self._watch = QFileSystemWatcher([str(self.path)], self)
            self._watch.fileChanged.connect(self._changed)
        self.reload()
        self.set_mode("web" if self.lang == "html" else "md" if self.lang == "markdown" else "code")

    # ── content ─────────────────────────────────────────────────────────────
    def source(self) -> str:
        if self._static is not None:
            return self._static
        try:
            if self.path.stat().st_size > MAX_BYTES:
                return f"(file is larger than {MAX_BYTES // 1000} KB — not previewed. Use Open or Download.)"
            return self.path.read_text(encoding="utf-8", errors="replace")
        except OSError as e:
            return f"(cannot read {self.path.name}: {e.strerror})"

    def reload(self) -> None:
        src = self.source()
        if src != self.code.toPlainText():
            bar = self.code.verticalScrollBar(); keep, at_end = bar.value(), bar.value() >= bar.maximum() - 2
            self.code.setPlainText(src)
            bar.setValue(bar.maximum() if at_end else keep)         # follow the end while it is being written
        if self.web is not None:
            base = QUrl.fromLocalFile(str(self.path.parent) + "/") if self.path else QUrl()
            (self.web.setHtml(src, base) if QWebEngineView is not None and isinstance(self.web, QWebEngineView)
             else self.web.setHtml(src))
        if self.md is not None:
            self.md.setMarkdown(src)
        self.info.setText(f"{self.title} · {self.lang} · {src.count(chr(10)) + 1} lines")

    def _changed(self, _p) -> None:
        if self.path and str(self.path) not in self._watch.files() and self.path.exists():
            self._watch.addPath(str(self.path))                      # editors that replace the file drop the watch
        self._timer.start(250)                                       # debounce a burst of writes

    def set_mode(self, mode: str) -> None:
        idx = {"code": 0, "web": 1 if self.web is not None else 0, "md": 1 if self.md is not None else 0}.get(mode, 0)
        self.stack.setCurrentIndex(idx)
        for k, b in self.modes.items():
            b.setChecked(k == mode)

    def mode(self) -> str:
        i = self.stack.currentIndex()
        return "code" if i == 0 else ("web" if self.web is not None else "md")

    # ── actions ─────────────────────────────────────────────────────────────
    def _browser(self) -> None:
        if self.path:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.path)))

    def _copy(self) -> None:
        QApplication.clipboard().setText(self.source())

    def _save(self) -> None:
        dest, _ = QFileDialog.getSaveFileName(self, "Save a copy", str(Path.home() / self.title))
        if dest:
            Path(dest).write_text(self.source(), encoding="utf-8")


class CodeViewer(QMainWindow):
    """A window of tabs, one per file; showing a file again re-uses its tab."""
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("CODE PREVIEW"); self.resize(980, 660)
        self.tabs = QTabWidget(); self.tabs.setTabsClosable(True); self.tabs.setDocumentMode(True)
        self.tabs.tabCloseRequested.connect(lambda i: self.tabs.removeTab(i))
        self.setCentralWidget(self.tabs)
        self.setStyleSheet("QMainWindow, QTabWidget { background:#020d14; } QTabBar::tab { background:#04202c; color:#9fe9f5; padding:6px 12px; }"
                           "QTabBar::tab:selected { background:#0b5b73; color:#fff; }")

    def _find(self, key: str) -> int:
        return next((i for i in range(self.tabs.count()) if self.tabs.widget(i).property("key") == key), -1)

    def open_file(self, path: str) -> CodePreview | None:
        p = Path(path)
        if not p.is_file():
            return None
        i = self._find(str(p.resolve()))
        if i < 0:
            w = CodePreview(str(p)); w.setProperty("key", str(p.resolve()))
            i = self.tabs.addTab(w, p.name)
        self.tabs.setCurrentIndex(i); self.tabs.widget(i).reload()
        self._front()
        return self.tabs.widget(i)

    def open_text(self, title: str, text: str, lang: str = "diff") -> CodePreview:
        i = self._find("text:" + title)
        if i >= 0:
            self.tabs.removeTab(i)
        w = CodePreview(text=text, title=title, lang=lang); w.setProperty("key", "text:" + title)
        self.tabs.setCurrentIndex(self.tabs.addTab(w, title)); self._front()
        return w

    def _front(self) -> None:
        self.show(); self.raise_(); self.activateWindow()
