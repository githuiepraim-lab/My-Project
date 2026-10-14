"""
Shared files: the place where anything the assistant produces for you — PDF
reports, screenshots, videos, zips — appears as a card you can view, download
or open in its folder, like an attachment in a chat.

Files are kept in  Documents/Ephraim Reports/Shared  (so they survive the
original being moved) and listed from a small JSON index there.
"""
from __future__ import annotations

import json
import shutil
import threading
import time
import zipfile
from pathlib import Path

from PyQt6.QtCore import QSize, Qt, QUrl, pyqtSignal
from PyQt6.QtGui import QDesktopServices, QFont, QPixmap
from PyQt6.QtWidgets import (QDialog, QFileDialog, QFrame, QHBoxLayout, QLabel, QListWidget,
                             QPlainTextEdit, QPushButton, QScrollArea, QVBoxLayout, QWidget)

SHARED_DIR = Path.home() / "Documents" / "Ephraim Reports" / "Shared"
MAX_COPY = 500 * 1024 * 1024
_IMG = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp"}
_VID = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"}
_TXT = {".txt", ".csv", ".log"}
_ICON = {"pdf": "📄", "image": "🖼", "video": "🎬", "zip": "🗜", "text": "📝", "web": "🌐", "code": "💻", "markdown": "📘", "file": "📎"}
_SENSITIVE_NAMES = ("secrets", "api_keys", ".env", "credentials", "id_rsa", "id_ed25519", "token", "password", ".pem")
_SENSITIVE_DIRS = (".ssh", ".aws", ".gnupg", "config")


def kind_of(path) -> str:
    s = Path(path).suffix.lower()
    from core.code_view import CODE_EXT
    return ("pdf" if s == ".pdf" else "image" if s in _IMG else "video" if s in _VID
            else "zip" if s == ".zip" else "web" if s in (".html", ".htm") else "markdown" if s in (".md", ".markdown")
            else "code" if s in CODE_EXT else "text" if s in _TXT else "file")


def is_sensitive(path) -> bool:
    """Refuse to put credentials into a place whose whole point is sharing."""
    p = Path(path)
    low = p.name.lower()
    return any(n in low for n in _SENSITIVE_NAMES) or any(part in _SENSITIVE_DIRS for part in p.parts[:-1])


def human_size(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024.0
    return f"{n} B"


class Registry:
    def __init__(self, folder: Path | None = None) -> None:
        self.dir = Path(folder) if folder else SHARED_DIR
        self._lock = threading.Lock()

    @property
    def index(self) -> Path:
        return self.dir / "shared.json"

    def list(self) -> list[dict]:
        try:
            rows = json.loads(self.index.read_text(encoding="utf-8"))
        except Exception:
            return []
        return [r for r in rows if Path(r.get("path", "")).is_file()]

    def add(self, path, note: str = "") -> dict:
        src = Path(path).expanduser()
        if not src.is_file():
            raise ValueError(f"not a file: {src.name}")
        if is_sensitive(src):
            raise ValueError(f"{src.name} looks like a credentials file and will not be shared")
        if src.stat().st_size > MAX_COPY:
            raise ValueError(f"{src.name} is over 500 MB")
        with self._lock:
            self.dir.mkdir(parents=True, exist_ok=True)
            dest = src
            if self.dir.resolve() not in src.resolve().parents:
                dest = self.dir / src.name
                i = 1
                while dest.exists() and dest.stat().st_size != src.stat().st_size:
                    dest = self.dir / f"{src.stem} ({i}){src.suffix}"
                    i += 1
                if not dest.exists():
                    shutil.copy2(src, dest)
            rows = self.list()
            row = next((r for r in rows if r["path"] == str(dest)), None)
            if row is None:
                row = {"path": str(dest), "name": dest.name, "kind": kind_of(dest),
                       "size": dest.stat().st_size, "time": time.time(), "note": note[:200]}
                rows.append(row)
                self.index.write_text(json.dumps(rows[-200:], indent=2), encoding="utf-8")
            return row

    def remove(self, path: str, delete_file: bool = False) -> None:
        with self._lock:
            rows = [r for r in self.list() if r["path"] != path]
            self.index.write_text(json.dumps(rows, indent=2), encoding="utf-8")
            if delete_file:
                Path(path).unlink(missing_ok=True)


# ── UI ──────────────────────────────────────────────────────────────────────
_QSS = """
QWidget { background:#020d14; color:#9fe9f5; font-family:'Courier New'; }
QLabel { background:transparent; }
QFrame#card { background:#04202c; border:1px solid #0b5b73; border-radius:6px; }
QPushButton { background:#00091a; color:#00d4ff; border:1px solid #0b5b73; border-radius:3px; padding:4px 9px; }
QPushButton:hover { background:#062a38; border-color:#00d4ff; }
QLabel#meta { color:#5a8f9c; font-size:10px; }
QLabel#empty { color:#5a8f9c; padding:24px; }
"""


class PreviewDialog(QDialog):
    """View a file without leaving the app. Anything that cannot be previewed
    still offers 'Open' in the system's own viewer."""

    def __init__(self, entry: dict, parent=None) -> None:
        super().__init__(parent)
        self.entry = entry
        self.setWindowTitle(entry["name"])
        self.setStyleSheet(_QSS)
        self.resize(820, 620)
        lay = QVBoxLayout(self)
        self.view = None
        try:
            self.view = self._build(entry)
        except Exception as e:
            self.view = QLabel(f"Preview is not available here ({type(e).__name__}). Use Open.")
        lay.addWidget(self.view, 1)
        row = QHBoxLayout()
        b = QPushButton("Open in default app")
        b.clicked.connect(lambda: QDesktopServices.openUrl(QUrl.fromLocalFile(entry["path"])))
        c = QPushButton("Close"); c.clicked.connect(self.accept)
        row.addStretch(1); row.addWidget(b); row.addWidget(c)
        lay.addLayout(row)

    def done(self, r) -> None:
        player = getattr(self, "_player", None)
        if player is not None:
            player.stop()
        super().done(r)

    def _build(self, e: dict) -> QWidget:
        k, path = e["kind"], e["path"]
        if k == "pdf":
            from PyQt6.QtPdf import QPdfDocument
            from PyQt6.QtPdfWidgets import QPdfView
            self._doc = QPdfDocument(self)
            self._doc.load(path)
            v = QPdfView(self)
            v.setDocument(self._doc)
            v.setPageMode(QPdfView.PageMode.MultiPage)
            v.setZoomMode(QPdfView.ZoomMode.FitToWidth)       # a whole page across, not a zoomed corner
            return v
        if k == "image":
            sc = QScrollArea(); lbl = QLabel()
            pm = QPixmap(path)
            lbl.setPixmap(pm.scaledToWidth(min(pm.width(), 780), Qt.TransformationMode.SmoothTransformation))
            sc.setWidget(lbl)
            return sc
        if k == "video":
            from PyQt6.QtMultimedia import QAudioOutput, QMediaPlayer
            from PyQt6.QtMultimediaWidgets import QVideoWidget
            vw = QVideoWidget(self)
            self._player = QMediaPlayer(self)
            self._audio = QAudioOutput(self)
            self._player.setAudioOutput(self._audio)
            self._player.setVideoOutput(vw)
            self._player.setSource(QUrl.fromLocalFile(path))
            self._player.play()
            return vw
        if k == "zip":
            lw = QListWidget()
            with zipfile.ZipFile(path) as z:
                for i in z.infolist()[:500]:
                    lw.addItem(f"{i.filename}    ({human_size(i.file_size)})")
            return lw
        if k in ("web", "code", "markdown"):
            from core.code_view import CodePreview
            return CodePreview(path)                    # highlighted code / rendered website / markdown
        if k == "text":
            t = QPlainTextEdit(); t.setReadOnly(True)
            t.setPlainText(Path(path).read_text(encoding="utf-8", errors="replace")[:200_000])
            return t
        return QLabel("No preview for this file type. Use Open or Download.")


class FileCard(QFrame):
    def __init__(self, entry: dict, panel: "FilesPanel") -> None:
        super().__init__()
        self.setObjectName("card")
        self.entry = entry
        lay = QVBoxLayout(self)
        top = QHBoxLayout()
        icon = QLabel(_ICON.get(entry["kind"], "📎")); icon.setFont(QFont("Segoe UI Emoji", 20))
        top.addWidget(icon)
        col = QVBoxLayout()
        name = QLabel(entry["name"]); name.setWordWrap(True); name.setFont(QFont("Courier New", 10, QFont.Weight.Bold))
        meta = QLabel(f"{entry['kind'].upper()} · {human_size(entry['size'])} · "
                      f"{time.strftime('%d %b %H:%M', time.localtime(entry['time']))}")
        meta.setObjectName("meta")
        col.addWidget(name); col.addWidget(meta)
        if entry.get("note"):
            n = QLabel(entry["note"]); n.setObjectName("meta"); n.setWordWrap(True); col.addWidget(n)
        top.addLayout(col, 1)
        lay.addLayout(top)
        row = QHBoxLayout()
        for text, fn in (("View", lambda: panel.view(entry)), ("Download", lambda: panel.download(entry)),
                         ("Folder", lambda: panel.reveal(entry)), ("✕", lambda: panel.remove(entry, self))):
            b = QPushButton(text); b.clicked.connect(fn); row.addWidget(b)
        lay.addLayout(row)


class FilesPanel(QWidget):
    status = pyqtSignal(str)
    countChanged = pyqtSignal(int)

    def __init__(self, registry: Registry | None = None, parent=None) -> None:
        super().__init__(parent)
        self.registry = registry or Registry()
        self.setStyleSheet(_QSS)
        self.setMinimumWidth(340)
        lay = QVBoxLayout(self)
        head = QLabel("📎  SHARED FILES"); head.setFont(QFont("Courier New", 11, QFont.Weight.Bold))
        lay.addWidget(head)
        self._empty = QLabel("Nothing yet. Reports, videos and zips the assistant makes for you will appear here.")
        self._empty.setObjectName("empty"); self._empty.setWordWrap(True)
        lay.addWidget(self._empty)
        self._holder = QWidget(); self._cards = QVBoxLayout(self._holder); self._cards.addStretch(1)
        sc = QScrollArea(); sc.setWidgetResizable(True); sc.setWidget(self._holder)
        lay.addWidget(sc, 1)
        self._n = 0
        for row in self.registry.list():
            self._add_card(row)

    def count(self) -> int:
        return self._n

    def _add_card(self, row: dict) -> None:
        self._cards.insertWidget(0, FileCard(row, self))
        self._n += 1
        self._empty.setVisible(False)
        self.countChanged.emit(self._n)

    def add(self, path, note: str = "") -> dict:
        row = self.registry.add(path, note)
        if not any(c.entry["path"] == row["path"] for c in self._card_widgets()):
            self._add_card(row)
        return row

    def _card_widgets(self) -> list[FileCard]:
        return [self._cards.itemAt(i).widget() for i in range(self._cards.count())
                if isinstance(self._cards.itemAt(i).widget(), FileCard)]

    def view(self, entry: dict) -> None:
        PreviewDialog(entry, self).exec()

    def download(self, entry: dict) -> str | None:
        """Save a copy wherever you choose (starts in Downloads)."""
        start = Path.home() / "Downloads"
        start = (start if start.exists() else Path.home()) / entry["name"]
        dest, _ = QFileDialog.getSaveFileName(self, "Save a copy", str(start))
        if not dest:
            return None
        shutil.copy2(entry["path"], dest)
        self.status.emit(f"SYS: Saved {entry['name']} to {dest}")
        return dest

    def reveal(self, entry: dict) -> None:
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(Path(entry["path"]).parent)))

    def remove(self, entry: dict, card: FileCard) -> None:
        self.registry.remove(entry["path"])          # the file itself is kept; only the card goes
        card.setParent(None); card.deleteLater()
        self._n -= 1
        self._empty.setVisible(self._n == 0)
        self.countChanged.emit(self._n)
