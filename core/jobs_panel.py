"""The JOBS panel: type a task and when, see everything queued, running and done."""
from __future__ import annotations

import time
from datetime import datetime

from PyQt6.QtCore import QTimer, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QFont
from PyQt6.QtWidgets import (QAbstractItemView, QDialog, QHBoxLayout, QHeaderView, QLabel, QLineEdit,
                             QPlainTextEdit, QPushButton, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget)

from core.jobs import JobStore, Notifier, add_from_text, announce, describe_when, parse_when

_QSS = """
QWidget { background:#020d14; color:#9fe9f5; font-family:'Courier New'; font-size:11px; }
QLabel { background:transparent; }
QLabel#hint { color:#5a8f9c; }
QLabel#when { color:#00d4ff; }
QLineEdit { background:#00091a; border:1px solid #0b5b73; padding:6px; color:#c8f6ff; }
QTableWidget { background:#00091a; gridline-color:#0b3a4a; border:1px solid #0b3a4a; }
QHeaderView::section { background:#04202c; color:#5fd4e6; border:0; padding:4px; }
QPushButton { background:#00091a; color:#00d4ff; border:1px solid #0b5b73; border-radius:3px; padding:5px 10px; }
QPushButton:hover { background:#062a38; border-color:#00d4ff; }
"""
_STATUS = {"queued": ("⏳ queued", "#ffcc66"), "running": ("▶ running", "#00d4ff"), "done": ("✅ done", "#66ff99"),
           "failed": ("❌ failed", "#ff6b6b"), "cancelled": ("⛔ cancelled", "#8899aa")}


class ResultDialog(QDialog):
    def __init__(self, job: dict, open_file, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle(f"Job #{job['id']}")
        self.setStyleSheet(_QSS)
        self.resize(620, 460)
        lay = QVBoxLayout(self)
        t = QPlainTextEdit(); t.setReadOnly(True)
        t.setPlainText(job.get("result") or job.get("error") or f"This job is {job['status']}.")
        lay.addWidget(t)
        row = QHBoxLayout(); row.addStretch(1)
        if job.get("file"):
            b = QPushButton("Open file in FILES"); b.clicked.connect(lambda: (open_file(job["file"], f"Job #{job['id']}"), self.accept())); row.addWidget(b)
        c = QPushButton("Close"); c.clicked.connect(self.accept); row.addWidget(c)
        lay.addLayout(row)


class JobsPanel(QWidget):
    status = pyqtSignal(str)
    countChanged = pyqtSignal(int)

    def __init__(self, store: JobStore | None = None, notifier: Notifier | None = None, open_file=None, parent=None) -> None:
        super().__init__(parent)
        self.store = store or JobStore()
        self.notifier = notifier or Notifier()
        self.open_file = open_file or (lambda p, n="": None)
        self.setStyleSheet(_QSS)
        self.setMinimumWidth(560)
        lay = QVBoxLayout(self)
        head = QLabel("📋  JOB QUEUE"); head.setFont(QFont("Courier New", 11, QFont.Weight.Bold))
        lay.addWidget(head)
        self.input = QLineEdit()
        self.input.setPlaceholderText('Task + when, e.g.  "write my Python study plan every weekday 7am"')
        self.input.textChanged.connect(self._preview)
        self.input.returnPressed.connect(self.add)
        lay.addWidget(self.input)
        self.when = QLabel(""); self.when.setObjectName("when")
        lay.addWidget(self.when)
        row = QHBoxLayout()
        for text, fn in (("Add job", self.add), ("Run now", self.run_now), ("Cancel", self.cancel),
                         ("View result", self.view), ("Clear finished", self.clear)):
            b = QPushButton(text); b.clicked.connect(fn); row.addWidget(b)
        lay.addLayout(row)
        self.table = QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(["#", "Task", "When", "Status", "Result"])
        for _c in (0, 2, 3):
            self.table.horizontalHeader().setSectionResizeMode(_c, QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(4, QHeaderView.ResizeMode.Stretch)
        self.table.verticalHeader().hide()
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.doubleClicked.connect(lambda *_: self.view())
        lay.addWidget(self.table, 1)
        self.note = QLabel(""); self.note.setObjectName("hint"); self.note.setWordWrap(True)
        lay.addWidget(self.note)
        self._timer = QTimer(self); self._timer.timeout.connect(self.refresh); self._timer.start(3000)
        self.refresh()

    def _preview(self, text: str) -> None:
        if not text.strip():
            self.when.setText("")
            return
        w, rest = parse_when(text)
        if w:
            rep = f" · repeats {w.repeat.replace(':', ' ')}" if w.repeat else ""
            self.when.setText(f"⏰ {describe_when(w.run_at.timestamp())}{rep}  —  “{rest}”")
        else:
            self.when.setText("⏰ no time found: it will run right away")

    def selected_id(self) -> int | None:
        r = self.table.currentRow()
        it = self.table.item(r, 0) if r >= 0 else None
        return int(it.text()) if it else None

    def refresh(self) -> None:
        keep = self.selected_id()
        jobs = self.store.list(limit=60)
        self.table.setRowCount(len(jobs))
        for r, j in enumerate(jobs):
            label, color = _STATUS.get(j["status"], (j["status"], "#9fe9f5"))
            cells = [str(j["id"]), j["title"] + (" ↻" if j.get("repeat") else ""),
                     describe_when(j["run_at"]) if j["status"] in ("queued", "running") else
                     datetime.fromtimestamp(j["finished"] or j["created"]).strftime("%d %b %H:%M"),
                     label, (j.get("result") or j.get("error") or "")[:80].replace("\n", " ")]
            for c, text in enumerate(cells):
                it = QTableWidgetItem(text)
                if c == 3:
                    it.setForeground(QColor(color))
                self.table.setItem(r, c, it)
            if keep == j["id"]:
                self.table.selectRow(r)
        self.countChanged.emit(sum(1 for j in jobs if j["status"] in ("queued", "running")))
        if not self.notifier.ready:
            self.note.setText("Telegram is not set up, so you will not get phone messages. "
                              "See docs/AI_UPGRADE.md → Job queue.")
        else:
            self.note.setText("")

    def add(self) -> dict | None:
        text = self.input.text().strip()
        if not text:
            return None
        try:
            job = add_from_text(self.store, text)
        except ValueError as e:
            self.status.emit(f"SYS: Could not add that job ({e}).")
            return None
        self.input.clear()
        sent = announce(job, self.notifier)
        self.status.emit(f"SYS: 📋 Job #{job['id']} queued for {describe_when(job['run_at'])}" + (" — Telegram confirmation sent." if sent else "."))
        self.refresh()
        return job

    def run_now(self) -> None:
        if (i := self.selected_id()) is not None:
            self.store.run_now(i); self.refresh()

    def cancel(self) -> None:
        if (i := self.selected_id()) is not None:
            self.store.cancel(i); self.refresh()

    def clear(self) -> None:
        self.store.delete_finished(); self.refresh()

    def view(self) -> None:
        i = self.selected_id()
        j = self.store.get(i) if i is not None else None
        if j:
            ResultDialog(j, self.open_file, self).exec()
