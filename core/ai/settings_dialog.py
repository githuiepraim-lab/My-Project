"""AI providers settings — add keys, pick models, set priorities, no source edits.

Keys typed here go straight to the OS keyring (or config/secrets.json); they are
never shown again, never written to providers.json and never logged. Network
work (fetch models, test) runs on a worker thread so the window never freezes.
"""
from __future__ import annotations

import threading

from PyQt6.QtCore import QObject, Qt, pyqtSignal
from PyQt6.QtWidgets import (QCheckBox, QComboBox, QDialog, QDoubleSpinBox, QFormLayout,
                             QHBoxLayout, QInputDialog, QLabel, QLineEdit, QPushButton,
                             QSpinBox, QTableWidget, QTableWidgetItem, QVBoxLayout,
                             QHeaderView, QAbstractItemView)

from . import config as C
from .hub import Hub, get_hub
from .redact import redact

_QSS = """
QDialog, QWidget { background:#020d14; color:#9fe9f5; font-family:'Courier New'; font-size:11px; }
QTableWidget { background:#00091a; gridline-color:#0b3a4a; border:1px solid #0b3a4a; }
QHeaderView::section { background:#04202c; color:#5fd4e6; border:0; padding:4px; }
QPushButton { background:#00091a; color:#00d4ff; border:1px solid #0b5b73; border-radius:3px; padding:5px 10px; }
QPushButton:hover { background:#062a38; border-color:#00d4ff; }
QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox { background:#00091a; border:1px solid #0b3a4a; padding:3px; color:#c8f6ff; }
QLabel#note { color:#5a8f9c; }
"""


class _Bus(QObject):
    done = pyqtSignal(str)


class AISettingsDialog(QDialog):
    COLS = ["On", "Provider", "Priority", "Key", "Memory", "Models  (id:tier, …)"]

    def __init__(self, parent=None, hub: Hub | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("AI providers")
        self.setStyleSheet(_QSS)
        self.resize(900, 560)
        self.hub = hub or get_hub()
        self._bus = _Bus()
        self._bus.done.connect(self._say)
        self._cfg = C.load(force=True)

        lay = QVBoxLayout(self)
        note = QLabel("Keys are stored in your OS keyring (or config/secrets.json) and are never "
                      "displayed. 'Memory' lets that provider see your stored facts — leave it off "
                      "for any service you do not fully trust with them.")
        note.setObjectName("note")
        note.setWordWrap(True)
        lay.addWidget(note)

        self.table = QTableWidget(0, len(self.COLS))
        self.table.setHorizontalHeaderLabels(self.COLS)
        self.table.horizontalHeader().setSectionResizeMode(5, QHeaderView.ResizeMode.Stretch)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.verticalHeader().hide()
        lay.addWidget(self.table, 1)

        row = QHBoxLayout()
        for text, fn in (("Set key…", self._set_key), ("Fetch models", self._fetch),
                         ("Test", self._test), ("Add provider…", self._add),
                         ("Remove", self._remove)):
            b = QPushButton(text)
            b.clicked.connect(fn)
            row.addWidget(b)
        row.addStretch(1)
        lay.addLayout(row)

        form = QFormLayout()
        self.strategy = QComboBox()
        self.strategy.addItems(["cost", "priority", "speed"])
        self.mode = QComboBox()
        self.mode.addItems(["auto", "manual"])
        self.pin = QComboBox()
        self.pin.setEditable(True)
        self.max_in = QSpinBox(); self.max_in.setRange(500, 200_000); self.max_in.setSingleStep(500)
        self.max_out = QSpinBox(); self.max_out.setRange(50, 32_000); self.max_out.setSingleStep(50)
        self.cap = QDoubleSpinBox(); self.cap.setRange(0, 10_000); self.cap.setPrefix("$ ")
        self.cap.setSpecialValueText("no limit")
        self.cache = QCheckBox("Cache repeatable answers")
        self.cross = QCheckBox("If every Gemini model fails, let another provider answer")
        self.down = QCheckBox("Allow a weaker model after a stronger one fails")
        form.addRow("Routing strategy", self.strategy)
        form.addRow("Model choice", self.mode)
        form.addRow("Pinned model (provider/model)", self.pin)
        form.addRow("Max input tokens / request", self.max_in)
        form.addRow("Max output tokens / reply", self.max_out)
        form.addRow("Monthly cost cap", self.cap)
        form.addRow("", self.cache)
        form.addRow("", self.cross)
        form.addRow("", self.down)
        lay.addLayout(form)

        self.status = QLabel("")
        self.status.setObjectName("note")
        self.status.setWordWrap(True)
        lay.addWidget(self.status)
        foot = QHBoxLayout()
        foot.addStretch(1)
        save = QPushButton("Save"); save.clicked.connect(self._save)
        close = QPushButton("Close"); close.clicked.connect(self.accept)
        foot.addWidget(save); foot.addWidget(close)
        lay.addLayout(foot)
        self._load()

    # ── data <-> widgets ────────────────────────────────────────────────────
    def _load(self) -> None:
        s = self._cfg["settings"]
        self.strategy.setCurrentText(s.get("strategy", "cost"))
        self.mode.setCurrentText(s.get("mode", "auto"))
        self.pin.setEditText(s.get("manual_model", ""))
        self.max_in.setValue(int(s.get("max_input_tokens", 6000)))
        self.max_out.setValue(int(s.get("max_output_tokens", 800)))
        self.cap.setValue(float(s.get("monthly_cost_cap_usd") or 0))
        self.cache.setChecked(bool(s.get("cache", True)))
        self.cross.setChecked(bool(s.get("cross_provider_fallback", True)))
        self.down.setChecked(bool(s.get("allow_downgrade", False)))
        self.table.setRowCount(0)
        for name, p in self._cfg["providers"].items():
            self._add_row(name, p)
        self._refresh_pins()

    def _add_row(self, name: str, p: dict) -> None:
        r = self.table.rowCount()
        self.table.insertRow(r)
        on = QCheckBox(); on.setChecked(bool(p.get("enabled")))
        mem = QCheckBox(); mem.setChecked(bool(p.get("share_memory", False)))
        pr = QSpinBox(); pr.setRange(0, 999); pr.setValue(int(p.get("priority", 100)))
        self.table.setCellWidget(r, 0, on)
        self.table.setItem(r, 1, QTableWidgetItem(name))
        self.table.item(r, 1).setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable)
        self.table.setCellWidget(r, 2, pr)
        self.table.setItem(r, 3, QTableWidgetItem(self._key_text(name, p)))
        self.table.item(r, 3).setFlags(Qt.ItemFlag.ItemIsEnabled)
        self.table.setCellWidget(r, 4, mem)
        models = ", ".join(f"{m['id']}:{m.get('tier', 1)}" for m in p.get("models", []))
        self.table.setItem(r, 5, QTableWidgetItem(models))

    @staticmethod
    def _key_text(name: str, p: dict) -> str:
        if p.get("requires_key", True) is False:
            return "not needed"
        return ("✓ " + C.key_source(name)) if C.get_api_key(name) else "— none"

    def _collect(self) -> dict:
        cfg = {"version": 1, "settings": dict(self._cfg["settings"]),
               "providers": {}, "agents": self._cfg.get("agents", {})}
        for r in range(self.table.rowCount()):
            name = self.table.item(r, 1).text()
            p = dict(self._cfg["providers"][name])
            p["enabled"] = self.table.cellWidget(r, 0).isChecked()
            p["priority"] = self.table.cellWidget(r, 2).value()
            p["share_memory"] = self.table.cellWidget(r, 4).isChecked()
            old = {m["id"]: m for m in p.get("models", [])}
            models = []
            for part in self.table.item(r, 5).text().split(","):
                part = part.strip()
                if not part:
                    continue
                mid, _, tier = part.rpartition(":") if part.rsplit(":", 1)[-1].isdigit() \
                    else (part, "", "1")
                mid = (mid or part).strip()
                m = dict(old.get(mid, {"id": mid, "caps": ["text", "stream"]}))
                m["id"], m["tier"] = mid, max(0, min(2, int(tier or 1)))
                models.append(m)
            p["models"] = models
            cfg["providers"][name] = p
        s = cfg["settings"]
        s.update(strategy=self.strategy.currentText(), mode=self.mode.currentText(),
                 manual_model=self.pin.currentText().strip(),
                 max_input_tokens=self.max_in.value(), max_output_tokens=self.max_out.value(),
                 monthly_cost_cap_usd=self.cap.value() or None, cache=self.cache.isChecked(),
                 cross_provider_fallback=self.cross.isChecked(),
                 allow_downgrade=self.down.isChecked())
        return cfg

    def _refresh_pins(self) -> None:
        cur = self.pin.currentText()
        self.pin.clear()
        for n, p in self._cfg["providers"].items():
            for m in p.get("models", []):
                self.pin.addItem(f"{n}/{m['id']}")
        self.pin.setEditText(cur or self._cfg["settings"].get("manual_model", ""))

    def _name(self) -> str | None:
        r = self.table.currentRow()
        return self.table.item(r, 1).text() if r >= 0 else None

    def _say(self, msg: str) -> None:
        self.status.setText(msg)

    # ── actions ─────────────────────────────────────────────────────────────
    def _save(self) -> None:
        self._cfg = self._collect()
        C.save(self._cfg)
        self.hub.health.reset()
        self._say("Saved. Changes apply to the next request.")

    def _set_key(self) -> None:
        name = self._name()
        if not name:
            return self._say("Select a provider first.")
        key, ok = QInputDialog.getText(self, f"API key for {name}", "Paste the key:",
                                       QLineEdit.EchoMode.Password)
        if ok and key.strip():
            where = C.set_api_key(name, key.strip())
            self.table.item(self.table.currentRow(), 3).setText(f"✓ {where}")
            self.hub.health.reset()
            self._say(f"Key stored in {where}.")

    def _work(self, fn, label: str) -> None:
        self._say(label + "…")
        def run():
            try:
                self._bus.done.emit(fn())
            except Exception as e:
                self._bus.done.emit("Failed: " + redact(e)[:200])
        threading.Thread(target=run, daemon=True).start()

    def _fetch(self) -> None:
        name = self._name()
        if not name:
            return self._say("Select a provider first.")
        self._save()
        def fn():
            ids = self.hub._adapter(name).list_models()
            return (f"{name} offers: " + ", ".join(ids[:25]) + (" …" if len(ids) > 25 else "")
                    if ids else f"{name} returned no model list (check the key / that it is running).")
        self._work(fn, f"Asking {name} for its models")

    def _test(self) -> None:
        name = self._name()
        if not name:
            return self._say("Select a provider first.")
        self._save()
        def fn():
            r = self.hub.ask_many("Reply with the single word: ready", [name], max_tokens=20,
                                  timeout=20)[0]
            return (f"{name}: {r.text!r} in {r.latency:.2f}s, {r.usage.input}+{r.usage.output} tokens"
                    if r.ok else f"{name}: {r.error}")
        self._work(fn, f"Testing {name}")

    def _add(self) -> None:
        name, ok = QInputDialog.getText(self, "Add provider", "Short name (letters/digits):")
        if not ok or not name.strip():
            return
        url, ok = QInputDialog.getText(self, "Add provider",
                                       "OpenAI-compatible base URL (e.g. https://api.together.xyz/v1):")
        if not ok or not url.strip():
            return
        model, ok = QInputDialog.getText(self, "Add provider", "A model id:")
        if not ok or not model.strip():
            return
        name = "".join(ch for ch in name.lower() if ch.isalnum() or ch in "-_")[:31] or "custom"
        p = {"kind": "openai_compat", "enabled": True, "priority": 60, "base_url": url.strip(),
             "share_memory": False, "stream_usage": False,
             "models": [{"id": model.strip(), "tier": 1, "caps": ["text", "stream", "tools"]}]}
        self._cfg["providers"][name] = p
        self._add_row(name, p)
        self._refresh_pins()
        self._say(f"Added {name}. Select it and press 'Set key…'.")

    def _remove(self) -> None:
        name = self._name()
        if not name:
            return
        self._cfg["providers"].pop(name, None)
        C.delete_api_key(name)
        self.table.removeRow(self.table.currentRow())
        self._refresh_pins()
        self._say(f"Removed {name} (and its stored key). Press Save to apply.")


def open_dialog(parent=None) -> None:
    AISettingsDialog(parent).exec()
