"""✈ TELEGRAM window: paste the bot token, send the pairing code to your bot, done."""
from __future__ import annotations

import threading

from PyQt6.QtCore import QObject, QUrl, pyqtSignal
from PyQt6.QtGui import QDesktopServices, QFont
from PyQt6.QtWidgets import QDialog, QHBoxLayout, QLabel, QLineEdit, QPushButton, QVBoxLayout

from core.telegram_setup import SetupError, TelegramSetup

_QSS = """
QDialog, QWidget { background:#020d14; color:#9fe9f5; font-family:'Courier New'; font-size:11px; }
QLabel { background:transparent; } QLabel#hint { color:#5a8f9c; } QLabel#code { font-size:30px; font-weight:bold; color:#00d4ff; letter-spacing:6px; }
QLineEdit { background:#00091a; border:1px solid #0b5b73; padding:7px; color:#c8f6ff; }
QPushButton { background:#00091a; color:#00d4ff; border:1px solid #0b5b73; border-radius:3px; padding:6px 12px; }
QPushButton:hover { border-color:#00d4ff; } QPushButton:disabled { color:#456; border-color:#234; }
"""


class _Bus(QObject):
    event = pyqtSignal(str, str)         # (kind, text): info | wait | ok | error


class TelegramDialog(QDialog):
    connected = pyqtSignal()

    def __init__(self, parent=None, setup: TelegramSetup | None = None) -> None:
        super().__init__(parent)
        self.setup = setup or TelegramSetup()
        self.setWindowTitle("Connect Telegram")
        self.setStyleSheet(_QSS)
        self.resize(520, 420)
        self._bus = _Bus()
        self._bus.event.connect(self._on_event)
        self._me: dict | None = None
        lay = QVBoxLayout(self)
        t = QLabel("✈  Connect Telegram"); t.setFont(QFont("Courier New", 12, QFont.Weight.Bold)); lay.addWidget(t)
        how = QLabel("1. In Telegram, message  @BotFather  → send  /newbot  → copy the token.\n2. Paste it below.\n"
                     "3. Send the code shown here to your new bot.\n\nWorks with the official Telegram app on iPhone and Android.")
        how.setObjectName("hint"); how.setWordWrap(True); lay.addWidget(how)
        self.token = QLineEdit(); self.token.setEchoMode(QLineEdit.EchoMode.Password)
        self.token.setPlaceholderText("123456789:ABCdef…  (hidden)"); self.token.returnPressed.connect(self.verify); lay.addWidget(self.token)
        self.go = QPushButton("Verify token and get my code"); self.go.clicked.connect(self.verify); lay.addWidget(self.go)
        self.code = QLabel(""); self.code.setObjectName("code"); lay.addWidget(self.code)
        self.link = QPushButton("Open my bot in Telegram"); self.link.hide(); self.link.clicked.connect(self._open_bot); lay.addWidget(self.link)
        self.status = QLabel(""); self.status.setWordWrap(True); lay.addWidget(self.status)
        row = QHBoxLayout(); row.addStretch(1)
        self.disc = QPushButton("Disconnect"); self.disc.clicked.connect(self.disconnect); row.addWidget(self.disc)
        close = QPushButton("Close"); close.clicked.connect(self.accept); row.addWidget(close); lay.addLayout(row)
        self._refresh_state()

    def _refresh_state(self) -> None:
        st = self.setup.status() if hasattr(self.setup, "status") else {"connected": False}
        self.disc.setVisible(bool(st.get("connected")))
        if st.get("connected"):
            self.status.setText("✅ Telegram is connected. Use the box above only if you want to connect a different bot.")

    def _open_bot(self) -> None:
        if self._me:
            QDesktopServices.openUrl(QUrl(f"https://t.me/{self._me['username']}"))

    def _run(self, fn) -> None:
        def go():
            try:
                fn()
            except SetupError as e:
                self._bus.event.emit("error", str(e))
            except Exception as e:
                self._bus.event.emit("error", f"{type(e).__name__}: {str(e)[:100]}")
        threading.Thread(target=go, daemon=True).start()

    def verify(self) -> None:
        raw = self.token.text()
        self.go.setEnabled(False)
        self.status.setText("Checking the token…")
        def job():
            me = self.setup.verify(raw)
            self._me = me
            code = self.setup.new_code()
            self._bus.event.emit("info", f"{code}|{me['name']}|{me['username']}")
            got = self.setup.pair(me["token"], code, timeout=180, on_wait=lambda left, ign: self._bus.event.emit("wait", f"{left}|{ign}"))
            self.setup.save(me["token"], got["chat_id"])
            self.setup.send_test(me["token"], got["chat_id"])
            self._bus.event.emit("ok", got["name"])
        self._run(job)

    def disconnect(self) -> None:
        self.setup.disconnect()
        self.status.setText("Disconnected. The token was removed from this computer.")
        self.disc.hide()
        self.connected.emit()

    def _on_event(self, kind: str, text: str) -> None:
        if kind == "info":
            code, name, user = text.split("|", 2)
            self.code.setText(code)
            self.link.show()
            self.status.setText(f"✓ Bot found: {name} (@{user}).\nOpen it in Telegram, press Start, and send the code above. Waiting…")
        elif kind == "wait":
            left, ignored = text.split("|")
            self.status.setText(f"Waiting for the code… {left}s left" + (f"  ({ignored} other message(s) ignored)" if ignored != "0" else ""))
        elif kind == "ok":
            self.code.setText("✓")
            self.status.setText(f"✅ Connected to {text}. A test message was sent to your Telegram.")
            self.token.clear(); self.go.setEnabled(True); self.disc.show(); self.link.hide()
            self.connected.emit()
        else:
            self.status.setText(f"✗ {text}")
            self.go.setEnabled(True)
