"""report_sender — make a PDF report, send files (PDF/photo/video/zip) to your
phone or inbox, or start a video call. Sending always waits for your
confirmation on the HUD, and only the files you name are ever attached."""
from __future__ import annotations

import json
import re
import webbrowser
from pathlib import Path

from core import confirm, reports
from core.ai import config as C

_SETUP = {
    "telegram": ("Telegram is not set up. In Telegram, message @BotFather → /newbot, copy the token, "
                 "send your new bot any message, then add \"telegram_chat_id\" to config/api_keys.json "
                 "and store the token with:  python -m core.ai key telegram"),
    "email": ("Email is not set up. Add \"smtp_host\", \"smtp_user\" (and optionally \"smtp_port\") to "
              "config/api_keys.json and store the app password with:  python -m core.ai key smtp"),
}


def _cfg() -> dict:
    try:
        return json.loads(C.LEGACY_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _split(v) -> list[str]:
    return [p.strip().strip('"') for p in re.split(r"[\n;,]+", str(v or "")) if p.strip()]


def _deliver(channel: str, files: list[Path], caption: str, to: str, zip_it: bool) -> str:
    if zip_it and len(files) > 1:
        files = [reports.zip_files(files)]
    cfg = _cfg()
    if channel == "email":
        tok = C.get_api_key("smtp")
        if not (cfg.get("smtp_host") and cfg.get("smtp_user") and tok):
            return _SETUP["email"]
        return reports.send_email({"host": cfg["smtp_host"], "user": cfg["smtp_user"],
                                   "port": cfg.get("smtp_port", 587)}, tok,
                                  to or cfg.get("email_to") or cfg["smtp_user"],
                                  caption[:80] or "Report", caption, files)
    token, chat = C.get_api_key("telegram"), str(cfg.get("telegram_chat_id", "")).strip()
    if not (token and chat):
        return _SETUP["telegram"]
    return reports.send_telegram(token, chat, files, caption)


def report_sender(parameters: dict, player=None) -> str:
    mode = str(parameters.get("mode", "make_pdf")).lower()
    try:
        if mode == "make_pdf":
            path = reports.make_pdf(str(parameters.get("title", "Report")),
                                    str(parameters.get("body", "")), _split(parameters.get("files")))
            return f"Report saved: {path}"
        if mode == "send":
            channel = "email" if str(parameters.get("channel", "telegram")).lower() == "email" else "telegram"
            files = reports.check_files(_split(parameters.get("files")))
            caption, to = str(parameters.get("caption", "")), str(parameters.get("to", ""))
            zip_it = str(parameters.get("zip", "")).lower() in ("1", "true", "yes")
            names = ", ".join(f"{f.name} ({f.stat().st_size // 1024} KB)" for f in files)
            return confirm.request("send_files", f"Send {len(files)} file(s) by {channel}",
                                   f"{names}" + (f" → {to}" if to else ""),
                                   lambda: _deliver(channel, files, caption, to, zip_it))
        if mode == "call":
            url = reports.call_link()
            webbrowser.open(url)
            note = ""
            if _cfg().get("telegram_chat_id") and C.get_api_key("telegram"):
                try:
                    reports.send_telegram  # noqa: B018  (link is sent as text below)
                    import requests
                    requests.post(f"https://api.telegram.org/bot{C.get_api_key('telegram')}/sendMessage",
                                  data={"chat_id": _cfg()["telegram_chat_id"],
                                        "text": f"Join the video call: {url}"}, timeout=20)
                    note = " I also sent the link to your Telegram so you can join from your phone."
                except Exception:
                    pass
            return (f"Video call room opened: {url}.{note} Note: I can talk with you here through "
                    f"the camera view; I cannot join the call itself as a participant.")
        return "Modes: make_pdf, send (asks you to confirm), call."
    except ValueError as e:
        return f"Sir, I cannot send that: {e}."
    except Exception as e:
        return f"Sir, report_sender failed: {type(e).__name__}: {str(e)[:100]}"


TOOL = {
    "name": "report_sender",
    "description": (
        "Make a PDF report from text; send files (PDF, photos, videos, zip) to the user's phone "
        "(Telegram) or e-mail — the user must confirm on screen; start a video call link. "
        "mode: make_pdf (title, body, files=images) | send (files, channel telegram|email, caption, to, zip) | call."),
    "parameters": {
        "type": "OBJECT",
        "properties": {
            "mode": {"type": "STRING", "description": "make_pdf | send | call"},
            "title": {"type": "STRING"}, "body": {"type": "STRING"},
            "files": {"type": "STRING", "description": "comma-separated file paths"},
            "channel": {"type": "STRING", "description": "telegram or email"},
            "caption": {"type": "STRING"}, "to": {"type": "STRING"},
            "zip": {"type": "STRING", "description": "yes to bundle into one zip"},
        },
        "required": ["mode"],
    },
    "handler": report_sender,
}
