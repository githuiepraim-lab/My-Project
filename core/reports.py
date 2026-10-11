"""Make a PDF report and deliver files (PDF, photos, video, zip) by Telegram or
e-mail, or start a video call. Delivery needs credentials you configure once;
nothing here runs without the on-screen confirmation in actions/report_sender."""
from __future__ import annotations

import html
import mimetypes
import secrets
import smtplib
import ssl
import zipfile
from email.message import EmailMessage
from pathlib import Path

MAX_FILES = 10
MAX_TOTAL = 45 * 1024 * 1024          # Telegram's bot limit is 50 MB; stay under it
_PHOTO = {".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp"}
_VIDEO = {".mp4", ".mov", ".mkv", ".avi", ".webm"}
OUT_DIR = Path.home() / "Documents" / "Ephraim Reports"


def make_pdf(title: str, body: str, images: list[str] | None = None, out: str | None = None) -> str:
    """Render a titled report (plain text with blank-line paragraphs, optional
    images) to PDF with Qt's own writer — no extra dependency."""
    from PyQt6.QtCore import QMarginsF, QSizeF
    from PyQt6.QtGui import QPageLayout, QPageSize, QPdfWriter, QTextDocument
    path = Path(out) if out else OUT_DIR / (
        "".join(c if c.isalnum() or c in "-_ " else "_" for c in title)[:50].strip() or "report") .replace(" ", "_")
    path = path.with_suffix(".pdf") if isinstance(path, Path) else Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    paras = "".join(f"<p>{html.escape(p).replace(chr(10), '<br>')}</p>" for p in body.split("\n\n") if p.strip())
    imgs = "".join(f'<p><img src="{Path(i).resolve().as_uri()}" width="480"></p>'
                   for i in (images or []) if Path(i).suffix.lower() in _PHOTO and Path(i).exists())
    doc = QTextDocument()
    doc.setHtml(f"<h1 style='color:#0b3a4a'>{html.escape(title)}</h1>"
                f"<p style='color:#777'>{__import__('time').strftime('%d %B %Y, %H:%M')}</p>{paras}{imgs}")
    w = QPdfWriter(str(path))
    w.setPageSize(QPageSize(QPageSize.PageSizeId.A4))
    w.setPageMargins(QMarginsF(18, 18, 18, 18), QPageLayout.Unit.Millimeter)
    doc.print(w)
    return str(path)


def check_files(paths: list[str]) -> list[Path]:
    files = [Path(p).expanduser() for p in paths if str(p).strip()]
    if not files:
        raise ValueError("no files given")
    if len(files) > MAX_FILES:
        raise ValueError(f"at most {MAX_FILES} files at a time")
    for f in files:
        if not f.is_file():
            raise ValueError(f"not a file: {f.name}")
    if sum(f.stat().st_size for f in files) > MAX_TOTAL:
        raise ValueError("files are over 45 MB in total; send fewer or zip and split them")
    return files


def zip_files(files: list[Path], name: str = "files") -> Path:
    out = Path(files[0].parent) / f"{name}.zip"
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for f in files:
            z.write(f, f.name)
    return out


def send_telegram(token: str, chat_id: str, files: list[Path], caption: str = "") -> str:
    import requests
    base = f"https://api.telegram.org/bot{token}"
    sent = 0
    for i, f in enumerate(files):
        ext = f.suffix.lower()
        method, field = (("sendPhoto", "photo") if ext in _PHOTO else
                         ("sendVideo", "video") if ext in _VIDEO else ("sendDocument", "document"))
        with f.open("rb") as fh:
            r = requests.post(f"{base}/{method}", data={"chat_id": chat_id, "caption": caption[:900] if i == 0 else ""},
                              files={field: (f.name, fh)}, timeout=120)
        if r.status_code != 200:
            raise RuntimeError(f"Telegram refused {f.name}: HTTP {r.status_code} "
                               f"{(r.json().get('description') if r.headers.get('content-type','').startswith('application/json') else '')}")
        sent += 1
    return f"Sent {sent} file(s) on Telegram."


def send_email(cfg: dict, password: str, to: str, subject: str, body: str, files: list[Path]) -> str:
    msg = EmailMessage()
    msg["From"], msg["To"], msg["Subject"] = cfg["user"], to, subject
    msg.set_content(body or "Sent by your assistant.")
    for f in files:
        ctype = mimetypes.guess_type(f.name)[0] or "application/octet-stream"
        main, sub = ctype.split("/", 1)
        msg.add_attachment(f.read_bytes(), maintype=main, subtype=sub, filename=f.name)
    host, port = cfg["host"], int(cfg.get("port", 587))
    ctx = ssl.create_default_context()
    if port == 465:
        with smtplib.SMTP_SSL(host, port, context=ctx, timeout=60) as s:
            s.login(cfg["user"], password)
            s.send_message(msg)
    else:
        with smtplib.SMTP(host, port, timeout=60) as s:
            s.starttls(context=ctx)
            s.login(cfg["user"], password)
            s.send_message(msg)
    return f"Emailed {len(files)} file(s) to {to}."


def call_link() -> str:
    """A fresh Jitsi Meet room: free, no account, works in any browser or the phone app."""
    return f"https://meet.jit.si/Ephraim-{secrets.token_urlsafe(9).replace('_', 'x').replace('-', 'y')}"
