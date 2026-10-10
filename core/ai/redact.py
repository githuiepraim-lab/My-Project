"""Secret redaction. Every error message, log line and diagnostic produced by
the AI layer passes through `redact()` so a key can never reach a log, a
dashboard or a prompt by accident."""
from __future__ import annotations

import re
import threading

_PATTERNS = [
    re.compile(r"sk-ant-[A-Za-z0-9_\-]{8,}"),
    re.compile(r"sk-or-[A-Za-z0-9_\-]{8,}"),
    re.compile(r"sk-[A-Za-z0-9_\-]{16,}"),
    re.compile(r"gsk_[A-Za-z0-9]{16,}"),
    re.compile(r"AIza[0-9A-Za-z_\-]{20,}"),
    re.compile(r"(?i)(bearer\s+)[A-Za-z0-9._\-]{12,}"),
    re.compile(r"(?i)((?:api[_-]?key|x-api-key|authorization|token)[\"'\s:=]+)[A-Za-z0-9._\-]{12,}"),
    re.compile(r"([?&]key=)[^&\s\"']+"),
]
_lock = threading.Lock()
_known: set[str] = set()


def register_secret(value: str) -> None:
    """Remember an exact key so it is scrubbed even if it matches no pattern."""
    v = (value or "").strip()
    if len(v) >= 8:
        with _lock:
            _known.add(v)


def redact(text) -> str:
    s = str(text)
    with _lock:
        known = sorted(_known, key=len, reverse=True)
    for k in known:
        if k in s:
            s = s.replace(k, "[REDACTED]")
    for pat in _PATTERNS:
        if pat.groups:
            s = pat.sub(lambda m: m.group(1) + "[REDACTED]", s)
        else:
            s = pat.sub("[REDACTED]", s)
    return s
