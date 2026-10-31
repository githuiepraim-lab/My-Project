"""cctv — watch cameras, alert on unknown people, analyse recorded footage."""
from __future__ import annotations

import json

from core import confirm
from core.ai import config as C
from core.cctv import CCTV, CCTV_DIR, analyze_video
from core.jobs import Notifier
from core.people import OpenCVFaces, People

_cc = None


def _cams() -> list[str]:
    try: return json.loads((CCTV_DIR / "cameras.json").read_text(encoding="utf-8"))
    except Exception: return []


def _get() -> CCTV:
    global _cc
    if _cc is None: _cc = CCTV(People(backend=OpenCVFaces()), Notifier.from_config())
    return _cc


def cctv(parameters: dict, player=None) -> str:
    mode, name = str(parameters.get("mode", "list")).lower(), str(parameters.get("name", "")).strip().lower()
    try:
        if mode == "list": return f"Cameras: {', '.join(_cams()) or 'none'}. Watching: {', '.join(_get().running()) or 'none'}."
        if mode == "add":
            url = str(parameters.get("url", "")).strip()
            if not name or not url: return "Give me a name and the camera address (rtsp://user:pass@192.168.1.20:554/stream, a USB camera number, or a file path)."
            def save():
                C.set_api_key(f"cctv_{name}", url)                    # may contain a password: credential store, not a file
                CCTV_DIR.mkdir(parents=True, exist_ok=True)
                (CCTV_DIR / "cameras.json").write_text(json.dumps(sorted(set(_cams()) | {name})), encoding="utf-8")
                return f"Camera '{name}' added."
            return confirm.request("cctv_add", f"Add camera '{name}'", "Its address is stored in the credential store.", save)
        if mode == "start": _get().start(name, C.get_api_key(f"cctv_{name}")); return f"Watching '{name}'. I will message you about unknown people."
        if mode == "stop": _get().stop(name or None); return "Stopped."
        if mode == "events": return "; ".join(f"{x['camera']}: {x['who']}" for x in _get().events(10)) or "No events yet."
        if mode == "analyze": return analyze_video(str(parameters.get("file", "")), _get().people)["summary"][:1500]
        return "Modes: list, add (name, url), start, stop, events, analyze (file)."
    except Exception as e:
        return f"Sir, cctv failed: {type(e).__name__}: {str(e)[:100]}"


TOOL = {"name": "cctv",
        "description": "CCTV: watch cameras (RTSP/USB/file), alert on unknown people, analyse footage. mode: list | add (name,url; confirms) | start/stop (name) | events | analyze (file).",
        "parameters": {"type": "OBJECT", "properties": {"mode": {"type": "STRING"}, "name": {"type": "STRING"}, "url": {"type": "STRING"}, "file": {"type": "STRING"}},
                       "required": ["mode"]}, "handler": cctv}
