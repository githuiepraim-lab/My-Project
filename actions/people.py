"""people — recognise people by face (camera) and voice, remember them by name, and ask YOU about strangers.
Switching it on, naming someone and deleting anyone all need your on-screen confirmation."""
from __future__ import annotations

from core import confirm
from core.people import OpenCVFaces, People, _thumb


def _frame():
    import cv2, numpy as np
    from actions import screen_processor as sp
    return cv2.imdecode(np.frombuffer(sp._capture_camera()[0], np.uint8), cv2.IMREAD_COLOR)


def _enroll_now(P, name: str, owner: bool) -> str:
    frame = _frame(); faces = P.backend.detect(frame)
    if len(faces) != 1: return f"I need exactly one face in view (I see {len(faces)})."
    P.enroll(name, faces[0][1], "face", _thumb(frame, faces[0][0]), owner)
    return f"Remembered {name}."


def people(parameters: dict, player=None) -> str:
    mode, name = str(parameters.get("mode", "status")).lower(), str(parameters.get("name", "")).strip()
    P = People()
    try:
        if mode == "status":
            return f"People recognition is {'ON' if P.enabled else 'OFF'}. Known: {', '.join(P.names()) or 'nobody yet'}. Waiting to be named: {len(P.pending())}."
        if mode == "enable":
            return confirm.request("people_on", "Turn on face/voice recognition", "Stored as numbers on this computer only. Tell people they may be recognised.",
                                   lambda: (P.enable(True), "People recognition is ON.")[1])
        if mode == "disable":
            P.enable(False); return "People recognition is OFF."
        if mode == "list":
            return "Known: " + (", ".join(P.names()) or "nobody") + ". Unknown waiting: " + (", ".join(f"#{k} ({v['kind']}, seen {v['seen']}x)" for k, v in P.pending().items()) or "none")
        if mode == "who":
            if not P.enabled: return "Turn people recognition on first."
            P.backend = OpenCVFaces(); res = P.see(_frame(), "camera")
            return "; ".join((r["name"] or f"someone I don't recognise (unknown #{r['pending_id']}) — who is it?") for r in res) or "I do not see anyone."
        if mode == "name":
            pid = int(str(parameters.get("id", "0")).lstrip("#") or 0)
            if not name or str(pid) not in P.pending(): return "Tell me the unknown person's number and the name."
            return confirm.request("people_name", f"Remember unknown #{pid} as {name}", "Adds this face/voice to the people I recognise.", lambda: P.resolve(pid, name))
        if mode == "enroll":
            P.backend = OpenCVFaces()
            return confirm.request("people_enroll", f"Remember the face in front of the camera as {name}", "Adds them to the people I recognise.",
                                   lambda: _enroll_now(P, name, bool(parameters.get("owner"))))
        if mode == "forget":
            return confirm.request("people_forget", f"Forget {name or 'EVERYONE'}", "Deletes the stored face/voice data.", lambda: f"Forgot {P.forget(name or None)} person(s).")
        return "Modes: status, enable, disable, list, who, name, enroll, forget."
    except Exception as e:
        return f"Sir, people recognition failed: {type(e).__name__}: {str(e)[:100]}"


TOOL = {"name": "people",
        "description": "Recognise people by face/voice; ask the owner about strangers. mode: status | enable (user confirms) | disable | list | who (camera) | "
                       "name (id, name; user confirms) | enroll (name; one face in view) | forget (name or all).",
        "parameters": {"type": "OBJECT", "properties": {"mode": {"type": "STRING"}, "name": {"type": "STRING"}, "id": {"type": "STRING"}, "owner": {"type": "BOOLEAN"}},
                       "required": ["mode"]}, "handler": people}
