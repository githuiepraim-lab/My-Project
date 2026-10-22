"""gestures — hand-gesture control with the webcam (pinch = close tab, spread = new tab, swipes, and more).
Turning the camera on needs the user's confirmation."""
from __future__ import annotations

from core import confirm
from core import gestures as G
from core.gestures import ACTIONS, GESTURES, Controller


def gestures(parameters: dict, player=None) -> str:
    mode = str(parameters.get("mode", "status")).lower()
    cmd = getattr(player, "gesture_command", None)
    try:
        if mode == "status":
            return getattr(player, "gesture_status", lambda: "Gesture control needs the app window.")()
        if mode == "list":
            from memory.config_manager import load_api_keys
            m = Controller.merged_map(load_api_keys().get("gesture_map"))
            return "; ".join(f"{g.replace('_', ' ')} → {a.replace('_', ' ')}" for g, a in m.items())
        if cmd is None:
            return "Gesture control needs the app window."
        if mode == "enable":
            return confirm.request("gestures_on", "Turn on hand-gesture control", "Uses the camera. Video is processed here and never stored or sent.",
                                   lambda: (cmd("on"), "Gesture control is switching on. Hold an open palm to arm it.")[1])
        if mode == "disable":
            cmd("off")
            return "Gesture control is off and the camera is released."
        if mode == "arm":
            cmd("arm")
            return "Armed for 10 seconds."
        if mode == "set":
            g, a = str(parameters.get("gesture", "")).strip().lower().replace(" ", "_"), str(parameters.get("action", "")).strip().lower().replace(" ", "_")
            if g not in GESTURES or a not in ACTIONS:
                return f"Gestures: {', '.join(GESTURES)}. Actions: {', '.join(ACTIONS)}."
            from memory.config_manager import _save_flag, load_api_keys
            m = Controller.merged_map({**(load_api_keys().get("gesture_map") or {}), g: a})
            _save_flag("gesture_map", m)
            return f"{g.replace('_', ' ')} now does: {a.replace('_', ' ')}. (Reopen the gesture window to see it.)"
        return "Modes: status, enable, disable, arm, list, set (gesture, action)."
    except Exception as e:
        return f"Sir, gesture control failed: {type(e).__name__}: {str(e)[:100]}"


TOOL = {
    "name": "gestures",
    "description": ("Hand-gesture control via webcam: pinch=close tab, spread=new tab, swipes=switch tab/scroll, fist=play/pause, "
                    "thumbs=volume, peace=screenshot. mode: status | enable (user confirms; uses camera) | disable | arm | list | "
                    "set (gesture, action). Needs an open palm held 1 s to arm."),
    "parameters": {"type": "OBJECT", "properties": {
        "mode": {"type": "STRING", "description": "status|enable|disable|arm|list|set"},
        "gesture": {"type": "STRING", "description": "e.g. pinch_in, spread, swipe_left, fist_hold, thumbs_up, peace, ok, three"},
        "action": {"type": "STRING", "description": "e.g. close_tab, new_tab, next_tab, scroll_down, play_pause, volume_up, screenshot, none"}},
        "required": ["mode"]},
    "handler": gestures,
}
