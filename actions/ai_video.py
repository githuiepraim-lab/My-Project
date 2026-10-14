"""ai_video — make a narrated video about any topic (AI script, AI pictures, voice, captions).
It runs in the background as a job, so the assistant keeps talking; the finished MP4 appears in
the FILES panel and on your Telegram. Nothing is uploaded anywhere automatically."""
from __future__ import annotations

from core.jobs import JobStore, Notifier, announce, describe_when, parse_when
import time


def ai_video(parameters: dict, player=None) -> str:
    topic = str(parameters.get("topic", "")).strip()
    if not topic:
        return "What should the video be about?"
    from core import video_maker
    if video_maker.ffmpeg_path() is None:
        return "I need ffmpeg to make videos. Run:  pip install imageio-ffmpeg"
    when, _ = parse_when(str(parameters.get("when", "")))
    payload = {"topic": topic, "seconds": int(parameters.get("seconds") or 45), "style": str(parameters.get("style", ""))}
    if parameters.get("approve"):
        payload["approve"] = True
    store = JobStore()
    job = store.add(f"Video: {topic}"[:120], "video", payload, when.run_at.timestamp() if when else time.time(), when.repeat if when else None)
    announce(job, Notifier.from_config())
    if player is not None and hasattr(player, "write_log"):
        player.write_log(f"SYS: 🎬 Video job #{job['id']} queued — see the JOBS panel.")
    return (f"Started video job #{job['id']} ({describe_when(job['run_at'])}). It takes a few minutes; I will tell you here and on "
            f"Telegram when it is ready.")


TOOL = {
    "name": "ai_video",
    "description": "Make a narrated AI video (script, AI pictures, voice, captions) about a topic, in the background. "
                   "Params: topic, seconds (15-180), style, when (optional, e.g. 'tomorrow 8am'), approve (true = ask on Telegram). "
                   "Does not upload anywhere.",
    "parameters": {"type": "OBJECT", "properties": {
        "topic": {"type": "STRING"}, "seconds": {"type": "INTEGER"}, "style": {"type": "STRING"},
        "when": {"type": "STRING"}, "approve": {"type": "BOOLEAN"}}, "required": ["topic"]},
    "handler": ai_video,
}
