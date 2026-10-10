"""
career_agent — a networking coach that can SEE (camera or screen) and write.

It never posts, sends or connects on its own. It looks, advises, drafts, and
anything that reaches another person goes through the on-screen confirmation
gate first.
"""
from __future__ import annotations

from core.ai import get_hub
from core.ai.types import AIError, Image, Request

_COACH = ("You are a sharp, warm career coach for a young software/electronics engineer. "
          "You are shown a live view (camera or screen) from the user's own device. Say, in "
          "under 70 words and plain speech, what you see that matters for connecting with "
          "people (name tag, profile, slide, message thread, setting) and the single best "
          "next action. Be concrete. Never identify a stranger from their face; only use "
          "text visible on screen or what the user tells you.")
_DRAFT = ("You draft messages for a young engineer building a career. Write ONE message for the "
          "platform and goal given: specific, human, under {n} words, no flattery clichés, "
          "no invented facts about the recipient, ends with one clear small ask. Output only "
          "the message.")
_PLAN = ("You are a career strategist for a young engineer. Give a short, concrete networking "
         "plan for the goal: 4-5 steps for the next two weeks, each one action under 20 words.")


def _untrusted_vision_providers(hub) -> frozenset:
    """Camera and screen images go only to providers the user marked trusted
    ('Memory' ticked) or that run locally."""
    return frozenset(m.provider for m in hub.models() if not (m.local or hub._share_ok(m.provider)))


def _capture(source: str) -> Image:
    from actions import screen_processor as sp
    data, mime = sp._capture_camera() if source == "camera" else sp._capture_screen()
    return Image(data, mime or "image/jpeg")


def _send(params: dict, player) -> str:
    from core import confirm
    from actions import send_message as sm
    platform, to, text = (str(params.get(k, "")).strip() for k in ("platform", "to", "text"))
    if not (platform and to and text):
        return "I need the platform, who it is for, and the message text."

    def run() -> str:
        return sm.send_message({"receiver": to, "message_text": text, "platform": platform},
                               player=player)
    return confirm.request("career_send", f"Send a message to {to} on {platform}",
                           f"“{text[:300]}”", run)


def career_agent(parameters: dict, player=None) -> str:
    mode = str(parameters.get("mode", "plan")).lower()
    goal = str(parameters.get("goal", "")).strip()
    hub = get_hub()
    try:
        if mode == "coach":
            src = "screen" if str(parameters.get("source", "camera")).lower() == "screen" else "camera"
            img = _capture(src)
            ask = goal or "What should I do here to make a good connection?"
            r = hub.ask(Request(messages=[{"role": "user", "content": ask}], system=_COACH,
                                images=[img], task="vision", max_output_tokens=220, cacheable=False),
                        exclude=_untrusted_vision_providers(hub))
            return r.text
        if mode == "draft":
            n = int(parameters.get("max_words") or 90)
            who = str(parameters.get("to", "")).strip()
            brief = (f"Platform: {parameters.get('platform', 'LinkedIn')}\nRecipient: {who or 'unknown'}\n"
                     f"Context: {parameters.get('context', '')}\nGoal: {goal}")
            return hub.ask(Request(messages=[{"role": "user", "content": brief}],
                                   system=_DRAFT.format(n=n), task="rewrite", min_tier=1,
                                   max_output_tokens=300, cacheable=False)).text
        if mode == "plan":
            return hub.ask(Request(messages=[{"role": "user", "content": goal or "Meet engineers at my target companies"}],
                                   system=_PLAN, task="plan", max_output_tokens=350)).text
        if mode == "send":
            return _send(parameters, player)
        return "Modes: coach (camera/screen), draft, plan, send (asks you to confirm first)."
    except AIError as e:
        return f"Sir, the career agent could not reach an AI: {e}"
    except Exception as e:
        return f"Sir, career_agent failed: {type(e).__name__}: {str(e)[:80]}"


TOOL = {
    "name": "career_agent",
    "description": (
        "Networking coach for the user's engineering career. mode: coach (look through camera or "
        "screen and say how to connect), draft (write a message; never sends), plan (2-week "
        "networking plan), send (send a drafted message — the user must confirm on screen). "
        "Use for networking, LinkedIn/WhatsApp outreach, events, reports to send."),
    "parameters": {
        "type": "OBJECT",
        "properties": {
            "mode": {"type": "STRING", "description": "coach|draft|plan|send"},
            "goal": {"type": "STRING", "description": "what the user wants"},
            "source": {"type": "STRING", "description": "camera or screen (coach)"},
            "platform": {"type": "STRING", "description": "LinkedIn, WhatsApp, Telegram..."},
            "to": {"type": "STRING", "description": "recipient"},
            "context": {"type": "STRING", "description": "background for the draft"},
            "text": {"type": "STRING", "description": "final message (send)"},
        },
        "required": ["mode"],
    },
    "handler": career_agent,
    "behavior": "NON_BLOCKING",
    "scheduling": "WHEN_IDLE",
}
