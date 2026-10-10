"""Keeping requests small: token estimates, history trimming and summarising,
relevant-tool selection and compact tool results."""
from __future__ import annotations

import hashlib
import json
import re

_WORD = re.compile(r"[A-Za-z0-9_]{2,}")


def estimate_tokens(text) -> int:
    """~4 chars per token for Latin text, ~1.5 for CJK. An estimate: providers'
    own counts are used whenever they report one."""
    s = text if isinstance(text, str) else json.dumps(text, default=str)
    if not s:
        return 0
    wide = sum(1 for c in s if ord(c) > 0x2E80)
    return int((len(s) - wide) / 4 + wide / 1.5) + 1


def message_tokens(m: dict) -> int:
    return 4 + estimate_tokens(m.get("content", "")) + (
        estimate_tokens(m["tool_calls"]) if m.get("tool_calls") else 0)


def compact_tool_result(text, max_chars: int = 1500) -> str:
    """Shrink a tool result: minify JSON, then keep head and tail."""
    s = text if isinstance(text, str) else json.dumps(text, default=str)
    st = s.strip()
    if st[:1] in "{[":
        try:
            s = json.dumps(json.loads(st), separators=(",", ":"), ensure_ascii=False)
        except ValueError:
            pass
    if len(s) <= max_chars:
        return s
    head = int(max_chars * 0.65)
    tail = max_chars - head
    return f"{s[:head]}\n…[{len(s) - max_chars} chars omitted]…\n{s[-tail:]}"


def dedupe_tool_results(messages: list[dict]) -> list[dict]:
    """Replace a tool output identical to an earlier one with a pointer, so the
    same large result is not paid for twice."""
    seen: dict[str, int] = {}
    out = []
    for i, m in enumerate(messages):
        if m.get("role") == "tool" and isinstance(m.get("content"), str) and len(m["content"]) > 200:
            h = hashlib.sha1(m["content"].encode("utf-8")).hexdigest()
            if h in seen:
                m = {**m, "content": f"[same output as message {seen[h]}]"}
            else:
                seen[h] = i
        out.append(m)
    return out


_FACT = re.compile(r"\d|\b(decid|agree|will|must|prefer|always|never|deadline|because|"
                   r"name|chose|plan|todo|need)\w*", re.I)


def extractive_summary(messages: list[dict], max_chars: int = 700) -> str:
    """Model-free fallback summary: keep sentences that carry facts, numbers
    or decisions. Used when no summariser is available or it fails."""
    picked: list[str] = []
    for m in messages:
        who = "User" if m.get("role") == "user" else "Assistant"
        for sent in re.split(r"(?<=[.!?])\s+|\n+", str(m.get("content", ""))):
            sent = sent.strip()
            if 12 <= len(sent) <= 220 and _FACT.search(sent):
                picked.append(f"{who}: {sent}")
    out, used = [], 0
    for line in picked:
        if used + len(line) + 1 > max_chars:
            break
        out.append(line)
        used += len(line) + 1
    return "\n".join(out)


def fit_history(messages: list[dict], budget: int, summarize=None,
                keep_last: int = 4) -> list[dict]:
    """Trim `messages` to about `budget` tokens.

    The newest `keep_last` messages are always kept whole. Older ones are
    replaced by ONE summary message (from `summarize(older) -> str` when
    given, else extractive) so decisions and facts survive the cut.
    """
    messages = dedupe_tool_results(list(messages))
    if sum(message_tokens(m) for m in messages) <= budget or len(messages) <= keep_last:
        return messages
    tail = messages[-keep_last:]
    # Never start the kept tail on a tool result whose call was cut away.
    while tail and tail[0].get("role") == "tool" and len(tail) < len(messages):
        tail = messages[-(len(tail) + 1):]
    older = messages[:len(messages) - len(tail)]
    summary = ""
    if summarize is not None:
        try:
            summary = (summarize(older) or "").strip()
        except Exception:
            summary = ""
    if not summary:
        summary = extractive_summary(older)
    head = []
    if summary:
        head = [{"role": "user", "content": f"[Earlier conversation, summarised]\n{summary}"}]
    out = head + tail
    # Last resort: drop whole tail messages from the front until it fits.
    while len(out) > 2 and sum(message_tokens(m) for m in out) > budget:
        out.pop(1 if head else 0)
    return out


def select_tools(tools: list[dict], query: str, limit: int = 8,
                 pinned: frozenset = frozenset()) -> list[dict]:
    """Send only the tool definitions that could matter for `query`."""
    if len(tools) <= limit:
        return tools
    q = {w.lower() for w in _WORD.findall(query or "")}
    scored = []
    for t in tools:
        text = f"{t.get('name', '')} {t.get('description', '')}".lower().replace("_", " ")
        words = set(_WORD.findall(text))
        name_words = set(t.get("name", "").lower().split("_"))
        s = len(q & words) + 3 * len(q & name_words)
        scored.append((-(10_000 if t.get("name") in pinned else s), t.get("name", ""), t))
    scored.sort(key=lambda x: (x[0], x[1]))
    return [t for _s, _n, t in scored[:limit]]
