"""show_code — display code in the app's own preview: highlighted source, a rendered website
for HTML, rendered Markdown, or a colour diff. It never opens an external editor."""
from __future__ import annotations

from pathlib import Path

from core.code_view import CODE_EXT


def show_code(parameters: dict, player=None) -> str:
    if player is None or getattr(player, "show_code", None) is None:
        return "The code preview is not available here."
    files = [f.strip().strip('"') for f in str(parameters.get("file", "")).replace("\n", ",").split(",") if f.strip()]
    code = str(parameters.get("code", ""))
    try:
        if files:
            ok = [f for f in files if Path(f).expanduser().is_file()]
            if not ok:
                return "I cannot find that file."
            player.show_code([str(Path(f).expanduser()) for f in ok])
            return f"Showing {len(ok)} file(s) in the code preview. HTML opens as a website; use the Code button for the source."
        if code.strip():
            lang = str(parameters.get("language", "text")).lower()
            lang = {"py": "python", "python": "python", "javascript": "js", "js": "js", "html": "html", "css": "css", "json": "json",
                    "markdown": "markdown", "md": "markdown", "diff": "diff"}.get(lang, "text")
            player.show_code_text(str(parameters.get("title", "snippet")), code[:200_000], lang)
            return "Showing it in the code preview."
        return "Give me a file path or some code to show."
    except Exception as e:
        return f"Sir, the code preview failed: {type(e).__name__}"


TOOL = {
    "name": "show_code",
    "description": "Show code in the app's own preview (never an external editor): file path(s) or pasted code; "
                   "HTML renders as a website, Markdown as formatted text, diffs in colour.",
    "parameters": {"type": "OBJECT", "properties": {
        "file": {"type": "STRING", "description": "path(s), comma-separated"},
        "code": {"type": "STRING", "description": "code text instead of a file"},
        "language": {"type": "STRING", "description": "python, js, html, css, json, markdown, diff"},
        "title": {"type": "STRING", "description": "tab title for pasted code"}},
        "required": []},
    "handler": show_code,
}
