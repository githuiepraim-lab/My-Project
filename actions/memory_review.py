"""memory_review — see, approve, discard, correct and delete what the assistant
has learned. Learning is opt-in per item: anything the assistant notices is
held as PENDING until the user approves it."""
from __future__ import annotations

from memory import memory_manager as M


def memory_review(parameters: dict, player=None) -> str:
    mode = str(parameters.get("mode", "list")).lower()
    ident = str(parameters.get("id", "")).strip()
    try:
        if mode == "list":
            rows = M.list_pending()
            if not rows:
                return "Nothing is waiting for your approval."
            return "Waiting for approval: " + "; ".join(
                f"[{r['id']}] {r['value']}" for r in rows)
        if mode == "approve":
            return M.approve_pending(ident or "all")
        if mode == "reject":
            return M.reject_pending(ident or "all")
        if mode == "edit":
            return M.edit_entry(str(parameters.get("category", "notes")), ident,
                                value=parameters.get("value"))
        if mode == "delete":
            return M.delete_entry(ident, str(parameters.get("category", "notes")))
        if mode == "history":
            h = M.history_of(ident, str(parameters.get("category", "notes")))
            return ("Earlier values: " + "; ".join(f"{x['value']} ({x['updated']})" for x in h)
                    if h else "No earlier values.")
        return "Modes: list, approve, reject, edit, delete, history."
    except Exception as e:
        return f"Sir, memory review failed: {type(e).__name__}"


TOOL = {
    "name": "memory_review",
    "description": (
        "Review what was noticed about the user and not yet saved. mode: list (pending items), "
        "approve/reject (id, or all), edit (category, id=key, value), delete (category, id=key), "
        "history (earlier values of a corrected fact). Use when the user says 'review what you "
        "learned', 'keep that', 'forget that'."),
    "parameters": {
        "type": "OBJECT",
        "properties": {
            "mode": {"type": "STRING", "description": "list|approve|reject|edit|delete|history"},
            "id": {"type": "STRING", "description": "pending id, or memory key"},
            "category": {"type": "STRING", "description": "memory category"},
            "value": {"type": "STRING", "description": "new value for edit"},
        },
        "required": ["mode"],
    },
    "handler": memory_review,
}
