import json
import re
from datetime import datetime
from threading import Lock
from pathlib import Path
import sys


def get_base_dir() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).parent
    return Path(__file__).resolve().parent.parent


BASE_DIR         = get_base_dir()
MEMORY_PATH      = BASE_DIR / "memory" / "long_term.json"
_lock            = Lock()
MAX_VALUE_LENGTH = 380

# ── Why there are two very different numbers here ────────────────────────────
#
# There used to be one: MEMORY_MAX_CHARS = 2200, applied to the whole store. It
# was a *storage* limit, and it existed only because the entire memory was
# pasted into the system prompt on every connect — so growing the memory grew
# every single request. When it filled, _trim_to_limit() deleted the oldest
# entries and printed one line to a console nobody reads. A memory described as
# "deeply remembers projects, preferences and personal context" was in practice
# two pages long, and quietly forgot your sister's name after a few weeks.
#
# Storage and prompt budget are now separate concerns:
#
#   MEMORY_MAX_CHARS  — a runaway guard, not a feature limit. Nothing normal
#                       reaches it; a bug writing in a loop does.
#   PROMPT_CORE_CHARS — what actually rides in the system prompt every session.
#                       Smaller than the old whole-memory dump, so sessions
#                       start *faster* than before, not slower.
#
# Everything above the core stays on disk and is fetched on demand by the
# recall_memory tool — see search_memory() and format_memory_for_prompt().
MEMORY_MAX_CHARS  = 200_000
PROMPT_CORE_CHARS = 900
PROMPT_INDEX_CHARS = 420
# Most entries any one category may contribute to the core block, so a person
# with forty stored preferences still gets their sister into the prompt.
PROMPT_MAX_PER_CATEGORY = 6

def _empty_memory() -> dict:
    return {
        "identity":      {},
        "preferences":   {},
        "projects":      {},
        "relationships": {},
        "wishes":        {},
        "notes":         {},
    }

def load_memory() -> dict:
    if not MEMORY_PATH.exists():
        return _empty_memory()
    with _lock:
        try:
            data = json.loads(MEMORY_PATH.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                base = _empty_memory()
                for key in base:
                    if key not in data:
                        data[key] = {}
                return data
            return _empty_memory()
        except Exception as e:
            print(f"[Memory] ⚠️ Load error: {e}")
            return _empty_memory()

def _all_entries(memory: dict) -> list[tuple]:
    entries = []
    for cat, items in memory.items():
        if not isinstance(items, dict):
            continue
        for key, entry in items.items():
            if isinstance(entry, dict) and "value" in entry:
                entries.append((cat, key, entry))
    return entries


# Set by main.py so a trim can reach the activity log. Deleting something a
# person told you and mentioning it only on stdout is how a memory loses trust.
_trim_notifier = None


def set_trim_notifier(fn) -> None:
    """Register a callable(str) that surfaces trims to the user."""
    global _trim_notifier
    _trim_notifier = fn


def _trim_to_limit(memory: dict) -> dict:
    if len(json.dumps(memory, ensure_ascii=False)) <= MEMORY_MAX_CHARS:
        return memory
    entries = _all_entries(memory)
    entries.sort(key=lambda t: t[2].get("updated", "0000-00-00"))
    dropped = []
    for cat, key, _ in entries:
        if len(json.dumps(memory, ensure_ascii=False)) <= MEMORY_MAX_CHARS:
            break
        del memory[cat][key]
        dropped.append(f"{cat}/{key}")
        print(f"[Memory] 🗑️  Trimmed {cat}/{key}")
    if dropped and _trim_notifier:
        try:
            _trim_notifier(
                f"SYS: Memory full — forgot {len(dropped)} oldest entries "
                f"({', '.join(dropped[:3])}{'…' if len(dropped) > 3 else ''})"
            )
        except Exception:
            pass
    return memory

def save_memory(memory: dict) -> None:
    if not isinstance(memory, dict):
        return
    memory = _trim_to_limit(memory)
    MEMORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with _lock:
        MEMORY_PATH.write_text(
            json.dumps(memory, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    _IDX["sig"] = None            # never trust the stamp alone: two saves can share an mtime
    _memory_changed()


def _memory_changed() -> None:
    """Any cached AI answer may have quoted a fact that just changed."""
    try:
        from core.ai import get_hub
        get_hub().invalidate_cache()
    except Exception:
        pass


def _truncate_value(val: str) -> str:
    if isinstance(val, str) and len(val) > MAX_VALUE_LENGTH:
        return val[:MAX_VALUE_LENGTH].rstrip() + "…"
    return val


def _recursive_update(target: dict, updates: dict, depth: int = 0) -> bool:
    changed = False
    for key, value in updates.items():
        if value is None:
            continue
        if isinstance(value, str) and not value.strip():
            continue
        if isinstance(value, dict) and "value" not in value:
            if key not in target or not isinstance(target[key], dict):
                target[key] = {}
                changed = True
            if _recursive_update(target[key], value, depth + 1):
                changed = True
        else:
            new_val  = _truncate_value(str(value["value"] if isinstance(value, dict) else value))
            today    = datetime.now().strftime("%Y-%m-%d")
            existing = target.get(key, {})

            # The same fact stored under a different key ("sister_name" and
            # "sisters_name") used to become two entries that drift apart. If
            # another entry in this category already says this, refresh it.
            if depth == 1 and not (isinstance(existing, dict) and "value" in existing):
                twin = _find_twin(target, new_val)
                if twin is not None:
                    if target[twin].get("updated") != today:
                        target[twin]["updated"] = today
                        changed = True
                    continue

            if not isinstance(existing, dict) or existing.get("value") != new_val:
                entry = {"value": new_val, "updated": today}
                # A changed value is a CORRECTION: keep what it replaced, so an
                # outdated fact can be inspected or restored, but only the
                # current one is ever shown to the model.
                if isinstance(existing, dict) and existing.get("value"):
                    hist = list(existing.get("history") or [])
                    hist.append({"value": existing["value"],
                                 "updated": existing.get("updated", "")})
                    entry["history"] = hist[-5:]
                target[key] = entry
                changed = True
    return changed

def update_memory(memory_update: dict) -> dict:
    if not isinstance(memory_update, dict) or not memory_update:
        return load_memory()
    memory = load_memory()
    if _recursive_update(memory, memory_update):
        save_memory(memory)
        print(f"[Memory] 💾 Saved: {list(memory_update.keys())}")
    return memory

def _entry_value(entry) -> str:
    """Accept both the {'value': ..., 'updated': ...} shape and a bare string,
    because early versions of the store wrote plain strings."""
    if isinstance(entry, dict):
        return str(entry.get("value", "") or "").strip()
    return str(entry or "").strip()


def _pretty(key: str) -> str:
    return key.replace("_", " ").strip()


# Identity is always in the prompt; these categories compete for the remaining
# budget by recency.
_CATEGORY_LABELS = {
    "preferences":   "Preferences",
    "projects":      "Active projects / goals",
    "relationships": "People in their life",
    "wishes":        "Wishes / plans",
    "notes":         "Notes",
}

_IDENTITY_FIELDS = ["name", "age", "birthday", "city", "job",
                    "language", "school", "nationality"]


def format_memory_for_prompt(memory: dict | None) -> str:
    """Build the memory block that goes into the system prompt.

    This used to dump everything. It now sends three things:

      1. IDENTITY  - always, in full. It is small, and it is wrong for the
         assistant to have to look up your name.
      2. RECENT    - the most recently updated entries from every other
         category, up to PROMPT_CORE_CHARS. Recency is the cheapest useful
         relevance signal available without embeddings.
      3. AN INDEX  - the *keys* of everything else, values omitted.

    Point 3 is what makes recall work at all. A model cannot decide to look
    something up if it does not know the thing exists: with only points 1 and 2,
    "who is Ayse?" would get "I don't know" while ayse_sister sat on disk
    unread. The index costs a few hundred characters and turns recall from a
    gamble into a lookup.

    Net effect on latency: this block is SMALLER than the old full dump, so
    every session connects with fewer tokens. Occasionally the model spends one
    extra round trip on recall_memory - covered by the acknowledgment it
    already speaks before any slow step."""
    if not memory:
        return ""

    core_lines: list[str] = []

    # 1. Identity - always, in full
    identity = memory.get("identity", {}) or {}
    for field in _IDENTITY_FIELDS:
        val = _entry_value(identity.get(field))
        if not val:
            continue
        if field == "language":
            # Labelled as an observation, not a setting. A bare "Language:
            # English" line written months ago reads like a standing order and
            # was one of the reasons a Turkish question came back in English.
            core_lines.append(
                f"Has spoken to you in: {val} (an observation about the past — "
                f"always answer in the language of their CURRENT message)")
        else:
            core_lines.append(f"{field.title()}: {val}")
    for key, entry in identity.items():
        if key in _IDENTITY_FIELDS:
            continue
        val = _entry_value(entry)
        if val:
            core_lines.append(f"{_pretty(key).title()}: {val}")

    # 2. Everything else, most recently updated first
    rest: list[tuple[str, str, str, str]] = []   # (updated, cat, key, value)
    for cat in _CATEGORY_LABELS:
        for key, entry in (memory.get(cat, {}) or {}).items():
            val = _entry_value(entry)
            if not val:
                continue
            updated = (entry.get("updated", "") if isinstance(entry, dict) else "") or "0000-00-00"
            rest.append((updated, cat, key, val))
    rest.sort(key=lambda t: t[0], reverse=True)

    used    = sum(len(l) + 1 for l in core_lines)
    shown: dict[str, list[str]] = {}
    overflow: dict[str, list[str]] = {}

    # Recency decides order, but no single category may take the whole budget.
    # Without the cap, someone with forty stored preferences gets a prompt that
    # is forty preferences and not one person's name — the categories that
    # matter most in conversation are also the ones that change least often, so
    # pure recency systematically buries them.
    per_cat_used: dict[str, int] = {}
    for _updated, cat, key, val in rest:
        line = f"  - {_pretty(key).title()}: {val}"
        if (per_cat_used.get(cat, 0) < PROMPT_MAX_PER_CATEGORY
                and used + len(line) + 1 <= PROMPT_CORE_CHARS):
            shown.setdefault(cat, []).append(line)
            per_cat_used[cat] = per_cat_used.get(cat, 0) + 1
            used += len(line) + 1
        else:
            overflow.setdefault(cat, []).append(_pretty(key))

    # The index is a table of contents, so it is interleaved across categories
    # rather than continuing in recency order. Sorted by recency it would list
    # twenty-four preferences before the first relationship, and the one entry
    # the index exists for — the old fact the model has no other way to know
    # about — would fall off the end.
    indexed: list[str] = []
    if overflow:
        cats  = [c for c in _CATEGORY_LABELS if overflow.get(c)]
        cursor = {c: 0 for c in cats}
        while cats:
            for cat in list(cats):
                i = cursor[cat]
                if i >= len(overflow[cat]):
                    cats.remove(cat)
                    continue
                indexed.append(overflow[cat][i])
                cursor[cat] = i + 1

    for cat, label in _CATEGORY_LABELS.items():
        if shown.get(cat):
            core_lines.append("")
            core_lines.append(f"{label}:")
            core_lines.extend(shown[cat])

    if not core_lines and not indexed:
        return ""

    out = [
        "[WHAT YOU KNOW ABOUT THIS PERSON — use naturally, never recite like a list]",
        *core_lines,
    ]

    # 3. The index of what is on disk but not in this prompt
    if indexed:
        budget, names = PROMPT_INDEX_CHARS, []
        for n in indexed:
            if budget - len(n) - 2 < 0:
                break
            names.append(n)
            budget -= len(n) + 2
        if names:
            out.append("")
            out.append(
                "[ALSO REMEMBERED — values not shown here. Call recall_memory "
                "with a keyword to read any of these before saying you do not know]"
            )
            out.append(", ".join(names)
                       + (f" (+{len(indexed) - len(names)} more)"
                          if len(indexed) > len(names) else ""))

    return "\n".join(out) + "\n"


# ── Recall ────────────────────────────────────────────────────────────────────

def _score(query_words: list[str], cat: str, key: str, value: str) -> int:
    """Cheap lexical relevance. No embeddings, no network, no model call - this
    runs in well under a millisecond, which is the entire point: recall must
    cost one model round trip, never two."""
    hay_key = _pretty(key).lower()
    hay_val = value.lower()
    score   = 0
    for w in query_words:
        if not w:
            continue
        if w == hay_key:
            score += 10
        elif w in hay_key:
            score += 6
        if w in hay_val:
            score += 3
        if w in cat:
            score += 1
    return score


def search_memory(query: str, limit: int = 8) -> str:
    """Find stored facts matching `query`. Backs the recall_memory tool.

    An empty query is treated as "show me everything you know", capped - the
    model asks that when the user says "what do you remember about me?"."""
    rows = search_entries(query, limit=10_000)
    if not rows:
        return (f"Nothing stored about '{query}'." if query
                else "I have not stored anything about this person yet.")
    shown = rows[:max(1, limit)]
    lines = [f"{r['category']}/{_pretty(r['key'])}: {r['value']}" for r in shown]
    head  = (f"Stored facts matching '{query}':" if query
             else "Everything currently stored:")
    more  = (f"\n(+{len(rows) - len(lines)} more — search with a narrower keyword)"
             if len(rows) > len(lines) else "")
    return head + "\n" + "\n".join(lines) + more


def all_entries_for_ui() -> list[dict]:
    """Flat list for the memory panel: what JARVIS knows, and when it learned it.
    Sorted newest first so the panel opens on what changed most recently."""
    memory = load_memory()
    rows = []
    for cat, items in memory.items():
        if not isinstance(items, dict):
            continue
        for key, entry in items.items():
            val = _entry_value(entry)
            if not val:
                continue
            rows.append({
                "category": cat,
                "key":      key,
                "value":    val,
                "updated":  (entry.get("updated", "") if isinstance(entry, dict) else ""),
            })
    rows.sort(key=lambda r: (r["updated"] or "0000-00-00"), reverse=True)
    return rows

def remember(key: str, value: str, category: str = "notes") -> str:
    valid = {"identity", "preferences", "projects", "relationships", "wishes", "notes"}
    if category not in valid:
        category = "notes"
    update_memory({category: {key: {"value": value}}})
    return f"Remembered: {category}/{key} = {value}"


def forget(key: str, category: str = "notes") -> str:
    memory = load_memory()
    cat    = memory.get(category, {})
    if key in cat:
        del cat[key]
        memory[category] = cat
        save_memory(memory)
        return f"Forgotten: {category}/{key}"
    return f"Not found: {category}/{key}"


forget_memory = forget


# ── Ranking, dedupe, editing and approved learning ────────────────────────────

_STOP = frozenset("""a an and are as at be but by do does for from had has have he her his i if in
into is it its me my of on or our she so than that the their them then there these they this to
was we were what when where which who why will with you your about tell know""".split())


def _stem(w: str) -> str:
    """Deliberately light: enough that 'sisters', 'sister' and 'sister's' meet."""
    if len(w) > 5 and w.endswith("ies"):
        return w[:-3] + "y"
    if len(w) > 4 and w.endswith("ie"):        # hoodie / hoodies meet at 'hoody'
        return w[:-2] + "y"
    for suf in ("ing", "ed", "es", "s"):
        if len(w) > len(suf) + 3 and w.endswith(suf):
            return w[: -len(suf)]
    return w


def _tokens(text: str) -> list[str]:
    return [_stem(w) for w in re.findall(r"[^\W_]+", (text or "").lower())
            if len(w) > 1 and w not in _STOP]


def _norm(text: str) -> str:
    return " ".join(re.findall(r"[^\W_]+", (text or "").lower()))


def _similar(a: str, b: str) -> bool:
    na, nb = _norm(a), _norm(b)
    if not na or not nb:
        return False
    if na == nb:
        return True
    ta, tb = set(_tokens(a)), set(_tokens(b))
    if len(ta) < 3 or len(tb) < 3:
        return False
    return len(ta & tb) / len(ta | tb) >= 0.8


def _find_twin(category: dict, value: str):
    for k, e in category.items():
        if isinstance(e, dict) and "value" in e and _similar(e["value"], value):
            return k
    return None


_IDX: dict = {"sig": None, "memory": None, "docs": None, "inv": None, "avg": 1.0}


def _build_index(memory: dict) -> tuple[list, dict, float]:
    """Documents plus an inverted index (term -> [(doc, weight)]) so a query
    touches only the entries that contain its words instead of re-tokenising
    the whole store on every call."""
    docs, inv = [], {}
    for cat, items in memory.items():
        if not isinstance(items, dict):
            continue
        for key, entry in items.items():
            val = _entry_value(entry)
            if not val:
                continue
            kt, vt, ct = _tokens(_pretty(key)), _tokens(val), _tokens(cat)
            tf = _tf(kt, vt, ct)
            di = len(docs)
            docs.append({"category": cat, "key": key, "value": val,
                         "updated": (entry.get("updated", "") if isinstance(entry, dict) else ""),
                         "len": len(kt) * 3 + len(vt) + len(ct),
                         "key_norm": _norm(_pretty(key))})
            for term, w in tf.items():
                inv.setdefault(term, []).append((di, w))
    avg = (sum(d["len"] for d in docs) / len(docs)) if docs else 1.0
    return docs, inv, (avg or 1.0)


def _memory_signature():
    try:
        stt = MEMORY_PATH.stat()
        return (str(MEMORY_PATH), stt.st_mtime_ns, stt.st_size)
    except OSError:
        return (str(MEMORY_PATH), None, None)


def search_entries(query: str, limit: int = 8, memory: dict | None = None) -> list[dict]:
    """Ranked entries for `query` (BM25 over key, value and category, with a
    stemmer, stop-words, prefix matching and a small recency boost). No
    embeddings and no network. The index is cached against the memory file's
    modification stamp, so it is rebuilt only when something was saved."""
    import math
    if memory is None:
        sig = _memory_signature()
        if _IDX["sig"] != sig or _IDX["docs"] is None:
            mem = load_memory()
            _IDX.update(sig=sig, memory=mem)
            _IDX["docs"], _IDX["inv"], _IDX["avg"] = _build_index(mem)
        docs, inv, avg = _IDX["docs"], _IDX["inv"], _IDX["avg"]
    else:
        docs, inv, avg = _build_index(memory)

    q = _tokens(query)
    if not q:
        rows = sorted(docs, key=lambda d: d["updated"] or "0000-00-00", reverse=True)
        return [_public(d) for d in rows[:limit]]
    n = len(docs) or 1
    scores: dict[int, float] = {}
    for t in dict.fromkeys(q):
        postings = list(inv.get(t, ()))
        if len(t) >= 4:                       # prefix: 'birth' finds 'birthday'
            for term, plist in inv.items():
                if term != t and term.startswith(t):
                    postings.extend((di, w * 0.6) for di, w in plist)
        if not postings:
            continue
        per_doc: dict[int, float] = {}
        for di, w in postings:
            per_doc[di] = per_doc.get(di, 0.0) + w
        idf = math.log(1 + (n - len(per_doc) + 0.5) / (len(per_doc) + 0.5))
        for di, f in per_doc.items():
            scores[di] = scores.get(di, 0.0) + idf * (f * 2.2) / (
                f + 1.2 * (0.25 + 0.75 * docs[di]["len"] / avg))
    if not scores:
        return []
    today = datetime.now().date()
    qn = _norm(query)
    scored = []
    for di, sc in scores.items():
        d = docs[di]
        if qn and qn == d["key_norm"]:
            sc *= 1.5
        try:
            age = (today - datetime.strptime(d["updated"], "%Y-%m-%d").date()).days
            sc *= 1.0 + 0.10 * max(0.0, 1.0 - age / 90.0)
        except ValueError:
            pass
        scored.append((sc, d))
    scored.sort(key=lambda x: (-x[0], x[1]["key"]))
    return [_public(d) for _s, d in scored[:limit]]


def _tf(kt, vt, ct) -> dict:
    tf: dict[str, float] = {}
    for t in kt:
        tf[t] = tf.get(t, 0) + 3.0      # a match in the KEY is worth three in the value
    for t in vt:
        tf[t] = tf.get(t, 0) + 1.0
    for t in ct:
        tf[t] = tf.get(t, 0) + 0.5
    return tf


def _hit(term: str, tf: dict) -> float:
    if term in tf:
        return tf[term]
    if len(term) >= 4:                   # prefix: 'birth' finds 'birthday'
        return sum(v for k, v in tf.items() if k.startswith(term)) * 0.6
    return 0.0


def _public(d: dict) -> dict:
    return {k: d[k] for k in ("category", "key", "value", "updated")}


def relevant_context(query: str, max_chars: int = 600) -> str:
    """The few stored facts worth sending with THIS request, as a compact block
    for `Request.private_context`. Identity first, then best matches."""
    memory = load_memory()
    lines, used = [], 0
    for field in ("name", "language"):
        v = _entry_value((memory.get("identity") or {}).get(field))
        if v:
            lines.append(f"{field}: {v}")
            used += len(lines[-1]) + 1
    for r in search_entries(query, limit=8, memory=memory):
        line = f"{_pretty(r['key'])}: {r['value']}"
        if used + len(line) + 1 > max_chars:
            break
        if line not in lines:
            lines.append(line)
            used += len(line) + 1
    return ("Known about the user:\n" + "\n".join(lines)) if lines else ""


def history_of(key: str, category: str = "notes") -> list[dict]:
    """Earlier values of a fact that was corrected, newest last."""
    e = (load_memory().get(category) or {}).get(key)
    return list(e.get("history") or []) if isinstance(e, dict) else []


def edit_entry(category: str, key: str, value: str | None = None,
               new_key: str | None = None) -> str:
    """User-driven edit of one fact: change its value and/or rename its key."""
    memory = load_memory()
    cat = memory.get(category)
    if not isinstance(cat, dict) or key not in cat:
        return f"Not found: {category}/{key}"
    entry = cat[key] if isinstance(cat[key], dict) else {"value": str(cat[key])}
    if value is not None and value.strip():
        hist = list(entry.get("history") or [])
        if entry.get("value") and entry["value"] != value:
            hist.append({"value": entry["value"], "updated": entry.get("updated", "")})
        entry = {**entry, "value": _truncate_value(value.strip()),
                 "updated": datetime.now().strftime("%Y-%m-%d"), "history": hist[-5:]}
    if new_key and new_key != key:
        del cat[key]
        key = new_key
    cat[key] = entry
    save_memory(memory)
    return f"Updated: {category}/{key}"


delete_entry = forget


# Approved learning. The assistant may NOTICE "I prefer short answers", but a
# noticed preference sits in a pending list until the user says yes; nothing is
# remembered permanently on the strength of an overheard sentence. Note this is
# a notebook, not training: no model is changed, only what it is told.
PENDING_PATH = BASE_DIR / "memory" / "pending.json"
_PENDING_MAX = 20
_LEARN = [
    (re.compile(r"\bfrom now on[, ]+(.{8,140})", re.I), "preferences"),
    (re.compile(r"\bi (?:really )?(?:prefer|always|never|usually|hate|dislike|love)\b[^.!?]{4,120}", re.I),
     "preferences"),
    (re.compile(r"\bmy name is ([A-Z][\w'-]{1,30})"), "identity"),
    (re.compile(r"\bcall me ([A-Z][\w'-]{1,30})"), "identity"),
    (re.compile(r"\bi (?:work|study) (?:at|as|in) [^.!?]{3,80}", re.I), "identity"),
    (re.compile(r"\bmy (?:wife|husband|sister|brother|mother|mum|dad|father|friend|boss) "
                r"(?:is )?(?:called |named )?[A-Z][\w'-]{1,30}"), "relationships"),
]


def _pending() -> list[dict]:
    try:
        data = json.loads(PENDING_PATH.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else []
    except Exception:
        return []


def _save_pending(rows: list[dict]) -> None:
    PENDING_PATH.parent.mkdir(parents=True, exist_ok=True)
    PENDING_PATH.write_text(json.dumps(rows[-_PENDING_MAX:], indent=2, ensure_ascii=False),
                            encoding="utf-8")


def observe(text: str) -> list[dict]:
    """Look at something the user said and queue anything worth remembering as
    PENDING. Returns the new candidates. Stores nothing permanently."""
    text = (text or "").strip()
    if len(text) < 8 or len(text) > 600:
        return []
    memory = load_memory()
    rows = _pending()
    added = []
    for pat, cat in _LEARN:
        m = pat.search(text)
        if not m:
            continue
        value = _truncate_value(m.group(0).strip(" ,.;"))
        if any(_similar(value, r["value"]) for r in rows):
            continue
        if any(_similar(value, _entry_value(e)) for e in (memory.get(cat) or {}).values()):
            continue                                  # already known
        words = [w for w in _tokens(value)][:3] or ["note"]
        row = {"id": hex(abs(hash((value, datetime.now().timestamp()))))[2:10],
               "category": cat, "key": "_".join(words), "value": value,
               "created": datetime.now().strftime("%Y-%m-%d")}
        rows.append(row)
        added.append(row)
    if added:
        _save_pending(rows)
    return added


def list_pending() -> list[dict]:
    return _pending()


def approve_pending(pid: str) -> str:
    rows = _pending()
    for r in rows:
        if r["id"] == pid or pid == "all":
            remember(r["key"], r["value"], r["category"])
            if pid != "all":
                _save_pending([x for x in rows if x["id"] != pid])
                return f"Remembered: {r['value']}"
    if pid == "all":
        n = len(rows)
        _save_pending([])
        return f"Remembered {n} things."
    return "No such pending item."


def reject_pending(pid: str) -> str:
    rows = _pending()
    keep = [r for r in rows if r["id"] != pid] if pid != "all" else []
    _save_pending(keep)
    return f"Discarded {len(rows) - len(keep)}."


# ── Session memory ─────────────────────────────────────────────────────────────

_SESSION_MAX = 3   # safety cap — in practice 0-1 entries after pop


def save_session_summary(summary: str, language: str = "") -> None:
    """Append a 1-2 sentence session summary to long_term.json['sessions']."""
    summary = (summary or "").strip()
    if not summary:
        return
    memory   = load_memory()
    sessions = memory.get("sessions", [])
    if not isinstance(sessions, list):
        sessions = []
    entry: dict = {
        "date":    datetime.now().strftime("%Y-%m-%d"),
        "summary": summary[:280],
    }
    if language:
        entry["language"] = language
    sessions.append(entry)
    memory["sessions"] = sessions[-_SESSION_MAX:]
    with _lock:
        MEMORY_PATH.parent.mkdir(parents=True, exist_ok=True)
        MEMORY_PATH.write_text(
            json.dumps(memory, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    print(f"[Memory] 📝 Session saved ({entry['date']}): {summary[:60]}…")


def pop_last_session() -> dict | None:
    """
    Return AND remove the most recent session entry.
    Calling this consumes the entry so it is never repeated in future briefings.
    """
    with _lock:
        if not MEMORY_PATH.exists():
            return None
        try:
            memory   = json.loads(MEMORY_PATH.read_text(encoding="utf-8"))
            sessions = memory.get("sessions", [])
            if not isinstance(sessions, list) or not sessions:
                return None
            entry = sessions.pop()          # remove the last entry
            memory["sessions"] = sessions
            MEMORY_PATH.write_text(
                json.dumps(memory, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            return entry
        except Exception as e:
            print(f"[Memory] ⚠️ pop_last_session error: {e}")
            return None