# Ephraim AI — Intelligence & Performance Upgrade

## What changed, in one paragraph
A new `core/ai/` package gives every AI call one front door: provider adapters
(Gemini, Anthropic, OpenAI-compatible — OpenAI, Groq, OpenRouter, LM Studio, any
URL you add — and Ollama), a router that picks the cheapest model that is good
enough, bounded failover, token budgets, caching, usage/cost tracking, and a
hub that can ask several AIs at once or let them discuss. Memory recall is
ranked and editable, learning is opt-in, and the avatar has a realistic face.
**Existing features are untouched**: the Gemini Live voice session, the Gemini
model ladder, the HUD, plugins, actions and computer control all run as before.

## Audit findings (what was found before anything was changed)
| # | Finding | Action |
|---|---|---|
| 1 | `core/llm_client.py` (Ollama/OpenAI-local) had **no callers** — dead code | Superseded by adapters; file kept, Ollama auto-start reused |
| 2 | 29 call sites go through `core.gemini.call`; when every Gemini rung failed the feature simply failed | Optional cross-provider fallback added inside `gemini.call` |
| 3 | No usage/latency/cost record anywhere | `logs/ai_usage.jsonl` + summary |
| 4 | Memory recall = substring score; "my sisters" missed "sister name"; filler words ("my") matched everything | BM25 + stemming + prefix + recency (see benchmark) |
| 5 | Same fact under two keys became two drifting entries; corrections overwrote silently | De-duplication; corrections keep history |
| 6 | `open_app` ran model-supplied names through `shell=True` on Windows (command injection) | **Fixed**; 10 tests |
| 7 | API key in plaintext `config/api_keys.json` | Gitignored already; other providers' keys go to OS keyring / env, never to config |
| 8 | `actions/desktop.py` executes model-written code in a restricted namespace; `dev_agent.py` runs shell commands | **Not changed** — flagged, see "Open items" |
| 9 | System prompt (≈2.3k tokens) | Audited: already lean, no duplication worth removing; left alone |

## Using several AIs
Settings drawer → **🤖 AI PROVIDERS**, or `python -m core.ai status`.
1. Select a provider → **Set key…** (hidden input; stored in the OS keyring, else `config/secrets.json`; or set `GROQ_API_KEY` etc. in the environment).
2. Tick **On**, set **Priority** (lower runs first), edit **Models** as `id:tier` (0 light, 1 standard, 2 strong), press **Fetch models** to see what the provider really offers, **Test**, **Save**.
3. **Add provider…** takes any OpenAI-compatible base URL — that is how you integrate anything of your choice.

Voice examples (the `multi_ai` tool): *"ask Claude and Groq what they think of X"* (compare) · *"have them discuss whether…"* (discuss) · *"draft it with Gemini and have Claude review it"* (chain) · *"which AIs are available?"* (status).
Named agents with roles go in `config/providers.json` → `"agents"` (see `providers.example.json`).

**Safety of AI-to-AI talk.** Discussions are text-only: no participant gets tools, so none can make another act on the computer. Each AI receives the others' words quoted as untrusted data, rounds are capped (≤6) and a total token budget ends the talk early; everyone saying `AGREED:` ends it sooner.

## Routing rules
* Capabilities are filtered first (vision / tools / JSON / audio) — a model lacking one is never tried.
* Tier is estimated free of charge from the task hint, length, code, reasoning keywords and images. Simple → light; code → standard; hard reasoning/planning → strong.
* Among adequate models: `strategy=cost` (default) cheapest first (local = free), `priority` your order, `speed` measured latency.
* Failure handling: rate-limit → switch provider immediately (cooldown from `Retry-After`); timeout/outage → one jittered retry then switch; auth/not-found → 6 h cooldown; bad request → no cooldown. Hard bounds: `max_attempts` (4) and `deadline_s` (45). A weaker model is used only if nothing adequate exists, or if you enable `allow_downgrade`.
* `mode=manual` pins a model; fallback still applies unless you disable it.
* `monthly_cost_cap_usd` restricts routing to free/local models once reached.

## Token & cost controls
Input/output budgets per request · history trimmed to budget with older turns summarised (model-free extractive fallback) · duplicate tool outputs collapsed · long tool results compacted · only relevant tool definitions sent when >12 offered · only relevant memories sent (`memory.relevant_context`) and only to providers you marked **Memory** · cache for repeatable tasks (classify/extract/summarize/translate/rewrite) invalidated whenever memory changes · Anthropic prompt caching for long stable system prompts · actual token counts used whenever the provider reports them, estimates flagged `est`.

## Memory
Ranked recall, de-duplication, corrections keep up to 5 earlier values (`history_of`), `edit_entry`/`delete_entry`. **Learning is opt-in per item:** preferences you state are queued in `memory/pending.json`; say *"review what you learned"* (tool `memory_review`) to approve, reject, edit or delete. Nothing is stored on the strength of an overheard sentence. *This is a notebook, not training: no model's weights ever change.*

## Face
**⚙ CONTROLS → 🧑 FACE** toggles REALISTIC / HOLOGRAM (default realistic). Real eyes (sclera, iris with fibres, limbal ring, pupil, catchlights, lashes, lid crease, socket shadow) that keep looking at you as the head turns, brows, lips with teeth and tongue when speaking, hair, lit skin shaded per pixel. Lip-sync, gaze, blink and status behaviour are shared with the hologram. Skin/eye colour: `avatar_skin` / `avatar_eye` (hex) in `config/api_keys.json`.

## Security
Keys: env → OS keyring → `config/secrets.json` (0600, gitignored); never in `providers.json`, logs, prompts or the dashboard. Every error string passes through `redact()`. Provider replies are size-bounded, control characters stripped, and any tool call the model invented (not offered) is dropped. Private context is sent only to providers with **Memory** on. `open_app` no longer uses a shell.

## Migration & rollback
First run creates `config/providers.json` from `api_keys.json` (Gemini key honoured; old `llm_*` settings become an Ollama/LM Studio provider). `api_keys.json` is never rewritten by this code. **Rollback:** delete `config/providers.json`; to remove the layer entirely revert the commit — nothing else depends on it.

## Tests, evals, benchmarks
```
pip install -r requirements-dev.txt
python -m pytest tests -q          # offline, mocked, no cost
python -m evals.run                # regression eval (mock);  --live uses your providers
python bench/benchmark.py --baseline /path/to/original/checkout
```

## Open items / honest limits
* **No real provider was called** (no keys were available): adapters are verified against mocked wire formats, not live servers. First live run: `python -m core.ai test`.
* The **Gemini Live voice conversation itself stays Gemini-only** — it is native audio streaming and no other provider offers the same session. Other AIs join through tools (`multi_ai`) and as automatic fallback for the 29 side calls.
* Default model ids/prices are starting points; vendors rename and reprice. Use **Fetch models** and edit.
* Windows-specific paths (`os.startfile`) and the avatar were tested on Linux/offscreen only.
* `desktop.py` code execution and `dev_agent.py` shell use remain; they warrant a confirmation prompt (`core/confirm.py` exists) — recommended next step.
* Realistic face: frame cost ≈16 ms at HUD size vs ≈8 ms for the hologram; switch to hologram on slow machines.

## Career agent (`career_agent` tool)
* **coach** — looks through the camera or the screen and says what matters for connecting and the single best next action. Frames go only to local or *Memory*-trusted providers. It never identifies a stranger from their face.
* **draft** / **plan** — writes one outreach message, or a two-week networking plan. Nothing is sent.
* **send** — hands a drafted message to `send_message`, but only after you confirm on the HUD (`core/confirm`). Named agents with roles (networker, mentor, editor) can be added under `"agents"` and used with `multi_ai`.
* Typing into Word etc. uses the existing `computer_control` tool and zip handling uses `file_controller`. **Not built:** sending PDF/photo/zip attachments through chat apps — it needs fragile UI automation I could not test here.

## Facial reactions
`HoloAvatar.react(name)` eases into happy / laugh / surprised / thinking / concerned / curious / serious / agree and back (smile, brows, eye widening, jaw, **head turn**, gaze), and `react_to_text` picks one from what the assistant just said (wired in `main.py`). Listening and thinking states show curious and thinking automatically.

## What was deliberately not built
A "no rules / uncensored" mode. The assistant keeps its safety behaviour; it can still be blunt, candid and direct, which is configurable in `core/prompt.txt`.

## Reports and images
* [Test & benchmark report](TEST_REPORT.md)
* Face before: `images/face_before.png` · after: `images/face_after.png` · states: `images/face_after_states.png` · expressions: `images/face_expressions.png`

![expressions](images/face_expressions.png)
