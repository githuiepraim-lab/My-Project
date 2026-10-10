"""
Provider configuration, secret storage and migration.

FILES
    config/providers.json   what providers exist, their models, priorities and
                            routing settings. Contains NO secrets.
    config/secrets.json     API keys, only when no OS keyring is available.
                            Gitignored; written with owner-only permissions.
    config/api_keys.json    the legacy file. Read for the Gemini key and the
                            old llm_* settings, never rewritten by this module.

KEY LOOKUP ORDER   environment variable  ->  OS keyring  ->  secrets.json
                   (-> api_keys.json for Gemini only)
"""
from __future__ import annotations

import copy
import json
import os
import re
import sys
import threading
from pathlib import Path

from .redact import register_secret

if getattr(sys, "frozen", False):
    BASE_DIR = Path(sys.executable).parent
else:
    BASE_DIR = Path(__file__).resolve().parent.parent.parent

CONFIG_DIR = BASE_DIR / "config"
PROVIDERS_FILE = CONFIG_DIR / "providers.json"
SECRETS_FILE = CONFIG_DIR / "secrets.json"
LEGACY_FILE = CONFIG_DIR / "api_keys.json"
KEYRING_SERVICE = "ephraim-ai"

_lock = threading.RLock()
_cache: dict | None = None
_cache_mtime: float = -1.0

_NAME_RE = re.compile(r"^[a-z][a-z0-9_\-]{0,31}$")

# Capability shorthands used in the catalogue below.
_T = ["text", "stream"]
_TV = ["text", "stream", "vision"]
_TVJ = ["text", "stream", "vision", "tools", "json"]
_TJ = ["text", "stream", "tools", "json"]

# Defaults. Model ids and prices are STARTING POINTS the user can edit in the
# settings dialog or providers.json; vendors rename and reprice models, so the
# dialog's "Fetch models" button asks the provider what it really offers.
_DEFAULT_PROVIDERS: dict = {
    "gemini": {
        "kind": "gemini", "enabled": False, "priority": 10, "share_memory": True,
        "models": [
            {"id": "gemini-2.5-flash-lite", "tier": 0, "caps": _TVJ,
             "price_in": 0.10, "price_out": 0.40, "context": 1_000_000},
            {"id": "gemini-2.5-flash", "tier": 1, "caps": _TVJ,
             "price_in": 0.30, "price_out": 2.50, "context": 1_000_000},
        ],
    },
    "anthropic": {
        "kind": "anthropic", "enabled": False, "priority": 20, "share_memory": False,
        "models": [
            {"id": "claude-haiku-5-5", "tier": 0, "caps": _TVJ, "context": 200_000},
            {"id": "claude-sonnet-5-5", "tier": 1, "caps": _TVJ, "context": 200_000},
            {"id": "claude-opus-5-5", "tier": 2, "caps": _TVJ, "context": 200_000},
        ],
    },
    "openai": {
        "kind": "openai_compat", "enabled": False, "priority": 30, "share_memory": False,
        "base_url": "https://api.openai.com/v1", "max_tokens_param": "max_completion_tokens",
        "stream_usage": True,
        "models": [
            {"id": "gpt-4o-mini", "tier": 0, "caps": _TVJ, "price_in": 0.15, "price_out": 0.60,
             "context": 128_000},
            {"id": "gpt-4o", "tier": 1, "caps": _TVJ, "price_in": 2.50, "price_out": 10.0,
             "context": 128_000},
        ],
    },
    "groq": {
        "kind": "openai_compat", "enabled": False, "priority": 40, "share_memory": False,
        "base_url": "https://api.groq.com/openai/v1", "stream_usage": True,
        "models": [
            {"id": "llama-3.1-8b-instant", "tier": 0, "caps": _TJ, "price_in": 0.05,
             "price_out": 0.08, "context": 128_000},
            {"id": "llama-3.3-70b-versatile", "tier": 1, "caps": _TJ, "price_in": 0.59,
             "price_out": 0.79, "context": 128_000},
        ],
    },
    "openrouter": {
        "kind": "openai_compat", "enabled": False, "priority": 50, "share_memory": False,
        "base_url": "https://openrouter.ai/api/v1", "stream_usage": True,
        "models": [
            {"id": "openrouter/auto", "tier": 1, "caps": _TVJ, "context": 128_000},
        ],
    },
    "ollama": {
        "kind": "ollama", "enabled": False, "priority": 5, "share_memory": True,
        "base_url": "http://localhost:11434", "requires_key": False,
        "models": [{"id": "llama3.2", "tier": 0, "caps": _T + ["tools"], "local": True}],
    },
    "lmstudio": {
        "kind": "openai_compat", "enabled": False, "priority": 6, "share_memory": True,
        "base_url": "http://localhost:1234/v1", "requires_key": False, "stream_usage": False,
        "models": [{"id": "local-model", "tier": 0, "caps": _T + ["tools"], "local": True}],
    },
}

DEFAULT_SETTINGS = {
    "strategy": "cost",            # cost | priority | speed
    "mode": "auto",                # auto | manual
    "manual_model": "",            # "provider/model" when mode == manual
    "fallback": True,
    "max_attempts": 4,
    "deadline_s": 45,
    "request_timeout_s": 25,
    "allow_downgrade": False,      # may a failed strong model fall to a weaker one?
    "cross_provider_fallback": True,   # core.gemini ladder exhausted -> other providers
    "cache": True,
    "cache_ttl_s": 900,
    "max_input_tokens": 6000,
    "max_output_tokens": 800,
    "monthly_cost_cap_usd": None,
    "share_memory_default": False,
}


def default_config() -> dict:
    return {"version": 1, "settings": dict(DEFAULT_SETTINGS),
            "providers": copy.deepcopy(_DEFAULT_PROVIDERS), "agents": {}}


# ── load / save / migrate ───────────────────────────────────────────────────

def _read_json(path: Path) -> dict:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _migrate_from_legacy() -> dict:
    """Build providers.json from the old api_keys.json. Additive: the legacy
    file is left exactly as it was, so rolling back is just deleting
    providers.json."""
    cfg = default_config()
    legacy = _read_json(LEGACY_FILE)
    if legacy.get("gemini_api_key"):
        cfg["providers"]["gemini"]["enabled"] = True
    prov = str(legacy.get("llm_provider", "")).strip().lower()
    if prov or legacy.get("llm_url") or legacy.get("llm_model"):
        url = str(legacy.get("llm_url") or "http://localhost:11434").rstrip("/")
        model = str(legacy.get("llm_model") or "llama3.2")
        if prov in ("openai", "lmstudio", "localai", "jan", "llamacpp"):
            p = cfg["providers"]["lmstudio"]
            p["base_url"] = url if url.endswith("/v1") else url + "/v1"
        else:
            p = cfg["providers"]["ollama"]
            p["base_url"] = url
        p["models"] = [{"id": model, "tier": 0, "caps": _T + ["tools"], "local": True}]
        p["enabled"] = True
    return cfg


def _validate(cfg: dict) -> dict:
    """Repair rather than reject: a hand-edited file with one bad entry must
    not take the whole assistant's AI layer down."""
    base = default_config()
    if not isinstance(cfg, dict):
        return base
    settings = dict(DEFAULT_SETTINGS)
    if isinstance(cfg.get("settings"), dict):
        settings.update(cfg["settings"])
    providers = {}
    for name, p in (cfg.get("providers") or {}).items():
        if not isinstance(p, dict) or not _NAME_RE.match(str(name)):
            continue
        if p.get("kind") not in ("gemini", "anthropic", "openai_compat", "ollama"):
            continue
        models = [m for m in (p.get("models") or [])
                  if isinstance(m, dict) and str(m.get("id", "")).strip()]
        q = dict(p)
        q["models"] = models
        q["enabled"] = bool(p.get("enabled", False))
        try:
            q["priority"] = int(p.get("priority", 100))
        except (TypeError, ValueError):
            q["priority"] = 100
        providers[name] = q
    agents = cfg.get("agents") if isinstance(cfg.get("agents"), dict) else {}
    return {"version": 1, "settings": settings, "providers": providers, "agents": agents}


def load(force: bool = False) -> dict:
    """The current config, migrating from the legacy file on first use. Cached
    and reloaded automatically when providers.json changes on disk."""
    global _cache, _cache_mtime
    with _lock:
        try:
            mtime = PROVIDERS_FILE.stat().st_mtime
        except OSError:
            mtime = -1.0
        if _cache is not None and not force and mtime == _cache_mtime:
            return _cache
        if mtime < 0:
            cfg = _validate(_migrate_from_legacy())
            try:
                save(cfg)
                mtime = PROVIDERS_FILE.stat().st_mtime
            except Exception:
                pass
        else:
            try:
                raw = json.loads(PROVIDERS_FILE.read_text(encoding="utf-8"))
                if not isinstance(raw, dict):
                    raise ValueError("not an object")
                cfg = _validate(raw)
            except Exception:
                # Hand-edit gone wrong: keep the broken file for the user to
                # fix, run on the defaults instead of on nothing.
                try:
                    os.replace(PROVIDERS_FILE, PROVIDERS_FILE.with_suffix(".json.bad"))
                except OSError:
                    pass
                cfg = _validate(_migrate_from_legacy())
                try:
                    save(cfg)
                    mtime = PROVIDERS_FILE.stat().st_mtime
                except Exception:
                    pass
        _cache, _cache_mtime = cfg, mtime
        return cfg


def save(cfg: dict) -> None:
    global _cache, _cache_mtime
    cfg = _validate(cfg)
    with _lock:
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        tmp = PROVIDERS_FILE.with_suffix(".tmp")
        tmp.write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, PROVIDERS_FILE)       # atomic: never a half-written file
        _cache = cfg
        try:
            _cache_mtime = PROVIDERS_FILE.stat().st_mtime
        except OSError:
            _cache_mtime = -1.0


def update_provider(name: str, **fields) -> None:
    cfg = copy.deepcopy(load())
    p = cfg["providers"].setdefault(name, {})
    p.update(fields)
    save(cfg)


def update_settings(**fields) -> None:
    cfg = copy.deepcopy(load())
    cfg["settings"].update(fields)
    save(cfg)


# ── secrets ─────────────────────────────────────────────────────────────────

def _env_names(provider: str) -> list[str]:
    up = re.sub(r"[^A-Z0-9]", "_", provider.upper())
    return [f"EPHRAIM_{up}_API_KEY", f"{up}_API_KEY"] + (
        ["GOOGLE_API_KEY"] if provider == "gemini" else [])


def _keyring():
    try:
        import keyring  # optional dependency
        return keyring
    except Exception:
        return None


def get_api_key(provider: str) -> str:
    """The key for `provider`, or "". Never raises, never logs."""
    for n in _env_names(provider):
        v = os.environ.get(n, "").strip()
        if v:
            register_secret(v)
            return v
    kr = _keyring()
    if kr is not None:
        try:
            v = kr.get_password(KEYRING_SERVICE, provider) or ""
            if v:
                register_secret(v)
                return v
        except Exception:
            pass
    v = str(_read_json(SECRETS_FILE).get(provider, "") or "").strip()
    if v:
        register_secret(v)
        return v
    if provider == "gemini":
        v = str(_read_json(LEGACY_FILE).get("gemini_api_key", "") or "").strip()
        if v:
            register_secret(v)
            return v
    return ""


def set_api_key(provider: str, key: str) -> str:
    """Store a key. Returns where it went: "keyring" or "secrets.json"."""
    key = (key or "").strip()
    if not key:
        raise ValueError("empty key")
    register_secret(key)
    kr = _keyring()
    if kr is not None:
        try:
            kr.set_password(KEYRING_SERVICE, provider, key)
            return "keyring"
        except Exception:
            pass
    with _lock:
        data = _read_json(SECRETS_FILE)
        data[provider] = key
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        tmp = SECRETS_FILE.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
        try:
            os.chmod(tmp, 0o600)       # no-op on Windows, where ACLs apply
        except OSError:
            pass
        os.replace(tmp, SECRETS_FILE)
    return "secrets.json"


def delete_api_key(provider: str) -> None:
    kr = _keyring()
    if kr is not None:
        try:
            kr.delete_password(KEYRING_SERVICE, provider)
        except Exception:
            pass
    with _lock:
        data = _read_json(SECRETS_FILE)
        if provider in data:
            del data[provider]
            SECRETS_FILE.write_text(json.dumps(data, indent=2), encoding="utf-8")


def key_source(provider: str) -> str:
    """Where the key for `provider` comes from, WITHOUT revealing it."""
    for n in _env_names(provider):
        if os.environ.get(n, "").strip():
            return f"env:{n}"
    kr = _keyring()
    if kr is not None:
        try:
            if kr.get_password(KEYRING_SERVICE, provider):
                return "keyring"
        except Exception:
            pass
    if _read_json(SECRETS_FILE).get(provider):
        return "secrets.json"
    if provider == "gemini" and _read_json(LEGACY_FILE).get("gemini_api_key"):
        return "api_keys.json"
    return ""


def is_available(name: str, p: dict) -> bool:
    if not p.get("enabled"):
        return False
    if p.get("requires_key", True) is False:
        return True
    return bool(get_api_key(name))
