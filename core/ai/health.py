"""Per-model health: cooldowns, a circuit breaker and measured latency.

This generalises the cooldown table that core/gemini.py keeps for its own
ladder to every provider, and it is what stops a failing service being paid
for on every call (and what makes retry loops finite)."""
from __future__ import annotations

import threading
import time

from .types import AIError


class Health:
    def __init__(self, clock=time.monotonic) -> None:
        self._clock = clock
        self._lock = threading.Lock()
        self._s: dict[str, dict] = {}

    def _st(self, key: str) -> dict:
        return self._s.setdefault(key, {
            "fails": 0, "until": 0.0, "ewma": None, "ok": 0, "err": 0, "last": ""})

    def available(self, key: str) -> bool:
        with self._lock:
            return self._clock() >= self._st(key)["until"]

    def remaining(self, key: str) -> float:
        with self._lock:
            return max(0.0, self._st(key)["until"] - self._clock())

    def success(self, key: str, latency: float) -> None:
        with self._lock:
            s = self._st(key)
            s["fails"] = 0
            s["until"] = 0.0
            s["ok"] += 1
            s["ewma"] = latency if s["ewma"] is None else 0.7 * s["ewma"] + 0.3 * latency

    def failure(self, key: str, err: AIError) -> float:
        """Record a failure; returns the cooldown applied, in seconds."""
        with self._lock:
            s = self._st(key)
            s["err"] += 1
            s["last"] = err.kind
            if err.kind in ("bad_request", "unsupported", "cancelled", "budget"):
                return 0.0          # the request's fault, not the model's
            s["fails"] += 1
            n = min(s["fails"], 6)
            if err.kind == "rate_limit":
                cool = err.retry_after if err.retry_after else 60.0 * (2 ** (n - 1))
                cool = min(cool, 900.0)
            elif err.kind in ("auth", "not_found"):
                cool = 6 * 3600.0   # will not fix itself; wait for a config change
            elif err.kind == "invalid_response":
                cool = 20.0 * n
            else:                   # timeout / unavailable / network
                cool = min(30.0 * (2 ** (n - 1)), 600.0)
            s["until"] = self._clock() + cool
            return cool

    def reset(self, key: str | None = None) -> None:
        with self._lock:
            if key is None:
                self._s.clear()
            else:
                self._s.pop(key, None)

    def latency(self, key: str) -> float | None:
        with self._lock:
            return self._st(key)["ewma"]

    def snapshot(self) -> dict:
        with self._lock:
            now = self._clock()
            return {k: {"ok": v["ok"], "errors": v["err"], "consecutive_fails": v["fails"],
                        "cooling_s": round(max(0.0, v["until"] - now), 1),
                        "latency_s": None if v["ewma"] is None else round(v["ewma"], 3),
                        "last_error": v["last"]} for k, v in self._s.items()}
