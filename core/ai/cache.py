"""Response cache with TTL, LRU eviction and explicit invalidation.

Only for requests where a repeat is genuinely the same question: no tools, no
private context, and a task that is deterministic by nature. The key includes
a namespace version so `invalidate()` retires every stale entry at once —
memory edits bump it, so an answer that quoted an old fact cannot be served
after the fact changed."""
from __future__ import annotations

import hashlib
import json
import threading
import time
from collections import OrderedDict

CACHEABLE_TASKS = frozenset({"classify", "extract", "summarize", "translate", "rewrite"})


class ResponseCache:
    def __init__(self, max_items: int = 256, clock=time.monotonic) -> None:
        self._d: OrderedDict[str, tuple[float, object]] = OrderedDict()
        self._max = max_items
        self._clock = clock
        self._lock = threading.Lock()
        self._versions: dict[str, int] = {}
        self.hits = 0
        self.misses = 0

    def version(self, ns: str = "default") -> int:
        return self._versions.get(ns, 0)

    def invalidate(self, ns: str = "default") -> None:
        """Retire every entry in `ns` (their keys embed the old version)."""
        with self._lock:
            self._versions[ns] = self._versions.get(ns, 0) + 1

    def key(self, model_key: str, system: str, messages: list, extra: dict,
            ns: str = "default") -> str:
        blob = json.dumps([self.version(ns), ns, model_key, system, messages, extra],
                          sort_keys=True, default=str)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def get(self, key: str):
        with self._lock:
            item = self._d.get(key)
            if item is None or item[0] < self._clock():
                if item is not None:
                    del self._d[key]
                self.misses += 1
                return None
            self._d.move_to_end(key)
            self.hits += 1
            return item[1]

    def put(self, key: str, value, ttl: float) -> None:
        with self._lock:
            self._d[key] = (self._clock() + ttl, value)
            self._d.move_to_end(key)
            while len(self._d) > self._max:
                self._d.popitem(last=False)

    def clear(self) -> None:
        with self._lock:
            self._d.clear()
