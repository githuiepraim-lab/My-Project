"""Token, latency and cost accounting.

Uses the provider's own usage numbers whenever they are reported; otherwise an
estimate, flagged as such so a dashboard never presents a guess as a fact."""
from __future__ import annotations

import json
import threading
import time
from collections import defaultdict
from pathlib import Path

from .config import BASE_DIR

LOG_FILE = BASE_DIR / "logs" / "ai_usage.jsonl"
_ROTATE_BYTES = 5 * 1024 * 1024


class UsageTracker:
    def __init__(self, path: Path | None = None) -> None:
        self.path = Path(path) if path else LOG_FILE
        self._lock = threading.Lock()
        self._rows: list[dict] = []        # this session, bounded
        self._month_cost = None            # lazily loaded from disk

    def record(self, *, provider: str, model: str, task: str, tokens_in: int,
               tokens_out: int, cached: int, estimated: bool, latency: float,
               cost: float | None, status: str, cache_hit: bool = False) -> None:
        row = {"ts": round(time.time(), 2), "provider": provider, "model": model,
               "task": task, "in": tokens_in, "out": tokens_out, "cached": cached,
               "est": estimated, "latency": round(latency, 3),
               "cost": None if cost is None else round(cost, 6),
               "status": status, "cache_hit": cache_hit}
        with self._lock:
            self._rows.append(row)
            del self._rows[:-2000]
            if cost and self._month_cost is not None:
                self._month_cost += cost
            try:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                if self.path.exists() and self.path.stat().st_size > _ROTATE_BYTES:
                    self.path.replace(self.path.with_suffix(".1.jsonl"))
                with self.path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(row) + "\n")
            except OSError:
                pass            # accounting must never break a request

    def month_cost(self) -> float:
        """USD spent this calendar month, from the on-disk log."""
        with self._lock:
            if self._month_cost is None:
                total = 0.0
                start = time.mktime(time.strptime(time.strftime("%Y-%m-01"), "%Y-%m-%d"))
                try:
                    for line in self.path.read_text(encoding="utf-8").splitlines():
                        try:
                            r = json.loads(line)
                        except ValueError:
                            continue
                        if r.get("ts", 0) >= start and r.get("cost"):
                            total += r["cost"]
                except OSError:
                    pass
                self._month_cost = total
            return self._month_cost

    def summary(self) -> dict:
        with self._lock:
            rows = list(self._rows)
        agg: dict = defaultdict(lambda: {"calls": 0, "in": 0, "out": 0, "cached": 0,
                                         "cost": 0.0, "latency": 0.0, "errors": 0})
        for r in rows:
            a = agg[f"{r['provider']}/{r['model']}"]
            a["calls"] += 1
            a["in"] += r["in"]
            a["out"] += r["out"]
            a["cached"] += r["cached"]
            a["cost"] += r["cost"] or 0.0
            a["latency"] += r["latency"]
            a["errors"] += r["status"] != "ok"
        for a in agg.values():
            a["avg_latency"] = round(a["latency"] / a["calls"], 3) if a["calls"] else 0.0
            a["cost"] = round(a["cost"], 6)
            del a["latency"]
        return dict(agg)
