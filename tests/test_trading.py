import time
from datetime import datetime

import numpy as np
import pytest
from core import trading as T


def series(kind, n=600, seed=1):
    r = np.random.RandomState(seed); drift = {"up": 0.002, "down": -0.002, "flat": 0.0}[kind]
    c = 100 * np.exp(np.cumsum(drift + r.normal(0, 0.01, n))); o = np.r_[c[0], c[:-1]]
    return o, c, r.uniform(50, 150, n)


def test_all_eight_strategies_run_and_return_valid_positions():
    o, c, v = series("up")
    assert len(T.STRATEGIES) == 8
    for name, fn in T.STRATEGIES.items():
        p = fn(c, v)
        assert len(p) == len(c) and set(np.unique(p)) <= {0.0, 1.0}, name


def test_backtest_shows_losses_and_charges_costs():
    o, c, v = series("down")
    res = T.backtest("trend", o, c, v)
    assert res.net_pct < 0 and res.buy_hold_pct < 0                          # a falling market loses; it is reported, not hidden
    free = T.backtest("trend", o, c, v, fee=0, slip=0); costly = T.backtest("trend", o, c, v, fee=0.01, slip=0.01)
    assert free.net_pct > costly.net_pct and "net" in res.line()


def test_trend_beats_cash_in_a_steady_uptrend_and_does_not_peek():
    o, c, v = series("up")
    assert T.backtest("trend", o, c, v).net_pct > 0
    shifted = c.copy(); shifted[-1] *= 3                                      # a future spike must not change earlier decisions
    assert np.array_equal(T.STRATEGIES["trend"](c, v)[:-1], T.STRATEGIES["trend"](shifted, v)[:-1])


def test_paper_ledger_reports_net_pnl_per_day_and_month(tmp_path):
    L = T.Ledger(tmp_path / "t.db")
    now = time.time()
    L.record("trend", "DOGEUSDT", 12.5, now); L.record("trend", "DOGEUSDT", -20.0, now)
    L.record("grid", "PEPEUSDT", 5.0, now - 86400 * 40)                       # an older month
    d, n = L.daily(); m, mn = L.monthly()
    assert d == pytest.approx(-7.5) and n == 2 and m == pytest.approx(-7.5)
    assert "today -7.50" in L.summary() and "losses included" in L.summary()
    assert L.monthly(datetime.fromtimestamp(now - 86400 * 40).strftime("%Y-%m"))[0] == pytest.approx(5.0)


def test_paper_trade_books_only_closed_round_trips(tmp_path):
    o, c, v = series("flat", seed=5)
    L = T.Ledger(tmp_path / "t.db")
    msg = T.paper_trade("mean_reversion", "DOGEUSDT", o, c, v, L)
    assert "closed paper trade" in msg and L.monthly()[1] >= 0


def test_data_sources_are_parsed_and_errors_are_clear():
    class R:
        def __init__(self, code, data): self.status_code, self._d = code, data
        def json(self): return self._d
    k = [[0, "1", "2", "0.5", "1.5", "10"], [3600000, "1.5", "2", "1", "1.8", "12"]]
    o, c, v, t = T.binance_klines("dogeusdt", get=lambda *a, **kw: R(200, k))
    assert list(c) == [1.5, 1.8] and list(o) == [1.0, 1.5]
    with pytest.raises(RuntimeError, match="HTTP 400"):
        T.binance_klines("NOPE", get=lambda *a, **kw: R(400, {}))
    oa = {"candles": [{"complete": True, "volume": 5, "mid": {"o": "1.1", "c": "1.2"}}, {"complete": False, "volume": 1, "mid": {"o": "1", "c": "1"}}]}
    o2, c2, *_ = T.oanda_candles("EUR_USD", "tok", get=lambda *a, **kw: R(200, oa))
    assert list(c2) == [1.2]
    with pytest.raises(RuntimeError, match="check the token"):
        T.oanda_candles("EUR_USD", "bad", get=lambda *a, **kw: R(401, {}))


def test_tool_never_places_real_orders_and_registers(monkeypatch, tmp_path):
    import actions.trading as A
    monkeypatch.setattr(T, "DB", tmp_path / "t.db")
    o, c, v = series("up")
    monkeypatch.setattr(T, "binance_klines", lambda *a, **k: (o, c, v, c))
    assert "Past results do not predict" in A.trading({"mode": "backtest", "symbol": "DOGEUSDT", "strategy": "trend"})
    assert "PAPER" in A.trading({"mode": "pnl"}) and "trend:" in A.trading({"mode": "strategies"})
    assert "No real orders" in A.trading({"mode": "buy"}) and "Strategies:" in A.trading({"mode": "backtest", "strategy": "nope"})
    from pathlib import Path
    from core.action_loader import discover_actions
    d = {x["name"]: x for x in discover_actions(Path("actions"), logger=lambda s: None).get_tool_declarations()}
    assert "never real orders" in d["trading"]["description"]
