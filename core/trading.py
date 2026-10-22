"""
Trading research + PAPER trading. It never places a real order.

  * 8 strategy modules, a backtester that charges fees and slippage and reports LOSSES as plainly as profits
  * a paper ledger (SQLite) and a net P&L report per day and per month
  * free public data: Binance klines (meme coins like DOGE/SHIB/PEPE vs USDT); forex via your own OANDA practice token

Honest limits: a backtest is history, not a promise. Most meme-coin and retail forex traders lose money.
Real-money execution (a broker/exchange key, or signing from your wallet) is deliberately NOT included.
"""
from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

DB = Path(__file__).resolve().parent.parent / "memory" / "trading.db"
FEE, SLIPPAGE = 0.001, 0.001            # 0.1% per side each: realistic for a CEX spot market; memecoin DEX costs are higher


def ema(x, n):
    a, out = 2 / (n + 1), np.empty(len(x))
    out[0] = x[0]
    for i in range(1, len(x)):
        out[i] = a * x[i] + (1 - a) * out[i - 1]
    return out


def sma(x, n):
    c = np.cumsum(np.insert(x, 0, 0.0))
    out = np.full(len(x), np.nan)
    out[n - 1:] = (c[n:] - c[:-n]) / n
    return out


def rsi(x, n=14):
    d = np.diff(x, prepend=x[0]); up, dn = np.maximum(d, 0), np.maximum(-d, 0)
    au, ad = ema(up, 2 * n - 1), ema(dn, 2 * n - 1)
    return 100 - 100 / (1 + au / np.maximum(ad, 1e-12))


def _hold(enter, exit_):
    """Stateful 0/1 position: in after `enter`, out after `exit_`."""
    pos, p = np.zeros(len(enter)), 0
    for i in range(len(enter)):
        p = 1 if enter[i] else (0 if exit_[i] else p)
        pos[i] = p
    return pos


def _roll(x, n, fn):
    return np.array([fn(x[max(0, i - n + 1):i + 1]) for i in range(len(x))])


def s_trend(c, v):        return (ema(c, 20) > ema(c, 50)).astype(float)
def s_meanrev(c, v):      r = rsi(c); return _hold(r < 30, r > 55)
def s_breakout(c, v):     hi = np.r_[np.nan, _roll(c, 20, np.max)[:-1]]; lo = np.r_[np.nan, _roll(c, 10, np.min)[:-1]]; return _hold(c > hi, c < lo)
def s_momentum(c, v):
    roc = np.r_[np.zeros(10), c[10:] / c[:-10] - 1]; vv = sma(v, 20)
    return _hold((roc > 0.05) & (v > 2 * np.nan_to_num(vv, nan=1e18)), roc < 0)
def s_grid(c, v):         m, sd = sma(c, 20), _roll(c, 20, np.std); return _hold(c < m - sd, c >= m)
def s_dca(c, v):          return ((c > np.nan_to_num(sma(c, 100), nan=1e18)) | (np.arange(len(c)) % 24 == 0)).astype(float) * (np.arange(len(c)) > 100)
def s_pullback(c, v):     r = rsi(c); up = ema(c, 50) > np.r_[c[:1], ema(c, 50)[:-1]]; return _hold(up & (r < 40), r > 60)
def s_squeeze(c, v):
    m, sd = sma(c, 20), _roll(c, 20, np.std); bw = np.nan_to_num(4 * sd / np.maximum(m, 1e-12), nan=9)
    low = bw <= np.r_[np.full(1, 9), _roll(bw, 60, lambda a: np.percentile(a, 20))[:-1]]
    return _hold(low & (c > np.nan_to_num(m + 2 * sd, nan=1e18)), c < np.nan_to_num(m, nan=-1))


STRATEGIES = {"trend": s_trend, "mean_reversion": s_meanrev, "breakout": s_breakout, "momentum": s_momentum,
              "grid": s_grid, "dca": s_dca, "pullback": s_pullback, "squeeze": s_squeeze}
INFO = {"trend": "EMA20 above EMA50", "mean_reversion": "buy RSI<30, sell RSI>55", "breakout": "20-bar high in, 10-bar low out",
        "momentum": "+5% in 10 bars on 2x volume", "grid": "buy 1σ under the 20-bar mean, sell at the mean", "dca": "periodic buys while above the 100-bar mean",
        "pullback": "RSI dip inside an uptrend", "squeeze": "breakout after a volatility squeeze"}


@dataclass
class Result:
    net_pct: float
    trades: int
    win_rate: float
    max_drawdown_pct: float
    buy_hold_pct: float

    def line(self) -> str:
        return (f"net {self.net_pct:+.1f}% over {self.trades} trades (win rate {self.win_rate:.0%}), worst drop {self.max_drawdown_pct:.1f}%; "
                f"just holding would have made {self.buy_hold_pct:+.1f}%")


def backtest(strategy: str, o, c, v, fee: float = FEE, slip: float = SLIPPAGE) -> Result:
    """Long-or-flat. A signal on bar i is traded at the OPEN of bar i+1 (no peeking), costs charged on every entry and exit."""
    o, c, v = (np.asarray(a, float) for a in (o, c, v))
    pos = STRATEGIES[strategy](c, v)
    eq, held, entry, wins, n = [1.0], 0, 0.0, 0, 0
    for i in range(1, len(c)):
        want = pos[i - 1]
        if want and not held:
            entry = o[i] * (1 + slip); eq.append(eq[-1] * (1 - fee)); held = 1; n += 1; eq[-1] *= c[i] / entry; continue
        if not want and held:
            sell = o[i] * (1 - slip); eq[-1] *= (sell / c[i - 1]) * (1 - fee); wins += sell > entry; held = 0
            eq.append(eq[-1]); continue
        eq.append(eq[-1] * (c[i] / c[i - 1]) if held else eq[-1])
    e = np.array(eq); dd = float(np.max(1 - e / np.maximum.accumulate(e))) * 100
    return Result((e[-1] - 1) * 100, n, wins / n if n else 0.0, dd, (c[-1] / c[0] - 1) * 100)


# ── free data (injectable for tests) ────────────────────────────────────────
def binance_klines(symbol: str, interval: str = "1h", limit: int = 500, get=None):
    import requests
    r = (get or requests.get)("https://api.binance.com/api/v3/klines", params={"symbol": symbol.upper(), "interval": interval, "limit": min(limit, 1000)}, timeout=20)
    if r.status_code != 200:
        raise RuntimeError(f"Binance said HTTP {r.status_code} for {symbol}")
    k = np.array(r.json(), dtype=float)
    return k[:, 1], k[:, 4], k[:, 5], k[:, 0]                       # open, close, volume, open time (ms)


def oanda_candles(instrument: str, token: str, granularity: str = "H1", count: int = 500, practice: bool = True, get=None):
    import requests
    host = "api-fxpractice.oanda.com" if practice else "api-fxtrade.oanda.com"
    r = (get or requests.get)(f"https://{host}/v3/instruments/{instrument}/candles", params={"granularity": granularity, "count": count, "price": "M"},
                              headers={"Authorization": f"Bearer {token}"}, timeout=20)
    if r.status_code != 200:
        raise RuntimeError(f"OANDA said HTTP {r.status_code}: check the token and the instrument name (e.g. EUR_USD)")
    cs = [x for x in r.json()["candles"] if x.get("complete")]
    return (np.array([float(x["mid"]["o"]) for x in cs]), np.array([float(x["mid"]["c"]) for x in cs]),
            np.array([float(x["volume"]) for x in cs]), np.array([0.0] * len(cs)))


# ── the paper ledger and the only numbers that are shown: net P&L per day / month ───────────────
class Ledger:
    def __init__(self, path=None) -> None:
        self.path = Path(path or DB); self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._c() as c:
            c.execute("CREATE TABLE IF NOT EXISTS trades(id INTEGER PRIMARY KEY, ts REAL, strategy TEXT, symbol TEXT, pnl REAL, paper INTEGER DEFAULT 1)")

    def _c(self): return sqlite3.connect(self.path)

    def record(self, strategy: str, symbol: str, pnl: float, ts: float | None = None) -> None:
        with self._c() as c:
            c.execute("INSERT INTO trades(ts,strategy,symbol,pnl) VALUES(?,?,?,1*?)", (ts or time.time(), strategy, symbol, float(pnl)))

    def net(self, fmt: str, key: str) -> tuple[float, int]:
        with self._c() as c:
            rows = [p for t, p in c.execute("SELECT ts, pnl FROM trades") if datetime.fromtimestamp(t).strftime(fmt) == key]
        return float(sum(rows)), len(rows)

    def daily(self, day: str | None = None): return self.net("%Y-%m-%d", day or datetime.now().strftime("%Y-%m-%d"))
    def monthly(self, month: str | None = None): return self.net("%Y-%m", month or datetime.now().strftime("%Y-%m"))

    def summary(self) -> str:
        d, dn = self.daily(); m, mn = self.monthly()
        return f"PAPER · today {d:+,.2f} ({dn} trades) · this month {m:+,.2f} ({mn} trades) — net of fees, losses included"


def paper_trade(strategy: str, symbol: str, o, c, v, ledger: Ledger, capital: float = 1000.0, fraction: float = 0.1) -> str:
    """Replay the strategy over the latest candles and book each CLOSED round trip into the paper ledger."""
    pos, held, entry = STRATEGIES[strategy](np.asarray(c, float), np.asarray(v, float)), 0, 0.0
    n = 0
    for i in range(1, len(c)):
        if pos[i - 1] and not held:
            entry, held = o[i] * (1 + SLIPPAGE), 1
        elif not pos[i - 1] and held:
            sell = o[i] * (1 - SLIPPAGE); size = capital * fraction
            ledger.record(strategy, symbol, size * (sell / entry - 1) - 2 * FEE * size); held = 0; n += 1
    return f"{symbol} · {strategy}: {n} closed paper trade(s) recorded. {ledger.summary()}"
