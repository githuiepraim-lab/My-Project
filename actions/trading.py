"""trading — research and PAPER trading only (no real orders). Shows net profit/loss per day and per month."""
from __future__ import annotations

from core import trading as T


def trading(parameters: dict, player=None) -> str:
    mode = str(parameters.get("mode", "pnl")).lower()
    try:
        if mode == "pnl":
            return T.Ledger().summary()
        if mode == "strategies":
            return "; ".join(f"{k}: {v}" for k, v in T.INFO.items())
        sym, strat = str(parameters.get("symbol", "DOGEUSDT")).upper(), str(parameters.get("strategy", "trend")).lower()
        if strat not in T.STRATEGIES:
            return "Strategies: " + ", ".join(T.STRATEGIES)
        o, c, v, _ = T.binance_klines(sym, str(parameters.get("interval", "1h")), 1000)
        if mode == "backtest":
            return f"{sym} · {strat} (history, fees included): " + T.backtest(strat, o, c, v).line() + ". Past results do not predict future ones."
        if mode == "paper":
            return T.paper_trade(strat, sym, o, c, v, T.Ledger())
        return "Modes: pnl, strategies, backtest (symbol, strategy), paper (symbol, strategy). No real orders are ever placed."
    except Exception as e:
        return f"Sir, trading failed: {type(e).__name__}: {str(e)[:100]}"


TOOL = {"name": "trading",
        "description": "Trading research + PAPER trading only (never real orders). mode: pnl (net profit/loss today and this month) | strategies | "
                       "backtest/paper (symbol e.g. DOGEUSDT, strategy: trend, mean_reversion, breakout, momentum, grid, dca, pullback, squeeze).",
        "parameters": {"type": "OBJECT", "properties": {"mode": {"type": "STRING"}, "symbol": {"type": "STRING"}, "strategy": {"type": "STRING"},
                                                          "interval": {"type": "STRING"}}, "required": ["mode"]},
        "handler": trading}
