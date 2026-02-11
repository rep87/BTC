from __future__ import annotations

import math

import pandas as pd


def compute_metrics(equity_curve: pd.DataFrame, fills: list[dict], status: str) -> dict:
    if equity_curve.empty:
        return {
            "total_return_pct": 0.0,
            "max_drawdown_pct": 0.0,
            "num_trades": 0,
            "num_closes": 0,
            "status": status,
            "start_equity": 0.0,
            "end_equity": 0.0,
        }

    start_equity = float(equity_curve["equity"].iloc[0])
    end_equity = float(equity_curve["equity"].iloc[-1])
    total_return_pct = ((end_equity / start_equity) - 1.0) * 100 if start_equity != 0 else math.nan
    max_drawdown_pct = float(equity_curve["drawdown"].min()) * 100.0
    num_trades = sum(1 for f in fills if f["action"] in {"open", "flip", "resize"})
    num_closes = sum(1 for f in fills if f["action"] == "close")

    return {
        "total_return_pct": float(total_return_pct),
        "max_drawdown_pct": max_drawdown_pct,
        "num_trades": int(num_trades),
        "num_closes": int(num_closes),
        "status": status,
        "start_equity": start_equity,
        "end_equity": end_equity,
    }
