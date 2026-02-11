from __future__ import annotations

from pathlib import Path

import pandas as pd


def _get_plt():
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "matplotlib is required for plotting outputs. Install dependencies (pip install -e .[dev])."
        ) from exc
    return plt


def save_equity_plot(equity_curve: pd.DataFrame, path: str | Path) -> None:
    plt = _get_plt()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 4))
    plt.plot(equity_curve.index, equity_curve["equity"])
    plt.title("Equity Curve")
    plt.xlabel("Time")
    plt.ylabel("Equity")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def save_drawdown_plot(equity_curve: pd.DataFrame, path: str | Path) -> None:
    plt = _get_plt()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 4))
    plt.plot(equity_curve.index, equity_curve["drawdown"])
    plt.title("Drawdown Curve")
    plt.xlabel("Time")
    plt.ylabel("Drawdown")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
