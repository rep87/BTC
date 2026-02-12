from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
import sys

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


from projectx.agents.rule_agent import RuleAgent
from projectx.config.defaults import DEFAULT_INTERVAL, DEFAULT_SYMBOL
from projectx.data.cache import get_or_download_klines
from projectx.features.indicators import add_indicators
from projectx.replay.window import ReplayWindow
from projectx.sim.engine import BacktestEngine
from projectx.sim.plotting import save_drawdown_plot, save_equity_plot
from projectx.sim.types import SimConfig


def main() -> None:
    symbol = DEFAULT_SYMBOL
    interval = DEFAULT_INTERVAL
    eval_start = pd.Timestamp("2025-01-10 00:00:00", tz="UTC")
    eval_end = pd.Timestamp("2025-01-12 00:00:00", tz="UTC")

    history_start = eval_start - pd.Timedelta(days=30)
    print("loading data (source=auto)...")
    raw = get_or_download_klines(
        symbol=symbol,
        interval=interval,
        start_ts=history_start,
        end_ts=eval_end,
        source="auto",
    )

    featured = add_indicators(raw, Path("src/projectx/config/indicator_config.yaml"))
    eval_df = featured[(featured.index >= eval_start) & (featured.index < eval_end)].copy()

    window = ReplayWindow(featured, eval_start=eval_start, eval_end=eval_end, step_hours=4)
    config = SimConfig()
    engine = BacktestEngine(df_eval=eval_df, config=config)
    result = engine.run(window=window, agent=RuleAgent())

    out_dir = Path("artifacts/step2")
    out_dir.mkdir(parents=True, exist_ok=True)

    fills_df = pd.DataFrame([asdict(f) for f in result["fills"]])
    fills_df.to_csv(out_dir / "trades.csv", index=False)
    result["equity_curve"].to_csv(out_dir / "equity.csv")
    with (out_dir / "metrics.json").open("w", encoding="utf-8") as fp:
        json.dump(result["metrics"], fp, ensure_ascii=False, indent=2)

    save_equity_plot(result["equity_curve"], out_dir / "equity.png")
    save_drawdown_plot(result["equity_curve"], out_dir / "drawdown.png")

    metrics = result["metrics"]
    print(
        f"start={metrics['start_equity']:.2f}, end={metrics['end_equity']:.2f}, "
        f"return={metrics['total_return_pct']:.2f}%, mdd={metrics['max_drawdown_pct']:.2f}%, "
        f"trades={metrics['num_trades']}, closes={metrics['num_closes']}, fills={metrics['num_fills']}, status={metrics['status']}"
    )
    print(f"saved: {out_dir / 'trades.csv'}")
    print(f"saved: {out_dir / 'equity.csv'}")
    print(f"saved: {out_dir / 'metrics.json'}")
    print(f"saved: {out_dir / 'equity.png'}")
    print(f"saved: {out_dir / 'drawdown.png'}")


if __name__ == "__main__":
    main()
