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

from projectx.agents.dummy_agent import DummyAgent
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
    eval_end = pd.Timestamp("2025-01-11 00:00:00", tz="UTC")  # 24h => 6 steps on 4h

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
    agent = DummyAgent(open_step=1, close_step=2, side="long", size_pct=0.5, leverage=2)
    engine = BacktestEngine(df_eval=eval_df, config=SimConfig())
    result = engine.run(window=window, agent=agent)

    out_dir = Path("artifacts/step2_dummy")
    out_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame([asdict(f) for f in result["fills"]]).to_csv(out_dir / "trades.csv", index=False)
    result["equity_curve"].to_csv(out_dir / "equity.csv")
    with (out_dir / "metrics.json").open("w", encoding="utf-8") as fp:
        json.dump(result["metrics"], fp, ensure_ascii=False, indent=2)

    save_equity_plot(result["equity_curve"], out_dir / "equity.png")
    save_drawdown_plot(result["equity_curve"], out_dir / "drawdown.png")

    m = result["metrics"]
    print(
        f"num_trades={m['num_trades']}, num_closes={m['num_closes']}, "
        f"num_fills={m['num_fills']}, end_equity={m['end_equity']:.2f}, status={m['status']}"
    )
    print(f"saved: {out_dir}")


if __name__ == "__main__":
    main()
