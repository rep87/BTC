from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from projectx.agents.interface import AgentInput, AgentOutput
from projectx.replay.window import ReplayWindow
from projectx.sim.engine import BacktestEngine
from projectx.sim.types import SimConfig


@dataclass
class DummyAgent:
    calls: int = 0

    def run(self, input_data: AgentInput) -> AgentOutput:
        self.calls += 1
        if self.calls == 1:
            return AgentOutput(decision={"action": "long", "size_pct": 1.0, "leverage": 1}, notes=[])
        return AgentOutput(decision={"action": "close"}, notes=[])


def _synthetic_df() -> pd.DataFrame:
    idx = pd.date_range("2025-01-01 00:00:00", periods=24, freq="1h", tz="UTC")
    close = [100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 110, 109, 108, 107, 106, 105, 104, 103, 102, 101, 100, 99]
    open_ = [100] + close[:-1]
    return pd.DataFrame(
        {
            "open": open_,
            "high": [c + 1 for c in close],
            "low": [c - 1 for c in close],
            "close": close,
            "volume": [10.0] * len(close),
        },
        index=idx,
    )


def test_engine_next_open_fills_and_equity_behavior():
    df = _synthetic_df()
    eval_start = pd.Timestamp("2025-01-01 04:00:00", tz="UTC")
    eval_end = pd.Timestamp("2025-01-01 12:00:00", tz="UTC")

    window = ReplayWindow(df, eval_start=eval_start, eval_end=eval_end, step_hours=4)
    engine = BacktestEngine(
        df_eval=df[(df.index >= eval_start) & (df.index < eval_end)],
        config=SimConfig(start_equity=100.0, fee_rate=0.001, step_hours=4, default_size_pct=1.0, default_leverage=1),
    )

    out = engine.run(window=window, agent=DummyAgent())
    fills = out["fills"]
    equity_curve = out["equity_curve"]
    metrics = out["metrics"]

    assert len(fills) >= 1
    assert fills[0].action == "open"
    assert pd.Timestamp(fills[0].time) == pd.Timestamp("2025-01-01 08:00:00+00:00")
    assert fills[0].fee > 0

    assert float(equity_curve["equity"].max()) >= 100.0
    assert float(equity_curve["drawdown"].min()) <= 0.0
    assert metrics["num_trades"] >= 1
