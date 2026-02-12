from __future__ import annotations

import pandas as pd

from projectx.agents.dummy_agent import DummyAgent
from projectx.agents.interface import AgentInput, AgentOutput
from projectx.replay.window import ReplayWindow
from projectx.sim.engine import BacktestEngine
from projectx.sim.types import SimConfig


class InvalidActionAgent:
    def run(self, input_data: AgentInput) -> AgentOutput:
        return AgentOutput(decision={"action": "INVALID_ACTION", "size_pct": 5.0, "leverage": 999}, notes=[])


def _df_5m_12h() -> pd.DataFrame:
    idx = pd.date_range("2025-01-01 00:00:00", periods=144, freq="5min", tz="UTC")
    close = [100.0 + i * 0.05 for i in range(len(idx))]
    open_ = [100.0] + close[:-1]
    return pd.DataFrame(
        {
            "open": open_,
            "high": [c + 0.2 for c in close],
            "low": [c - 0.2 for c in close],
            "close": close,
            "volume": [10.0] * len(idx),
        },
        index=idx,
    )


def test_dummy_agent_generates_open_and_close_fills():
    df = _df_5m_12h()
    eval_start = pd.Timestamp("2025-01-01 00:00:00", tz="UTC")
    eval_end = pd.Timestamp("2025-01-01 12:00:00", tz="UTC")

    window = ReplayWindow(df, eval_start=eval_start, eval_end=eval_end, step_hours=4)
    agent = DummyAgent(open_step=1, close_step=2, side="long", size_pct=0.5, leverage=2)
    engine = BacktestEngine(df_eval=df[(df.index >= eval_start) & (df.index < eval_end)], config=SimConfig())

    out = engine.run(window=window, agent=agent)
    fills = out["fills"]
    eq = out["equity_curve"]
    m = out["metrics"]

    assert len(fills) >= 2
    assert fills[0].action in {"open", "flip"}
    assert any(f.action == "close" for f in fills)
    assert all(f.fee > 0 for f in fills)
    assert m["num_trades"] >= 1
    assert m["num_closes"] >= 1
    assert m["num_fills"] == len(fills)
    assert float(m["end_equity"]) != float(m["start_equity"])
    assert eq["position_side"].iloc[-1] == "flat"


def test_invalid_action_is_normalized_and_warning_recorded():
    df = _df_5m_12h()
    eval_start = pd.Timestamp("2025-01-01 00:00:00", tz="UTC")
    eval_end = pd.Timestamp("2025-01-01 04:00:00", tz="UTC")

    window = ReplayWindow(df, eval_start=eval_start, eval_end=eval_end, step_hours=4)
    engine = BacktestEngine(df_eval=df[(df.index >= eval_start) & (df.index < eval_end)], config=SimConfig())
    out = engine.run(window=window, agent=InvalidActionAgent())

    assert out["metrics"]["decision_warnings"]
    assert out["metrics"]["num_fills"] == 0
