from __future__ import annotations

import pandas as pd

from projectx.agents.interface import AgentInput
from projectx.agents.rule_agent import RuleAgent


def _df() -> pd.DataFrame:
    idx = pd.date_range("2025-01-01", periods=4, freq="5min", tz="UTC")
    return pd.DataFrame(
        {
            "open": [1.0, 1.1, 1.2, 1.3],
            "high": [1.1, 1.2, 1.3, 1.4],
            "low": [0.9, 1.0, 1.1, 1.2],
            "close": [1.0, 1.2, 1.1, 1.4],
            "volume": [10.0, 11.0, 9.0, 12.0],
            "rsi_14": [50.0, 55.0, 60.0, 72.0],
        },
        index=idx,
    )


def test_rule_agent_output_shape():
    df = _df()
    agent = RuleAgent()
    out = agent.run(
        AgentInput(
            context_df=df.iloc[:1],
            revealed_df=df.iloc[:2],
            new_chunk_df=df.iloc[2:],
            metadata={"symbol": "BTCUSDT", "interval": "5m", "step_hours": 4, "objective": "offline_research"},
            indicators_enabled=["rsi"],
        )
    )

    assert isinstance(out.decision, dict)
    assert {"regime", "action", "confidence"}.issubset(out.decision.keys())
    assert isinstance(out.notes, list)
    assert len(out.notes) <= 3
