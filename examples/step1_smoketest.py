from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import yaml

from projectx.agents.interface import AgentInput
from projectx.agents.rule_agent import RuleAgent
from projectx.config.defaults import DEFAULT_INTERVAL, DEFAULT_SYMBOL
from projectx.data.cache import get_or_download_klines
from projectx.features.indicators import add_indicators
from projectx.replay.window import ReplayWindow


def main() -> None:
    symbol = DEFAULT_SYMBOL
    interval = DEFAULT_INTERVAL
    eval_start = pd.Timestamp("2025-01-10 00:00:00", tz="UTC")
    eval_end = pd.Timestamp("2025-01-10 12:00:00", tz="UTC")

    history_start = eval_start - pd.Timedelta(days=30)
    print("loading klines (source=auto)...")
    raw = get_or_download_klines(
        symbol=symbol,
        interval=interval,
        start_ts=history_start,
        end_ts=eval_end,
        source="auto",
    )
    print(f"loaded rows: {len(raw)}")

    indicator_yaml = Path("src/projectx/config/indicator_config.yaml")
    featured = add_indicators(raw, indicator_yaml)

    replay = ReplayWindow(featured, eval_start=eval_start, eval_end=eval_end, step_hours=4)
    agent = RuleAgent()

    with indicator_yaml.open("r", encoding="utf-8") as handle:
        indicator_cfg = yaml.safe_load(handle) or {}
    enabled = list(indicator_cfg.get("enabled", []))

    context = replay.get_context()
    print(f"context rows: {len(context)}")

    for i in range(2):
        chunk = replay.step()
        revealed = replay.get_revealed()
        if chunk.empty:
            print(f"step {i + 1}: empty chunk")
            continue

        ai_input = AgentInput(
            context_df=context,
            revealed_df=revealed,
            new_chunk_df=chunk,
            metadata={
                "symbol": symbol,
                "interval": interval,
                "step_hours": 4,
                "objective": "offline_research",
            },
            indicators_enabled=enabled,
        )
        output = agent.run(ai_input)
        print(
            f"step {i + 1}: chunk {chunk.index.min()} -> {chunk.index.max()}, rows={len(chunk)}, decision={output.decision}"
        )


if __name__ == "__main__":
    main()
