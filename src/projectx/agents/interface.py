from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import pandas as pd


@dataclass(slots=True)
class AgentInput:
    context_df: pd.DataFrame
    revealed_df: pd.DataFrame
    new_chunk_df: pd.DataFrame
    metadata: dict[str, Any]
    indicators_enabled: list[str]


@dataclass(slots=True)
class AgentOutput:
    decision: dict[str, Any]
    notes: list[str]
    strategy_patch: dict[str, Any] | None = None


class BaseAgent(Protocol):
    def run(self, input_data: AgentInput) -> AgentOutput:
        """Run deterministic inference for one replay step."""
        ...
