from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(slots=True)
class SimConfig:
    start_equity: float = 100.0
    target_equity: float = 10000.0
    fee_rate: float = 0.0004
    step_hours: int = 4
    max_leverage: int = 10
    min_leverage: int = 1
    default_size_pct: float = 0.1
    default_leverage: int = 1
    allow_negative_equity: bool = False


@dataclass(slots=True)
class Position:
    side: Literal["flat", "long", "short"] = "flat"
    qty: float = 0.0
    entry_price: float = 0.0
    leverage: int = 1
    margin: float = 0.0
    entry_time: str | None = None


@dataclass(slots=True)
class Fill:
    time: str
    side: Literal["long", "short"]
    action: Literal["open", "close", "flip", "resize"]
    price: float
    qty: float
    notional: float
    fee: float
    equity_after: float


@dataclass(slots=True)
class StepDecision:
    action: Literal["hold", "long", "short", "close"]
    size_pct: float | None = None
    leverage: int | None = None


def clamp_leverage(value: int | None, min_leverage: int, max_leverage: int, default_leverage: int) -> int:
    if value is None:
        return default_leverage
    return int(max(min_leverage, min(max_leverage, int(value))))


def clamp_size_pct(value: float | None, default_size_pct: float) -> float:
    if value is None:
        value = default_size_pct
    return float(max(0.0, min(1.0, float(value))))
