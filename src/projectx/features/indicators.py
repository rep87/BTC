from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import yaml


def add_ema(df: pd.DataFrame, spans: list[int]) -> pd.DataFrame:
    out = df.copy()
    for span in spans:
        out[f"ema_{span}"] = out["close"].ewm(span=span, adjust=False).mean()
    return out


def add_rsi(df: pd.DataFrame, period: int) -> pd.DataFrame:
    out = df.copy()
    delta = out["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss
    out[f"rsi_{period}"] = 100 - (100 / (1 + rs))
    return out


def add_macd(df: pd.DataFrame, fast: int, slow: int, signal: int) -> pd.DataFrame:
    out = df.copy()
    fast_ema = out["close"].ewm(span=fast, adjust=False).mean()
    slow_ema = out["close"].ewm(span=slow, adjust=False).mean()
    out["macd"] = fast_ema - slow_ema
    out["macd_signal"] = out["macd"].ewm(span=signal, adjust=False).mean()
    out["macd_hist"] = out["macd"] - out["macd_signal"]
    return out


def add_atr(df: pd.DataFrame, period: int) -> pd.DataFrame:
    out = df.copy()
    prev_close = out["close"].shift(1)
    true_range = pd.concat(
        [
            (out["high"] - out["low"]).abs(),
            (out["high"] - prev_close).abs(),
            (out["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    out[f"atr_{period}"] = true_range.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    return out


def _load_indicator_config(indicator_yaml_path: str | Path) -> dict[str, Any]:
    path = Path(indicator_yaml_path)
    if not path.exists():
        raise FileNotFoundError(f"Indicator config not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    if not isinstance(config, dict):
        raise ValueError(f"Indicator config must be a mapping, got: {type(config)}")
    return config


def add_indicators(df: pd.DataFrame, indicator_yaml_path: str | Path) -> pd.DataFrame:
    """Add indicators deterministically based on YAML enabled list and params."""
    config = _load_indicator_config(indicator_yaml_path)
    enabled = set(config.get("enabled", []))
    params = config.get("params", {})

    out = df.copy()
    if "ema" in enabled:
        spans = [int(s) for s in params.get("ema_spans", [20])]
        out = add_ema(out, spans=spans)
    if "rsi" in enabled:
        period = int(params.get("rsi_period", 14))
        out = add_rsi(out, period=period)
    if "macd" in enabled:
        fast = int(params.get("macd_fast", 12))
        slow = int(params.get("macd_slow", 26))
        signal = int(params.get("macd_signal", 9))
        out = add_macd(out, fast=fast, slow=slow, signal=signal)
    if "atr" in enabled:
        period = int(params.get("atr_period", 14))
        out = add_atr(out, period=period)
    return out
