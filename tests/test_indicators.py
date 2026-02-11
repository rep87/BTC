from __future__ import annotations

import pandas as pd
import yaml

from projectx.features.indicators import add_indicators


def _base_df() -> pd.DataFrame:
    idx = pd.date_range("2025-01-01", periods=200, freq="5min", tz="UTC")
    df = pd.DataFrame(index=idx)
    df["open"] = 100.0 + (pd.Series(range(200), index=idx) * 0.1)
    df["high"] = df["open"] + 1.0
    df["low"] = df["open"] - 1.0
    df["close"] = df["open"] + 0.2
    df["volume"] = 10.0
    return df


def test_enabled_indicators_create_columns(tmp_path):
    cfg = {
        "enabled": ["ema", "rsi", "macd", "atr"],
        "params": {
            "ema_spans": [20, 50],
            "rsi_period": 14,
            "macd_fast": 12,
            "macd_slow": 26,
            "macd_signal": 9,
            "atr_period": 14,
        },
    }
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")

    df = _base_df()
    out = add_indicators(df, cfg_path)

    assert len(out) == len(df)
    expected = {"ema_20", "ema_50", "rsi_14", "macd", "macd_signal", "macd_hist", "atr_14"}
    assert expected.issubset(set(out.columns))


def test_disabling_indicators_omits_columns(tmp_path):
    cfg = {"enabled": [], "params": {}}
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")

    df = _base_df()
    out = add_indicators(df, cfg_path)

    assert len(out) == len(df)
    assert set(out.columns) == set(df.columns)
