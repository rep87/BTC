from __future__ import annotations

from pathlib import Path

import pandas as pd

from projectx.config.defaults import cache_file_path, ensure_supported_interval
from projectx.data.binance_futures_klines import download_klines


NUMERIC_COLS = ["open", "high", "low", "close", "volume"]


def _empty_klines() -> pd.DataFrame:
    return pd.DataFrame(columns=NUMERIC_COLS, index=pd.DatetimeIndex([], name="timestamp", tz="UTC"))


def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return _empty_klines()
    out = df.copy()
    out.index = pd.to_datetime(out.index, utc=True)
    out = out.sort_index()
    out = out[~out.index.duplicated(keep="last")]
    for col in NUMERIC_COLS:
        out[col] = out[col].astype(float)
    return out


def _merge(existing: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    if existing.empty:
        return _normalize(new)
    if new.empty:
        return _normalize(existing)
    merged = pd.concat([existing, new], axis=0)
    return _normalize(merged)


def _missing_ranges(df: pd.DataFrame, start_ts: pd.Timestamp, end_ts: pd.Timestamp) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    if df.empty:
        return [(start_ts, end_ts)]

    available = df[(df.index >= start_ts) & (df.index < end_ts)]
    if available.empty:
        return [(start_ts, end_ts)]

    spans: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    first_idx = available.index.min()
    last_idx = available.index.max()
    if start_ts < first_idx:
        spans.append((start_ts, first_idx))
    if (last_idx + pd.Timedelta(milliseconds=1)) < end_ts:
        spans.append((last_idx + pd.Timedelta(milliseconds=1), end_ts))
    return spans


def get_or_download_klines(
    symbol: str,
    interval: str,
    start_ts: pd.Timestamp,
    end_ts: pd.Timestamp,
    cache_root: str = "data_cache",
) -> pd.DataFrame:
    """Load klines from cache and download missing ranges, then persist parquet cache."""
    interval = ensure_supported_interval(interval)
    start = pd.Timestamp(start_ts, tz="UTC")
    end = pd.Timestamp(end_ts, tz="UTC")
    if start >= end:
        raise ValueError(f"start_ts must be before end_ts: {start} >= {end}")

    cache_path: Path = cache_file_path(symbol, interval, cache_root=cache_root)
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    if cache_path.exists():
        cached = pd.read_parquet(cache_path)
        cached.index = pd.to_datetime(cached.index, utc=True)
        cached = _normalize(cached)
    else:
        cached = _empty_klines()

    missing = _missing_ranges(cached, start, end)
    merged = cached
    for miss_start, miss_end in missing:
        downloaded = download_klines(symbol=symbol, interval=interval, start_ts=miss_start, end_ts=miss_end)
        merged = _merge(merged, downloaded)

    merged.to_parquet(cache_path)
    return merged[(merged.index >= start) & (merged.index < end)].copy()
