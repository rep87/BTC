from __future__ import annotations

from pathlib import Path

import pandas as pd

from projectx.config.defaults import cache_file_path, ensure_supported_interval
from projectx.data.sources import NUMERIC_COLS, get_kline_source, normalize_ohlcv_frame, to_utc_timestamp

INTERVAL_TO_FREQ = {
    "1m": "1min",
    "5m": "5min",
}


def _empty_klines() -> pd.DataFrame:
    empty_idx = pd.DatetimeIndex([], name="timestamp", tz="UTC")
    return pd.DataFrame(columns=NUMERIC_COLS, index=empty_idx)


def _missing_ranges(
    df: pd.DataFrame,
    start_ts: pd.Timestamp,
    end_ts: pd.Timestamp,
    interval: str,
) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    if df.empty:
        return [(start_ts, end_ts)]

    freq = INTERVAL_TO_FREQ[interval]
    expected = pd.date_range(start=start_ts, end=end_ts - pd.Timedelta(minutes=1), freq=freq, tz="UTC")
    covered = df[(df.index >= start_ts) & (df.index < end_ts)].index

    missing = expected.difference(covered)
    if missing.empty:
        return []

    ranges: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    range_start = missing[0]
    prev = missing[0]
    step = pd.Timedelta(freq)

    for ts in missing[1:]:
        if ts != prev + step:
            ranges.append((range_start, prev + step))
            range_start = ts
        prev = ts
    ranges.append((range_start, prev + step))
    return ranges


def get_or_download_klines(
    symbol: str,
    interval: str,
    start_ts,
    end_ts,
    cache_root: str = "data_cache",
    source: str = "auto",
) -> pd.DataFrame:
    """Load klines from cache and download missing ranges, then persist parquet cache."""
    interval = ensure_supported_interval(interval)
    start = to_utc_timestamp(start_ts)
    end = to_utc_timestamp(end_ts)
    if start >= end:
        raise ValueError(f"start_ts must be before end_ts: {start} >= {end}")

    cache_path: Path = cache_file_path(symbol, interval, cache_root=cache_root)
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    if cache_path.exists():
        cached = pd.read_parquet(cache_path)
        cached = normalize_ohlcv_frame(cached)
    else:
        cached = _empty_klines()

    missing = _missing_ranges(cached, start, end, interval)
    if not missing:
        return cached[(cached.index >= start) & (cached.index < end)].copy()

    kline_source = get_kline_source(mode=source, cache_root=cache_root)
    merged = cached
    for miss_start, miss_end in missing:
        downloaded = kline_source.fetch(symbol=symbol, interval=interval, start_ts=miss_start, end_ts=miss_end)
        downloaded = normalize_ohlcv_frame(downloaded)
        if downloaded.empty:
            continue
        if merged.empty:
            merged = downloaded
            continue
        merged = normalize_ohlcv_frame(pd.concat([merged, downloaded], axis=0))

    merged.to_parquet(cache_path)
    return merged[(merged.index >= start) & (merged.index < end)].copy()
