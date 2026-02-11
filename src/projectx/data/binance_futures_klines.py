from __future__ import annotations

import time
from datetime import timezone

import pandas as pd
import requests

BINANCE_FUTURES_KLINES_URL = "https://fapi.binance.com/fapi/v1/klines"
MAX_LIMIT = 1500
REQUEST_TIMEOUT = 15


def _standardize_klines(raw_rows: list[list]) -> pd.DataFrame:
    if not raw_rows:
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"]).set_index(
            pd.DatetimeIndex([], name="timestamp")
        )

    frame = pd.DataFrame(
        raw_rows,
        columns=[
            "open_time",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "close_time",
            "quote_asset_volume",
            "number_of_trades",
            "taker_buy_base_asset_volume",
            "taker_buy_quote_asset_volume",
            "ignore",
        ],
    )
    frame["timestamp"] = pd.to_datetime(frame["open_time"], unit="ms", utc=True)
    out = frame[["timestamp", "open", "high", "low", "close", "volume"]].copy()
    numeric_cols = ["open", "high", "low", "close", "volume"]
    out[numeric_cols] = out[numeric_cols].astype(float)
    out = out.set_index("timestamp").sort_index()
    out.index = out.index.tz_convert(timezone.utc)
    return out


def download_klines(
    symbol: str,
    interval: str,
    start_ts: pd.Timestamp,
    end_ts: pd.Timestamp,
    limit: int = MAX_LIMIT,
) -> pd.DataFrame:
    """Download Binance USD-M futures klines in [start_ts, end_ts) with pagination."""
    if limit <= 0 or limit > MAX_LIMIT:
        raise ValueError(f"limit must be between 1 and {MAX_LIMIT}, got {limit}")

    start = pd.Timestamp(start_ts, tz="UTC")
    end = pd.Timestamp(end_ts, tz="UTC")
    if start >= end:
        raise ValueError(f"start_ts must be before end_ts: {start} >= {end}")

    rows: list[list] = []
    current_start_ms = int(start.timestamp() * 1000)
    end_ms = int(end.timestamp() * 1000)
    backoff_seconds = 1.0

    while current_start_ms < end_ms:
        params = {
            "symbol": symbol.upper(),
            "interval": interval,
            "startTime": current_start_ms,
            "endTime": end_ms,
            "limit": limit,
        }

        try:
            response = requests.get(BINANCE_FUTURES_KLINES_URL, params=params, timeout=REQUEST_TIMEOUT)
            if response.status_code in (418, 429):
                time.sleep(backoff_seconds)
                backoff_seconds = min(backoff_seconds * 2, 16.0)
                continue
            response.raise_for_status()
            batch = response.json()
        except requests.RequestException as exc:
            time.sleep(backoff_seconds)
            backoff_seconds = min(backoff_seconds * 2, 16.0)
            if backoff_seconds >= 16.0:
                raise RuntimeError(f"Failed downloading klines after retries: {exc}") from exc
            continue

        backoff_seconds = 1.0
        if not batch:
            break

        rows.extend(batch)
        last_open_time_ms = int(batch[-1][0])
        next_start_ms = last_open_time_ms + 1
        if next_start_ms <= current_start_ms:
            break
        current_start_ms = next_start_ms

        if len(batch) < limit:
            break

    frame = _standardize_klines(rows)
    if frame.empty:
        return frame
    return frame[(frame.index >= start) & (frame.index < end)]
