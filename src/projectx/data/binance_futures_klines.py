from __future__ import annotations

import pandas as pd

from projectx.data.sources import BinanceFapiKlinesSource, TimestampLike


def download_klines(
    symbol: str,
    interval: str,
    start_ts: TimestampLike,
    end_ts: TimestampLike,
    limit: int = 1500,
) -> pd.DataFrame:
    """Backward-compatible wrapper for FAPI source only."""
    if limit != 1500:
        # kept for compatibility with existing function signature
        raise ValueError("Custom limit is not supported in Step 1 wrapper; use data source classes directly.")
    source = BinanceFapiKlinesSource()
    return source.fetch(symbol=symbol, interval=interval, start_ts=start_ts, end_ts=end_ts)
