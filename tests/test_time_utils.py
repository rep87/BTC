from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd

from projectx.data.sources import iter_months, to_utc_timestamp
from projectx.replay.window import ReplayWindow


def test_to_utc_timestamp_from_milliseconds_int():
    ts = to_utc_timestamp(1735689600000)
    assert ts == pd.Timestamp("2025-01-01 00:00:00", tz="UTC")


def test_to_utc_timestamp_from_naive_and_aware_datetime():
    naive = datetime(2025, 1, 1, 0, 0, 0)
    aware = datetime(2025, 1, 1, 9, 0, 0, tzinfo=timezone.utc)

    naive_out = to_utc_timestamp(naive)
    aware_out = to_utc_timestamp(aware)

    assert naive_out == pd.Timestamp("2025-01-01 00:00:00", tz="UTC")
    assert aware_out == pd.Timestamp("2025-01-01 09:00:00", tz="UTC")


def test_to_utc_timestamp_from_iso_string():
    ts = to_utc_timestamp("2025-01-01T00:00:00+09:00")
    assert ts == pd.Timestamp("2024-12-31 15:00:00", tz="UTC")


def test_iter_months_accepts_tz_aware_inputs_without_exception():
    start = pd.Timestamp("2025-01-15T00:00:00+00:00")
    end = pd.Timestamp("2025-03-02T00:00:00+00:00")
    assert iter_months(start, end) == [(2025, 1), (2025, 2), (2025, 3)]


def test_replay_window_accepts_tz_aware_inputs_without_exception():
    idx = pd.date_range("2025-01-01", periods=12, freq="1h", tz="UTC")
    df = pd.DataFrame({"open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "volume": 1.0}, index=idx)
    window = ReplayWindow(
        df,
        eval_start=pd.Timestamp("2025-01-01T02:00:00+00:00"),
        eval_end=pd.Timestamp("2025-01-01T08:00:00+00:00"),
        step_hours=4,
    )
    chunk = window.step()
    assert not chunk.empty
    assert chunk.index.min() == pd.Timestamp("2025-01-01 02:00:00", tz="UTC")
