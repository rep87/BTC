from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd

from projectx.data.sources import to_utc_timestamp


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
