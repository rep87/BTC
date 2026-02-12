from __future__ import annotations

import pandas as pd

from projectx.replay.window import ReplayWindow


def _df_5m() -> pd.DataFrame:
    idx = pd.date_range("2025-01-01", periods=24 * 12, freq="5min", tz="UTC")
    return pd.DataFrame(
        {
            "open": 1.0,
            "high": 2.0,
            "low": 0.5,
            "close": 1.5,
            "volume": 10.0,
        },
        index=idx,
    )


def test_context_and_step_window_no_leakage():
    df = _df_5m()
    eval_start = pd.Timestamp("2025-01-01 08:00:00", tz="UTC")
    eval_end = pd.Timestamp("2025-01-01 16:00:00", tz="UTC")

    rw = ReplayWindow(df, eval_start=eval_start, eval_end=eval_end, step_hours=4)

    context = rw.get_context()
    assert not context.empty
    assert context.index.max() < eval_start

    chunk1 = rw.step()
    assert len(chunk1) == 48  # 4h * 12 candles/hour for 5m base interval
    assert chunk1.index.min() == eval_start
    assert chunk1.index.max() < eval_start + pd.Timedelta(hours=4)

    revealed = rw.get_revealed()
    assert revealed.index.max() < rw.pointer

    chunk2 = rw.step()
    assert len(chunk2) == 48
    assert rw.is_done()
    assert rw.get_revealed().index.max() < eval_end
