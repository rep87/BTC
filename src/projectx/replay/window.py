from __future__ import annotations

import pandas as pd


def _ensure_utc_timestamp(value: pd.Timestamp) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts.tz is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


class ReplayWindow:
    """Reveal base-interval candles in rolling wall-clock windows."""

    def __init__(self, df: pd.DataFrame, eval_start: pd.Timestamp, eval_end: pd.Timestamp, step_hours: int = 4):
        self._df = df.copy().sort_index()
        self._df.index = pd.to_datetime(self._df.index, utc=True)
        self.eval_start = _ensure_utc_timestamp(eval_start)
        self.eval_end = _ensure_utc_timestamp(eval_end)
        if self.eval_start >= self.eval_end:
            raise ValueError(f"eval_start must be before eval_end: {self.eval_start} >= {self.eval_end}")
        if step_hours <= 0:
            raise ValueError(f"step_hours must be positive, got {step_hours}")
        self.step_delta = pd.Timedelta(hours=step_hours)
        self.pointer = self.eval_start

    def get_context(self) -> pd.DataFrame:
        return self._df[self._df.index < self.eval_start].copy()

    def reset(self) -> None:
        self.pointer = self.eval_start

    def step(self) -> pd.DataFrame:
        if self.is_done():
            return self._df.iloc[0:0].copy()
        end = min(self.pointer + self.step_delta, self.eval_end)
        chunk = self._df[(self._df.index >= self.pointer) & (self._df.index < end)].copy()
        self.pointer = end
        return chunk

    def get_revealed(self) -> pd.DataFrame:
        return self._df[(self._df.index >= self.eval_start) & (self._df.index < self.pointer)].copy()

    def is_done(self) -> bool:
        return self.pointer >= self.eval_end
