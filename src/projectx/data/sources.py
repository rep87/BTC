from __future__ import annotations

import io
import time
import zipfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Protocol

import pandas as pd
import requests

BINANCE_FAPI_URL = "https://fapi.binance.com/fapi/v1/klines"
BINANCE_VISION_BASE = "https://data.binance.vision/data/futures/um"
REQUEST_TIMEOUT_SECONDS = 20
MAX_LIMIT = 1500
NUMERIC_COLS = ["open", "high", "low", "close", "volume"]
VISION_COLS = [
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
]

TimestampLike = int | float | str | datetime | pd.Timestamp

def _ensure_utc_timestamp(value: pd.Timestamp | datetime) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts.tz is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


class FapiBlockedError(RuntimeError):
    """Raised when FAPI endpoint is inaccessible due to policy/geo restriction."""


class KlineSource(Protocol):
    def fetch(self, symbol: str, interval: str, start_ts: TimestampLike, end_ts: TimestampLike) -> pd.DataFrame:
        """Return standardized klines with UTC datetime index and OHLCV float columns."""


@dataclass(slots=True)
class BinanceFapiKlinesSource:
    request_timeout: int = REQUEST_TIMEOUT_SECONDS

    def fetch(self, symbol: str, interval: str, start_ts: TimestampLike, end_ts: TimestampLike) -> pd.DataFrame:
        start = to_utc_timestamp(start_ts)
        end = to_utc_timestamp(end_ts)
        if start >= end:
            raise ValueError(f"start_ts must be before end_ts: {start} >= {end}")

        rows: list[list] = []
        current_start_ms = int(start.timestamp() * 1000)
        end_ms = int(end.timestamp() * 1000)
        backoff_seconds = 1.0
        failures = 0

        while current_start_ms < end_ms:
            params = {
                "symbol": symbol.upper(),
                "interval": interval,
                "startTime": current_start_ms,
                "endTime": end_ms,
                "limit": MAX_LIMIT,
            }
            try:
                response = requests.get(BINANCE_FAPI_URL, params=params, timeout=self.request_timeout)
                if response.status_code in (401, 403, 451):
                    detail = f"FAPI blocked with HTTP {response.status_code}"
                    raise FapiBlockedError(detail)
                if response.status_code in (418, 429):
                    time.sleep(backoff_seconds)
                    backoff_seconds = min(backoff_seconds * 2, 16.0)
                    continue
                response.raise_for_status()
                batch = response.json()
                failures = 0
            except FapiBlockedError:
                raise
            except requests.RequestException as exc:
                failures += 1
                time.sleep(backoff_seconds)
                backoff_seconds = min(backoff_seconds * 2, 16.0)
                if failures >= 5:
                    raise RuntimeError(f"Failed downloading from FAPI after retries: {exc}") from exc
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
            if len(batch) < MAX_LIMIT:
                break

        frame = standardize_raw_klines(rows)
        if frame.empty:
            return frame
        return frame[(frame.index >= start) & (frame.index < end)]


@dataclass(slots=True)
class BinanceVisionKlinesSource:
    cache_root: str = "data_cache"
    request_timeout: int = REQUEST_TIMEOUT_SECONDS

    def fetch(self, symbol: str, interval: str, start_ts: TimestampLike, end_ts: TimestampLike) -> pd.DataFrame:
        start = to_utc_timestamp(start_ts)
        end = to_utc_timestamp(end_ts)
        if start >= end:
            raise ValueError(f"start_ts must be before end_ts: {start} >= {end}")

        raw_dir = Path(self.cache_root) / symbol.upper() / interval / "vision_raw"
        raw_dir.mkdir(parents=True, exist_ok=True)

        parts: list[pd.DataFrame] = []
        missing_months: list[tuple[int, int]] = []

        for y, m in iter_months(start, end):
            monthly_name = f"{symbol.upper()}-{interval}-{y:04d}-{m:02d}.zip"
            monthly_url = f"{BINANCE_VISION_BASE}/monthly/klines/{symbol.upper()}/{interval}/{monthly_name}"
            monthly_zip = self._get_zip_cached(raw_dir, monthly_name, monthly_url)
            if monthly_zip is None:
                missing_months.append((y, m))
                continue
            parts.append(parse_vision_zip(monthly_zip))

        for y, m in missing_months:
            month_start = _ensure_utc_timestamp(pd.Timestamp(year=y, month=m, day=1))
            next_month = (month_start + pd.offsets.MonthBegin(1)).to_pydatetime()
            month_end = _ensure_utc_timestamp(next_month)
            day_start = max(start, month_start)
            day_end = min(end, month_end)
            for day in iter_days(day_start, day_end):
                daily_name = f"{symbol.upper()}-{interval}-{day.strftime('%Y-%m-%d')}.zip"
                daily_url = f"{BINANCE_VISION_BASE}/daily/klines/{symbol.upper()}/{interval}/{daily_name}"
                daily_zip = self._get_zip_cached(raw_dir, daily_name, daily_url)
                if daily_zip is None:
                    continue
                parts.append(parse_vision_zip(daily_zip))

        if not parts:
            raise RuntimeError(
                "Vision download returned no data. Check symbol/interval range or network availability. "
                "Try a wider historical date range or verify connectivity to data.binance.vision."
            )

        merged = normalize_ohlcv_frame(pd.concat(parts, axis=0))
        out = merged[(merged.index >= start) & (merged.index < end)]
        if out.empty:
            raise RuntimeError(
                "Vision source produced empty data in requested range. "
                "Try adjusting eval window or confirm data exists for symbol/interval."
            )
        return out

    def _get_zip_cached(self, raw_dir: Path, filename: str, url: str) -> Path | None:
        file_path = raw_dir / filename
        if file_path.exists() and file_path.stat().st_size > 0:
            try:
                with zipfile.ZipFile(file_path, "r"):
                    return file_path
            except zipfile.BadZipFile:
                file_path.unlink(missing_ok=True)

        retries = 3
        for _ in range(retries):
            try:
                response = requests.get(url, timeout=self.request_timeout)
                if response.status_code == 404:
                    return None
                response.raise_for_status()
                temp_path = file_path.with_suffix(file_path.suffix + ".tmp")
                temp_path.write_bytes(response.content)
                try:
                    with zipfile.ZipFile(temp_path, "r"):
                        pass
                except zipfile.BadZipFile:
                    temp_path.unlink(missing_ok=True)
                    time.sleep(0.5)
                    continue
                temp_path.replace(file_path)
                return file_path
            except requests.RequestException:
                time.sleep(0.75)
                continue

        file_path.unlink(missing_ok=True)
        return None


@dataclass(slots=True)
class AutoKlinesSource:
    cache_root: str = "data_cache"

    def fetch(self, symbol: str, interval: str, start_ts: TimestampLike, end_ts: TimestampLike) -> pd.DataFrame:
        fapi = BinanceFapiKlinesSource()
        try:
            return fapi.fetch(symbol=symbol, interval=interval, start_ts=start_ts, end_ts=end_ts)
        except FapiBlockedError:
            print("FAPI blocked (HTTP 451). Falling back to Binance Vision static data.")
        except RuntimeError:
            pass

        vision = BinanceVisionKlinesSource(cache_root=self.cache_root)
        try:
            return vision.fetch(symbol=symbol, interval=interval, start_ts=start_ts, end_ts=end_ts)
        except Exception as exc:
            raise RuntimeError(
                "Both FAPI and Vision data sources failed. "
                "Hint: verify network access, check symbol/interval, and ensure requested dates exist in Vision archives."
            ) from exc


def get_kline_source(mode: str, cache_root: str = "data_cache") -> KlineSource:
    mode_norm = mode.strip().lower()
    if mode_norm == "auto":
        return AutoKlinesSource(cache_root=cache_root)
    if mode_norm == "fapi":
        return BinanceFapiKlinesSource()
    if mode_norm == "vision":
        return BinanceVisionKlinesSource(cache_root=cache_root)
    raise ValueError(f"Unsupported source mode '{mode}'. Use one of: auto, fapi, vision")


def to_utc_timestamp(value: TimestampLike) -> pd.Timestamp:
    if isinstance(value, pd.Timestamp):
        ts = value
    elif isinstance(value, datetime):
        ts = pd.Timestamp(value)
    elif isinstance(value, (int, float)):
        unit = "ms" if abs(value) >= 10**12 else "s"
        ts = pd.to_datetime(value, unit=unit, utc=True)
    elif isinstance(value, str):
        ts = pd.Timestamp(value)
    else:
        raise TypeError(f"Unsupported timestamp type: {type(value)}")

    return _ensure_utc_timestamp(ts)


def standardize_raw_klines(raw_rows: list[list]) -> pd.DataFrame:
    if not raw_rows:
        empty_idx = pd.DatetimeIndex([], name="timestamp", tz="UTC")
        return pd.DataFrame(columns=NUMERIC_COLS, index=empty_idx)

    frame = pd.DataFrame(raw_rows, columns=VISION_COLS)
    frame["timestamp"] = pd.to_datetime(frame["open_time"], unit="ms", utc=True)
    out = frame[["timestamp", *NUMERIC_COLS]].set_index("timestamp")
    return normalize_ohlcv_frame(out)


def normalize_ohlcv_frame(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        empty_idx = pd.DatetimeIndex([], name="timestamp", tz="UTC")
        return pd.DataFrame(columns=NUMERIC_COLS, index=empty_idx)

    out = df.copy()
    out.index = pd.to_datetime(out.index, utc=True)
    out = out.sort_index()
    out = out[~out.index.duplicated(keep="last")]
    for col in NUMERIC_COLS:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out = out.dropna(subset=NUMERIC_COLS)
    for col in NUMERIC_COLS:
        out[col] = out[col].astype(float)
    return out


def parse_vision_csv_text(csv_text: str) -> pd.DataFrame:
    # Read as strings first so header/no-header formats are both handled safely.
    raw = pd.read_csv(io.StringIO(csv_text), header=None, dtype=str)
    if raw.empty:
        empty_idx = pd.DatetimeIndex([], name="timestamp", tz="UTC")
        return pd.DataFrame(columns=NUMERIC_COLS, index=empty_idx)

    first_cell = str(raw.iloc[0, 0]).strip().lower()
    has_header = first_cell == "open_time" or first_cell == "timestamp"
    if has_header:
        raw = raw.iloc[1:].reset_index(drop=True)

    # Ensure all required columns exist even if CSV is short/corrupt.
    raw = raw.reindex(columns=range(len(VISION_COLS)))
    raw.columns = VISION_COLS

    raw["open_time"] = pd.to_numeric(raw["open_time"], errors="coerce")
    for col in NUMERIC_COLS:
        raw[col] = pd.to_numeric(raw[col], errors="coerce")

    cleaned = raw.dropna(subset=["open_time", *NUMERIC_COLS]).copy()
    if cleaned.empty:
        empty_idx = pd.DatetimeIndex([], name="timestamp", tz="UTC")
        return pd.DataFrame(columns=NUMERIC_COLS, index=empty_idx)

    cleaned["timestamp"] = pd.to_datetime(cleaned["open_time"].astype("int64"), unit="ms", utc=True)
    out = cleaned[["timestamp", *NUMERIC_COLS]].set_index("timestamp")
    return normalize_ohlcv_frame(out)


def parse_vision_zip(zip_path: Path) -> pd.DataFrame:
    try:
        with zipfile.ZipFile(zip_path, "r") as archive:
            csv_files = [name for name in archive.namelist() if name.endswith(".csv")]
            if not csv_files:
                raise RuntimeError(f"No CSV found in zip: {zip_path}")
            with archive.open(csv_files[0], "r") as handle:
                csv_text = handle.read().decode("utf-8")
    except zipfile.BadZipFile as exc:
        raise RuntimeError(f"Corrupt Vision zip file: {zip_path}") from exc
    return parse_vision_csv_text(csv_text)


def iter_months(start: pd.Timestamp, end: pd.Timestamp) -> list[tuple[int, int]]:
    start_utc = _ensure_utc_timestamp(start)
    end_utc = _ensure_utc_timestamp(end)

    start_month = _ensure_utc_timestamp(pd.Timestamp(year=start_utc.year, month=start_utc.month, day=1))
    end_floor = end_utc - pd.Timedelta(milliseconds=1)
    end_month = _ensure_utc_timestamp(pd.Timestamp(year=end_floor.year, month=end_floor.month, day=1))

    out: list[tuple[int, int]] = []
    cursor = start_month
    while cursor <= end_month:
        out.append((cursor.year, cursor.month))
        cursor = (cursor + pd.offsets.MonthBegin(1)).to_pydatetime()
        cursor = _ensure_utc_timestamp(cursor)
    return out


def iter_days(start: pd.Timestamp, end: pd.Timestamp) -> list[pd.Timestamp]:
    start_utc = _ensure_utc_timestamp(start)
    end_utc = _ensure_utc_timestamp(end)
    if start_utc >= end_utc:
        return []
    cursor = _ensure_utc_timestamp(pd.Timestamp(year=start_utc.year, month=start_utc.month, day=start_utc.day))
    end_floor = end_utc - pd.Timedelta(milliseconds=1)
    last_day = _ensure_utc_timestamp(pd.Timestamp(year=end_floor.year, month=end_floor.month, day=end_floor.day))
    days: list[pd.Timestamp] = []
    while cursor <= last_day:
        days.append(cursor)
        cursor = cursor + pd.Timedelta(days=1)
    return days
