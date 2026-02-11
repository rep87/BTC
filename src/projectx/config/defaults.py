from __future__ import annotations

from pathlib import Path

DEFAULT_SYMBOL = "BTCUSDT"
DEFAULT_INTERVAL = "5m"
SUPPORTED_INTERVALS = {"1m", "5m"}
DEFAULT_STEP_HOURS = 4
DEFAULT_CACHE_ROOT = "data_cache"
DEFAULT_WARMUP_DAYS = 30


def ensure_supported_interval(interval: str) -> str:
    """Validate and normalize interval for Step 1."""
    normalized = interval.strip().lower()
    if normalized not in SUPPORTED_INTERVALS:
        raise ValueError(
            f"Unsupported interval '{interval}'. Supported intervals: {sorted(SUPPORTED_INTERVALS)}"
        )
    return normalized


def indicator_config_path() -> Path:
    """Return default indicator YAML path in the package."""
    return Path(__file__).resolve().parent / "indicator_config.yaml"


def cache_file_path(symbol: str, interval: str, cache_root: str = DEFAULT_CACHE_ROOT) -> Path:
    """Build canonical cache parquet path."""
    interval_norm = ensure_supported_interval(interval)
    return Path(cache_root) / symbol.upper() / interval_norm / "klines.parquet"
