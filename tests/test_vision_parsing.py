from __future__ import annotations

import pandas as pd

from projectx.data.sources import parse_vision_csv_text


def _assert_common_schema(out: pd.DataFrame) -> None:
    assert list(out.columns) == ["open", "high", "low", "close", "volume"]
    assert out.index.tz is not None
    assert out.index.dtype == "datetime64[ns, UTC]"
    assert out["close"].dtype == float
    assert len(out) == 2


def test_parse_vision_csv_text_without_header():
    csv_text = "\n".join(
        [
            "1735689600000,100,101,99,100.5,123.45,1735689899999,0,0,0,0,0",
            "1735689900000,100.5,102,100,101,120.0,1735690199999,0,0,0,0,0",
        ]
    )

    out = parse_vision_csv_text(csv_text)

    _assert_common_schema(out)
    assert out.index[0] == pd.Timestamp("2025-01-01 00:00:00", tz="UTC")


def test_parse_vision_csv_text_with_header():
    csv_text = "\n".join(
        [
            "open_time,open,high,low,close,volume,close_time,quote_asset_volume,number_of_trades,taker_buy_base_asset_volume,taker_buy_quote_asset_volume,ignore",
            "1735689600000,100,101,99,100.5,123.45,1735689899999,0,0,0,0,0",
            "1735689900000,100.5,102,100,101,120.0,1735690199999,0,0,0,0,0",
        ]
    )

    out = parse_vision_csv_text(csv_text)

    _assert_common_schema(out)
    assert out.index[1] == pd.Timestamp("2025-01-01 00:05:00", tz="UTC")
