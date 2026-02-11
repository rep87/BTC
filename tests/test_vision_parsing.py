from __future__ import annotations

import pandas as pd

from projectx.data.sources import parse_vision_csv_text


def test_parse_vision_csv_text_normalizes_schema_and_types():
    csv_text = "\n".join(
        [
            "1735689600000,100,101,99,100.5,123.45,1735689899999,0,0,0,0,0",
            "1735689900000,100.5,102,100,101,120.0,1735690199999,0,0,0,0,0",
        ]
    )

    out = parse_vision_csv_text(csv_text)

    assert list(out.columns) == ["open", "high", "low", "close", "volume"]
    assert out.index.tz is not None
    assert out.index[0] == pd.Timestamp("2025-01-01 00:00:00", tz="UTC")
    assert out["close"].dtype == float
    assert len(out) == 2
