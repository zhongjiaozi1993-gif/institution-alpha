"""breakout_close_quality 因子单元测试。

覆盖：
  1. rolling_high 不包含当天
  2. 突破日/非突破日正确区分
  3. 组件取值范围
  4. 缺失行情处理（high==low, 窗口不足）
  5. 多股票 panel 模式
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJ))
from src.features.price_action.breakout import breakout_close_quality


def _make_panel(symbol: str, dates: list[str], ohlcv: list[tuple]) -> pd.DataFrame:
    rows = []
    for i, (d, (op, hi, lo, cl, vo)) in enumerate(zip(dates, ohlcv)):
        rows.append({"trade_date": pd.Timestamp(d), "symbol": symbol,
                     "open": float(op), "high": float(hi), "low": float(lo),
                     "close": float(cl), "volume": float(vo)})
    return pd.DataFrame(rows)


# 10 日连续上升趋势：每天 open/high/low/close 递增
DATES = [f"2025-09-{d:02d}" for d in range(1, 16)]
OHLCV = [(10.0 + i, 10.5 + i, 9.8 + i, 10.3 + i, 1e6 + i * 1e5) for i in range(len(DATES))]


# ---------- 1. rolling_high 不包含当天 ----------
def test_rolling_high_excludes_today():
    panel = _make_panel("000001", DATES, OHLCV)
    L = 5
    result = breakout_close_quality(panel, L=L, ATR_N=14, VOL_N=20)

    # 对第 i 天，rolling_high 应该是 high[i-L:i] 的 max（不含 i）
    h = panel["high"].values
    for i in range(L, len(DATES)):
        expected_rh = h[i - L : i].max()
        row = result[result["trade_date"] == pd.Timestamp(DATES[i])]
        if row["breakout_event"].iloc[0]:
            # 突破日：close > rolling_high
            assert h[i] <= expected_rh or panel["close"].iloc[i] > expected_rh


# ---------- 2. 突破日正确标记 ----------
def test_breakout_event_detection():
    # 构造明显突破：前 22 天盘整提供足够 ATR/VOL 窗口，第 23 天大幅突破
    dates = [f"2025-10-{d:02d}" for d in range(1, 26)]
    ohlcv = []
    # padding: 22 天平稳盘整，满足 ATR_N=14 (需 ≥15 天) 和 VOL_N=20 (需 ≥21 天)
    for i in range(22):
        ohlcv.append((10 + i * 0.1, 11 + i * 0.1, 9 + i * 0.1, 10.5 + i * 0.1, 1e6))
    # day 22 (index 22): 突破！close=14.5 > rolling_high(L=3)≈11.7
    ohlcv.append((11.5, 15, 11, 14.5, 5e6))
    # 之后回落
    ohlcv.append((14.5, 15.5, 14, 15, 2e6))
    ohlcv.append((15, 16, 14.5, 15.2, 1.5e6))
    panel = _make_panel("000002", dates, ohlcv)
    result = breakout_close_quality(panel, L=3, ATR_N=14, VOL_N=20)

    b22 = result[result["trade_date"] == pd.Timestamp("2025-10-23")]
    assert b22["breakout_event"].iloc[0], "day 22 should be breakout"
    assert b22["factor_value"].iloc[0] > 0, f"day 22 factor should be > 0, got {b22['factor_value'].iloc[0]}"

    b0 = result[result["trade_date"] == pd.Timestamp("2025-10-01")]
    assert not b0["breakout_event"].iloc[0], "day 0 should not be breakout (window too short)"
    assert b0["factor_value"].iloc[0] == 0 or np.isnan(b0["factor_value"].iloc[0])


# ---------- 3. 非突破日 factor = 0 ----------
def test_non_breakout_is_zero():
    panel = _make_panel("000003", DATES, OHLCV)
    # 用超大 L 使几乎所有日都不是突破
    result = breakout_close_quality(panel, L=60, ATR_N=14, VOL_N=20)
    non_b = result[~result["breakout_event"] & result["factor_value"].notna()]
    assert (non_b["factor_value"] == 0).all(), "非突破日 factor 应为 0"


# ---------- 4. 因子值在 [0,1] 范围 ----------
def test_factor_range():
    panel = _make_panel("000004", DATES, OHLCV)
    result = breakout_close_quality(panel, L=20, ATR_N=14, VOL_N=20)
    valid = result["factor_value"].dropna()
    assert valid.between(0, 1).all(), f"factor 超出 [0,1]: min={valid.min()}, max={valid.max()}"


# ---------- 5. high==low 不报错，返回 NaN ----------
def test_zero_range_returns_nan():
    dates = ["2025-11-01", "2025-11-02", "2025-11-03", "2025-11-04", "2025-11-05"]
    ohlcv = [
        (10, 10, 10, 10, 0),       # 一字板 high==low
        (11, 12, 10.5, 11.5, 1e6),
        (12, 12.5, 11.5, 12.2, 1e6),
        (12.2, 12.3, 12.1, 12.25, 5e5),
        (12.25, 13, 12, 12.8, 1e6),
    ]
    panel = _make_panel("000005", dates, ohlcv)
    result = breakout_close_quality(panel, L=3, ATR_N=14, VOL_N=20)
    d0 = result[result["trade_date"] == pd.Timestamp("2025-11-01")]
    assert np.isnan(d0["factor_value"].iloc[0]), "high==low 应返回 NaN"


# ---------- 6. 窗口不足返回 NaN ----------
def test_insufficient_window_returns_nan():
    dates = [f"2025-12-{d:02d}" for d in range(1, 6)]
    ohlcv = [(10 + i, 11 + i, 9 + i, 10.5 + i, 1e6) for i in range(5)]
    panel = _make_panel("000006", dates, ohlcv)
    result = breakout_close_quality(panel, L=20, ATR_N=14, VOL_N=20)
    # 所有日期的 L 窗口都不足，应为 NaN
    assert result["factor_value"].isna().all(), "窗口不足应全为 NaN"


# ---------- 7. 多股票模式 ----------
def test_multi_stock():
    panel1 = _make_panel("A", DATES, OHLCV)
    panel2 = _make_panel("B", DATES, [(o + 5, hi + 5, lo + 5, cl + 5, vo) for o, hi, lo, cl, vo in OHLCV])
    panel = pd.concat([panel1, panel2], ignore_index=True)
    result = breakout_close_quality(panel, L=20, ATR_N=14, VOL_N=20)
    assert set(result["symbol"].unique()) == {"A", "B"}
    assert list(result.columns) == ["trade_date", "symbol", "factor_value", "breakout_event", "raw_quality"]


# ---------- 8. 缺少列报错 ----------
def test_missing_columns_raises():
    panel = pd.DataFrame({"trade_date": [], "symbol": []})
    with pytest.raises(ValueError, match="缺少列"):
        breakout_close_quality(panel)


# ---------- 9. raw_quality 列 ----------
def test_raw_quality_column():
    """raw_quality: BO日=factor_value, 非BO日=NaN。"""
    panel = _make_panel("000007", DATES, OHLCV)
    result = breakout_close_quality(panel, L=20, ATR_N=14, VOL_N=20)
    assert "raw_quality" in result.columns
    bo = result[result["breakout_event"]]
    nbo = result[~result["breakout_event"] & result["factor_value"].notna()]
    # BO 日: raw_quality == factor_value（因子即 raw，非突破=0）
    if len(bo) > 0:
        assert np.allclose(bo["raw_quality"].to_numpy(), bo["factor_value"].to_numpy(), equal_nan=True)
    # 非BO日: raw_quality = NaN
    if len(nbo) > 0:
        assert nbo["raw_quality"].isna().all()
    # range [0,1]
    valid_raw = result["raw_quality"].dropna()
    if len(valid_raw) > 0:
        assert valid_raw.between(0, 1).all()
