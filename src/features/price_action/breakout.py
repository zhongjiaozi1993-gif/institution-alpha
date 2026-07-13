"""breakout_close_quality — 突破K线收盘质量因子。

规格来源：institution-alpha-research/fangfangtu-price-action/factor_specs/breakout_close_quality.md

因子评价突破发生当天K线的质量（实体、收盘位置、ATR标准化突破幅度、成交量异常）。
非突破日 factor=0，不可交易/数据不足 factor=NaN。
T 日收盘后生成，最早用于 T+1 open。
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def breakout_close_quality(
    panel: pd.DataFrame,
    L: int = 20,
    ATR_N: int = 20,
    VOL_N: int = 20,
) -> pd.DataFrame:
    """为多股票日线 panel 计算 breakout_close_quality 因子。

    Args:
        panel: [trade_date, symbol, open, high, low, close, volume]，按 symbol 分组、日期升序。
        L: 突破 lookback 窗口。
        ATR_N: ATR 计算窗口。
        VOL_N: 成交量 z-score 窗口。

    Returns:
        [trade_date, symbol, factor_value, breakout_event]
        factor_value: [0, 1] 连续质量分，非突破日为 0，无效为 NaN。
        breakout_event: bool，当日是否发生向上突破。
    """
    required = {"trade_date", "symbol", "open", "high", "low", "close", "volume"}
    missing = required - set(panel.columns)
    if missing:
        raise ValueError(f"panel 缺少列: {missing}")

    out = []
    for symbol, g in panel.sort_values("trade_date").groupby("symbol"):
        g = g.reset_index(drop=True)
        res = _single_stock(g, L, ATR_N, VOL_N)
        res["symbol"] = symbol
        out.append(res)

    if not out:
        return pd.DataFrame(columns=["trade_date", "symbol", "factor_value", "breakout_event", "raw_quality"])

    result = pd.concat(out, ignore_index=True)
    return result[["trade_date", "symbol", "factor_value", "breakout_event", "raw_quality"]]


def _single_stock(
    df: pd.DataFrame,
    L: int,
    ATR_N: int,
    VOL_N: int,
) -> pd.DataFrame:
    """单只股票的 breakout_close_quality 计算。"""
    n = len(df)
    out = pd.DataFrame({"trade_date": df["trade_date"].values})

    o = df["open"].to_numpy(float)
    h = df["high"].to_numpy(float)
    l = df["low"].to_numpy(float)
    c = df["close"].to_numpy(float)
    v = df["volume"].to_numpy(float)

    # ---- 突破事件 ----
    rolling_high = np.full(n, np.nan)
    for i in range(L, n):
        rolling_high[i] = np.max(h[i - L : i])  # i-L to i-1 (excludes i)

    breakout = (c > rolling_high) & (~np.isnan(rolling_high))

    # ---- 组件 ----
    rng = h - l
    body_ratio = np.where(rng > 0, np.maximum(c - o, 0.0) / rng, np.nan)
    close_pos = np.where(rng > 0, (c - l) / rng, np.nan)

    # ATR (使用 T-1 及以前的 True Range，不含 T 日)
    tr = np.full(n, np.nan)
    for i in range(1, n):
        tr[i] = max(h[i] - l[i], abs(h[i] - c[i - 1]), abs(l[i] - c[i - 1]))
    atr = np.full(n, np.nan)
    for i in range(ATR_N + 1, n):
        atr[i] = np.nanmean(tr[i - ATR_N : i])  # T-ATR_N to T-1

    depth_atr = np.where(atr > 0, (c - rolling_high) / atr, np.nan)

    # Volume z-score (T-1 及以前窗口)
    log_v = np.log(v + 1.0)
    vol_mean = np.full(n, np.nan)
    vol_std = np.full(n, np.nan)
    for i in range(VOL_N + 1, n):
        window = log_v[i - VOL_N : i]  # T-VOL_N to T-1
        vol_mean[i] = np.nanmean(window)
        vol_std[i] = np.nanstd(window)

    vol_z = np.where(vol_std > 0, (log_v - vol_mean) / vol_std, np.nan)
    vol_z = np.clip(vol_z, -5.0, 5.0)
    vol_score = np.clip((vol_z - 0.5) / 2.0, 0.0, 1.0)

    # Depth score
    depth_score = np.clip(depth_atr / 1.5, 0.0, 1.0)

    # ---- 因子合成 ----
    raw = (
        0.35 * np.clip(body_ratio, 0.0, 1.0)
        + 0.30 * np.clip(close_pos, 0.0, 1.0)
        + 0.20 * depth_score
        + 0.15 * vol_score
    )

    # 非突破日 factor = 0
    factor = np.where(breakout, raw, 0.0)

    # 不可计算的情况 → NaN
    valid_bar = (rng > 0) & (~np.isnan(rolling_high))
    factor = np.where(valid_bar, factor, np.nan)

    out["factor_value"] = factor
    out["breakout_event"] = breakout & valid_bar
    out["raw_quality"] = np.where(valid_bar & breakout, raw, np.nan)
    return out
