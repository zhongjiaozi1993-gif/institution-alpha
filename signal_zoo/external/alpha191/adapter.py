"""GTJA Alpha191 Signal Adapter.

Wraps selected GTJA191 factor formulas into the unified Signal interface.
Formulas sourced from aurumq-rl (yupoet/aurumq-rl, MIT License).

Each factor is a per-stock time-series computation on daily OHLCV data.
Output: standardized signal DataFrame with columns
[trade_date, stock_code, signal_id, signal_value, signal_name, source].

Reference: Guotai Junan 2017 "基于短周期量价特征的多因子选股体系"
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional

PROJECT = Path(__file__).resolve().parent.parent.parent.parent
DAILY_DIR = PROJECT / "data" / "daily"
SIGNAL_DIR = PROJECT / "data" / "processed" / "signals" / "price_alpha191"

# ============================================================
# Core factor formulas (per-stock time-series, pandas)
# ============================================================


def _ts_rank(series: pd.Series, window: int) -> pd.Series:
    """Rolling rank: rank of last value within each rolling window.

    Equivalent to Polars ts_rank: for each position i, rank of
    series[i] among series[i-window+1 : i+1], normalized to [0, 1].
    """
    def _rank_last(x):
        if len(x) < window:
            return np.nan
        # rank of last element within window
        return (x.rank().iloc[-1] - 1) / (window - 1)
    return series.rolling(window, min_periods=window).apply(_rank_last, raw=False)


def gtja_002_reversal(df: pd.DataFrame) -> pd.Series:
    """GTJA #002: Williams %R proxy — 1-period delta of mid-range position.

    Formula: -1 * Δ(((C-L) - (H-C)) / (H-L), 1)
    Category: mean_reversion
    Direction: reverse (negative value = buy signal)
    """
    h_l = df["high"] - df["low"]
    mid_pos = ((df["close"] - df["low"]) - (df["high"] - df["close"])) / h_l.replace(0, np.nan)
    return -1.0 * mid_pos.diff(1)


def gtja_004_volume_price(df: pd.DataFrame) -> pd.Series:
    """GTJA #004: Trend regime conditional with volume gate.

    Formula:
        if (MA(C,8) + STD(C,8)) < MA(C,2): -1
        elif MA(C,2) < (MA(C,8) - STD(C,8)): 1
        elif V / MA(V,20) >= 1: 1
        else: -1

    Category: mean_reversion (practical: volume_price pattern)
    Direction: normal (positive = buy signal)
    """
    c = df["close"]
    v = df["volume"]
    ma8 = c.rolling(8, min_periods=8).mean()
    std8 = c.rolling(8, min_periods=8).std()
    ma2 = c.rolling(2, min_periods=2).mean()
    vol_ratio = v / v.rolling(20, min_periods=20).mean()

    cond1 = (ma8 + std8) < ma2
    cond2 = ma2 < (ma8 - std8)
    cond3 = vol_ratio >= 1.0

    result = pd.Series(0.0, index=df.index)
    result[cond1] = -1.0
    result[cond2] = 1.0
    result[~cond1 & ~cond2 & cond3] = 1.0
    result[~cond1 & ~cond2 & ~cond3] = -1.0
    return result


def gtja_070_volatility(df: pd.DataFrame) -> pd.Series:
    """GTJA #070: 6-day rolling std of amount.

    Formula: STD(amount, 6)
    Category: volatility
    Direction: normal (positive = higher volatility)
    """
    return df["amount"].rolling(6, min_periods=6).std()


def gtja_085_momentum(df: pd.DataFrame) -> pd.Series:
    """GTJA #085: Volume-ratio TS-rank × negated close-delta TS-rank.

    Formula: TSRANK(V / MA(V,20), 20) × TSRANK(-Δ(C,7), 8)
    Category: momentum
    Direction: reverse (negative value = buy signal)
    """
    v = df["volume"]
    c = df["close"]
    vol_ratio = v / v.rolling(20, min_periods=20).mean()
    arm1 = _ts_rank(vol_ratio, 20)
    arm2 = _ts_rank(-1.0 * c.diff(7), 8)
    return arm1 * arm2


# ============================================================
# Factor registry
# ============================================================

FACTOR_REGISTRY = {
    "gtja_002": {
        "signal_id": "Signal017",
        "signal_name": "Alpha191_Reversal_GTJA002",
        "category": "Price",
        "source": "External",
        "source_library": "aurumq-rl/GTJA191",
        "source_formula_id": "gtja_002",
        "data_requirement": "Daily OHLCV",
        "frequency": "Daily",
        "description": "Williams %R proxy: mid-range position reversal",
        "compute_fn": gtja_002_reversal,
    },
    "gtja_004": {
        "signal_id": "Signal018",
        "signal_name": "Alpha191_VolumePrice_GTJA004",
        "category": "Price",
        "source": "External",
        "source_library": "aurumq-rl/GTJA191",
        "source_formula_id": "gtja_004",
        "data_requirement": "Daily OHLCV",
        "frequency": "Daily",
        "description": "Trend regime ternary with volume gate",
        "compute_fn": gtja_004_volume_price,
    },
    "gtja_070": {
        "signal_id": "Signal019",
        "signal_name": "Alpha191_Volatility_GTJA070",
        "category": "Price",
        "source": "External",
        "source_library": "aurumq-rl/GTJA191",
        "source_formula_id": "gtja_070",
        "data_requirement": "Daily OHLCV",
        "frequency": "Daily",
        "description": "6-day rolling std of amount",
        "compute_fn": gtja_070_volatility,
    },
    "gtja_085": {
        "signal_id": "Signal020",
        "signal_name": "Alpha191_Momentum_GTJA085",
        "category": "Price",
        "source": "External",
        "source_library": "aurumq-rl/GTJA191",
        "source_formula_id": "gtja_085",
        "data_requirement": "Daily OHLCV",
        "frequency": "Daily",
        "description": "TS-rank of volume ratio × TS-rank of negated price delta",
        "compute_fn": gtja_085_momentum,
    },
}


# ============================================================
# Adapter: load data, compute factor, output Signal DataFrame
# ============================================================


def load_daily_data(stock_code: str) -> Optional[pd.DataFrame]:
    """Load daily OHLCV parquet for a stock."""
    p = DAILY_DIR / f"{stock_code}.parquet"
    if not p.exists():
        return None
    df = pd.read_parquet(p)
    df = df.sort_values("date").reset_index(drop=True)
    # Standardize column names (handle possible Chinese columns)
    col_map = {}
    for c in df.columns:
        cl = c.lower().strip()
        if cl in ("date", "trade_date"):
            col_map[c] = "date"
        elif cl in ("open",):
            col_map[c] = "open"
        elif cl in ("high",):
            col_map[c] = "high"
        elif cl in ("low",):
            col_map[c] = "low"
        elif cl in ("close",):
            col_map[c] = "close"
        elif cl in ("volume", "vol"):
            col_map[c] = "volume"
        elif cl in ("amount", "amt"):
            col_map[c] = "amount"
    if col_map:
        df = df.rename(columns=col_map)
    return df


def compute_signal_for_stock(
    stock_code: str,
    factor_key: str,
    start_date: str = "2025-01-01",
    end_date: str = "2025-12-31",
) -> Optional[pd.DataFrame]:
    """Compute one GTJA191 factor for one stock over a date range.

    Returns DataFrame with columns:
        trade_date, stock_code, signal_value
    """
    df = load_daily_data(stock_code)
    if df is None:
        return None
    info = FACTOR_REGISTRY[factor_key]
    fn = info["compute_fn"]

    mask = (df["date"] >= start_date) & (df["date"] <= end_date)
    df = df[mask].copy()

    values = fn(df)
    out = pd.DataFrame({
        "trade_date": df["date"].values,
        "stock_code": stock_code,
        "signal_value": values.values,
    })
    out["signal_value"] = out["signal_value"].replace([np.inf, -np.inf], np.nan)
    return out


def compute_signal_batch(
    stock_codes: list[str],
    factor_key: str,
    start_date: str = "2025-01-01",
    end_date: str = "2025-12-31",
    zscore: bool = True,
) -> pd.DataFrame:
    """Compute one GTJA191 factor for multiple stocks.

    Returns unified Signal DataFrame:
        trade_date, stock_code, signal_id, signal_value, signal_name, source

    If zscore=True, cross-sectionally z-score signal_value per date.
    """
    info = FACTOR_REGISTRY[factor_key]
    frames = []
    for code in stock_codes:
        df = compute_signal_for_stock(code, factor_key, start_date, end_date)
        if df is not None and len(df) > 0:
            frames.append(df)

    if not frames:
        return pd.DataFrame()

    result = pd.concat(frames, ignore_index=True)

    if zscore:
        # Cross-sectional z-score per date
        g = result.groupby("trade_date")["signal_value"]
        result["signal_value"] = (result["signal_value"] - g.transform("mean")) / g.transform("std").replace(0, np.nan)

    result["signal_id"] = info["signal_id"]
    result["signal_name"] = info["signal_name"]
    result["source"] = "aurumq-rl/GTJA191"
    return result[["trade_date", "stock_code", "signal_id", "signal_value", "signal_name", "source"]]
