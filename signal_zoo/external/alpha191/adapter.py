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
# Helpers
# ============================================================


def _ts_rank(series: pd.Series, window: int) -> pd.Series:
    """Rolling rank: rank of last value within each rolling window, normalized to [0, 1]."""
    def _rank_last(x):
        if len(x) < window:
            return np.nan
        return (x.rank().iloc[-1] - 1) / (window - 1)
    return series.rolling(window, min_periods=window).apply(_rank_last, raw=False)


def _safe_div(a: pd.Series, b: pd.Series) -> pd.Series:
    """Element-wise division with zero → NaN."""
    return a / b.replace(0, np.nan)


# ============================================================
# Sprint 2 factors (Signal017-020)
# ============================================================


def gtja_002_reversal(df: pd.DataFrame) -> pd.Series:
    h_l = df["high"] - df["low"]
    mid_pos = ((df["close"] - df["low"]) - (df["high"] - df["close"])) / h_l.replace(0, np.nan)
    return -1.0 * mid_pos.diff(1)


def gtja_004_volume_price(df: pd.DataFrame) -> pd.Series:
    c = df["close"]; v = df["volume"]
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
    return df["amount"].rolling(6, min_periods=6).std()


def gtja_085_momentum(df: pd.DataFrame) -> pd.Series:
    v = df["volume"]; c = df["close"]
    vol_ratio = v / v.rolling(20, min_periods=20).mean()
    arm1 = _ts_rank(vol_ratio, 20)
    arm2 = _ts_rank(-1.0 * c.diff(7), 8)
    return arm1 * arm2


# ============================================================
# Sprint 3: Momentum factors (Signal021-026)
# ============================================================


def gtja_014_momentum(df: pd.DataFrame) -> pd.Series:
    """5-day simple price momentum."""
    return df["close"] - df["close"].shift(5)


def gtja_053_momentum(df: pd.DataFrame) -> pd.Series:
    """12-day up-day percentage."""
    up = (df["close"] > df["close"].shift(1)).astype(float)
    return up.rolling(12, min_periods=12).mean() * 100


def gtja_088_momentum(df: pd.DataFrame) -> pd.Series:
    """20-day percentage price change."""
    return (df["close"] / df["close"].shift(20) - 1) * 100


def gtja_106_momentum(df: pd.DataFrame) -> pd.Series:
    """20-day absolute price change."""
    return df["close"] - df["close"].shift(20)


def gtja_112_momentum(df: pd.DataFrame) -> pd.Series:
    """12-day Chande Momentum Oscillator."""
    dc = df["close"].diff()
    pos = dc.clip(lower=0).rolling(12, min_periods=12).sum()
    neg = (-dc.clip(upper=0)).rolling(12, min_periods=12).sum()
    return _safe_div(pos - neg, pos + neg) * 100


def gtja_167_momentum(df: pd.DataFrame) -> pd.Series:
    """12-day cumulative up-move."""
    dc = df["close"].diff()
    return dc.clip(lower=0).rolling(12, min_periods=12).sum()


# ============================================================
# Sprint 3: Reversal factors (Signal027-030)
# ============================================================


def gtja_046_reversal(df: pd.DataFrame) -> pd.Series:
    """Multi-MA price ratio (>1 = oversold)."""
    c = df["close"]
    ma_avg = (c.rolling(3).mean() + c.rolling(6).mean() +
              c.rolling(12).mean() + c.rolling(24).mean()) / 4
    return ma_avg / c


def gtja_065_reversal(df: pd.DataFrame) -> pd.Series:
    """6-day MA / price ratio."""
    return df["close"].rolling(6, min_periods=6).mean() / df["close"]


def gtja_066_reversal(df: pd.DataFrame) -> pd.Series:
    """Percent deviation from 6-day MA."""
    ma6 = df["close"].rolling(6, min_periods=6).mean()
    return (df["close"] - ma6) / ma6 * 100


def gtja_078_reversal(df: pd.DataFrame) -> pd.Series:
    """CCI-type oscillator on typical price."""
    tp = (df["high"] + df["low"] + df["close"]) / 3
    ma_tp = tp.rolling(12, min_periods=12).mean()
    mad = (tp - ma_tp).abs().rolling(12, min_periods=12).mean()
    return (tp - ma_tp) / (0.015 * mad.replace(0, np.nan))


# ============================================================
# Sprint 3: Volume-Price factors (Signal031-037)
# ============================================================


def gtja_011_volume_price(df: pd.DataFrame) -> pd.Series:
    """6-day sum of volume-weighted intraday position."""
    h, l, c, v = df["high"], df["low"], df["close"], df["volume"]
    pos = _safe_div(2 * c - l - h, h - l) * v
    return pos.rolling(6, min_periods=6).sum()


def gtja_032_volume_price(df: pd.DataFrame) -> pd.Series:
    """Negative sum of ranked hi-vol correlation."""
    h_rank = _ts_rank(df["high"], 3)
    v_rank = _ts_rank(df["volume"], 3)
    corr = h_rank.rolling(3, min_periods=3).corr(v_rank)
    corr_rank = _ts_rank(corr, 3)
    return -corr_rank.rolling(3, min_periods=3).sum()


def gtja_084_volume_price(df: pd.DataFrame) -> pd.Series:
    """20-day signed volume (buying pressure)."""
    sign = np.sign(df["close"].diff()).fillna(0)
    return (sign * df["volume"]).rolling(20, min_periods=20).sum()


def gtja_102_volume_price(df: pd.DataFrame) -> pd.Series:
    """Volume RSI: SMA(pos_dV) / SMA(abs_dV) * 100."""
    dv = df["volume"].diff()
    pos = dv.clip(lower=0).rolling(10, min_periods=10).mean()
    abs_dv = dv.abs().rolling(10, min_periods=10).mean()
    return _safe_div(pos, abs_dv) * 100


def gtja_128_volume_price(df: pd.DataFrame) -> pd.Series:
    """14-day Money Flow Index."""
    tp = (df["high"] + df["low"] + df["close"]) / 3
    mf = tp * df["volume"]
    tp_diff = tp.diff()
    pos_mf = mf.where(tp_diff > 0, 0).rolling(14, min_periods=14).sum()
    neg_mf = mf.where(tp_diff < 0, 0).rolling(14, min_periods=14).sum()
    return _safe_div(pos_mf, pos_mf + neg_mf) * 100


def gtja_150_volume_price(df: pd.DataFrame) -> pd.Series:
    """Typical price × log(volume)."""
    tp = (df["close"] + df["high"] + df["low"]) / 3
    return tp * np.log(df["volume"].replace(0, np.nan))


def gtja_178_volume_price(df: pd.DataFrame) -> pd.Series:
    """Volume-weighted daily return."""
    ret = df["close"].pct_change()
    return ret * df["volume"]


# ============================================================
# Sprint 3: Volatility factors (Signal038-042)
# ============================================================


def gtja_049_volatility(df: pd.DataFrame) -> pd.Series:
    """12-day asymmetric range: down-day range share."""
    tr = df["high"] - df["low"]
    down_tr = tr.where(df["close"] < df["close"].shift(1), 0)
    return _safe_div(
        down_tr.rolling(12, min_periods=12).sum(),
        tr.rolling(12, min_periods=12).sum(),
    )


def gtja_076_volatility(df: pd.DataFrame) -> pd.Series:
    """CV of volume-adjusted absolute returns over 20 days."""
    ret = df["close"].pct_change().abs()
    adj = _safe_div(ret, df["volume"])
    std_adj = adj.rolling(20, min_periods=20).std()
    mean_adj = adj.rolling(20, min_periods=20).mean()
    return _safe_div(std_adj, mean_adj)


def gtja_095_volatility(df: pd.DataFrame) -> pd.Series:
    """20-day amount standard deviation."""
    return df["amount"].rolling(20, min_periods=20).std()


def gtja_158_volatility(df: pd.DataFrame) -> pd.Series:
    """Normalized daily range (high-low / close)."""
    return (df["high"] - df["low"]) / df["close"].replace(0, np.nan)


def gtja_161_volatility(df: pd.DataFrame) -> pd.Series:
    """12-day average true range."""
    c_prev = df["close"].shift(1)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - c_prev).abs(),
        (df["low"] - c_prev).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(12, min_periods=12).mean()


# ============================================================
# Sprint 3: Trend / MA / Breakout factors (Signal043-046)
# ============================================================


def gtja_089_trend(df: pd.DataFrame) -> pd.Series:
    """MACD-type oscillator: 2 × (MACD - signal)."""
    c = df["close"]
    macd = c.rolling(13, min_periods=13).mean() - c.rolling(27, min_periods=27).mean()
    signal = macd.rolling(10, min_periods=10).mean()
    return 2 * (macd - signal)


def gtja_096_trend(df: pd.DataFrame) -> pd.Series:
    """Double-smoothed Stochastic %K (9,3,3)."""
    llv = df["low"].rolling(9, min_periods=9).min()
    hhv = df["high"].rolling(9, min_periods=9).max()
    raw_k = _safe_div(df["close"] - llv, hhv - llv) * 100
    k = raw_k.rolling(3, min_periods=3).mean()
    return k.rolling(3, min_periods=3).mean()


def gtja_153_trend(df: pd.DataFrame) -> pd.Series:
    """BBI: average of 4 MAs (3, 6, 12, 24)."""
    c = df["close"]
    return (c.rolling(3).mean() + c.rolling(6).mean() +
            c.rolling(12).mean() + c.rolling(24).mean()) / 4


def gtja_172_trend(df: pd.DataFrame) -> pd.Series:
    """ADX-type: 6-day average of 14-period DX."""
    h, l, c = df["high"], df["low"], df["close"]
    c_prev = c.shift(1)
    tr = pd.concat([h - l, (h - c_prev).abs(), (l - c_prev).abs()], axis=1).max(axis=1)
    atr = tr.rolling(14, min_periods=14).mean()

    up_move = h - h.shift(1)
    down_move = l.shift(1) - l
    plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0)
    minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0)

    plus_di = _safe_div(plus_dm.rolling(14, min_periods=14).mean(), atr) * 100
    minus_di = _safe_div(minus_dm.rolling(14, min_periods=14).mean(), atr) * 100
    dx = _safe_div((plus_di - minus_di).abs(), plus_di + minus_di) * 100
    return dx.rolling(6, min_periods=6).mean()


# ============================================================
# Factor registry (30 entries)
# ============================================================

def _mk_entry(sid, sname, cat, fid, desc, fn):
    return {
        "signal_id": sid,
        "signal_name": sname,
        "category": "Price",
        "sub_category": cat,
        "source": "External",
        "source_library": "aurumq-rl/GTJA191",
        "source_formula_id": fid,
        "data_requirement": "Daily OHLCV",
        "frequency": "Daily",
        "description": desc,
        "compute_fn": fn,
    }


FACTOR_REGISTRY = {
    # Sprint 2
    "gtja_002": _mk_entry("Signal017", "Alpha191_Reversal_GTJA002", "Reversal", "gtja_002",
                          "Williams %R proxy: mid-range position reversal", gtja_002_reversal),
    "gtja_004": _mk_entry("Signal018", "Alpha191_VolumePrice_GTJA004", "Volume-Price", "gtja_004",
                          "Trend regime ternary with volume gate", gtja_004_volume_price),
    "gtja_070": _mk_entry("Signal019", "Alpha191_Volatility_GTJA070", "Volatility", "gtja_070",
                          "6-day rolling std of amount", gtja_070_volatility),
    "gtja_085": _mk_entry("Signal020", "Alpha191_Momentum_GTJA085", "Momentum", "gtja_085",
                          "TS-rank of volume ratio × TS-rank of negated price delta", gtja_085_momentum),
    # Sprint 3: Momentum
    "gtja_014": _mk_entry("Signal021", "Alpha191_Momentum_GTJA014", "Momentum", "gtja_014",
                          "5-day simple price momentum", gtja_014_momentum),
    "gtja_053": _mk_entry("Signal022", "Alpha191_Momentum_GTJA053", "Momentum", "gtja_053",
                          "12-day up-day percentage", gtja_053_momentum),
    "gtja_088": _mk_entry("Signal023", "Alpha191_Momentum_GTJA088", "Momentum", "gtja_088",
                          "20-day pct price change", gtja_088_momentum),
    "gtja_106": _mk_entry("Signal024", "Alpha191_Momentum_GTJA106", "Momentum", "gtja_106",
                          "20-day absolute price change", gtja_106_momentum),
    "gtja_112": _mk_entry("Signal025", "Alpha191_Momentum_GTJA112", "Momentum", "gtja_112",
                          "12-day Chande Momentum Oscillator", gtja_112_momentum),
    "gtja_167": _mk_entry("Signal026", "Alpha191_Momentum_GTJA167", "Momentum", "gtja_167",
                          "12-day cumulative up-move", gtja_167_momentum),
    # Sprint 3: Reversal
    "gtja_046": _mk_entry("Signal027", "Alpha191_Reversal_GTJA046", "Reversal", "gtja_046",
                          "Multi-MA price ratio (oversold detector)", gtja_046_reversal),
    "gtja_065": _mk_entry("Signal028", "Alpha191_Reversal_GTJA065", "Reversal", "gtja_065",
                          "6-day MA / price ratio", gtja_065_reversal),
    "gtja_066": _mk_entry("Signal029", "Alpha191_Reversal_GTJA066", "Reversal", "gtja_066",
                          "Pct deviation from 6-day MA", gtja_066_reversal),
    "gtja_078": _mk_entry("Signal030", "Alpha191_Reversal_GTJA078", "Reversal", "gtja_078",
                          "CCI-type oscillator on typical price", gtja_078_reversal),
    # Sprint 3: Volume-Price
    "gtja_011": _mk_entry("Signal031", "Alpha191_VolPrice_GTJA011", "Volume-Price", "gtja_011",
                          "6d sum of volume-weighted intraday position", gtja_011_volume_price),
    "gtja_032": _mk_entry("Signal032", "Alpha191_VolPrice_GTJA032", "Volume-Price", "gtja_032",
                          "Negative sum of ranked hi-vol correlation", gtja_032_volume_price),
    "gtja_084": _mk_entry("Signal033", "Alpha191_VolPrice_GTJA084", "Volume-Price", "gtja_084",
                          "20-day signed volume (buying pressure)", gtja_084_volume_price),
    "gtja_102": _mk_entry("Signal034", "Alpha191_VolPrice_GTJA102", "Volume-Price", "gtja_102",
                          "Volume RSI: SMA(pos_dV) / SMA(abs_dV)", gtja_102_volume_price),
    "gtja_128": _mk_entry("Signal035", "Alpha191_VolPrice_GTJA128", "Volume-Price", "gtja_128",
                          "14-day Money Flow Index", gtja_128_volume_price),
    "gtja_150": _mk_entry("Signal036", "Alpha191_VolPrice_GTJA150", "Volume-Price", "gtja_150",
                          "Typical price × log(volume)", gtja_150_volume_price),
    "gtja_178": _mk_entry("Signal037", "Alpha191_VolPrice_GTJA178", "Volume-Price", "gtja_178",
                          "Volume-weighted daily return", gtja_178_volume_price),
    # Sprint 3: Volatility
    "gtja_049": _mk_entry("Signal038", "Alpha191_Volatility_GTJA049", "Volatility", "gtja_049",
                          "12-day asymmetric range (down-day share)", gtja_049_volatility),
    "gtja_076": _mk_entry("Signal039", "Alpha191_Volatility_GTJA076", "Volatility", "gtja_076",
                          "CV of volume-adjusted absolute returns", gtja_076_volatility),
    "gtja_095": _mk_entry("Signal040", "Alpha191_Volatility_GTJA095", "Volatility", "gtja_095",
                          "20-day amount std", gtja_095_volatility),
    "gtja_158": _mk_entry("Signal041", "Alpha191_Volatility_GTJA158", "Volatility", "gtja_158",
                          "Normalized daily range (H-L)/C", gtja_158_volatility),
    "gtja_161": _mk_entry("Signal042", "Alpha191_Volatility_GTJA161", "Volatility", "gtja_161",
                          "12-day average true range", gtja_161_volatility),
    # Sprint 3: Trend/MA/Breakout
    "gtja_089": _mk_entry("Signal043", "Alpha191_Trend_GTJA089", "Trend", "gtja_089",
                          "MACD-type: 2×(MACD-signal)", gtja_089_trend),
    "gtja_096": _mk_entry("Signal044", "Alpha191_Trend_GTJA096", "Trend", "gtja_096",
                          "Double-smoothed Stochastic %%K (9,3,3)", gtja_096_trend),
    "gtja_153": _mk_entry("Signal045", "Alpha191_Trend_GTJA153", "Trend", "gtja_153",
                          "BBI: average of MA3/6/12/24", gtja_153_trend),
    "gtja_172": _mk_entry("Signal046", "Alpha191_Trend_GTJA172", "Trend", "gtja_172",
                          "ADX-type: 6d avg of 14-period DX", gtja_172_trend),
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
    stock_code: str, factor_key: str,
    start_date: str = "2025-01-01", end_date: str = "2025-12-31",
) -> Optional[pd.DataFrame]:
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
    stock_codes: list[str], factor_key: str,
    start_date: str = "2025-01-01", end_date: str = "2025-12-31",
    zscore: bool = True,
) -> pd.DataFrame:
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
        g = result.groupby("trade_date")["signal_value"]
        result["signal_value"] = (result["signal_value"] - g.transform("mean")) / g.transform("std").replace(0, np.nan)
    result["signal_id"] = info["signal_id"]
    result["signal_name"] = info["signal_name"]
    result["source"] = "aurumq-rl/GTJA191"
    result["source_formula_id"] = info.get("source_formula_id", factor_key)
    return result[["trade_date", "stock_code", "signal_id", "signal_value", "signal_name", "source", "source_formula_id"]]
