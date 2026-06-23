"""
市场环境检测
牛/熊/震荡市识别，自适应仓位调整
"""
from __future__ import annotations
import numpy as np
import pandas as pd
try:
    from loguru import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)


def detect_regime(
    index_data: pd.DataFrame,
    lookback: int = 60,
) -> str:
    """
    检测当前市场环境

    基于趋势方向和波动率：
    - BULL:    趋势>10% 且 波动率<20%
    - BEAR:    趋势<-10% 且 波动率>25%
    - OSCILLATE: 其他情况

    Returns: "BULL" | "BEAR" | "OSCILLATE"
    """
    if index_data.empty or len(index_data) < lookback:
        return "OSCILLATE"

    recent = index_data.tail(lookback).copy()
    close = recent["close"]

    returns = close.pct_change().dropna()
    trend = close.iloc[-1] / close.iloc[0] - 1
    volatility = returns.std() * np.sqrt(252)

    if trend > 0.1 and volatility < 0.2:
        return "BULL"
    elif trend < -0.1 and volatility > 0.25:
        return "BEAR"
    else:
        return "OSCILLATE"


REGIME_ADJUSTMENTS = {
    "BULL": {
        "position_multiplier": 1.2,
        "stop_loss_relax": 0.02,  # 放宽止损2%
        "description": "牛市 — 趋势向上、低波动",
    },
    "BEAR": {
        "position_multiplier": 0.5,
        "stop_loss_tighten": 0.03,  # 收紧止损3%
        "ignore_tier_C": True,
        "description": "熊市 — 趋势向下、高波动",
    },
    "OSCILLATE": {
        "position_multiplier": 0.8,
        "time_stop_shorten": 10,  # 缩短持有期到10天
        "description": "震荡市 — 趋势不明",
    },
}


def adjust_for_regime(
    signals: pd.DataFrame,
    regime: str,
    adjustments: dict | None = None,
) -> pd.DataFrame:
    """
    根据市场环境调整信号强度
    """
    if signals.empty or regime not in REGIME_ADJUSTMENTS:
        return signals

    adj = adjustments or REGIME_ADJUSTMENTS[regime]
    df = signals.copy()
    mult = adj.get("position_multiplier", 1.0)
    df["regime"] = regime
    df["strength"] = (df["strength"] * mult).clip(upper=1.0)

    if adj.get("ignore_tier_C") and "tier" in df.columns:
        df = df[df["tier"] != "C"]

    logger.info(f"市场环境: {regime} — {adj['description']}，仓位乘数: {mult}")
    return df
