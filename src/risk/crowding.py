"""
信号拥挤度检测
某股票近期被过多机构买入 → 信号可能失效
"""
from __future__ import annotations
import pandas as pd
try:
    from loguru import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)


def calculate_crowding(
    stock_code: str,
    signals_history: pd.DataFrame,
    lookback: int = 20,
    threshold: float = 0.8,
) -> tuple[str, float]:
    """
    检测单只股票的信号拥挤度

    signal_history: [stock_code, lhb_date, seat_name, strength, ...]
    lookback: 回顾窗口（天）
    threshold: 拥挤阈值

    Returns: ("CROWDED" | "WARM" | "NORMAL", crowding_score)
    """
    recent = signals_history[
        (signals_history["stock_code"] == stock_code) &
        (signals_history["lhb_date"] >= pd.Timestamp.now() - pd.Timedelta(days=lookback))
    ]

    if len(recent) == 0:
        return "NORMAL", 0.0

    signal_density = len(recent) / lookback

    if "strength" in recent.columns:
        avg_strength = recent["strength"].mean()
        std_strength = recent["strength"].std()
        cv = std_strength / avg_strength if avg_strength > 0 else 1.0
    else:
        cv = 0.5

    # 拥挤度 = 信号密度 * (1 - 变异系数) → 密度高且信号一致表示拥挤
    crowding_score = signal_density * (1 - min(cv, 1))

    if crowding_score > threshold:
        status = "CROWDED"
        logger.debug(f"{stock_code} 信号拥挤 (score={crowding_score:.2f})")
    elif crowding_score > threshold * 0.5:
        status = "WARM"
    else:
        status = "NORMAL"

    return status, crowding_score


def filter_crowded_stocks(
    signals: pd.DataFrame,
    signal_history: pd.DataFrame,
    lookback: int = 20,
    threshold: float = 0.8,
) -> pd.DataFrame:
    """
    过滤拥挤度过高的股票信号
    """
    if signals.empty:
        return signals

    mask = pd.Series(True, index=signals.index)
    for idx, row in signals.iterrows():
        status, score = calculate_crowding(
            row["stock_code"], signal_history, lookback, threshold
        )
        if status == "CROWDED":
            mask[idx] = False

    return signals[mask].reset_index(drop=True)
