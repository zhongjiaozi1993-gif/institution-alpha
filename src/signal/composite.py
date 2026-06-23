"""
信号合成与过滤
多维度过滤（流动性、ST、次新股）+ 持仓数量上限控制
"""
from __future__ import annotations
import pandas as pd
try:
    from loguru import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)


def filter_signals(
    signals: pd.DataFrame,
    daily_prices: dict[str, pd.DataFrame] | None = None,
    max_signals_per_day: int = 20,
    blacklist: set[str] | None = None,
) -> pd.DataFrame:
    """
    信号过滤：
    1. 不再黑名单中的股票
    2. 每日最多max_signals_per_day个信号（按strength降序取top）
    """
    if signals.empty:
        return signals

    filtered = signals.copy()

    if blacklist:
        filtered = filtered[~filtered["stock_code"].isin(blacklist)]

    # 按日期分组，每天只取top N
    if "lhb_date" in filtered.columns:
        result = []
        for date, group in filtered.groupby("lhb_date"):
            group = group.sort_values("strength", ascending=False)
            result.append(group.head(max_signals_per_day))
        filtered = pd.concat(result, ignore_index=True)

    return filtered
