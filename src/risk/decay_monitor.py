"""
机构Alpha衰减监控
连续亏损检测 → Alpha失效预警
"""
from __future__ import annotations
import numpy as np
import pandas as pd
try:
    from loguru import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)


class AlphaDecayMonitor:
    """监控机构Alpha能力是否失效"""

    def __init__(self, window: int = 10, threshold: float = -0.05):
        self.window = window
        self.threshold = threshold

    def check_decay(
        self,
        seat_name: str,
        trade_history: pd.DataFrame,
    ) -> str:
        """
        检查机构Alpha是否衰减

        Returns: "ACTIVE" | "WARNING" | "DECAYED" | "INSUFFICIENT_DATA"
        """
        recent = trade_history[trade_history["seat_name"] == seat_name]
        recent = recent.sort_values("lhb_date").tail(self.window)

        if len(recent) < self.window:
            return "INSUFFICIENT_DATA"

        ret_col = "ret_20d" if "ret_20d" in recent.columns else None
        if ret_col is None:
            for col in recent.columns:
                if col.startswith("ret_"):
                    ret_col = col
                    break
        if ret_col is None:
            return "INSUFFICIENT_DATA"

        recent_returns = recent[ret_col].dropna()
        if len(recent_returns) < self.window:
            return "INSUFFICIENT_DATA"

        cum_ret = recent_returns.sum()
        win_rate = (recent_returns > 0).mean()

        if cum_ret < self.threshold and win_rate < 0.3:
            return "DECAYED"  # Alpha失效
        elif cum_ret < 0:
            return "WARNING"  # 预警
        else:
            return "ACTIVE"  # Alpha正常

    def filter_decayed_seats(
        self,
        seat_scores: pd.DataFrame,
        trade_history: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        从评分表中移除已衰减的机构
        """
        if seat_scores.empty:
            return seat_scores

        mask = pd.Series(True, index=seat_scores.index)
        for idx, row in seat_scores.iterrows():
            status = self.check_decay(row["seat_name"], trade_history)
            if status == "DECAYED":
                mask[idx] = False
                logger.info(f"机构 {row['seat_name']} Alpha已衰减，从信号中排除")

        return seat_scores[mask].reset_index(drop=True)
