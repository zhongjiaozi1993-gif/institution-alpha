"""
信号衰减模型
信号强度随时间指数衰减，模拟信息扩散导致的Alpha衰减
"""
from __future__ import annotations
import numpy as np
import pandas as pd


def decay_signals(
    signals: pd.DataFrame,
    days_elapsed: int = 0,
    half_life: int = 5,
) -> pd.DataFrame:
    """
    计算信号衰减后的强度

    signals: [strength, ...]
    days_elapsed: 距离信号生成的交易日数
    half_life: 信号半衰期（交易日数，默认5天）

    Returns: 添加 current_strength, is_expired 列
    """
    if signals.empty:
        return signals

    df = signals.copy()
    decay_factor = np.exp(-days_elapsed * np.log(2) / half_life)
    df["current_strength"] = df["strength"] * decay_factor
    df["days_elapsed"] = days_elapsed
    df["is_expired"] = df["current_strength"] < 0.2
    return df


class SignalDecayManager:
    """管理持仓中信号的衰减跟踪"""

    def __init__(self, half_life: int = 5):
        self.half_life = half_life

    def get_current_strength(self, signal: pd.Series, days_held: int) -> float:
        """获取某信号持有N天后的当前强度"""
        decay = np.exp(-days_held * np.log(2) / self.half_life)
        return signal["strength"] * decay

    def is_alive(self, signal: pd.Series, days_held: int, threshold: float = 0.2) -> bool:
        """信号是否仍然有效（强度>阈值）"""
        return self.get_current_strength(signal, days_held) >= threshold
