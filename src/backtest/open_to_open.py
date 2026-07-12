"""单股交易日历口径的远期 open→open 收益。

独立小模块，供回测归因与组合构造共用。核心：按**每只股票自己的有效日线序列**（停牌=缺失行）
计算 open[entry+h]/open[entry]-1，与 src/features/label_builder._open_to_open_labels 完全同口径，
不用全局交易日历，避免停牌日错位。此文件是历史归因口径修复的落点，便于单独审计/回滚。
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def forward_open_return(pdf: pd.DataFrame, entry_date: str, h: int) -> float:
    """open[entry+h]/open[entry]-1，entry+h 为该股**自己**日线序列中 entry 后第 h 个有效日。

    pdf: 单只股票日线（含 date_str, open_yuan，按日期升序、index 0..n-1）。
    与 label_builder._open_to_open_labels 同口径：停牌日在 pdf 中缺失，shift 自动跳过。
    """
    ds = pdf["date_str"].to_numpy()
    pos = np.where(ds == entry_date)[0]
    if len(pos) == 0:
        return np.nan
    i = int(pos[0])
    ei = i + h
    if ei >= len(ds):
        return np.nan
    o_in = float(pdf["open_yuan"].iloc[i])
    o_out = float(pdf["open_yuan"].iloc[ei])
    if o_in <= 0:
        return np.nan
    return o_out / o_in - 1.0
