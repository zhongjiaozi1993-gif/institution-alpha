"""Phase 7A 口径一致性：40d/60d 单股票有效日序列 forward_open_return 与 label 一致。

证明在含停牌/缺失交易日的股票上：
  1. forward_open_return(pdf, entry_date, 40) 与 label_40d 一致；
  2. forward_open_return(pdf, entry_date, 60) 与 label_60d 一致；
  3. 越界（该股无 entry+h 有效日）返回 NaN；
  4. 无停牌时每股口径与全局口径一致（40d/60d 回归保护）。
"""
from pathlib import Path

import numpy as np
import pandas as pd

from src.backtest.open_to_open import forward_open_return

# 含停牌的单票：缺失 2025-09-04、09-05（模拟 2 日停牌），周末 09-06/07 天然缺失。
# 序列拉长以便测试 40d/60d 步进。
SUSP_DATES = [
    "2025-07-01", "2025-07-02", "2025-07-03", "2025-07-04", "2025-07-07",
    "2025-07-08", "2025-07-09", "2025-07-10", "2025-07-11", "2025-07-14",
    "2025-07-15", "2025-07-16", "2025-07-17", "2025-07-18", "2025-07-21",
    # 停牌: 07-22 ~ 07-25 缺失
    "2025-07-28", "2025-07-29", "2025-07-30", "2025-07-31",
    "2025-08-01", "2025-08-04", "2025-08-05", "2025-08-06", "2025-08-07",
    "2025-08-08", "2025-08-11", "2025-08-12", "2025-08-13", "2025-08-14",
    "2025-08-15", "2025-08-18", "2025-08-19", "2025-08-20", "2025-08-21",
    "2025-08-22", "2025-08-25", "2025-08-26", "2025-08-27", "2025-08-28",
    "2025-08-29",
    "2025-09-01", "2025-09-02", "2025-09-03",
    # 停牌: 09-04, 09-05 缺失
    "2025-09-08", "2025-09-09", "2025-09-10", "2025-09-11", "2025-09-12",
    "2025-09-15", "2025-09-16", "2025-09-17", "2025-09-18", "2025-09-19",
]
SUSP_OPENS = [float(i + 10) for i in range(len(SUSP_DATES))]


def _susp_pdf():
    return pd.DataFrame({"date_str": SUSP_DATES, "open_yuan": SUSP_OPENS})


# ---------- 1. forward_open_return 与 label 口径一致（40d/60d）----------
def test_calendar_agreement_40d():
    pdf = _susp_pdf()
    open_arr = np.array(SUSP_OPENS)
    from src.features.label_builder import _open_to_open_labels
    labels = _open_to_open_labels(open_arr, [40])["label_40d"]
    h = 40
    for i in range(len(SUSP_DATES)):
        if i + 1 >= len(SUSP_DATES):
            continue
        entry_date = SUSP_DATES[i + 1]  # 信号行 i → 建仓行 i+1
        attr = forward_open_return(pdf, entry_date, h)
        entry_pos = i + 1
        exit_pos = entry_pos + h
        shift = (SUSP_OPENS[exit_pos] / SUSP_OPENS[entry_pos] - 1.0
                 if exit_pos < len(SUSP_OPENS) else np.nan)
        lab = labels[i]
        if np.isnan(lab):
            assert np.isnan(attr) and np.isnan(shift), f"h=40 i={i} 应三者皆 NaN"
        else:
            assert abs(attr - lab) < 1e-12, f"forward_open_return≠label h=40 i={i}"
            assert abs(shift - lab) < 1e-12, f"shift≠label h=40 i={i}"


def test_calendar_agreement_60d():
    pdf = _susp_pdf()
    open_arr = np.array(SUSP_OPENS)
    from src.features.label_builder import _open_to_open_labels
    labels = _open_to_open_labels(open_arr, [60])["label_60d"]
    h = 60
    for i in range(len(SUSP_DATES)):
        if i + 1 >= len(SUSP_DATES):
            continue
        entry_date = SUSP_DATES[i + 1]
        attr = forward_open_return(pdf, entry_date, h)
        entry_pos = i + 1
        exit_pos = entry_pos + h
        shift = (SUSP_OPENS[exit_pos] / SUSP_OPENS[entry_pos] - 1.0
                 if exit_pos < len(SUSP_OPENS) else np.nan)
        lab = labels[i]
        if np.isnan(lab):
            assert np.isnan(attr) and np.isnan(shift), f"h=60 i={i} 应三者皆 NaN"
        else:
            assert abs(attr - lab) < 1e-12, f"forward_open_return≠label h=60 i={i}"
            assert abs(shift - lab) < 1e-12, f"shift≠label h=60 i={i}"


# ---------- 2. 越界返回 NaN ----------
def test_out_of_range_returns_nan():
    pdf = _susp_pdf()
    assert np.isnan(forward_open_return(pdf, SUSP_DATES[-1], 40))
    assert np.isnan(forward_open_return(pdf, SUSP_DATES[-1], 60))
    assert np.isnan(forward_open_return(pdf, "2025-09-04", 40))
    assert np.isnan(forward_open_return(pdf, "2025-09-04", 60))


# ---------- 3. 无停牌时每股口径与全局口径一致 ----------
def test_contiguous_calendar_matches_global_40d():
    dates = list(pd.bdate_range("2025-01-02", "2025-07-31").strftime("%Y-%m-%d"))
    opens = [10.0 + i * 0.1 for i in range(len(dates))]
    pdf = pd.DataFrame({"date_str": dates, "open_yuan": opens})
    h = 40
    for i in range(min(len(dates) - 1, 20)):  # 抽样前20个建仓日
        entry = dates[i]
        val = forward_open_return(pdf, entry, h)
        exit_pos = i + h
        exp = opens[exit_pos] / opens[i] - 1.0 if exit_pos < len(opens) else np.nan
        if np.isnan(exp):
            assert np.isnan(val)
        else:
            assert abs(val - exp) < 1e-12


def test_contiguous_calendar_matches_global_60d():
    dates = list(pd.bdate_range("2025-01-02", "2025-07-31").strftime("%Y-%m-%d"))
    opens = [10.0 + i * 0.1 for i in range(len(dates))]
    pdf = pd.DataFrame({"date_str": dates, "open_yuan": opens})
    h = 60
    for i in range(min(len(dates) - 1, 20)):
        entry = dates[i]
        val = forward_open_return(pdf, entry, h)
        exit_pos = i + h
        exp = opens[exit_pos] / opens[i] - 1.0 if exit_pos < len(opens) else np.nan
        if np.isnan(exp):
            assert np.isnan(val)
        else:
            assert abs(val - exp) < 1e-12
