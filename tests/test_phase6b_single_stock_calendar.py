"""Phase 6B 口径一致性：open→open 归因必须按**每只股票自己的有效日线序列**计算。

证明在含停牌/缺失交易日的股票上，三种口径给出同一个远期收益：
  1. label horizon —— label_builder._open_to_open_labels（生产标签口径）；
  2. open→open attribution —— run_alpha_fusion_backtest.trade_oo_gross_return
     （委托 portfolio_construction.forward_open_return）；
  3. 单股票有效交易日 shift —— 直接按该股自己行位置 entry_pos+h 取价。

并额外证明：若错用**全局交易日历**（把停牌日也算进 horizon 步进），结果会偏离标签口径，
即本修复是 load-bearing 而非等价改写。
"""
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

PROJ = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location(
    "run_alpha_fusion_backtest", PROJ / "scripts" / "run_alpha_fusion_backtest.py")
bt = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bt)

import sys
sys.path.insert(0, str(PROJ))
from src.features.label_builder import _open_to_open_labels
from src.backtest.open_to_open import forward_open_return


# 含停牌的单票：缺失 2025-09-04、09-05（模拟 2 日停牌），周末 09-06/07 天然缺失。
SUSP_DATES = ["2025-09-01", "2025-09-02", "2025-09-03",
              "2025-09-08", "2025-09-09", "2025-09-10", "2025-09-11", "2025-09-12"]
SUSP_OPENS = [10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0]


def _susp_pdf():
    return pd.DataFrame({"date_str": SUSP_DATES, "open_yuan": SUSP_OPENS})


# ---------- 1. 三口径在停牌股票上完全一致 ----------
def test_three_calibers_agree_under_suspension():
    pdf = _susp_pdf()
    prices = {"000001": pdf}
    open_arr = np.array(SUSP_OPENS)
    for h in (1, 3, 5):
        labels = _open_to_open_labels(open_arr, [h])[f"label_{h}d"]
        for i in range(len(SUSP_DATES)):
            # 信号行 i → 建仓行 i+1（该股自己序列）
            if i + 1 >= len(SUSP_DATES):
                continue
            entry_date = SUSP_DATES[i + 1]
            # (2) attribution 口径
            attr = bt.trade_oo_gross_return(prices, "000001", entry_date, h)
            # (3) 单股票 shift 口径：entry_pos+h 越界→NaN
            entry_pos = i + 1
            exit_pos = entry_pos + h
            shift = (open_arr[exit_pos] / open_arr[entry_pos] - 1.0
                     if exit_pos < len(open_arr) else np.nan)
            # (1) label 口径（按信号行 i 索引）
            lab = labels[i]
            if np.isnan(lab):
                assert np.isnan(attr) and np.isnan(shift), f"h={h} i={i} 应三者皆 NaN"
            else:
                assert abs(attr - lab) < 1e-12, f"attribution≠label h={h} i={i}"
                assert abs(shift - lab) < 1e-12, f"shift≠label h={h} i={i}"


# ---------- 2. 错用全局日历会偏离标签口径（证明修复是必要的）----------
def test_global_calendar_would_diverge_under_suspension():
    pdf = _susp_pdf()
    prices = {"000001": pdf}
    # 全局业务日日历把停牌日 09-04/05 也算进步进
    global_dates = list(pd.bdate_range("2025-09-01", "2025-09-12").strftime("%Y-%m-%d"))
    h = 3
    entry_date = "2025-09-03"                 # 该股建仓日
    # 每股口径（正确）：09-03 是该股第 2 行 → 第 5 行 = 09-10, open=15 → 15/12-1
    per_stock = bt.trade_oo_gross_return(prices, "000001", entry_date, h)
    assert abs(per_stock - (15.0 / 12.0 - 1.0)) < 1e-12

    # 全局日历口径（错误重现）：09-03 全局 index2 +3 = index5 = 09-08，该股 09-08 open=13 → 13/12-1
    gi = {d: i for i, d in enumerate(global_dates)}
    wrong_exit = global_dates[gi[entry_date] + h]           # 2025-09-08
    o_in = float(pdf.loc[pdf["date_str"] == entry_date, "open_yuan"].iloc[0])
    o_out = float(pdf.loc[pdf["date_str"] == wrong_exit, "open_yuan"].iloc[0])
    global_caliber = o_out / o_in - 1.0
    assert abs(global_caliber - (13.0 / 12.0 - 1.0)) < 1e-12
    # 两者必须不同 → 修复 load-bearing
    assert abs(per_stock - global_caliber) > 1e-6


# ---------- 3. 越界（该股无 entry+h 有效日）→ NaN ----------
def test_out_of_range_returns_nan():
    pdf = _susp_pdf()
    prices = {"000001": pdf}
    # 末行建仓、h=1 → 无下一有效日
    assert np.isnan(bt.trade_oo_gross_return(prices, "000001", SUSP_DATES[-1], 1))
    # entry 不在该股序列中 → NaN
    assert np.isnan(bt.trade_oo_gross_return(prices, "000001", "2025-09-04", 3))
    # forward_open_return 直接调用同样处理越界
    assert np.isnan(forward_open_return(pdf, SUSP_DATES[-1], 3))


# ---------- 4. 无停牌时每股口径与全局口径一致（回归保护）----------
def test_contiguous_calendar_matches_global():
    dates = list(pd.bdate_range("2025-09-01", "2025-09-12").strftime("%Y-%m-%d"))
    opens = [10.0 + i for i in range(len(dates))]
    pdf = pd.DataFrame({"date_str": dates, "open_yuan": opens})
    prices = {"000001": pdf}
    h = 3
    for i in range(len(dates) - 1):
        entry = dates[i]
        per_stock = bt.trade_oo_gross_return(prices, "000001", entry, h)
        exit_pos = i + h
        exp = opens[exit_pos] / opens[i] - 1.0 if exit_pos < len(opens) else np.nan
        if np.isnan(exp):
            assert np.isnan(per_stock)
        else:
            assert abs(per_stock - exp) < 1e-12
