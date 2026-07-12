"""Phase 6B 公共窗口 / 基准同口径回归测试。

覆盖组合对比脚本的 eval_equity 截断与信息比率/主动回撤（Section VIII）：
  1. equity 统一截断到 [test_start, 四方案实际 exit_date 最大值]；
  2. 信息比率/主动回撤只在组合与基准**重叠交易日**上计算（inner join）；
  3. 重叠不足（<3 日）→ (nan, nan)；
  4. 组合持续跑输 → 主动回撤为负；持续跑赢 → 主动回撤≈0。
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJ))
_spec = importlib.util.spec_from_file_location(
    "run_phase6b_portfolio_comparison", PROJ / "scripts" / "run_phase6b_portfolio_comparison.py")
comp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(comp)


# ---------- 1. eval_equity 截断到最后一个 exit_date ----------
def test_eval_equity_truncates_to_last_exit():
    eq = pd.DataFrame({
        "date": pd.bdate_range("2025-09-01", "2025-12-31").strftime("%Y-%m-%d"),
        "nav": np.linspace(1_000_000, 1_100_000, len(pd.bdate_range("2025-09-01", "2025-12-31")))})
    trades = pd.DataFrame({
        "stock": ["000001", "000002"], "signal_date": ["2025-09-05", "2025-09-08"],
        "entry_date": ["2025-09-08", "2025-09-09"], "exit_date": ["2025-10-15", "2025-09-20"],
        "net_return_pct": [1.2, -0.5]})
    r = comp.eval_equity(eq, trades, "2025-09-01")
    assert r["eval_end"] == "2025-10-15"                       # 四方案 exit_date 最大值
    assert pd.Timestamp(r["first_day"]) >= pd.Timestamp("2025-09-01")
    assert r["n_days"] > 0


# ---------- 2/4. 信息比率 / 主动回撤：重叠日 + 跑输为负 ----------
def test_ir_active_dd_underperform_negative():
    dates = pd.bdate_range("2025-09-01", "2025-09-30").strftime("%Y-%m-%d")
    port = pd.DataFrame({"date": dates, "nav": np.linspace(1.0, 0.95, len(dates))})   # 单调下跌
    bench = pd.DataFrame({"date": dates, "nav": np.linspace(1.0, 1.05, len(dates))})  # 单调上涨
    ir, add = comp.ir_active_dd(port, bench)
    assert not np.isnan(ir) and ir < 0
    assert not np.isnan(add) and add < 0


def test_ir_active_dd_outperform_nonneg_drawdown():
    dates = pd.bdate_range("2025-09-01", "2025-09-30").strftime("%Y-%m-%d")
    port = pd.DataFrame({"date": dates, "nav": np.linspace(1.0, 1.10, len(dates))})   # 持续跑赢
    bench = pd.DataFrame({"date": dates, "nav": np.linspace(1.0, 1.02, len(dates))})
    ir, add = comp.ir_active_dd(port, bench)
    assert ir > 0
    assert add >= -1e-9                                        # 相对基准无回撤


# ---------- 3. 重叠不足 → (nan, nan) ----------
def test_ir_active_dd_insufficient_overlap():
    port = pd.DataFrame({"date": ["2025-09-01", "2025-09-02"], "nav": [1.0, 1.01]})
    bench = pd.DataFrame({"date": ["2025-11-01", "2025-11-02"], "nav": [1.0, 1.0]})   # 无重叠
    ir, add = comp.ir_active_dd(port, bench)
    assert np.isnan(ir) and np.isnan(add)


# ---------- 2b. 只用重叠交易日（inner join）----------
def test_ir_uses_only_overlapping_dates():
    dates_all = pd.bdate_range("2025-09-01", "2025-09-30").strftime("%Y-%m-%d")
    port = pd.DataFrame({"date": dates_all, "nav": np.linspace(1.0, 1.05, len(dates_all))})
    # 基准只覆盖后半段
    half = dates_all[len(dates_all) // 2:]
    bench = pd.DataFrame({"date": half, "nav": np.linspace(1.0, 1.01, len(half))})
    ir, add = comp.ir_active_dd(port, bench)
    # 仅重叠段参与计算 → 有限值，不因非重叠日报错或引入 NaN
    assert not np.isnan(ir)
    assert not np.isnan(add)
