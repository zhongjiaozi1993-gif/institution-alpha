"""Phase 6A.1 回测窗口与实际持仓归因修复的回归测试。

覆盖：equity 统一截断到 common_eval_end、基准与方案同起止日、暴露用实际成交、
候选池与实际成交暴露分离、verdict 无硬编码指标、open→open 退出与 label horizon 一致。
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


def _controls_one_day(date="2025-09-05"):
    """5 只股票、单日：log_amount 递增 → 截面百分位 0.2..1.0。"""
    syms = [f"{i:06d}" for i in range(1, 6)]
    return pd.DataFrame({
        "trade_date": pd.Timestamp(date), "symbol": syms,
        "log_mktcap": [1.0, 2, 3, 4, 5], "log_amount": [10.0, 20, 30, 40, 50],
        "turnover": [0.01, 0.02, 0.03, 0.04, 0.05]})


# ---------- 1. equity 截断到 common_eval_end ----------
def test_equity_stops_at_common_eval_end():
    t1 = pd.DataFrame({"exit_date": ["2025-12-20", "2026-01-05"]})
    t2 = pd.DataFrame({"exit_date": ["2026-01-16", "2025-11-30"]})
    cee = bt.common_eval_end([t1, t2])
    assert cee == pd.Timestamp("2026-01-16")            # 四方案 exit_date 最大值
    eq = pd.DataFrame({"date": pd.bdate_range("2025-09-01", "2026-03-31").strftime("%Y-%m-%d"),
                       "nav": 1.0})
    tr = bt.truncate_equity(eq, bt.TEST_START, cee)
    assert pd.to_datetime(tr["date"]).max() == cee
    assert (pd.to_datetime(tr["date"]) <= cee).all()
    assert (pd.to_datetime(tr["date"]) >= pd.Timestamp(bt.TEST_START)).all()


# ---------- 2. 基准与方案同起止日 ----------
def test_benchmark_uses_identical_date_range():
    start, end = "2025-09-01", "2026-01-16"
    m, nav = bt.benchmark_metrics(start, end)
    assert not nav.empty
    assert pd.to_datetime(nav["date"]).min() >= pd.Timestamp(start)
    assert pd.to_datetime(nav["date"]).max() <= pd.Timestamp(end)
    assert m["date_start"] >= start and m["date_end"] <= end
    # 同起止日两次调用结果一致（determinism / 完全相同区间）
    m2, _ = bt.benchmark_metrics(start, end)
    assert m2["n_days"] == m["n_days"] and m2["portfolio_total_return"] == m["portfolio_total_return"]


# ---------- 3. 暴露用实际成交 ----------
def test_exposure_uses_executed_trades():
    controls = _controls_one_day()
    # 只成交最高 log_amount 的 000005 → executed 暴露应≈1.0
    trades = pd.DataFrame({"stock": ["000005"], "signal_date": ["2025-09-05"]})
    ex = bt.executed_entry_exposure(trades, controls)
    assert abs(ex["log_amount"] - 1.0) < 1e-9
    # 换只成交最低的 000001 → 暴露应≈0.2，证明确实取实际成交而非固定值
    trades_low = pd.DataFrame({"stock": ["000001"], "signal_date": ["2025-09-05"]})
    ex_low = bt.executed_entry_exposure(trades_low, controls)
    assert abs(ex_low["log_amount"] - 0.2) < 1e-9


# ---------- 4. 候选池与实际成交暴露分离 ----------
def test_candidate_and_executed_exposure_are_separate():
    controls = _controls_one_day()
    buys = pd.DataFrame({"stock_code": [f"{i:06d}" for i in range(1, 6)],
                         "signal_date": ["2025-09-05"] * 5})      # 全部候选
    trades = pd.DataFrame({"stock": ["000005"], "signal_date": ["2025-09-05"]})  # 只成交高流动性
    cand = bt.candidate_exposure(buys, controls)
    exe = bt.executed_entry_exposure(trades, controls)
    assert abs(cand["log_amount"] - 0.6) < 1e-9      # 候选池均值 (0.2..1.0)=0.6
    assert abs(exe["log_amount"] - 1.0) < 1e-9       # 实际成交=1.0
    assert cand["log_amount"] != exe["log_amount"]   # 两者必须不同口径


# ---------- 5. verdict 无硬编码指标 ----------
def _synthetic_scheme(ret, sharpe, dd, ricir, amt_pct):
    return {"portfolio_total_return": ret, "sharpe": sharpe, "max_drawdown": dd,
            "test_rankicir": ricir, "executed_exposure": {"log_amount": amt_pct},
            "best_stock": "000999", "best_contrib": 12.3, "monthly": {"2025-09": 1.0, "2025-10": 2.0}}


def test_report_contains_no_hardcoded_metrics():
    # 用与真实结果不同的 sentinel 数值构造，verdict 应反映之，且不得出现旧硬编码字面量
    all_res = {}
    for h in (5, 10):
        all_res[h] = {
            "best_single": _synthetic_scheme(0.4242, 5.55, -0.11, 0.99, 0.07),
            "equal_weight": _synthetic_scheme(0.1111, 0.33, -0.12, 0.88, 0.44),
            "icir_weight": _synthetic_scheme(0.1010, 0.22, -0.13, 0.77, 0.40),
            "stability_weight": _synthetic_scheme(0.1212, 0.44, -0.14, 0.66, 0.41)}
    bench = {5: {"portfolio_total_return": 0.9999, "sharpe_ratio": 1.11},
             10: {"portfolio_total_return": 0.8888, "sharpe_ratio": 1.22}}
    meta = {h: {"signal_start": "2025-09-01", "common_eval_end": "2026-01-16",
                "equity_n_days": 90} for h in (5, 10)}
    txt = bt.verdict(all_res, bench, meta)
    # 动态值必须出现
    assert "+99.99%" in txt          # 基准 sentinel
    assert "+0.07" in txt            # executed log成交额分位 sentinel (best_single)
    # 旧硬编码字面量必须消失
    for forbidden in ["1.58", "+59.7", "26–38%", "14–16%", "0.34→0.51", "0.36→0.56", "2.51"]:
        assert forbidden not in txt, f"发现硬编码残留: {forbidden}"


# ---------- 6. open→open 与 label horizon 一致 ----------
def test_open_to_open_exit_matches_label_horizon():
    dates = list(pd.bdate_range("2025-09-01", "2025-09-12").strftime("%Y-%m-%d"))
    opens = [10.0 + i for i in range(len(dates))]     # 10,11,12,...
    pdf = pd.DataFrame({"date_str": dates, "open_yuan": opens})
    prices = {"000001": pdf}
    h = 3
    entry = dates[1]                                   # 建仓日
    exp = opens[1 + h] / opens[1] - 1.0                # open[entry+h]/open[entry]-1
    got = bt.trade_oo_gross_return(prices, "000001", entry, h)
    assert abs(got - exp) < 1e-12
    # 越界（无 entry+h）→ NaN
    assert np.isnan(bt.trade_oo_gross_return(prices, "000001", dates[-1], h))
