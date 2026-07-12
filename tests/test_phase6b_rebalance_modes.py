"""Phase 6B 组合构造引擎的 rebalance 机制回归测试。

覆盖：fill_slots 与底层引擎完全一致；periodic/daily 目标权重 T+1 open 执行；
诊断指标口径与边界；stock_contrib 归因符号；run_portfolio 分发与非法 mode。
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJ))
from src.backtest import portfolio_construction as pc
from src.backtest.signal_backtester import SignalBacktester, BacktestConfig

DATES = ["2025-09-01", "2025-09-02", "2025-09-03", "2025-09-04",
         "2025-09-05", "2025-09-08", "2025-09-09", "2025-09-10"]
TRENDS = {"000001": 0.02, "000002": 0.005, "000003": -0.01}   # 日涨幅：赢家/中性/输家


def _prices():
    prices = {}
    for s, g in TRENDS.items():
        opens = [10.0 * (1 + g) ** i for i in range(len(DATES))]
        closes = [o * 1.001 for o in opens]
        prices[s] = pd.DataFrame({"date_str": DATES, "open_yuan": opens, "close_yuan": closes})
    return prices


def _scores(order=("000001", "000002", "000003")):
    """恒定截面排序（best→worst），final_score 越高越好。"""
    rows = []
    for d in DATES:
        for r, s in enumerate(order):
            rows.append({"trade_date": pd.Timestamp(d), "symbol": s,
                         "final_score": float(len(order) - r)})
    return pd.DataFrame(rows)


def _cfg(top_n=2, h=3):
    return pc.PortfolioConfig(top_n=top_n, holding_days=h, cost_bps=20, slippage_bps=10,
                              initial_capital=1_000_000.0, name="t")


# ---------- 1. fill_slots 与底层引擎一致 ----------
def test_fill_slots_matches_engine():
    prices, scores, cfg = _prices(), _scores(), _cfg(top_n=2, h=3)
    res = pc.run_portfolio("fixed_holding_fill_slots", scores, prices, None, cfg)
    # 直接用 SignalBacktester 复算相同买入信号
    buys = pc._scores_to_buys(scores, cfg.top_n)
    bcfg = BacktestConfig(holding_days=3, max_positions=2, cost_bps=20, slippage_bps=10,
                          initial_capital=1_000_000.0, name="t")
    direct = SignalBacktester(bcfg).run(buys, prices, tradable_flags=None)
    assert len(res["trades"]) == len(direct["trades"])
    a = res["equity"]["nav"].to_numpy()
    b = direct["equity_curve"]["nav"].to_numpy()
    assert np.allclose(a, b)


# ---------- 2. T+1 open 执行：信号日当天不成交 ----------
def test_rebalance_executes_next_day():
    prices, scores = _prices(), _scores()
    res = pc.run_portfolio("daily_rebalance_topN", scores, prices, None, _cfg(top_n=2, h=1))
    hd = res["holdings_daily"]
    assert hd[0]["date"] == DATES[0]
    assert hd[0]["holdings"] == {}                    # 首日（信号日）无持仓
    assert len(hd[1]["holdings"]) > 0                 # 次日才建仓
    # 每笔买入的 exec_date 必须晚于 signal_date（无 T+0）
    tr = res["trades"]
    buys = tr[tr["side"] == "buy"] if not tr.empty else tr
    for _, t in buys.iterrows():
        assert t["exec_date"] > t["signal_date"]


# ---------- 3. periodic：rebalance_every 超过日历 → 只建仓一次 ----------
def test_periodic_single_rebalance_when_interval_large():
    prices, scores = _prices(), _scores()
    cfg = pc.PortfolioConfig(top_n=2, holding_days=len(DATES) + 5)  # 间隔 > 交易日数
    res = pc.run_rebalance(scores, prices, None, cfg, rebalance_every=cfg.holding_days)
    # 只有首个信号日进入 reb_days → 仅一次目标建立（后续无再平衡卖出/买入循环）
    tr = res["trades"]
    assert not tr.empty
    assert tr["signal_date"].nunique() == 1
    assert tr[tr["side"] == "buy"]["signal_date"].iloc[0] == DATES[0]


# ---------- 4. 诊断指标口径与边界 ----------
def test_diagnostics_bounds():
    prices, scores = _prices(), _scores()
    for mode in ("fixed_holding_fill_slots", "periodic_rebalance_topN", "daily_rebalance_topN"):
        res = pc.run_portfolio(mode, scores, prices, None, _cfg(top_n=2, h=2))
        d = res["diagnostics"]
        assert 0.0 <= d["execution_rate"] <= 1.0
        if not np.isnan(d["mean_topn_overlap"]):
            assert 0.0 <= d["mean_topn_overlap"] <= 1.0
        if not np.isnan(d["mean_holding_rank"]):
            assert d["mean_holding_rank"] >= 1.0
        assert d["candidate_slots"] >= d["executed_fills"] >= 0


# ---------- 5. NAV 恒正、现金不为负 ----------
def test_nav_positive_cash_nonneg():
    prices, scores = _prices(), _scores()
    for mode in ("periodic_rebalance_topN", "daily_rebalance_topN"):
        res = pc.run_rebalance(scores, prices, None, _cfg(top_n=2, h=2),
                               rebalance_every=(2 if mode == "periodic_rebalance_topN" else 1))
        eq = res["equity"]
        assert (eq["nav"] > 0).all()
        assert (eq["cash"] >= -1e-6).all()


# ---------- 6. 单票贡献符号：赢家为正、输家为负 ----------
def test_stock_contribution_sign():
    prices, scores = _prices(), _scores()
    res = pc.run_portfolio("daily_rebalance_topN", scores, prices, None, _cfg(top_n=2, h=1))
    sc = res["stock_contrib"]
    # 000001 持续上涨且长期在 Top-2 → 贡献为正
    assert "000001" in sc.index
    assert sc.loc["000001"] > 0


# ---------- 7. 非法 mode ----------
def test_unknown_mode_raises():
    prices, scores = _prices(), _scores()
    with pytest.raises(ValueError):
        pc.run_portfolio("nope", scores, prices, None, _cfg())
