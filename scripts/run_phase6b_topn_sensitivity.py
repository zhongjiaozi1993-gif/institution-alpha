"""Phase 6B 驱动脚本 2：Top-N 敏感性（Section IV）。

固定 6A 严格 OOS 切分（train≤2025-08-31, test 2025-09-01..2025-12-31），主方案 equal_weight，
只把 Top-N 从 10/20/30/50/100 扫一遍，**观察平滑性**（是否单调/突变），不据此挑最优 Top-N。

每个 Top-N 输出：收益/超额/Sharpe/最大回撤/年化换手/成交率/单票集中度/前5贡献占比/
实际成交的市值·成交额·换手截面分位。构造用 fixed_holding_fill_slots（与主分析一致）。

产出 reports 片段并入 phase6b_portfolio_construction_report.md（本脚本单独可跑，打印+返回结果）。
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))
from src.fusion import alpha_rule_fusion as arf
from src.backtest import portfolio_construction as pc
from src.backtest.metrics import compute_full_metrics

_spec = importlib.util.spec_from_file_location(
    "run_alpha_fusion_backtest", PROJECT / "scripts" / "run_alpha_fusion_backtest.py")
bt = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bt)

FLAGS_PATH = PROJECT / "data" / "processed" / "tradable" / "tradable_flags.parquet"

TOPN_GRID = [10, 20, 30, 50, 100]
HORIZONS = (5, 10)
MAIN = "equal_weight"
TEST_START = arf.TEST_START
PRICE_END = "2026-03-31"


def _contribution_concentration(stock_contrib: pd.Series) -> dict:
    """单票贡献集中度：前1/5/10 名占|总贡献|之比 + 去掉最大贡献票后剩余总贡献。"""
    if stock_contrib is None or len(stock_contrib) == 0:
        return {"top1": np.nan, "top5": np.nan, "top10": np.nan, "ex_top1_total": np.nan}
    s = stock_contrib.sort_values(ascending=False)
    total_abs = float(s.abs().sum())
    total = float(s.sum())
    pos = s[s > 0]
    top = lambda k: float(pos.head(k).sum() / total_abs) if total_abs > 0 else np.nan
    return {"top1": top(1), "top5": top(5), "top10": top(10),
            "ex_top1_total": total - float(s.iloc[0])}


def run_topn(scores, prices, flags, controls, top_n, h) -> dict:
    cfg = pc.PortfolioConfig(top_n=top_n, holding_days=h, cost_bps=bt.COST_BPS,
                             slippage_bps=bt.SLIP_BPS, initial_capital=bt.INIT_CAP, name=f"n{top_n}")
    res = pc.run_fill_slots(scores, prices, flags, cfg)
    trades, equity = res["trades"], res["equity"]
    cee = pd.to_datetime(trades["exit_date"]).max() if (trades is not None and not trades.empty) \
        else pd.Timestamp(TEST_START)
    eq = bt.truncate_equity(equity, TEST_START, cee)
    mt = compute_full_metrics(eq, trades if trades is not None else pd.DataFrame())
    bench, _ = bt.benchmark_metrics(TEST_START, cee)
    ret = float(mt.get("portfolio_total_return", np.nan))
    bench_ret = float(bench.get("portfolio_total_return", np.nan)) if bench else np.nan
    n_days = int(len(eq))
    conc = _contribution_concentration(res["stock_contrib"])
    ee = bt.executed_entry_exposure(trades, controls)
    return {
        "top_n": top_n, "total_return": ret, "excess": ret - bench_ret,
        "sharpe": mt.get("sharpe_ratio", np.nan), "max_drawdown": mt.get("max_drawdown", np.nan),
        "ann_turnover": bt.annualized_turnover(len(trades), top_n, n_days),
        "execution_rate": res["diagnostics"]["execution_rate"],
        "n_trades": int(len(trades)), "bench_return": bench_ret,
        "conc_top1": conc["top1"], "conc_top5": conc["top5"], "conc_top10": conc["top10"],
        "exec_log_mktcap": ee["log_mktcap"], "exec_log_amount": ee["log_amount"],
        "exec_turnover": ee["turnover"],
    }


def smoothness(rows: list[dict], key: str) -> dict:
    """相邻 Top-N 的一阶差分幅度（判断平滑/突变），返回最大单步变化与是否单调。"""
    vals = [r[key] for r in rows]
    diffs = [b - a for a, b in zip(vals, vals[1:]) if not (np.isnan(a) or np.isnan(b))]
    if not diffs:
        return {"max_step": np.nan, "monotonic": False}
    mono = all(d >= -1e-9 for d in diffs) or all(d <= 1e-9 for d in diffs)
    return {"max_step": max(abs(d) for d in diffs), "monotonic": mono}


def run(panel=None, fwd=None, controls=None, flags=None, master_prices=None) -> dict:
    if panel is None:
        panel = arf.load_alpha_panel("2025-01-01", "2025-12-31")
    if fwd is None:
        fwd = arf.load_fwd(HORIZONS)
    if controls is None:
        controls = arf.load_exposure_controls()
    if flags is None:
        flags = pd.read_parquet(FLAGS_PATH)

    out = {}
    for h in HORIZONS:
        fit = arf.fit_fusion(panel, fwd, h)
        _, test_panel, _ = arf.purge_train_test(panel, h)
        scores = arf.build_scheme_scores(test_panel, fit["schemes"][MAIN])
        codes = sorted(test_panel["symbol"].unique())
        if master_prices is None:
            prices = bt.load_prices(codes, TEST_START, PRICE_END)
        else:
            prices = {c: master_prices[c] for c in codes if c in master_prices}
        rows = [run_topn(scores, prices, flags, controls, n, h) for n in TOPN_GRID]
        out[h] = {"rows": rows,
                  "smooth_return": smoothness(rows, "total_return"),
                  "smooth_sharpe": smoothness(rows, "sharpe")}
        print(f"\n=== {h}d Top-N 敏感性 ===")
        for r in rows:
            print(f"  N={r['top_n']:3d} ret={_pct(r['total_return'])} exc={_pct(r['excess'])} "
                  f"sharpe={_num(r['sharpe'])} maxDD={_pct(r['max_drawdown'])} "
                  f"execRate={_pct(r['execution_rate'])} top5={_pct(r['conc_top5'])} "
                  f"execAmtPct={_num(r['exec_log_amount'])}")
    return out


def _pct(x):
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x*100:+.2f}%"


def _num(x):
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:+.2f}"


if __name__ == "__main__":
    run()
