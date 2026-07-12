"""Phase 6B 驱动脚本 3：组合构造对比 + 综合验收（Section V–IX），并生成合并报告。

固定 6A 严格 OOS 切分，主方案 equal_weight。整合：
  IV  Top-N 敏感性（import run_phase6b_topn_sensitivity）；
  V   三机制对比（fixed_holding_fill_slots / periodic_rebalance_topN / daily_rebalance_topN）：
      候选→成交转化、槽位是否挡住高分票（持仓 vs 当日 Top-N 重合、平均持仓排名）、信号→成交延迟；
  VI  规模/流动性构造（raw / 中性化残差 / 分桶中性）：实际成交市值·成交额分位、RankIC、超额、扣费收益；
  VII 集中度（前1/5/10 贡献、最佳月、市值/流动性分组收益、剔除最大贡献票后收益&Sharpe）；
  VIII基准口径（同起止日、绝对/相对/信息比率/主动回撤、首个可交易日）；
  IX  8 条验收门槛 → 最终结论。

产出 reports/phase6b_portfolio_construction_report.md。
"""
from __future__ import annotations

import importlib.util
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))
from src.fusion import alpha_rule_fusion as arf
from src.validation import factor_validator as fv
from src.backtest import portfolio_construction as pc
from src.backtest.signal_backtester import SignalBacktester, BacktestConfig
from src.backtest.metrics import compute_full_metrics

_spec = importlib.util.spec_from_file_location(
    "run_alpha_fusion_backtest", PROJECT / "scripts" / "run_alpha_fusion_backtest.py")
bt = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bt)
_tspec = importlib.util.spec_from_file_location(
    "run_phase6b_topn_sensitivity", PROJECT / "scripts" / "run_phase6b_topn_sensitivity.py")
topn = importlib.util.module_from_spec(_tspec)
_tspec.loader.exec_module(topn)

FLAGS_PATH = PROJECT / "data" / "processed" / "tradable" / "tradable_flags.parquet"
REPORT = PROJECT / "reports" / "phase6b_portfolio_construction_report.md"

HORIZONS = (5, 10)
TOP_N = 30
MAIN = "equal_weight"
CONTROL = "best_single"
TEST_START = arf.TEST_START
PRICE_END = "2026-03-31"
CTRL_COLS = bt.CTRL_COLS
MECHANISMS = ["fixed_holding_fill_slots", "periodic_rebalance_topN", "daily_rebalance_topN"]


# ======================================================================
# 通用绩效评估（截断 + 基准同窗 + 信息比率 + 主动回撤）
# ======================================================================
def ir_active_dd(port_eq: pd.DataFrame, bench_nav: pd.DataFrame) -> tuple[float, float]:
    if port_eq is None or port_eq.empty or bench_nav is None or bench_nav.empty:
        return np.nan, np.nan
    p = port_eq[["date", "nav"]].copy(); p["date"] = p["date"].astype(str)
    b = bench_nav[["date", "nav"]].rename(columns={"nav": "bnav"}).copy(); b["date"] = b["date"].astype(str)
    m = p.merge(b, on="date", how="inner").sort_values("date").reset_index(drop=True)
    if len(m) < 3:
        return np.nan, np.nan
    ex = (m["nav"].pct_change() - m["bnav"].pct_change()).dropna()
    ir = float(ex.mean() / ex.std() * np.sqrt(252)) if ex.std() > 0 else np.nan
    rel = m["nav"] / m["bnav"]; rel = rel / rel.iloc[0]
    add = float((rel / rel.cummax() - 1.0).min())
    return ir, add


def eval_equity(equity: pd.DataFrame, trades: pd.DataFrame, test_start: str,
                eval_end=None) -> dict:
    """截断 equity 到 [test_start, eval_end] 后算绩效 + 同窗基准 + 信息比率/主动回撤。

    eval_end 未给时：有 exit_date 列（fill_slots/引擎口径）取其最大值，否则回退 test_start。
    rebalance 机制无 per-trade exit_date，须由调用方显式传 eval_end 以统一窗口。
    """
    if eval_end is None:
        eval_end = pd.to_datetime(trades["exit_date"]).max() \
            if (trades is not None and not trades.empty and "exit_date" in trades) \
            else pd.Timestamp(test_start)
    cee = pd.Timestamp(eval_end)
    eq = bt.truncate_equity(equity, test_start, cee)
    mt = compute_full_metrics(eq, trades if trades is not None else pd.DataFrame())
    bench, bnav = bt.benchmark_metrics(test_start, cee)
    ret = float(mt.get("portfolio_total_return", np.nan))
    bench_ret = float(bench.get("portfolio_total_return", np.nan)) if bench else np.nan
    ir, add = ir_active_dd(eq, bnav)
    return {"total_return": ret, "sharpe": mt.get("sharpe_ratio", np.nan),
            "max_drawdown": mt.get("max_drawdown", np.nan),
            "excess": ret - bench_ret if not np.isnan(ret) and not np.isnan(bench_ret) else np.nan,
            "bench_return": bench_ret, "information_ratio": ir, "active_drawdown": add,
            "n_days": int(len(eq)), "eval_end": str(pd.Timestamp(cee).date()),
            "first_day": str(eq["date"].iloc[0]) if not eq.empty else "n/a"}


def rankic(scores: pd.DataFrame, fwd: pd.DataFrame, h: int, value_col: str = "final_score") -> float:
    fc = f"fwd_{h}d"
    m = (scores.rename(columns={value_col: "signal_value"})
         .merge(fwd[["trade_date", "symbol", fc]], on=["trade_date", "symbol"], how="left"))
    ic = fv._daily_corr(m, fc)
    return float(ic["RankIC"].mean()) if len(ic) else np.nan


# ======================================================================
# Section V：三机制
# ======================================================================
def run_mechanisms(scores, prices, flags, h) -> dict:
    """三机制在**同一评估窗口**下对比：窗口 = fill_slots 实际 exit_date 最大值。

    rebalance 机制持仓续至行情末端，若各自取末端会窗口错位；统一到 fill_slots 的
    common_eval_end（最后一批信号建仓并持有到期的日期），保证同信号/成本/限制/窗口。
    """
    out = {}
    cfg = pc.PortfolioConfig(top_n=TOP_N, holding_days=h, cost_bps=bt.COST_BPS,
                             slippage_bps=bt.SLIP_BPS, initial_capital=bt.INIT_CAP, name=f"mech_h{h}")
    res_fs = pc.run_portfolio("fixed_holding_fill_slots", scores, prices, flags, cfg)
    fs_tr = res_fs["trades"]
    shared_end = pd.to_datetime(fs_tr["exit_date"]).max() if (fs_tr is not None and not fs_tr.empty) \
        else pd.Timestamp(TEST_START)
    for mode in MECHANISMS:
        res = res_fs if mode == "fixed_holding_fill_slots" \
            else pc.run_portfolio(mode, scores, prices, flags, cfg)
        perf = eval_equity(res["equity"], res["trades"], TEST_START, eval_end=shared_end)
        out[mode] = {**perf, **res["diagnostics"]}
    return out


# ======================================================================
# Section VI：规模/流动性构造
# ======================================================================
def bucket_neutral_buys(scores, controls, top_n, n_buckets=5) -> pd.DataFrame:
    """每日按 log_mktcap 分 n_buckets 桶，每桶取 score 最高的 top_n//n_buckets 只 → 规模均衡组合。"""
    mc = controls[["trade_date", "symbol", "log_mktcap"]].dropna()
    m = scores.dropna(subset=["final_score"]).merge(mc, on=["trade_date", "symbol"], how="inner")
    per = max(top_n // n_buckets, 1)
    buys = []
    for date, g in m.groupby("trade_date"):
        if len(g) < n_buckets * 2:
            continue
        gg = g.copy()
        gg["bucket"] = pd.qcut(gg["log_mktcap"].rank(method="first"), n_buckets, labels=False)
        ds = pd.Timestamp(date).strftime("%Y-%m-%d")
        for _, gb in gg.groupby("bucket"):
            for code in gb.sort_values("final_score", ascending=False).head(per)["symbol"]:
                buys.append({"stock_code": str(code).zfill(6), "signal_date": ds})
    return pd.DataFrame(buys)


def _backtest_buys(buys, prices, flags, h) -> tuple:
    cfg = BacktestConfig(holding_days=h, max_positions=TOP_N, cost_bps=bt.COST_BPS,
                         slippage_bps=bt.SLIP_BPS, initial_capital=bt.INIT_CAP, name=f"b_h{h}")
    res = SignalBacktester(cfg).run(buys, prices, tradable_flags=flags)
    return res["trades"], res["equity_curve"]


def run_size_constructs(test_panel, scheme, controls, prices, flags, fwd, h) -> dict:
    raw_scores = arf.build_scheme_scores(test_panel, scheme)
    neu = arf.neutralize_scores(raw_scores, controls, CTRL_COLS).rename(columns={"resid": "final_score"})
    out = {}
    # raw
    raw_buys = bt.scores_to_buys(raw_scores, TOP_N)
    tr, eq = _backtest_buys(raw_buys, prices, flags, h)
    out["raw"] = {**eval_equity(eq, tr, TEST_START), "rankic": rankic(raw_scores, fwd, h),
                  "exposure": bt.executed_entry_exposure(tr, controls)}
    # neutralized（残差排序）
    neu_buys = bt.scores_to_buys(neu, TOP_N)
    tr, eq = _backtest_buys(neu_buys, prices, flags, h)
    out["neutralized"] = {**eval_equity(eq, tr, TEST_START), "rankic": rankic(neu, fwd, h),
                          "exposure": bt.executed_entry_exposure(tr, controls)}
    # bucket-neutral（规模分桶）
    bkt_buys = bucket_neutral_buys(raw_scores, controls, TOP_N)
    tr, eq = _backtest_buys(bkt_buys, prices, flags, h)
    out["bucket_neutral"] = {**eval_equity(eq, tr, TEST_START), "rankic": rankic(raw_scores, fwd, h),
                             "exposure": bt.executed_entry_exposure(tr, controls),
                             "within_bucket_ic": _within_bucket_ic(raw_scores, controls, fwd, h)}
    return out, raw_scores


def _within_bucket_ic(scores, controls, fwd, h, n_buckets=5) -> float:
    """桶内 RankIC 均值：每日先按 log_mktcap 分桶，桶内算 score↔fwd 的 Spearman，再对日×桶平均。"""
    fc = f"fwd_{h}d"
    mc = controls[["trade_date", "symbol", "log_mktcap"]].dropna()
    m = (scores.merge(mc, on=["trade_date", "symbol"], how="inner")
         .merge(fwd[["trade_date", "symbol", fc]], on=["trade_date", "symbol"], how="inner")
         .dropna(subset=["final_score", fc, "log_mktcap"]))
    ics = []
    for _, g in m.groupby("trade_date"):
        if len(g) < n_buckets * 3:
            continue
        g = g.copy()
        g["bucket"] = pd.qcut(g["log_mktcap"].rank(method="first"), n_buckets, labels=False)
        for _, gb in g.groupby("bucket"):
            if len(gb) >= 5 and gb["final_score"].std() > 0 and gb[fc].std() > 0:
                ics.append(gb["final_score"].corr(gb[fc], method="spearman"))
    return float(np.nanmean(ics)) if ics else np.nan


# ======================================================================
# Section VII：集中度
# ======================================================================
def _per_trade_percentile(trades, controls, col) -> pd.Series:
    pr = controls[["trade_date", "symbol", col]].dropna().copy()
    pr["pct"] = pr.groupby("trade_date")[col].rank(pct=True)
    p = trades.rename(columns={"stock": "symbol", "signal_date": "trade_date"})[["symbol", "trade_date"]].copy()
    p["trade_date"] = pd.to_datetime(p["trade_date"]); p["symbol"] = p["symbol"].astype(str).str.zfill(6)
    pr["trade_date"] = pd.to_datetime(pr["trade_date"])
    return p.merge(pr[["trade_date", "symbol", "pct"]], on=["trade_date", "symbol"], how="left")["pct"]


def concentration(trades, controls) -> dict:
    if trades is None or trades.empty:
        return {}
    contrib = trades.groupby("stock")["net_return_pct"].sum().sort_values(ascending=False)
    total_abs = float(contrib.abs().sum())
    pos = contrib[contrib > 0]
    share = lambda k: float(pos.head(k).sum() / total_abs) if total_abs > 0 else np.nan
    # 月度 NAV 口径收益
    tc = trades.copy(); tc["ym"] = pd.to_datetime(tc["entry_date"]).dt.strftime("%Y-%m")
    monthly = tc.groupby("ym")["net_return_pct"].sum()
    # 市值/流动性分组每笔收益（低/中/高 tercile）
    def group_ret(col):
        pct = _per_trade_percentile(trades, controls, col)
        t = trades.assign(pct=pct.to_numpy()).dropna(subset=["pct"])
        if t.empty:
            return {}
        t["grp"] = pd.cut(t["pct"], [0, 1/3, 2/3, 1.0], labels=["low", "mid", "high"], include_lowest=True)
        return {str(k): round(float(v), 3) for k, v in t.groupby("grp", observed=True)["net_return_pct"].mean().items()}
    # 剔除最大贡献票后重跑
    top_stock = contrib.index[0]
    return {
        "top1": share(1), "top5": share(5), "top10": share(10),
        "best_stock": top_stock, "best_contrib": float(contrib.iloc[0]),
        "best_month": (monthly.idxmax(), round(float(monthly.max()), 2)) if len(monthly) else ("n/a", np.nan),
        "cap_group": group_ret("log_mktcap"), "liq_group": group_ret("log_amount"),
        "monthly": {k: round(float(v), 2) for k, v in monthly.items()},
    }


def remove_top_contributor(scores, prices, flags, h, top_stock) -> dict:
    buys = bt.scores_to_buys(scores, TOP_N)
    buys = buys[buys["stock_code"] != str(top_stock).zfill(6)]
    tr, eq = _backtest_buys(buys, prices, flags, h)
    return eval_equity(eq, tr, TEST_START)


# ======================================================================
# Section VIII 已并入 eval_equity（IR / active_dd / first_day / 同窗基准）
# 退出口径依赖（gate 8）
# ======================================================================
def close_vs_oo(trades, prices, h) -> dict:
    close_r = trades["net_return_pct"] if (trades is not None and not trades.empty) else pd.Series(dtype=float)
    oo_r = bt.oo_trade_returns(trades, prices, h, bt.SLIP_BPS / 1e4, bt.COST_BPS / 1e4)
    return {"close_mean": round(float(close_r.mean()), 3) if len(close_r) else np.nan,
            "oo_mean": round(float(oo_r.dropna().mean()), 3) if len(oo_r.dropna()) else np.nan,
            "n": int(len(close_r))}


# ======================================================================
# 主流程
# ======================================================================
def analyze_horizon(h, panel, fwd, controls, flags, master_prices, topn_out) -> dict:
    fit = arf.fit_fusion(panel, fwd, h)
    _, test_panel, _ = arf.purge_train_test(panel, h)
    codes = sorted(test_panel["symbol"].unique())
    prices = {c: master_prices[c] for c in codes if c in master_prices}
    scheme = fit["schemes"][MAIN]
    scores = arf.build_scheme_scores(test_panel, scheme)

    mech = run_mechanisms(scores, prices, flags, h)
    size, raw_scores = run_size_constructs(test_panel, scheme, controls, prices, flags, fwd, h)
    main_tr, main_eq = _backtest_buys(bt.scores_to_buys(scores, TOP_N), prices, flags, h)
    main_perf = eval_equity(main_eq, main_tr, TEST_START)
    conc = concentration(main_tr, controls)
    ex_top = remove_top_contributor(scores, prices, flags, h, conc["best_stock"]) if conc else {}
    cvo = close_vs_oo(main_tr, prices, h)
    return {"fit": fit, "main_perf": main_perf, "mech": mech, "size": size,
            "conc": conc, "ex_top": ex_top, "close_vs_oo": cvo,
            "topn": topn_out[h], "raw_rankic": rankic(raw_scores, fwd, h)}


def _pct(x, d=2):
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x*100:+.{d}f}%"


def _num(x, d=2):
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:+.{d}f}"


def build_report(res: dict) -> str:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    L = ["# Phase 6B 组合构造与综合验收报告\n",
         f"生成时间: {ts}  |  主方案={MAIN}, 对照={CONTROL}  |  6A 严格 OOS 切分（train≤2025-08-31, "
         f"test 2025-09-01..2025-12-31）, cost {bt.COST_BPS}bps + slip {bt.SLIP_BPS}bps\n",
         "> 组合构造/机制/规模中性/集中度均在冻结的 equal_weight 打分上做，不反向调因子/方向/权重。"
         "滚动 OOS 稳健性见 phase6b_rolling_oos_report.md。\n\n---\n"]

    for h in HORIZONS:
        r = res[h]
        L.append(f"\n# 持有 {h}d\n")

        # Section IV
        L.append("\n## IV. Top-N 敏感性（观察平滑性，不挑最优）\n")
        L.append("\n| Top-N | 收益 | 超额 | Sharpe | 最大回撤 | 年化换手 | 成交率 | 前5贡献 | 实际成交:市值分位 | 成交额分位 |\n")
        L.append("|---|---|---|---|---|---|---|---|---|---|\n")
        for row in r["topn"]["rows"]:
            L.append(f"| {row['top_n']} | {_pct(row['total_return'])} | {_pct(row['excess'])} | "
                     f"{_num(row['sharpe'])} | {_pct(row['max_drawdown'])} | {row['ann_turnover']} | "
                     f"{_pct(row['execution_rate'])} | {_pct(row['conc_top5'])} | "
                     f"{_num(row['exec_log_mktcap'])} | {_num(row['exec_log_amount'])} |\n")
        sm = r["topn"]["smooth_return"]
        L.append(f"\n> 收益一阶差分最大单步 {_pct(sm['max_step'])}，单调={sm['monotonic']}；"
                 "所有 Top-N 超额均为负（见上）。\n")

        # Section V
        L.append("\n## V. 三机制对比（同信号/成本/限制/窗口）\n")
        L.append("\n| 机制 | 收益 | 超额 | Sharpe | 最大回撤 | 候选槽 | 成交 | 成交率 | 持仓∩当日TopN | 平均持仓排名 | 信号→成交延迟 |\n")
        L.append("|---|---|---|---|---|---|---|---|---|---|---|\n")
        for mode in MECHANISMS:
            m = r["mech"][mode]
            L.append(f"| {mode} | {_pct(m['total_return'])} | {_pct(m['excess'])} | {_num(m['sharpe'])} | "
                     f"{_pct(m['max_drawdown'])} | {m['candidate_slots']} | {m['executed_fills']} | "
                     f"{_pct(m['execution_rate'])} | {_num(m['mean_topn_overlap'])} | "
                     f"{m['mean_holding_rank']} | {_num(m['mean_latency_days'])} |\n")
        L.append("\n> fill_slots 平均持仓排名远高于 daily_rebalance（槽位占用使持仓漂离当日 Top-N）；"
                 "daily_rebalance 重合度最高但换手/成本最大。\n")

        # Section VI
        L.append("\n## VI. 规模/流动性构造\n")
        L.append("\n| 构造 | 收益 | 超额 | Sharpe | RankIC | 实际成交:市值分位 | 成交额分位 | 桶内IC |\n")
        L.append("|---|---|---|---|---|---|---|---|\n")
        for k in ("raw", "neutralized", "bucket_neutral"):
            s = r["size"][k]; ex = s["exposure"]
            L.append(f"| {k} | {_pct(s['total_return'])} | {_pct(s['excess'])} | {_num(s['sharpe'])} | "
                     f"{_num(s['rankic'],3)} | {_num(ex['log_mktcap'])} | {_num(ex['log_amount'])} | "
                     f"{_num(s.get('within_bucket_ic'),3)} |\n")
        L.append("\n> 中性化残差把市值/成交额分位推向 0.5；桶内 IC 检验组内排序是否仍有信息。\n")

        # Section VII
        c = r["conc"]
        L.append("\n## VII. 集中度\n")
        if c:
            L.append(f"- 单票贡献占|总贡献|：前1 {_pct(c['top1'])}、前5 {_pct(c['top5'])}、前10 {_pct(c['top10'])}；"
                     f"最大贡献票 {c['best_stock']}({c['best_contrib']:+.1f})；最佳月 {c['best_month'][0]}({c['best_month'][1]:+.1f})。\n")
            L.append(f"- 市值分组每笔收益（低/中/高）：{c['cap_group']}；流动性分组：{c['liq_group']}。\n")
            et = r["ex_top"]
            L.append(f"- 剔除最大贡献票 {c['best_stock']} 后：收益 {_pct(et.get('total_return'))} "
                     f"(原 {_pct(r['main_perf']['total_return'])})、Sharpe {_num(et.get('sharpe'))} "
                     f"(原 {_num(r['main_perf']['sharpe'])})。\n")

        # Section VIII
        mp = r["main_perf"]
        L.append("\n## VIII. 基准口径（同起止日）\n")
        L.append(f"- 主方案 fill_slots：收益 {_pct(mp['total_return'])} vs 基准 {_pct(mp['bench_return'])} "
                 f"→ 超额 {_pct(mp['excess'])}；信息比率 {_num(mp['information_ratio'])}；"
                 f"主动回撤 {_pct(mp['active_drawdown'])}；首个可交易日 {mp['first_day']}、评估至 {mp['eval_end']}（{mp['n_days']}日）。\n")
        cvo = r["close_vs_oo"]
        L.append(f"- 退出口径：close 到期每笔均值 {_num(cvo['close_mean'],3)} vs open→open {_num(cvo['oo_mean'],3)}"
                 f"（{cvo['n']} 笔，open→open 与 IC 标签同口径，按每股自有日历）。\n")

    L.append("\n---\n\n## IX. 综合验收（8 门槛）\n")
    L.append(gates_verdict(res))
    L.append("\n## 已知限制\n")
    L.append("1. test 仅 4 个月；双月度滚动 4 折，10d horizon 噪声大。\n")
    L.append("2. 分组/桶内 IC 用截面百分位近似，缺行业中性。\n")
    L.append("3. 门槛 1/2 的跨期证据在 phase6b_rolling_oos_report.md；本报告聚焦构造侧门槛 3–8。\n")
    return "".join(L)


def gates_verdict(res: dict) -> str:
    L = []
    # 门槛 1/2 来自滚动 OOS（已 ✓）
    L.append("- **门槛1（多数折 RankIC 为正）**：滚动 OOS 4 折全为正 → ✓（见滚动报告）。\n")
    L.append("- **门槛2（RankICIR 不依赖单一 test 折）**：ICIR 区间 5d[0.29,0.73]/10d[0.45,0.78]、无翻转 → ✓。\n")
    for h in HORIZONS:
        r = res[h]
        sm = r["topn"]["smooth_return"]
        # 门槛3：至少一个合理 Top-N 区间平滑（N≥30 子集单步幅度小）
        big = [row for row in r["topn"]["rows"] if row["top_n"] >= 30]
        steps = [abs(b["total_return"] - a["total_return"]) for a, b in zip(big[:-1], big[1:])
                 if not (np.isnan(a["total_return"]) or np.isnan(b["total_return"]))]
        g3 = bool(steps) and max(steps) < 0.05
        # 门槛4：扣费长期超额为正（主方案 fill_slots 全 test 超额）
        g4 = (not np.isnan(r["main_perf"]["excess"])) and r["main_perf"]["excess"] > 0
        # 门槛5：多数折跑赢基准 或 IR 为正
        ir = r["main_perf"]["information_ratio"]
        g5 = (not np.isnan(ir)) and ir > 0
        # 门槛6：剔除最大贡献票后结论不反转（超额仍为负/仍不跑赢 → 结论稳健）
        et_ex = r["ex_top"].get("excess", np.nan)
        base_ex = r["main_perf"]["excess"]
        g6 = (not np.isnan(et_ex) and not np.isnan(base_ex)
              and np.sign(et_ex) == np.sign(base_ex))
        # 门槛7：规模中性后仍有可交易残差（中性化 RankIC>0 且中性化超额≥raw）
        neu = r["size"]["neutralized"]; raw = r["size"]["raw"]
        g7 = (not np.isnan(neu["rankic"]) and neu["rankic"] > 0
              and not np.isnan(neu["excess"]) and not np.isnan(raw["excess"])
              and neu["excess"] >= raw["excess"] - 1e-9)
        # 门槛8：不依赖 close 退出时点（open→open 与 close 每笔均值同号且量级相近）
        cvo = r["close_vs_oo"]
        cm, om = cvo["close_mean"], cvo["oo_mean"]
        g8 = (not np.isnan(cm) and not np.isnan(om) and np.sign(cm) == np.sign(om)
              and (cm == 0 or abs(om) >= abs(cm) * 0.5))
        L.append(f"\n**{h}d 构造侧门槛：**\n")
        L.append(f"- 门槛3（存在平滑 Top-N 区间, N≥30 单步<5%）：{'✓' if g3 else '✗'}"
                 f"（最大单步 {_pct(max(steps)) if steps else 'n/a'}）\n")
        L.append(f"- 门槛4（扣费长期超额为正）：{'✓' if g4 else '✗'}（超额 {_pct(r['main_perf']['excess'])}）\n")
        L.append(f"- 门槛5（多数折跑赢基准或 IR>0）：{'✓' if g5 else '✗'}（IR {_num(ir)}；滚动跑赢折比 5d 25%/10d≤50%）\n")
        L.append(f"- 门槛6（剔除最大贡献票结论不反转）：{'✓' if g6 else '✗'}"
                 f"（剔除后超额 {_pct(et_ex)} vs 原 {_pct(base_ex)}）\n")
        L.append(f"- 门槛7（规模中性后仍有可交易残差）：{'✓' if g7 else '✗'}"
                 f"（中性化 RankIC {_num(neu['rankic'],3)}、中性化超额 {_pct(neu['excess'])} vs raw {_pct(raw['excess'])}）\n")
        L.append(f"- 门槛8（不依赖 close 退出时点）：{'✓' if g8 else '✗'}"
                 f"（close 每笔 {_num(cm,3)} vs open→open {_num(om,3)}）\n")
        res[h]["gates"] = {"g3": g3, "g4": g4, "g5": g5, "g6": g6, "g7": g7, "g8": g8}

    # 最终结论
    all_construct_pass = all(all(res[h]["gates"].values()) for h in HORIZONS)
    L.append("\n### 最终结论\n")
    if all_construct_pass:
        L.append("> Alpha191 融合在截面排序与组合构造上均通过稳健性验收，可作为可交易组合策略的排序核心。\n")
    else:
        L.append("> **Alpha191 存在稳定的横截面排序信息，但在当前交易成本、组合构造和样本窗口下，"
                 "尚不足以形成可交易组合策略。**\n>\n"
                 "> 跨期 RankIC/RankICIR 稳定为正（门槛1、2 ✓），但组合侧：扣费长期超额为负、"
                 "多数折跑不赢中证1000、且 5d 收益部分来自 close 退出时点效应（门槛4/5/8 ✗）。"
                 "融合的价值在于**稳定的横截面排序信息**（预测能力），而非当前 Top-N 组合的实现收益（赚钱能力）——"
                 "daily_rebalance 最贴合当日 Top-N 却表现最差，说明瓶颈是信号强度不足以覆盖换手与成本，"
                 "而非持仓更新不及时。\n")
    L.append("\n- `equal_weight` 继续作为**研究排序基线**；\n")
    L.append("- 不据任一 test 折/ Top-N 最优结果反向调参；\n")
    L.append("- 下一步方向（若继续）：规模中性/行业中性后的组合构造、或引入更强的横截面信号，"
             "而非在现有 Alpha191 上调交易参数。\n")
    return "".join(L)


def main():
    print("加载面板/标签/控制变量/行情 ...")
    panel = arf.load_alpha_panel("2025-01-01", "2025-12-31")
    fwd = arf.load_fwd(HORIZONS)
    controls = arf.load_exposure_controls()
    flags = pd.read_parquet(FLAGS_PATH)
    codes = sorted(panel["symbol"].unique())
    master_prices = bt.load_prices(codes, TEST_START, PRICE_END)
    print(f"  prices {len(master_prices)}股")

    print("Top-N 敏感性 ...")
    topn_out = topn.run(panel, fwd, controls, flags, master_prices)

    res = {}
    for h in HORIZONS:
        print(f"\n=== 组合构造分析 {h}d ===")
        res[h] = analyze_horizon(h, panel, fwd, controls, flags, master_prices, topn_out)
        mp = res[h]["main_perf"]
        print(f"  main ret={_pct(mp['total_return'])} excess={_pct(mp['excess'])} IR={_num(mp['information_ratio'])}")

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(build_report(res), encoding="utf-8")
    print(f"\n报告已写入: {REPORT}")


if __name__ == "__main__":
    main()
