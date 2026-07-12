"""Phase 6B 驱动脚本 1：滚动 OOS（Section III）。

对多个时间切分**逐折重跑**因子筛选/方向/去相关代表/权重（只在该折 train 上），
逐 horizon purge（label_end<test_start）+ embargo=6，test 冻结评估。记录每折入选因子、
方向、权重与 Top-N 回测，检验融合的**跨期稳健性**（而非单一 test 窗口的最优结果）。

两种切分（test 区块统一双月度，10d horizon 下月度过薄）：
  - expanding：train 从数据起点扩张；
  - rolling：train 固定约 4 个月滚动。
主方案 = equal_weight（研究排序基线）；best_single 作高规模暴露对照。
不根据任何 test 折的最优结果反向修改筛选门槛/方向/权重。

产出 reports/phase6b_rolling_oos_report.md。
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
from src.backtest.signal_backtester import SignalBacktester, BacktestConfig
from src.backtest.metrics import compute_full_metrics

# 复用 6A.1 脚本的 load_prices / benchmark_metrics / truncate_equity / scores_to_buys
_spec = importlib.util.spec_from_file_location(
    "run_alpha_fusion_backtest", PROJECT / "scripts" / "run_alpha_fusion_backtest.py")
bt = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bt)

FLAGS_PATH = PROJECT / "data" / "processed" / "tradable" / "tradable_flags.parquet"
REPORT = PROJECT / "reports" / "phase6b_rolling_oos_report.md"

DATA_START = "2025-01-02"
PRICE_END = "2026-03-31"
TOP_N = 30
MAX_POS = 30
HORIZONS = (5, 10)
EMBARGO = arf.EMBARGO
MAIN = "equal_weight"                    # Section I：主方案
CONTROL = "best_single"                  # 高规模暴露对照

# 双月度 test 区块：(train_end, test_start, test_end)。expanding 全用 DATA_START 起 train；
# rolling 用 train_start≈test_start 前 4 个月。fold C = 6A 锚点（train≤2025-08-31, test 从 09-01）。
BLOCKS = [
    ("2025-04-30", "2025-05-01", "2025-06-30"),
    ("2025-06-30", "2025-07-01", "2025-08-31"),
    ("2025-08-31", "2025-09-01", "2025-10-31"),
    ("2025-10-31", "2025-11-01", "2025-12-31"),
]
ROLL_TRAIN_START = {
    "2025-04-30": "2025-01-02", "2025-06-30": "2025-03-01",
    "2025-08-31": "2025-05-01", "2025-10-31": "2025-07-01",
}


def slice_prices(master: dict, start: str, end: str) -> dict:
    out = {}
    for c, df in master.items():
        sub = df[(df["date_str"] >= start) & (df["date_str"] <= end)]
        if not sub.empty:
            out[c] = sub.reset_index(drop=True)
    return out


def fold_metrics(scores, fwd, prices, flags, test_start, test_end, h) -> dict:
    """单折单方案：test RankIC/RankICIR + Top-N 回测（截断到实际 exit_date）+ 基准同窗超额。"""
    fc = f"fwd_{h}d"
    m = (scores.rename(columns={"final_score": "signal_value"})
         .merge(fwd[["trade_date", "symbol", fc]], on=["trade_date", "symbol"], how="left"))
    ic = fv._daily_corr(m, fc)
    rankic = float(ic["RankIC"].mean()) if len(ic) else np.nan
    rankicir = float(rankic / ic["RankIC"].std()) if len(ic) and ic["RankIC"].std() > 0 else np.nan

    buys = bt.scores_to_buys(scores, TOP_N)
    cfg = BacktestConfig(holding_days=h, max_positions=MAX_POS, cost_bps=bt.COST_BPS,
                         slippage_bps=bt.SLIP_BPS, initial_capital=bt.INIT_CAP, name=f"f_h{h}")
    res = SignalBacktester(cfg).run(buys, prices, tradable_flags=flags)
    trades, equity = res["trades"], res["equity_curve"]
    cee = pd.to_datetime(trades["exit_date"]).max() if (trades is not None and not trades.empty) \
        else pd.Timestamp(test_end)
    eq = bt.truncate_equity(equity, test_start, cee)
    mt = compute_full_metrics(eq, trades if trades is not None else pd.DataFrame())
    bench, _ = bt.benchmark_metrics(test_start, cee)
    ret = float(mt.get("portfolio_total_return", np.nan))
    bench_ret = float(bench.get("portfolio_total_return", np.nan)) if bench else np.nan
    return {
        "rankic": rankic, "rankicir": rankicir,
        "total_return": ret, "sharpe": mt.get("sharpe_ratio", np.nan),
        "max_drawdown": mt.get("max_drawdown", np.nan),
        "n_trades": int(len(trades)) if trades is not None else 0,
        "bench_return": bench_ret,
        "excess": (ret - bench_ret) if (not np.isnan(ret) and not np.isnan(bench_ret)) else np.nan,
        "eval_end": str(pd.Timestamp(cee).date()), "n_days": int(len(eq)),
    }


def run_fold(mode: str, panel, fwd, master_prices, flags, block, h) -> dict:
    train_end, test_start, test_end = block
    train_start = DATA_START if mode == "expanding" else ROLL_TRAIN_START[train_end]
    sub_panel = panel[panel["trade_date"] >= pd.Timestamp(train_start)]
    fit = arf.fit_fusion(sub_panel, fwd, h, train_end=train_end, test_start=test_start, embargo=EMBARGO)
    rec = {"mode": mode, "train_start": train_start, "train_end": train_end,
           "test_start": test_start, "test_end": test_end, "horizon": h}
    if "error" in fit:
        rec["error"] = fit["error"]; rec["kept"] = []
        return rec
    kept = fit["kept"]
    signs = {c: float(fit["schemes"][MAIN]["signs"][c]) for c in kept}
    rec.update({"kept": kept, "signs": signs, "n_kept": len(kept),
                "best_single_factor": fit["schemes"][CONTROL]["factors"][0],
                "purged_days": fit["purge"].get("purged_rows", np.nan)})

    test_block = panel[(panel["trade_date"] >= pd.Timestamp(test_start))
                       & (panel["trade_date"] <= pd.Timestamp(test_end))]
    price_end = (pd.Timestamp(test_end) + pd.Timedelta(days=45)).strftime("%Y-%m-%d")
    prices = slice_prices(master_prices, test_start, min(price_end, PRICE_END))
    for tag, scheme_name in (("main", MAIN), ("control", CONTROL)):
        scores = arf.build_scheme_scores(test_block, fit["schemes"][scheme_name])
        rec[tag] = fold_metrics(scores, fwd, prices, flags, test_start, test_end, h)
    return rec


# ======================================================================
# 聚合稳健性指标
# ======================================================================
def aggregate(folds: list[dict]) -> dict:
    ok = [f for f in folds if "error" not in f]
    n = len(ok)
    if n == 0:
        return {"n_folds": 0}
    ric = [f["main"]["rankic"] for f in ok]
    ricir = [f["main"]["rankicir"] for f in ok]
    ret = [f["main"]["total_return"] for f in ok]
    exc = [f["main"]["excess"] for f in ok]
    # 因子入选频率 & 方向翻转
    freq: dict[str, int] = {}
    dir_by_factor: dict[str, set] = {}
    for f in ok:
        for c in f["kept"]:
            freq[c] = freq.get(c, 0) + 1
            dir_by_factor.setdefault(c, set()).add(f["signs"][c])
    flipped = [c for c, s in dir_by_factor.items() if len(s) > 1]
    # kept 集合稳定性：相邻折 Jaccard 均值
    jac = []
    for a, b in zip(ok[:-1], ok[1:]):
        sa, sb = set(a["kept"]), set(b["kept"])
        if sa or sb:
            jac.append(len(sa & sb) / len(sa | sb))
    return {
        "n_folds": n,
        "rankic_pos_ratio": float(np.mean([r > 0 for r in ric])),
        "rankicir_min": float(np.nanmin(ricir)), "rankicir_max": float(np.nanmax(ricir)),
        "rankicir_mean": float(np.nanmean(ricir)),
        "positive_return_ratio": float(np.mean([r > 0 for r in ret if not np.isnan(r)])),
        "beat_bench_ratio": float(np.mean([e > 0 for e in exc if not np.isnan(e)])),
        "n_kept_min": min(f["n_kept"] for f in ok), "n_kept_max": max(f["n_kept"] for f in ok),
        "factor_freq": dict(sorted(freq.items(), key=lambda kv: -kv[1])),
        "direction_flipped": flipped,
        "kept_jaccard_mean": float(np.mean(jac)) if jac else np.nan,
    }


# ======================================================================
# 报告
# ======================================================================
def _p(x, d=3, pct=False):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "n/a"
    return f"{x*100:+.{d}f}%" if pct else f"{x:+.{d}f}"


def build_report(results: dict, agg: dict) -> str:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    L = ["# Phase 6B 滚动 OOS 稳健性报告\n",
         f"生成时间: {ts}  |  主方案={MAIN}, 对照={CONTROL}  |  Top-{TOP_N}, 逐折重筛/去相关/权重, "
         f"逐 horizon purge + embargo={EMBARGO}, test 冻结\n",
         "> 每折在其**自己的 train** 上独立确定因子/方向/权重；不根据任何 test 折结果反向调参。"
         "test 区块为双月度（10d horizon 下月度样本过薄）。基准=中证1000，同窗口起止。\n\n---\n"]

    for h in HORIZONS:
        L.append(f"\n## 持有 {h}d\n")
        for mode in ("expanding", "rolling"):
            folds = results[(h, mode)]
            L.append(f"\n### {mode} window\n")
            L.append("\n| 折 | train | test | 入选数 | 主RankIC | 主RankICIR | 主收益 | 基准 | 超额 | 主Sharpe | 主回撤 | 对照收益 | 对照Sharpe |\n")
            L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|\n")
            for f in folds:
                if "error" in f:
                    L.append(f"| {f['test_start'][:7]} | {f['train_start'][:7]}→{f['train_end'][:7]} | "
                             f"{f['test_start'][:7]}→{f['test_end'][:7]} | 0 | 无因子通过 | | | | | | | | |\n")
                    continue
                mm, cc = f["main"], f["control"]
                L.append(f"| {f['test_start'][:7]} | {f['train_start'][:7]}→{f['train_end'][:7]} | "
                         f"{f['test_start'][:7]}→{f['test_end'][:7]} | {f['n_kept']} | "
                         f"{_p(mm['rankic'])} | {_p(mm['rankicir'])} | {_p(mm['total_return'],2,True)} | "
                         f"{_p(mm['bench_return'],2,True)} | {_p(mm['excess'],2,True)} | "
                         f"{_p(mm['sharpe'],2)} | {_p(mm['max_drawdown'],2,True)} | "
                         f"{_p(cc['total_return'],2,True)} | {_p(cc['sharpe'],2)} |\n")
            a = agg[(h, mode)]
            if a.get("n_folds"):
                L.append(f"\n**{mode} 聚合（{a['n_folds']} 折）**：RankIC 正比 {_p(a['rankic_pos_ratio'],0,True)}；"
                         f"RankICIR [{_p(a['rankicir_min'],2)}, {_p(a['rankicir_max'],2)}] 均值 {_p(a['rankicir_mean'],2)}；"
                         f"正收益折比 {_p(a['positive_return_ratio'],0,True)}；跑赢基准折比 {_p(a['beat_bench_ratio'],0,True)}；"
                         f"入选数 {a['n_kept_min']}–{a['n_kept_max']}；kept 相邻 Jaccard {_p(a['kept_jaccard_mean'],2)}；"
                         f"方向翻转因子 {a['direction_flipped'] or '无'}。\n")
                L.append(f"\n  因子入选频率：" + ", ".join(f"{k}×{v}" for k, v in a["factor_freq"].items()) + "\n")

    L.append("\n---\n\n## 验收判定（Section IX 部分门槛）\n")
    L.append(verdict(agg))
    L.append("\n## 已知限制\n")
    L.append("1. 全样本仅 2025 全年；双月度 test 区块 4 折，10d horizon 每折信号日少、噪声大。\n")
    L.append("2. rolling 首折 train 起点=数据起点，与 expanding 首折等价。\n")
    L.append("3. 主方案 equal_weight 权重恒为 1/n_kept，权重稳定性以 kept 集合/方向稳定性表征。\n")
    return "".join(L)


def verdict(agg: dict) -> str:
    L = []
    for h in HORIZONS:
        for mode in ("expanding", "rolling"):
            a = agg[(h, mode)]
            if not a.get("n_folds"):
                L.append(f"- **{h}d {mode}**：无有效折。\n"); continue
            g1 = a["rankic_pos_ratio"] >= 0.5                       # 门槛1：多数折 RankIC 正
            span = a["rankicir_max"] - a["rankicir_min"]
            g2 = a["rankicir_min"] > 0 and span <= abs(a["rankicir_mean"]) * 2 + 1e-9  # 门槛2：不依赖单一折
            g5 = a["beat_bench_ratio"] >= 0.5                       # 门槛5：多数折跑赢基准
            L.append(f"- **{h}d {mode}**：门槛1 多数折RankIC正 {'✓' if g1 else '✗'}"
                     f"（{_p(a['rankic_pos_ratio'],0,True)}）；门槛2 RankICIR不依赖单一折 "
                     f"{'✓' if g2 else '✗'}（区间[{_p(a['rankicir_min'],2)},{_p(a['rankicir_max'],2)}]）；"
                     f"门槛5 多数折跑赢基准 {'✓' if g5 else '✗'}（{_p(a['beat_bench_ratio'],0,True)}）。\n")
    L.append("\n> 综合门槛判定见 phase6b_portfolio_construction_report.md（Top-N/机制/规模中性/集中度合并结论）。\n")
    return "".join(L)


def main():
    print("加载面板/标签/行情/flags ...")
    panel = arf.load_alpha_panel(DATA_START, "2025-12-31")
    fwd = arf.load_fwd(HORIZONS)
    flags = pd.read_parquet(FLAGS_PATH)
    codes = sorted(panel["symbol"].unique())
    master_prices = bt.load_prices(codes, DATA_START, PRICE_END)
    print(f"  panel {panel['trade_date'].nunique()}日 {len(codes)}股 | prices {len(master_prices)}股")

    results, agg = {}, {}
    for h in HORIZONS:
        for mode in ("expanding", "rolling"):
            folds = [run_fold(mode, panel, fwd, master_prices, flags, blk, h) for blk in BLOCKS]
            results[(h, mode)] = folds
            agg[(h, mode)] = aggregate(folds)
            a = agg[(h, mode)]
            print(f"  {h}d {mode}: folds={a.get('n_folds')} RankICpos={_p(a.get('rankic_pos_ratio'),0,True)} "
                  f"beatBench={_p(a.get('beat_bench_ratio'),0,True)} ICIR[{_p(a.get('rankicir_min'),2)},{_p(a.get('rankicir_max'),2)}]")

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(build_report(results, agg), encoding="utf-8")
    print(f"\n报告已写入: {REPORT}")


if __name__ == "__main__":
    main()
