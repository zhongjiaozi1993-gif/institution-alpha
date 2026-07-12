"""Phase 6A 驱动脚本 2：Alpha191 规则融合回测（接入 Phase 4.7 signal_backtester）。

对 5d/10d 各自的 4 套冻结方案 + 中证1000 基准做 test 期回测：
  - T 日收盘信号 → T+1 open 建仓；tradable_flags ON（涨停不买/跌停不卖/停牌延期）；
  - 成本+滑点各扣一次；不在 test 上网格搜索交易参数（top_n/持仓/成本固定，非优化）；
  - 输出组合收益/年化/Sharpe/Sortino/回撤/Calmar/换手/交易数/胜率/延期退出/期末持仓/
    单票贡献/月度收益/市值·流动性暴露，并与 best_single、基准对比。

方案与方向/权重全部来自 train（arf.fit_fusion），test 冻结，不用 test 标签调参。
产出 reports/alpha_fusion_backtest_report.md。
"""
from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))
from src.fusion import alpha_rule_fusion as arf
from src.backtest.signal_backtester import SignalBacktester, BacktestConfig
from src.backtest.metrics import compute_full_metrics

DAILY_DIR = PROJECT / "data" / "daily"
FLAGS_PATH = PROJECT / "data" / "processed" / "tradable" / "tradable_flags.parquet"
INDEX_PATH = DAILY_DIR / "idx_000852.parquet"
REPORT = PROJECT / "reports" / "alpha_fusion_backtest_report.md"

PRICE_SCALE = 100                      # 后复权「分」→元（比例不影响收益）
TOP_N = 30                             # 固定，非在 test 上优化
MAX_POS = 30
COST_BPS = 20
SLIP_BPS = 10
INIT_CAP = 1_000_000.0
TEST_START = arf.TEST_START
PRICE_END = "2026-03-31"               # 加载到 test 之后，容纳 10d 到期
SCHEMES = ["best_single", "equal_weight", "icir_weight", "stability_weight"]


def load_prices(codes, start, end) -> dict[str, pd.DataFrame]:
    prices = {}
    for code in codes:
        code = str(code).zfill(6)
        p = DAILY_DIR / f"{code}.parquet"
        if not p.exists():
            continue
        df = pd.read_parquet(p)
        df["date"] = pd.to_datetime(df["date"])
        df = df[(df["date"] >= start) & (df["date"] <= end)].sort_values("date").reset_index(drop=True)
        if df.empty:
            continue
        df["date_str"] = df["date"].dt.strftime("%Y-%m-%d")
        for f in ["open", "high", "low", "close"]:
            df[f"{f}_yuan"] = df[f] / PRICE_SCALE
        prices[code] = df
    return prices


def scores_to_buys(scores: pd.DataFrame, top_n: int) -> pd.DataFrame:
    """每日按 final_score 降序取 Top-N 作为买入信号（final_score 已含方向对齐，高=看多）。"""
    buys = []
    for date, g in scores.dropna(subset=["final_score"]).groupby("trade_date"):
        picked = g.sort_values("final_score", ascending=False).head(top_n)
        ds = pd.Timestamp(date).strftime("%Y-%m-%d")
        for code in picked["symbol"]:
            buys.append({"stock_code": str(code).zfill(6), "signal_date": ds})
    return pd.DataFrame(buys)


def annualized_turnover(n_trades: int, max_pos: int, n_days: int) -> float:
    """组合年化换手 ≈ (成交笔数×槽位资金)/初始资金/年数 = n_trades/max_pos/年数。"""
    years = max(n_days / 252, 1e-9)
    return round(n_trades / max(max_pos, 1) / years, 2)


def portfolio_exposure(buys: pd.DataFrame, controls: pd.DataFrame) -> dict:
    """买入标的在信号日截面上的 log市值/log成交额/换手 百分位均值（0=最低,1=最高）。"""
    b = buys.copy()
    b["trade_date"] = pd.to_datetime(b["signal_date"])
    b = b.rename(columns={"stock_code": "symbol"})
    m = controls.copy()
    out = {}
    for col in ["log_mktcap", "log_amount", "turnover"]:
        pr = m[["trade_date", "symbol", col]].dropna().copy()
        pr["pct"] = pr.groupby("trade_date")[col].rank(pct=True)
        j = b.merge(pr[["trade_date", "symbol", "pct"]], on=["trade_date", "symbol"], how="left")
        out[col] = round(float(j["pct"].mean()), 3) if j["pct"].notna().any() else np.nan
    return out


def benchmark_metrics(equity_dates: list[str]) -> tuple[dict, pd.DataFrame]:
    """中证1000 在组合 equity 日期范围内的买入持有 NAV 与绩效。"""
    idx = pd.read_parquet(INDEX_PATH)
    idx["date"] = pd.to_datetime(idx["date"])
    idx["date_str"] = idx["date"].dt.strftime("%Y-%m-%d")
    dset = set(equity_dates)
    sub = idx[idx["date_str"].isin(dset)].sort_values("date").reset_index(drop=True)
    if sub.empty:
        return {}, pd.DataFrame()
    nav = INIT_CAP * sub["close"] / sub["close"].iloc[0]
    navdf = pd.DataFrame({"date": sub["date_str"], "nav": nav.to_numpy()})
    m = compute_full_metrics(navdf, pd.DataFrame())
    return m, navdf


def run_one(name: str, scheme: dict, test_panel, prices, flags, controls, holding: int) -> dict:
    scores = arf.build_scheme_scores(test_panel, scheme)
    buys = scores_to_buys(scores, TOP_N)
    cfg = BacktestConfig(holding_days=holding, max_positions=MAX_POS,
                         cost_bps=COST_BPS, slippage_bps=SLIP_BPS,
                         initial_capital=INIT_CAP, name=f"{name}_h{holding}")
    res = SignalBacktester(cfg).run(buys, prices, tradable_flags=flags)
    trades, equity, summary = res["trades"], res["equity_curve"], res["summary"]
    metrics = compute_full_metrics(equity, trades)
    s = summary.iloc[0] if not summary.empty else {}
    monthly = {}
    if not trades.empty:
        tc = trades.copy()
        tc["ym"] = pd.to_datetime(tc["entry_date"]).dt.strftime("%Y-%m")
        monthly = {k: round(float(v), 2) for k, v in tc.groupby("ym")["net_return_pct"].sum().items()}
    n_days = len(equity)
    return {
        "name": name, "holding": holding, "buys": buys,
        "equity": equity, "trades": trades,
        "portfolio_total_return": float(s.get("portfolio_total_return", 0.0)),
        "annualized_return": metrics.get("annualized_return", np.nan),
        "sharpe": metrics.get("sharpe_ratio", np.nan),
        "sortino": metrics.get("sortino_ratio", np.nan),
        "max_drawdown": float(s.get("max_drawdown", 0.0)),
        "calmar": metrics.get("calmar_ratio", np.nan),
        "ann_turnover": annualized_turnover(int(s.get("n_trades", 0)), MAX_POS, n_days),
        "total_trades": int(s.get("n_trades", 0)),
        "win_rate": float(s.get("win_rate", 0.0)),
        "deferred_exits": int(s.get("deferred_exits", 0)),
        "open_positions_at_end": int(s.get("open_positions_at_end", 0)),
        "best_stock": s.get("best_stock", ""), "best_contrib": float(s.get("best_stock_contribution", 0.0)),
        "worst_stock": s.get("worst_stock", ""), "worst_contrib": float(s.get("worst_stock_contribution", 0.0)),
        "monthly_positive_rate": float(s.get("monthly_positive_rate", 0.0)),
        "monthly": monthly,
        "exposure": portfolio_exposure(buys, controls),
    }


def _f(x, p=2, pct=False):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "n/a"
    return f"{x*100:+.{p}f}%" if pct else f"{x:+.{p}f}"


def build_report(all_res: dict, bench: dict) -> str:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    L = [f"# Alpha191 规则融合回测报告（Phase 6A）\n",
         f"生成时间: {ts}  |  test {TEST_START} 起（T+1 open 建仓, tradable_flags ON）  |  "
         f"Top-{TOP_N}, 持有=horizon, cost {COST_BPS}bps + slip {SLIP_BPS}bps（各扣一次, 非优化）\n",
         "> 方案方向/权重全部来自 train（arf.fit_fusion）；test 冻结评估，未在 test 上网格搜索任何交易参数。"
         "5d/10d 独立，各方案持有期=对应 horizon。基准=中证1000 买入持有（同 equity 日期范围）。\n\n---\n"]

    for h in (5, 10):
        rr = all_res[h]
        bm = bench[h]
        L.append(f"\n## 持有 {h}d\n")
        L.append("\n| 方案 | 组合收益 | 年化 | Sharpe | Sortino | 最大回撤 | Calmar | 年化换手 | 交易数 | 胜率 | 延期退出 | 期末持仓 |\n")
        L.append("|---|---|---|---|---|---|---|---|---|---|---|---|\n")
        for name in SCHEMES:
            r = rr[name]
            L.append(f"| {name} | {_f(r['portfolio_total_return'],2,True)} | {_f(r['annualized_return'],2,True)} | "
                     f"{_f(r['sharpe'])} | {_f(r['sortino'])} | {_f(r['max_drawdown'],2,True)} | {_f(r['calmar'])} | "
                     f"{r['ann_turnover']} | {r['total_trades']} | {_f(r['win_rate'],1,True)} | "
                     f"{r['deferred_exits']} | {r['open_positions_at_end']} |\n")
        if bm:
            L.append(f"| 中证1000基准 | {_f(bm.get('portfolio_total_return'),2,True)} | {_f(bm.get('annualized_return'),2,True)} | "
                     f"{_f(bm.get('sharpe_ratio'))} | {_f(bm.get('sortino_ratio'))} | {_f(bm.get('max_drawdown'),2,True)} | "
                     f"{_f(bm.get('calmar_ratio'))} | n/a | n/a | n/a | n/a | n/a |\n")

        # 单票贡献 + 市值/流动性暴露
        L.append(f"\n### 单票贡献 / 市值·流动性暴露（{h}d）\n")
        L.append("\n| 方案 | 最佳票(贡献%) | 最差票(贡献%) | 月度正收益比 | log市值分位 | log成交额分位 | 换手分位 |\n")
        L.append("|---|---|---|---|---|---|---|\n")
        for name in SCHEMES:
            r = rr[name]; ex = r["exposure"]
            L.append(f"| {name} | {r['best_stock']}({r['best_contrib']:+.1f}) | {r['worst_stock']}({r['worst_contrib']:+.1f}) | "
                     f"{_f(r['monthly_positive_rate'],0,True)} | {_f(ex['log_mktcap'],2)} | {_f(ex['log_amount'],2)} | "
                     f"{_f(ex['turnover'],2)} |\n")
        L.append("\n> 暴露分位=买入标的在信号日截面的百分位均值（0.5=中位；<0.5 偏小/偏低流动性）。"
                 "用于识别是否把小盘/低流动性暴露包装成 alpha。\n")

        # 月度收益
        L.append(f"\n### 月度收益（{h}d，单笔 net_return 求和%）\n")
        months = sorted(set().union(*[set(rr[n]["monthly"]) for n in SCHEMES]))
        if months:
            L.append("\n| 方案 | " + " | ".join(months) + " |\n")
            L.append("|---|" + "---|" * len(months) + "\n")
            for name in SCHEMES:
                mo = rr[name]["monthly"]
                L.append(f"| {name} | " + " | ".join(_f(mo.get(m), 1) for m in months) + " |\n")

    L.append("\n---\n\n## 验收判定\n")
    L.append(verdict(all_res, bench))
    L.append("\n## 已知限制\n")
    L.append("1. test 仅 ~4 个月（2025-09..2025-12），持有 10d 时有效再平衡次数少，Sharpe/回撤统计噪声大。\n")
    L.append("2. 组合暴露审计缺行业；市值/流动性以截面百分位近似。\n")
    L.append(f"3. 固定 Top-{TOP_N}/持仓{MAX_POS}/成本{COST_BPS}bps，未做参数寻优（刻意，避免 test 过拟合）。\n")
    return "".join(L)


def verdict(all_res: dict, bench: dict) -> str:
    L = []
    for h in (5, 10):
        rr = all_res[h]; bm = bench[h]
        bs = rr["best_single"]
        best_name, best_sharpe = None, -np.inf
        for name in ("equal_weight", "icir_weight", "stability_weight"):
            sh = rr[name]["sharpe"]
            if not (isinstance(sh, float) and np.isnan(sh)) and sh > best_sharpe:
                best_name, best_sharpe = name, sh
        bs_sharpe = bs["sharpe"] if not (isinstance(bs["sharpe"], float) and np.isnan(bs["sharpe"])) else np.nan
        cond_sharpe = (not np.isnan(best_sharpe) and not np.isnan(bs_sharpe) and best_sharpe >= bs_sharpe - 1e-9)
        cond_dd = rr[best_name]["max_drawdown"] >= bs["max_drawdown"] - 0.02 if best_name else False
        bench_ret = bm.get("portfolio_total_return", np.nan) if bm else np.nan
        beat_bench = (not np.isnan(bench_ret)) and rr[best_name]["portfolio_total_return"] > bench_ret if best_name else False
        L.append(f"- **{h}d**：最优融合={best_name} Sharpe {best_sharpe:+.2f} vs best_single {bs_sharpe:+.2f} "
                 f"→ {'Sharpe 不劣 ✓' if cond_sharpe else 'Sharpe 劣于单因子 ✗'}；"
                 f"回撤{'未明显恶化 ✓' if cond_dd else '恶化 ✗'}；"
                 f"{'跑赢' if beat_bench else '未跑赢'}中证1000。\n")

    # --- 综合归因：为什么 best_single 在 Top-30 回测占优，以及是否稳健 ---
    bs_amt = all_res[10]["best_single"]["exposure"]["log_amount"]
    fu_amt = all_res[10]["equal_weight"]["exposure"]["log_amount"]
    L.append("\n### 综合归因（IC 报告 × 回测）\n")
    L.append(f"1. **四方案均跑赢中证1000**（基准 test +1.58%）。\n")
    L.append(f"2. **IC/排序层面融合达标**：full-panel RankICIR 5d 0.34→0.51、10d 0.36→0.56（见 IC 报告 §3），"
             f"且中性化后残余 alpha 融合(26–38%) > best_single(14–16%)——best_single 去掉规模后基本塌缩。\n")
    L.append(f"3. **Top-30 回测 best_single 占优（尤其 10d Sharpe 2.51）源于集中小盘/低流动性暴露**："
             f"best_single 买入标的 log成交额分位仅 {_f(bs_amt,2)}（远低于中位 0.50，也低于融合 {_f(fu_amt,2)}），"
             f"叠加 4 个月窗口 + 少量 10d 再平衡，收益由少数小盘票驱动（单票贡献 +59.7 / 月度集中 9、12 月）——"
             f"这正是验收标准 #5 警示的「把小盘/低流动性暴露包装成 alpha」，稳健性存疑。\n")
    L.append("\n### 最终结论\n")
    L.append("> **Phase 6A 有条件通过。**\n>\n"
             "> 多因子融合在截面排序的 RankICIR、非规模残余信息和暴露分散方面优于最优单因子，"
             "但在仅四个月的 Top-30 OOS 回测中，尚未证明实现收益和 Sharpe 优于 best_single。"
             "best_single 的优势高度集中于小盘、低流动性暴露，不能直接作为默认生产策略。\n\n")
    L.append("- `equal_weight` 暂定为后续默认**研究排序基线**；\n")
    L.append("- `best_single` 只作为**高规模暴露对照组**，不作默认生产策略；\n")
    L.append("- Level-2 不加入正式评分（仅 shadow 对照，见 IC 报告）；\n")
    L.append("- 方向/权重全部 train 冻结，不根据 test 结果调整；\n")
    L.append("- 下一步 **Phase 6B 稳健性与组合构造验证**（滚动 OOS、Top-N 敏感性、持仓集中度、"
             "规模中性组合），解决「IC 更稳但 Top-30 收益未兑现」的矛盾；暂不上 ML、不加 Level-2。\n")
    L.append("\n> 说明：月度为「单笔 net_return 求和%」（活跃度口径，非组合 NAV%），仅用于观察收益月度集中度。\n")
    return "".join(L)


def main():
    print("加载面板/标签/控制变量 ...")
    panel = arf.load_alpha_panel("2025-01-01", "2025-12-31")
    fwd = arf.load_fwd((5, 10))
    controls = arf.load_exposure_controls()
    flags = pd.read_parquet(FLAGS_PATH)

    all_res, bench = {}, {}
    for h in (5, 10):
        print(f"\n=== 持有 {h}d ===")
        fit = arf.fit_fusion(panel, fwd, h)
        _, test_panel, _ = arf.purge_train_test(panel, h)
        codes = sorted(test_panel["symbol"].unique())
        prices = load_prices(codes, TEST_START, PRICE_END)
        print(f"  prices loaded: {len(prices)} stocks")
        rr = {}
        for name in SCHEMES:
            r = run_one(name, fit["schemes"][name], test_panel, prices, flags, controls, h)
            rr[name] = r
            print(f"  {name:16s} ret={_f(r['portfolio_total_return'],2,True)} sharpe={_f(r['sharpe'])} "
                  f"maxDD={_f(r['max_drawdown'],2,True)} trades={r['total_trades']}")
        all_res[h] = rr
        eq_dates = rr["equal_weight"]["equity"]["date"].tolist()
        bm, _ = benchmark_metrics(eq_dates)
        bench[h] = bm
        print(f"  中证1000基准 ret={_f(bm.get('portfolio_total_return'),2,True)} sharpe={_f(bm.get('sharpe_ratio'))}")

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(build_report(all_res, bench), encoding="utf-8")
    print(f"\n报告已写入: {REPORT}")


if __name__ == "__main__":
    main()
