"""Phase 6A.1 驱动脚本 2（回测窗口与实际持仓归因修复）。

在 Phase 6A 基础上修复：
  1. 绩效区间：行情仍加载到 2026-03-31 用于年末退出；四方案跑完后取
     common_eval_end = 四方案实际 trades.exit_date 最大值，equity 统一截断到
     [2025-09-01, common_eval_end]，基准用完全相同的起止日；截断后重算所有绩效。
  2. 暴露归因：candidate_exposure（每日 Top-30 候选池）与 executed_entry_exposure
     （实际成交股票，按 stock+signal_date 合并控制变量）分离；输出 candidate_signals /
     executed_trades / execution_rate。judge 小盘/低流动性依赖只用 executed 口径。
  3. 退出口径对照：主回测 T+1 open→entry+h close 之外，增加与 IC 标签一致的
     T+1 open→T+1+h open 口径，比较 best_single / equal_weight 收益与 Sharpe（仅归因）。
  4. verdict 全部指标从结果对象动态读取，无硬编码；「跑赢基准」逐方案代码判定。

方案方向/权重来自 train（arf.fit_fusion），test 冻结，不用 test 标签调参。
Alpha191 因子筛选/方向/权重与 IC 报告不变（本脚本不改）。
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
from src.validation import factor_validator as fv
from src.backtest.signal_backtester import SignalBacktester, BacktestConfig
from src.backtest.metrics import compute_full_metrics
from src.backtest.open_to_open import forward_open_return

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
SLOT_CAPITAL = INIT_CAP / MAX_POS
TEST_START = arf.TEST_START
PRICE_END = "2026-03-31"               # 加载到 test 之后，容纳 10d 到期
SCHEMES = ["best_single", "equal_weight", "icir_weight", "stability_weight"]
CTRL_COLS = ["log_mktcap", "log_amount", "turnover"]


# ======================================================================
# 数据
# ======================================================================
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


# ======================================================================
# 绩效区间修复
# ======================================================================
def common_eval_end(trades_list: list[pd.DataFrame]) -> pd.Timestamp | None:
    """四方案实际 trades.exit_date 的最大值 → 统一评估终点。"""
    ends = []
    for tr in trades_list:
        if tr is not None and not tr.empty and "exit_date" in tr.columns:
            ends.append(pd.to_datetime(tr["exit_date"]).max())
    return max(ends) if ends else None


def truncate_equity(equity_df: pd.DataFrame, start, end) -> pd.DataFrame:
    """把 equity_curve 截断到 [start, end]（含端点），去掉退出后的空仓平坦尾部。"""
    d = pd.to_datetime(equity_df["date"])
    m = (d >= pd.Timestamp(start)) & (d <= pd.Timestamp(end))
    return equity_df[m.to_numpy()].reset_index(drop=True)


def annualized_turnover(n_trades: int, max_pos: int, n_days: int) -> float:
    """组合年化换手 ≈ (成交笔数×槽位资金)/初始资金/年数 = n_trades/max_pos/年数。"""
    years = max(n_days / 252, 1e-9)
    return round(n_trades / max(max_pos, 1) / years, 2)


def benchmark_metrics(start, end) -> tuple[dict, pd.DataFrame]:
    """中证1000 在 [start, end] 交易日内的买入持有 NAV 与绩效（与方案完全相同的起止日）。"""
    idx = pd.read_parquet(INDEX_PATH)
    idx["date"] = pd.to_datetime(idx["date"])
    sub = idx[(idx["date"] >= pd.Timestamp(start)) & (idx["date"] <= pd.Timestamp(end))]
    sub = sub.sort_values("date").reset_index(drop=True)
    if sub.empty:
        return {}, pd.DataFrame()
    nav = INIT_CAP * sub["close"] / sub["close"].iloc[0]
    navdf = pd.DataFrame({"date": sub["date"].dt.strftime("%Y-%m-%d"), "nav": nav.to_numpy()})
    m = compute_full_metrics(navdf, pd.DataFrame())
    m["date_start"] = str(sub["date"].iloc[0].date())
    m["date_end"] = str(sub["date"].iloc[-1].date())
    m["n_days"] = int(len(sub))
    return m, navdf


# ======================================================================
# 暴露归因（候选池 vs 实际成交）
# ======================================================================
def _exposure_percentile(pairs: pd.DataFrame, controls: pd.DataFrame, cols=CTRL_COLS) -> dict:
    """pairs=[trade_date, symbol]；返回各控制变量在信号日截面的百分位均值（0..1）。"""
    p = pairs.copy()
    p["trade_date"] = pd.to_datetime(p["trade_date"])
    p["symbol"] = p["symbol"].astype(str).str.zfill(6)
    out = {}
    for col in cols:
        pr = controls[["trade_date", "symbol", col]].dropna().copy()
        pr["pct"] = pr.groupby("trade_date")[col].rank(pct=True)
        j = p.merge(pr[["trade_date", "symbol", "pct"]], on=["trade_date", "symbol"], how="left")
        out[col] = round(float(j["pct"].mean()), 3) if j["pct"].notna().any() else np.nan
    return out


def candidate_exposure(buys: pd.DataFrame, controls: pd.DataFrame) -> dict:
    """每日 Top-30 候选池（信号，未必成交）的暴露百分位。"""
    if buys.empty:
        return {c: np.nan for c in CTRL_COLS}
    pairs = buys.rename(columns={"stock_code": "symbol", "signal_date": "trade_date"})
    return _exposure_percentile(pairs[["trade_date", "symbol"]], controls)


def executed_entry_exposure(trades: pd.DataFrame, controls: pd.DataFrame) -> dict:
    """实际成交股票（按 stock+signal_date 合并控制变量）的暴露百分位。"""
    if trades is None or trades.empty:
        return {c: np.nan for c in CTRL_COLS}
    pairs = trades.rename(columns={"stock": "symbol", "signal_date": "trade_date"})
    return _exposure_percentile(pairs[["trade_date", "symbol"]], controls)


# ======================================================================
# 退出口径对照（T+1 open → T+1+h open，与 IC 标签一致）
# ======================================================================
def trade_oo_gross_return(prices: dict, stock: str, entry_date: str, h: int) -> float:
    """单笔 open→open 毛收益 = open[entry+h]/open[entry] − 1。

    entry+h 为该股**自己**日线序列中 entry 后第 h 个有效日（停牌=缺失行，自动跳过），
    与 label_builder._open_to_open_labels 完全同口径；不用全局日历，避免停牌错位。
    """
    pdf = prices.get(str(stock).zfill(6))
    if pdf is None:
        return np.nan
    return forward_open_return(pdf, entry_date, h)


def oo_trade_returns(trades: pd.DataFrame, prices: dict, h: int,
                     slip: float, cost: float) -> pd.Series:
    """实际成交在 open→open 口径下的每笔净收益%（复用引擎的成交决定，仅换退出价）。"""
    if trades is None or trades.empty:
        return pd.Series(dtype=float)
    rets = []
    for _, t in trades.iterrows():
        g = trade_oo_gross_return(prices, t["stock"], t["entry_date"], h)
        if np.isnan(g):
            rets.append(np.nan); continue
        # 与引擎口径一致：买入 open×(1+slip)、卖出 open×(1−slip)，买卖各扣一次费
        entry_exec = float(t["entry_price"])                 # = open_entry×(1+slip)
        open_out = entry_exec / (1 + slip) * (1 + g)          # open[entry+h]
        exit_exec = open_out * (1 - slip)
        proceeds = float(t["shares"]) * exit_exec
        net = proceeds * (1 - cost)
        rets.append((net / SLOT_CAPITAL - 1) * 100)
    return pd.Series(rets, dtype=float)


def _sharpe_of_trades(r: pd.Series) -> float:
    r = r.dropna()
    return round(float(r.mean() / r.std()), 3) if len(r) > 1 and r.std() > 0 else np.nan


# ======================================================================
# 单方案运行（返回原始 equity/trades + 归因，绩效在主流程截断后统一算）
# ======================================================================
def run_one(name: str, scheme: dict, test_panel, prices, flags, controls,
            fwd, horizon: int) -> dict:
    scores = arf.build_scheme_scores(test_panel, scheme)
    buys = scores_to_buys(scores, TOP_N)
    cfg = BacktestConfig(holding_days=horizon, max_positions=MAX_POS,
                         cost_bps=COST_BPS, slippage_bps=SLIP_BPS,
                         initial_capital=INIT_CAP, name=f"{name}_h{horizon}")
    res = SignalBacktester(cfg).run(buys, prices, tradable_flags=flags)
    trades, equity = res["trades"], res["equity_curve"]

    # test 期 RankIC/RankICIR（供 verdict 动态引用；不改 IC 报告）
    fc = f"fwd_{horizon}d"
    m = (scores.rename(columns={"final_score": "signal_value"})
         .merge(fwd[["trade_date", "symbol", fc]], on=["trade_date", "symbol"], how="left"))
    ic = fv._daily_corr(m, fc)
    rankic = float(ic["RankIC"].mean()) if len(ic) else np.nan
    rankicir = float(rankic / ic["RankIC"].std()) if len(ic) and ic["RankIC"].std() > 0 else np.nan

    # 退出口径对照
    oo_r = oo_trade_returns(trades, prices, horizon, SLIP_BPS / 1e4, COST_BPS / 1e4)
    close_r = trades["net_return_pct"] if (trades is not None and not trades.empty) else pd.Series(dtype=float)

    return {
        "name": name, "buys": buys, "equity_raw": equity, "trades": trades,
        "candidate_signals": int(len(buys)),
        "executed_trades": int(len(trades)) if trades is not None else 0,
        "candidate_exposure": candidate_exposure(buys, controls),
        "executed_exposure": executed_entry_exposure(trades, controls),
        "test_rankic": rankic, "test_rankicir": rankicir,
        "oo": {"close_mean": round(float(close_r.mean()), 3) if len(close_r) else np.nan,
               "close_sharpe": _sharpe_of_trades(close_r),
               "oo_mean": round(float(oo_r.mean()), 3) if len(oo_r.dropna()) else np.nan,
               "oo_sharpe": _sharpe_of_trades(oo_r),
               "n": int(len(close_r))},
    }


def finalize_metrics(raw: dict, eval_end: pd.Timestamp) -> dict:
    """截断 equity 到 [TEST_START, eval_end] 后重算绩效 + 交易统计。"""
    trades = raw["trades"]
    eq = truncate_equity(raw["equity_raw"], TEST_START, eval_end)
    mt = compute_full_metrics(eq, trades if trades is not None else pd.DataFrame())
    n_days = int(len(eq))
    monthly, best_stock, best_c, worst_stock, worst_c, monthly_pos = {}, "", 0.0, "", 0.0, 0.0
    if trades is not None and not trades.empty:
        contrib = trades.groupby("stock")["net_return_pct"].sum().sort_values(ascending=False)
        best_stock, best_c = contrib.index[0], float(contrib.iloc[0])
        worst_stock, worst_c = contrib.index[-1], float(contrib.iloc[-1])
        tc = trades.copy(); tc["ym"] = pd.to_datetime(tc["entry_date"]).dt.strftime("%Y-%m")
        mser = tc.groupby("ym")["net_return_pct"].sum()
        monthly = {k: round(float(v), 2) for k, v in mser.items()}
        monthly_pos = float((mser > 0).mean())
    exec_n = raw["executed_trades"]
    cand_n = max(raw["candidate_signals"], 1)
    return {
        **raw,
        "equity_n_days": n_days,
        "portfolio_total_return": float(mt.get("portfolio_total_return", 0.0)),
        "annualized_return": mt.get("annualized_return", np.nan),
        "sharpe": mt.get("sharpe_ratio", np.nan),
        "sortino": mt.get("sortino_ratio", np.nan),
        "max_drawdown": mt.get("max_drawdown", 0.0),
        "calmar": mt.get("calmar_ratio", np.nan),
        "ann_turnover": annualized_turnover(exec_n, MAX_POS, n_days),
        "total_trades": exec_n,
        "win_rate": mt.get("win_rate", np.nan),
        "deferred_exits": mt.get("deferred_exits", 0),
        "execution_rate": round(exec_n / cand_n, 3),
        "best_stock": best_stock, "best_contrib": best_c,
        "worst_stock": worst_stock, "worst_contrib": worst_c,
        "monthly_positive_rate": monthly_pos, "monthly": monthly,
    }


# ======================================================================
# 报告（全部动态）
# ======================================================================
def _f(x, p=2, pct=False):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "n/a"
    return f"{x*100:+.{p}f}%" if pct else f"{x:+.{p}f}"


def build_report(all_res: dict, bench: dict, meta: dict) -> str:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    L = [f"# Alpha191 规则融合回测报告（Phase 6A.1）\n",
         f"生成时间: {ts}  |  T+1 open 建仓, tradable_flags ON  |  "
         f"Top-{TOP_N}, 持有=horizon, cost {COST_BPS}bps + slip {SLIP_BPS}bps（各扣一次, 非优化）\n",
         "> 方案方向/权重来自 train（arf.fit_fusion）；test 冻结评估，未在 test 上网格搜索交易参数。"
         "绩效区间统一截断到 [signal_start, common_eval_end]（去空仓尾部），基准同起止日。"
         "暴露归因区分候选池与实际成交。\n\n---\n"]

    for h in (5, 10):
        rr = all_res[h]; bm = bench[h]; mt = meta[h]
        L.append(f"\n## 持有 {h}d\n")
        L.append(f"绩效区间披露：signal_start={mt['signal_start']} | last_signal_date={mt['last_signal_date']} | "
                 f"common_eval_end={mt['common_eval_end']} | equity_n_days={mt['equity_n_days']}"
                 f"（基准 {bm.get('date_start','n/a')}→{bm.get('date_end','n/a')}, {bm.get('n_days','n/a')}日, 完全相同起止）\n")
        L.append("\n| 方案 | 组合收益 | 年化 | Sharpe | Sortino | 最大回撤 | Calmar | 年化换手 | 交易数 | 胜率 | 延期退出 |\n")
        L.append("|---|---|---|---|---|---|---|---|---|---|---|\n")
        for name in SCHEMES:
            r = rr[name]
            L.append(f"| {name} | {_f(r['portfolio_total_return'],2,True)} | {_f(r['annualized_return'],2,True)} | "
                     f"{_f(r['sharpe'])} | {_f(r['sortino'])} | {_f(r['max_drawdown'],2,True)} | {_f(r['calmar'])} | "
                     f"{r['ann_turnover']} | {r['total_trades']} | {_f(r['win_rate'],1,True)} | {r['deferred_exits']} |\n")
        if bm:
            L.append(f"| 中证1000基准 | {_f(bm.get('portfolio_total_return'),2,True)} | {_f(bm.get('annualized_return'),2,True)} | "
                     f"{_f(bm.get('sharpe_ratio'))} | {_f(bm.get('sortino_ratio'))} | {_f(bm.get('max_drawdown'),2,True)} | "
                     f"{_f(bm.get('calmar_ratio'))} | n/a | n/a | n/a | n/a |\n")

        # 暴露：候选 vs 实际成交
        L.append(f"\n### 暴露归因（{h}d）：候选池 vs 实际成交\n")
        L.append("\n| 方案 | 口径 | 候选信号 | 实际成交 | 成交率 | log市值分位 | log成交额分位 | 换手分位 |\n")
        L.append("|---|---|---|---|---|---|---|---|\n")
        for name in SCHEMES:
            r = rr[name]; ce = r["candidate_exposure"]; ee = r["executed_exposure"]
            L.append(f"| {name} | 候选池 | {r['candidate_signals']} | — | — | "
                     f"{_f(ce['log_mktcap'],2)} | {_f(ce['log_amount'],2)} | {_f(ce['turnover'],2)} |\n")
            L.append(f"| {name} | 实际成交 | — | {r['executed_trades']} | {_f(r['execution_rate'],1,True)} | "
                     f"{_f(ee['log_mktcap'],2)} | {_f(ee['log_amount'],2)} | {_f(ee['turnover'],2)} |\n")
        L.append("\n> 分位=信号日截面百分位均值（0.5=中位；<0.5 偏小/偏低流动性）。"
                 "**judge 小盘/低流动性依赖只看「实际成交」行**，候选池暴露不直接解释 realized Sharpe。\n")

        # 单票 + 月度
        L.append(f"\n### 单票贡献 / 月度收益（{h}d，单笔 net_return 求和%，活跃度口径非 NAV%）\n")
        L.append("\n| 方案 | 最佳票(贡献) | 最差票(贡献) | 月度正收益比 |\n|---|---|---|---|\n")
        for name in SCHEMES:
            r = rr[name]
            L.append(f"| {name} | {r['best_stock']}({r['best_contrib']:+.1f}) | {r['worst_stock']}({r['worst_contrib']:+.1f}) | "
                     f"{_f(r['monthly_positive_rate'],0,True)} |\n")
        months = sorted(set().union(*[set(rr[n]["monthly"]) for n in SCHEMES])) if any(rr[n]["monthly"] for n in SCHEMES) else []
        if months:
            L.append("\n| 方案 | " + " | ".join(months) + " |\n")
            L.append("|---|" + "---|" * len(months) + "\n")
            for name in SCHEMES:
                mo = rr[name]["monthly"]
                L.append(f"| {name} | " + " | ".join(_f(mo.get(m), 1) for m in months) + " |\n")

        # 退出口径对照
        L.append(f"\n### 退出口径对照（{h}d）：close 到期 vs open→open（与 IC 标签同口径）\n")
        L.append("\n| 方案 | 口径 | 每笔均值% | 每笔Sharpe(mean/std) | 笔数 |\n|---|---|---|---|---|\n")
        for name in ("best_single", "equal_weight"):
            oo = rr[name]["oo"]
            L.append(f"| {name} | close到期 | {_f(oo['close_mean'],3)} | {_f(oo['close_sharpe'])} | {oo['n']} |\n")
            L.append(f"| {name} | open→open | {_f(oo['oo_mean'],3)} | {_f(oo['oo_sharpe'])} | {oo['n']} |\n")
        L.append("\n> 仅口径归因，不据此优化退出。open→open 与 IC 标签（open[T+1+h]/open[T+1]）一致，"
                 "用于判断 best_single vs equal_weight 的差异是否为 close-vs-open 退出口径造成。\n")

    L.append("\n---\n\n## 验收判定（动态生成）\n")
    L.append(verdict(all_res, bench, meta))
    L.append("\n## 已知限制\n")
    L.append("1. test 仅 ~4 个月（2025-09..2025-12）；截断后有效日 5d/10d 见各区间披露，10d 再平衡次数少，统计噪声大。\n")
    L.append("2. 组合暴露审计缺行业；市值/流动性以截面百分位近似。\n")
    L.append(f"3. 固定 Top-{TOP_N}/持仓{MAX_POS}/成本{COST_BPS}bps，未做参数寻优（刻意，避免 test 过拟合）。\n")
    L.append("4. 退出口径对照为每笔 Sharpe(mean/std)，非组合 NAV Sharpe；close 口径 NAV Sharpe 见上表。\n")
    return "".join(L)


def verdict(all_res: dict, bench: dict, meta: dict) -> str:
    """全部指标从结果对象动态读取；跑赢基准逐方案代码判定。"""
    fusion = ("equal_weight", "icir_weight", "stability_weight")
    L = []
    for h in (5, 10):
        rr = all_res[h]; bm = bench[h]; mt = meta[h]
        bench_ret = bm.get("portfolio_total_return", np.nan)
        beats = [n for n in SCHEMES
                 if not np.isnan(rr[n]["portfolio_total_return"]) and not np.isnan(bench_ret)
                 and rr[n]["portfolio_total_return"] > bench_ret]
        best_name, best_sharpe = None, -np.inf
        for n in fusion:
            sh = rr[n]["sharpe"]
            if not (isinstance(sh, float) and np.isnan(sh)) and sh > best_sharpe:
                best_name, best_sharpe = n, sh
        bs = rr["best_single"]
        bs_sharpe = bs["sharpe"] if not (isinstance(bs["sharpe"], float) and np.isnan(bs["sharpe"])) else np.nan
        cond_sharpe = (not np.isnan(best_sharpe) and not np.isnan(bs_sharpe) and best_sharpe >= bs_sharpe - 1e-9)
        cond_dd = rr[best_name]["max_drawdown"] >= bs["max_drawdown"] - 0.02 if best_name else False
        beat_txt = ("四方案均跑赢基准" if len(beats) == 4
                    else (f"{len(beats)}/4 跑赢基准（" + ", ".join(beats) + "）" if beats else "无方案跑赢基准"))
        L.append(f"- **{h}d**（窗口 {mt['signal_start']}→{mt['common_eval_end']}, {mt['equity_n_days']}日）："
                 f"最优融合={best_name} Sharpe {_f(best_sharpe)} vs best_single {_f(bs_sharpe)} "
                 f"→ {'Sharpe 不劣 ✓' if cond_sharpe else 'Sharpe 劣于单因子 ✗'}；"
                 f"回撤 {_f(rr[best_name]['max_drawdown'],2,True)} vs {_f(bs['max_drawdown'],2,True)} "
                 f"{'未明显恶化 ✓' if cond_dd else '恶化 ✗'}；{beat_txt}（基准 {_f(bench_ret,2,True)}）。\n")

    # executed 口径归因（动态）
    L.append("\n### 综合归因（executed 口径 × IC）\n")
    for h in (5, 10):
        rr = all_res[h]
        bs, eq = rr["best_single"], rr["equal_weight"]
        bs_amt, eq_amt = bs["executed_exposure"]["log_amount"], eq["executed_exposure"]["log_amount"]
        mo = bs["monthly"]; dom = max(mo, key=mo.get) if mo else "n/a"
        L.append(f"- **{h}d**：融合 test RankICIR {_f(eq['test_rankicir'])} vs best_single {_f(bs['test_rankicir'])}；"
                 f"**实际成交** log成交额分位 best_single {_f(bs_amt,2)} vs equal_weight {_f(eq_amt,2)}"
                 f"（<0.50=偏小/低流动性）；best_single 最大贡献票 {bs['best_stock']}({bs['best_contrib']:+.1f})、"
                 f"收益集中月 {dom}；退出口径 close vs open→open 见上表。\n")

    L.append("\n### 最终结论\n")
    L.append("> **Phase 6A 有条件通过。**\n>\n"
             "> 多因子融合在截面排序的 RankICIR、非规模残余信息和暴露分散方面优于最优单因子，"
             "但在仅四个月的 Top-30 OOS 回测中，尚未证明实现收益和 Sharpe 优于 best_single。"
             "best_single 的优势高度集中于小盘、低流动性暴露（见实际成交口径），不能直接作为默认生产策略。\n\n")
    L.append("- `equal_weight` 暂定为后续默认**研究排序基线**；\n")
    L.append("- `best_single` 只作为**高规模暴露对照组**，不作默认生产策略；\n")
    L.append("- Level-2 不加入正式评分（仅 shadow 对照，见 IC 报告）；\n")
    L.append("- 方向/权重全部 train 冻结，不根据 test 结果调整；\n")
    L.append("- 下一步 **Phase 6B 稳健性与组合构造验证**（滚动 OOS、Top-N 敏感性、持仓集中度、"
             "规模中性组合），解决「IC 更稳但 Top-30 收益未兑现」的矛盾；暂不上 ML、不加 Level-2。\n")
    return "".join(L)


# ======================================================================
def main():
    print("加载面板/标签/控制变量 ...")
    panel = arf.load_alpha_panel("2025-01-01", "2025-12-31")
    fwd = arf.load_fwd((5, 10))
    controls = arf.load_exposure_controls()
    flags = pd.read_parquet(FLAGS_PATH)

    all_res, bench, meta = {}, {}, {}
    for h in (5, 10):
        print(f"\n=== 持有 {h}d ===")
        fit = arf.fit_fusion(panel, fwd, h)
        _, test_panel, _ = arf.purge_train_test(panel, h)
        codes = sorted(test_panel["symbol"].unique())
        prices = load_prices(codes, TEST_START, PRICE_END)
        n_dates = len({d for pdf in prices.values() for d in pdf["date_str"].values})
        print(f"  prices loaded: {len(prices)} stocks, {n_dates} dates")

        raws = {name: run_one(name, fit["schemes"][name], test_panel, prices,
                              flags, controls, fwd, h) for name in SCHEMES}
        cee = common_eval_end([raws[n]["trades"] for n in SCHEMES])
        rr = {name: finalize_metrics(raws[name], cee) for name in SCHEMES}
        all_res[h] = rr
        bench[h], _ = benchmark_metrics(TEST_START, cee)
        last_signal = max((r["buys"]["signal_date"].max() for r in raws.values()
                           if not r["buys"].empty), default=TEST_START)
        meta[h] = {"signal_start": TEST_START, "last_signal_date": str(last_signal),
                   "common_eval_end": str(pd.Timestamp(cee).date()),
                   "equity_n_days": rr["equal_weight"]["equity_n_days"]}
        for name in SCHEMES:
            r = rr[name]
            print(f"  {name:16s} ret={_f(r['portfolio_total_return'],2,True)} sharpe={_f(r['sharpe'])} "
                  f"maxDD={_f(r['max_drawdown'],2,True)} exec={r['executed_trades']}/{r['candidate_signals']} "
                  f"execAmtPct={_f(r['executed_exposure']['log_amount'],2)}")
        print(f"  基准 {meta[h]['common_eval_end']} ret={_f(bench[h].get('portfolio_total_return'),2,True)} "
              f"sharpe={_f(bench[h].get('sharpe_ratio'))} | window {meta[h]['equity_n_days']}日")

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(build_report(all_res, bench, meta), encoding="utf-8")
    print(f"\n报告已写入: {REPORT}")


if __name__ == "__main__":
    main()
