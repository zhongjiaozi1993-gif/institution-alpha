"""Phase 6B 组合构造引擎与口径工具。

提供：
  1. 三种组合机制（同信号/成本/交易限制/评估窗口）：
       - fixed_holding_fill_slots：复用 signal_backtester（固定持有期 + 槽位补仓）。
       - periodic_rebalance_topN：每 h 个交易日按当日 Top-N 目标权重重建组合。
       - daily_rebalance_topN：每日按 Top-N 目标权重调仓（研究排序兑现，非生产方案）。
     统一 T+1 open 执行、买卖各扣一次成本+滑点、tradable_flags 约束、逐日 mark-to-market。
  2. 诊断：候选→成交转化率、实际持仓与当日 Top-N 重合度、平均持仓排名、信号→成交延迟、
     单票收益贡献（逐日持仓 mark-to-market 归因）。

远期 open→open 口径（按每股自有日历）见 src/backtest/open_to_open.forward_open_return。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.backtest.signal_backtester import SignalBacktester, BacktestConfig


# ======================================================================
# 组合配置
# ======================================================================
@dataclass
class PortfolioConfig:
    top_n: int = 30
    holding_days: int = 5              # fill_slots 持有期 / periodic 再平衡间隔
    cost_bps: float = 20
    slippage_bps: float = 10
    initial_capital: float = 1_000_000.0
    name: str = "port"


def _scores_to_buys(scores: pd.DataFrame, top_n: int) -> pd.DataFrame:
    buys = []
    for date, g in scores.dropna(subset=["final_score"]).groupby("trade_date"):
        picked = g.sort_values("final_score", ascending=False).head(top_n)
        ds = pd.Timestamp(date).strftime("%Y-%m-%d")
        for code in picked["symbol"]:
            buys.append({"stock_code": str(code).zfill(6), "signal_date": ds})
    return pd.DataFrame(buys)


def _daily_topn_and_rank(scores: pd.DataFrame, top_n: int):
    """→ (day_topn[date_str]=[Top-N codes], day_rank[date_str]={code:rank(1=best)})。"""
    day_topn, day_rank = {}, {}
    for d, g in scores.dropna(subset=["final_score"]).groupby("trade_date"):
        ds = pd.Timestamp(d).strftime("%Y-%m-%d")
        gg = g.sort_values("final_score", ascending=False)
        codes = [str(s).zfill(6) for s in gg["symbol"]]
        day_topn[ds] = codes[:top_n]
        day_rank[ds] = {c: r for r, c in enumerate(codes, 1)}
    return day_topn, day_rank


def _px_maps(prices: dict):
    O, C = {}, {}
    for s, pdf in prices.items():
        O[s] = dict(zip(pdf["date_str"], pdf["open_yuan"].astype(float)))
        C[s] = dict(zip(pdf["date_str"], pdf["close_yuan"].astype(float)))
    return O, C


# ======================================================================
# 目标权重再平衡引擎（periodic / daily）
# ======================================================================
def run_rebalance(scores: pd.DataFrame, prices: dict, flags: pd.DataFrame | None,
                  cfg: PortfolioConfig, rebalance_every: int) -> dict:
    """每 rebalance_every 个信号日按 Top-N 等权目标重建组合；T+1 open 执行，差额调仓。"""
    top_n = cfg.top_n
    cost = cfg.cost_bps / 1e4
    slip = cfg.slippage_bps / 1e4
    day_topn, day_rank = _daily_topn_and_rank(scores, top_n)
    O, C = _px_maps(prices)
    nb, ns, susp, nt = SignalBacktester._build_flag_lookups(flags)
    all_dates = sorted({d for pdf in prices.values() for d in pdf["date_str"].values})
    sig_sorted = [pd.Timestamp(x).strftime("%Y-%m-%d") for x in sorted(scores["trade_date"].unique())]
    reb_days = set(sig_sorted[::max(rebalance_every, 1)])

    cash = cfg.initial_capital
    holdings: dict[str, float] = {}
    last_close: dict[str, float] = {}
    trades, equity, holdings_daily = [], [], []
    pending = None
    cand_slots, exec_fills, traded_notional = 0, 0, 0.0
    latency = []

    for today in all_dates:
        # ---- 执行昨日目标（T+1 open）----
        if pending is not None:
            tgt = pending["stocks"]; sigd = pending["signal_date"]
            tgt_set = set(tgt)
            nav = cash + sum(holdings[s] * O[s].get(today, last_close.get(s, 0.0))
                             for s in holdings if (O[s].get(today) or last_close.get(s)))
            tgt_val = nav / top_n if top_n else 0.0
            # 卖出：非目标或超配
            for s in list(holdings):
                px = O[s].get(today)
                if px is None or (s, today) in ns or (s, today) in susp:
                    continue
                cur = holdings[s] * px
                desired = tgt_val if s in tgt_set else 0.0
                if cur > desired + 1e-6:
                    sh_sell = min(holdings[s], (cur - desired) / px)
                    exec_p = px * (1 - slip); proceeds = sh_sell * exec_p; fee = proceeds * cost
                    cash += proceeds - fee; traded_notional += sh_sell * px
                    holdings[s] -= sh_sell
                    if holdings[s] <= 1e-9:
                        del holdings[s]
                    trades.append({"side": "sell", "stock": s, "signal_date": sigd,
                                   "exec_date": today, "value": round(sh_sell * px, 2)})
            # 买入：目标欠配（按排名顺序）
            for s in tgt:
                cand_slots += 1
                px = O[s].get(today)
                if px is None or (s, today) in nb or (s, today) in nt or (s, today) in susp:
                    continue
                cur = holdings.get(s, 0.0) * px
                if tgt_val > cur + 1e-6 and cash > 1e-6:
                    buy_val = min(tgt_val - cur, cash)
                    exec_p = px * (1 + slip); fee = buy_val * cost; sh_buy = (buy_val - fee) / exec_p
                    cash -= buy_val; traded_notional += buy_val
                    holdings[s] = holdings.get(s, 0.0) + sh_buy
                    exec_fills += 1
                    latency.append(all_dates.index(today) - all_dates.index(sigd))
                    trades.append({"side": "buy", "stock": s, "signal_date": sigd,
                                   "exec_date": today, "value": round(buy_val, 2)})
            pending = None

        # ---- 今日信号 → 明日目标 ----
        if today in reb_days:
            pending = {"signal_date": today, "stocks": day_topn.get(today, [])}

        # ---- mark-to-market ----
        pv = 0.0
        for s, sh in holdings.items():
            px = C[s].get(today, last_close.get(s))
            if px:
                last_close[s] = px; pv += sh * px
        equity.append({"date": today, "nav": round(cash + pv, 4), "cash": round(cash, 4),
                       "n_positions": len(holdings)})
        holdings_daily.append({"date": today, "holdings": dict(holdings)})

    diag = _diagnostics(holdings_daily, day_topn, day_rank, top_n, cand_slots, exec_fills, latency)
    stock_contrib = _stock_contribution(holdings_daily, C, cfg.initial_capital)
    return {"equity": pd.DataFrame(equity), "trades": pd.DataFrame(trades),
            "holdings_daily": holdings_daily, "diagnostics": diag,
            "stock_contrib": stock_contrib, "traded_notional": traded_notional}


# ======================================================================
# fill_slots：复用 signal_backtester
# ======================================================================
def run_fill_slots(scores: pd.DataFrame, prices: dict, flags: pd.DataFrame | None,
                   cfg: PortfolioConfig) -> dict:
    buys = _scores_to_buys(scores, cfg.top_n)
    bcfg = BacktestConfig(holding_days=cfg.holding_days, max_positions=cfg.top_n,
                          cost_bps=cfg.cost_bps, slippage_bps=cfg.slippage_bps,
                          initial_capital=cfg.initial_capital, name=cfg.name)
    res = SignalBacktester(bcfg).run(buys, prices, tradable_flags=flags)
    trades, equity = res["trades"], res["equity_curve"]
    _, C = _px_maps(prices)
    day_topn, day_rank = _daily_topn_and_rank(scores, cfg.top_n)
    # 重建逐日持仓快照（entry_date≤today<exit_date 视为持有）
    all_dates = sorted({d for pdf in prices.values() for d in pdf["date_str"].values})
    holdings_daily = _reconstruct_holdings(trades, all_dates)
    latency = ([all_dates.index(e) - all_dates.index(s)
                for s, e in zip(trades["signal_date"], trades["entry_date"])
                if s in all_dates and e in all_dates] if not trades.empty else [])
    cand_slots = int(len(buys)); exec_fills = int(len(trades))
    diag = _diagnostics(holdings_daily, day_topn, day_rank, cfg.top_n, cand_slots, exec_fills, latency)
    stock_contrib = _stock_contribution(holdings_daily, C, cfg.initial_capital)
    return {"equity": equity, "trades": trades, "holdings_daily": holdings_daily,
            "diagnostics": diag, "stock_contrib": stock_contrib,
            "traded_notional": float(cfg.initial_capital / cfg.top_n * (len(trades) * 2))}


def _reconstruct_holdings(trades: pd.DataFrame, all_dates: list[str]) -> list[dict]:
    """由 fill_slots trades（含 entry_date/exit_date/shares）重建逐日持仓股票集合（份额=1 占位）。"""
    idx = {d: i for i, d in enumerate(all_dates)}
    held_by_day = {d: {} for d in all_dates}
    if trades is None or trades.empty:
        return [{"date": d, "holdings": {}} for d in all_dates]
    for _, t in trades.iterrows():
        e, x = t["entry_date"], t["exit_date"]
        if e not in idx:
            continue
        xi = idx.get(x, len(all_dates) - 1)
        for i in range(idx[e], xi):                # 持有区间 [entry, exit)
            held_by_day[all_dates[i]][str(t["stock"]).zfill(6)] = float(t["shares"])
    return [{"date": d, "holdings": held_by_day[d]} for d in all_dates]


# ======================================================================
# 诊断与归因
# ======================================================================
def _diagnostics(holdings_daily, day_topn, day_rank, top_n, cand_slots, exec_fills, latency) -> dict:
    overlaps, avg_ranks = [], []
    for snap in holdings_daily:
        d = snap["date"]; held = set(snap["holdings"])
        if d in day_topn and held:
            overlaps.append(len(held & set(day_topn[d])) / max(top_n, 1))
        if d in day_rank and held:
            ranks = [day_rank[d][s] for s in held if s in day_rank[d]]
            if ranks:
                avg_ranks.append(float(np.mean(ranks)))
    return {
        "candidate_slots": int(cand_slots), "executed_fills": int(exec_fills),
        "execution_rate": round(exec_fills / cand_slots, 3) if cand_slots else np.nan,
        "mean_topn_overlap": round(float(np.mean(overlaps)), 3) if overlaps else np.nan,
        "mean_holding_rank": round(float(np.mean(avg_ranks)), 1) if avg_ranks else np.nan,
        "mean_latency_days": round(float(np.mean(latency)), 2) if latency else np.nan,
    }


def _stock_contribution(holdings_daily, C: dict, init_cap: float) -> pd.Series:
    """逐日持仓 mark-to-market 归因：contrib[s] = Σ shares_prev·(close_t − close_{t-1}) / init ×100(%)。"""
    contrib: dict[str, float] = {}
    for prev, cur in zip(holdings_daily[:-1], holdings_daily[1:]):
        d0, d1 = prev["date"], cur["date"]
        for s, sh in prev["holdings"].items():
            c0 = C.get(s, {}).get(d0); c1 = C.get(s, {}).get(d1)
            if c0 is not None and c1 is not None:
                contrib[s] = contrib.get(s, 0.0) + sh * (c1 - c0) / init_cap * 100
    return pd.Series(contrib).sort_values(ascending=False) if contrib else pd.Series(dtype=float)


# ======================================================================
# 统一入口
# ======================================================================
def run_portfolio(mode: str, scores: pd.DataFrame, prices: dict,
                  flags: pd.DataFrame | None, cfg: PortfolioConfig) -> dict:
    if mode == "fixed_holding_fill_slots":
        return run_fill_slots(scores, prices, flags, cfg)
    if mode == "periodic_rebalance_topN":
        return run_rebalance(scores, prices, flags, cfg, rebalance_every=cfg.holding_days)
    if mode == "daily_rebalance_topN":
        return run_rebalance(scores, prices, flags, cfg, rebalance_every=1)
    raise ValueError(f"unknown mode: {mode}")
