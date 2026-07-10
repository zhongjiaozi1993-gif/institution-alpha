"""
Unified signal backtest engine (Phase 4.6 rewrite).

Strategy mode: fixed_holding_fill_slots
  - 固定持有期 N 日；共 max_positions 个槽位；有空槽且现金足够才补仓。
  - 尚未支持 daily_rebalance_topN（后续）。

Trading timeline (无未来函数)
  - 信号在 T 日收盘后产生，仅用 T 及之前信息。
  - T 日只生成 pending_order，**不进入 open_positions**。
  - T+1 当天开盘后才真正建仓（entry_date = T+1，入场价 = T+1 开盘 × (1+买滑点)）。
  - mark-to-market 只统计 entry_date <= today 的仓位（信号日 T 的 NAV 不含未来仓位）。
  - 到期 = entry_date + holding_days 日，按当日收盘卖出。

Portfolio accounting（份额制，成本不重复扣）
  - 每个槽位固定投入 slot_capital = initial_capital / max_positions（gross 现金）。
  - 买入: buy_fee = slot_capital × cost;  notional = slot_capital − buy_fee;
          shares = notional / (open × (1+买滑点));  cash −= slot_capital。
  - 卖出: exec = price × (1−卖滑点);  proceeds = shares × exec;
          sell_fee = proceeds × cost;  cash += proceeds − sell_fee。
  - 买手续费/买滑点/卖手续费/卖滑点各体现一次，不重复。
  - 组合收益以 equity_curve 的 NAV 为准（summary.portfolio_total_return）。
    summary.trade_return_sum 仅为「单笔净收益求和」，非组合口径。

Tradable flags（可选，data/processed/tradable/tradable_flags.parquet）
  - 买入必须同时满足 buyable_flag=True 且 tradable_flag=True（且非停牌）。
  - 卖出遇 跌停/停牌(sellable_flag=False 或 suspend) → 不丢失退出事件：
    置 pending_exit，之后第一个可卖日按 **当日开盘价** 卖出（deferred=True）。
  - flags=None 时无约束。
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class BacktestConfig:
    """Single source of truth for all backtest parameters."""
    holding_days: int = 5                 # fixed holding horizon (days)
    stop_loss: float | None = None        # e.g. -0.10 (decimal)
    take_profit: float | None = None      # e.g. 0.30
    cooldown_days: int = 0                # from exit_date
    max_positions: int = 999
    cost_bps: float = 20                  # 20 bps = 0.20% per side (commission)
    slippage_bps: float = 10              # 10 bps = 0.10% per side (slippage)
    initial_capital: float = 1_000_000.0
    stock_trend_filter: str | None = None  # "ma20" or "ma60" (uses T & prior only)
    strategy_mode: str = "fixed_holding_fill_slots"
    name: str = "default"


@dataclass
class _Position:
    stock: str
    signal_date: str
    entry_date: str
    entry_exec_price: float               # open × (1+buy slippage)
    shares: float
    gross_alloc: float                    # cash removed at entry (incl buy fee)
    exit_date: str                        # scheduled maturity date
    stop_loss_price: float | None = None
    take_profit_price: float | None = None
    pending_exit_reason: str | None = None
    last_close: float = 0.0               # for mark-to-market when close missing


def _get_price(prices: dict, stock: str, date_str: str, field: str) -> float | None:
    pdf = prices.get(stock)
    if pdf is None:
        return None
    row = pdf[pdf["date_str"] == date_str]
    if row.empty:
        return None
    col = f"{field}_yuan"
    if col not in row.columns:
        return None
    v = float(row.iloc[0][col])
    return v if v > 0 else None


def _check_stock_trend(prices: dict, stock: str, signal_date: str, ma_window: int) -> bool:
    """True if close > MA on signal_date (T). Uses only T and prior data."""
    pdf = prices.get(stock)
    if pdf is None:
        return False
    idx_arr = pdf.index[pdf["date_str"] == signal_date]
    if len(idx_arr) == 0:
        return False
    ti = int(idx_arr[0])
    if ti < ma_window:
        return False
    window = pdf.iloc[ti - ma_window + 1:ti + 1]["close_yuan"]
    if len(window) < ma_window:
        return False
    return float(pdf.iloc[ti]["close_yuan"]) > float(window.mean())


class SignalBacktester:
    """Unified backtest engine for institution BUY signals (fixed_holding_fill_slots)."""

    def __init__(self, config: BacktestConfig):
        self.config = config

    @staticmethod
    def _build_flag_lookups(flags: pd.DataFrame | None) -> tuple[set, set, set, set]:
        """把 tradable_flags 转为 O(1) 查询集合，键 (symbol, 'YYYY-MM-DD')。

        返回 (not_buyable, not_sellable, suspended, not_tradable)。
        """
        not_buyable: set = set()
        not_sellable: set = set()
        suspended: set = set()
        not_tradable: set = set()
        if flags is None or len(flags) == 0:
            return not_buyable, not_sellable, suspended, not_tradable
        f = flags.copy()
        f["date_str"] = pd.to_datetime(f["trade_date"]).dt.strftime("%Y-%m-%d")
        f["symbol"] = f["symbol"].astype(str).str.zfill(6)
        has_tradable = "tradable_flag" in f.columns
        for sym, ds, buyable, sellable, susp in zip(
            f["symbol"], f["date_str"],
            f["buyable_flag"], f["sellable_flag"], f["suspend_flag"],
        ):
            if not buyable:
                not_buyable.add((sym, ds))
            if not sellable:
                not_sellable.add((sym, ds))
            if susp:
                suspended.add((sym, ds))
        if has_tradable:
            for sym, ds, trad in zip(f["symbol"], f["date_str"], f["tradable_flag"]):
                if not trad:
                    not_tradable.add((sym, ds))
        return not_buyable, not_sellable, suspended, not_tradable

    def run(
        self,
        signals: pd.DataFrame,
        prices: dict[str, pd.DataFrame],
        tradable_flags: pd.DataFrame | None = None,
    ) -> dict[str, pd.DataFrame]:
        """Execute walk-forward backtest.

        signals : DataFrame [stock_code, signal_date] — signal_date is T (post-close).
        prices  : dict[stock_code -> DataFrame with date_str, open_yuan..close_yuan].
        tradable_flags : optional [trade_date, symbol, buyable_flag, sellable_flag,
                         suspend_flag, tradable_flag].
        Returns dict: trades, equity_curve, summary.
        """
        cfg = self.config
        h = cfg.holding_days
        cost_frac = cfg.cost_bps / 1e4
        slip_frac = cfg.slippage_bps / 1e4
        slot_capital = cfg.initial_capital / max(cfg.max_positions, 1)

        not_buyable, not_sellable, suspended, not_tradable = self._build_flag_lookups(tradable_flags)

        sig = signals[["stock_code", "signal_date"]].drop_duplicates()
        # 引擎内部统一补零到 6 位，避免仅依赖入口脚本（int/"1" 均可匹配 prices/flags 的 6 位键）
        sig["stock_code"] = sig["stock_code"].astype(str).str.zfill(6)
        sig_by_date: dict[str, list[str]] = defaultdict(list)
        for _, row in sig.iterrows():
            sig_by_date[row["signal_date"]].append(row["stock_code"])

        all_dates = sorted(set(d for pdf in prices.values() for d in pdf["date_str"].values))
        date_to_idx = {d: i for i, d in enumerate(all_dates)}

        open_positions: dict[str, _Position] = {}
        cooldown_until: dict[str, str] = {}
        pending_orders: list[dict] = []      # from previous day's signals: [{stock, signal_date}]
        trades: list[dict] = []
        equity: list[dict] = []
        cash = cfg.initial_capital

        def _record_sell(pos: _Position, today: str, price: float, reason: str, deferred: bool):
            nonlocal cash
            exec_sell = price * (1 - slip_frac)
            gross_proceeds = pos.shares * exec_sell
            sell_fee = gross_proceeds * cost_frac
            net_proceeds = gross_proceeds - sell_fee
            cash += net_proceeds
            net_ret = (net_proceeds / pos.gross_alloc - 1) * 100 if pos.gross_alloc else 0.0
            gross_ret = (exec_sell / pos.entry_exec_price - 1) * 100
            hold = date_to_idx.get(today, 0) - date_to_idx.get(pos.entry_date, 0)
            trades.append({
                "stock": pos.stock,
                "signal_date": pos.signal_date,
                "entry_date": pos.entry_date,
                "exit_date": today,
                "entry_price": round(pos.entry_exec_price, 6),
                "exit_price": round(exec_sell, 6),
                "shares": round(pos.shares, 8),
                "holding_days": hold,
                "gross_return_pct": round(gross_ret, 4),
                "net_return_pct": round(net_ret, 4),
                "exit_reason": reason,
                "deferred": deferred,
            })
            if cfg.cooldown_days > 0:
                exit_idx = date_to_idx.get(today, 0)
                cd_end = exit_idx + cfg.cooldown_days
                if cd_end < len(all_dates):
                    cooldown_until[pos.stock] = all_dates[cd_end]

        for day_idx, today in enumerate(all_dates):
            # ---- Step 1: exits (positions entered on prior days) ----
            for stock, pos in list(open_positions.items()):
                if (stock, today) in suspended:
                    # 停牌当日不可交易；若已到期，登记 pending maturity（下一可卖日按开盘价成交，deferred）
                    if pos.pending_exit_reason is None and day_idx >= date_to_idx.get(pos.exit_date, day_idx):
                        pos.pending_exit_reason = "maturity"
                    continue
                sellable = (stock, today) not in not_sellable

                # 先处理已挂起的退出（跌停/停牌顺延）
                if pos.pending_exit_reason is not None:
                    if sellable:
                        open_t = _get_price(prices, stock, today, "open")
                        if open_t is not None:
                            _record_sell(pos, today, open_t, pos.pending_exit_reason, deferred=True)
                            del open_positions[stock]
                    continue

                close_t = _get_price(prices, stock, today, "close")
                if close_t is None:
                    continue
                high_t = _get_price(prices, stock, today, "high")
                low_t = _get_price(prices, stock, today, "low")

                reason = None
                price = None
                if pos.take_profit_price is not None and high_t is not None and high_t >= pos.take_profit_price:
                    reason, price = "take_profit", pos.take_profit_price
                elif pos.stop_loss_price is not None and low_t is not None and low_t <= pos.stop_loss_price:
                    reason, price = "stop_loss", pos.stop_loss_price
                elif day_idx >= date_to_idx.get(pos.exit_date, day_idx):
                    reason, price = "maturity", close_t

                if reason is not None:
                    if sellable:
                        _record_sell(pos, today, price, reason, deferred=False)
                        del open_positions[stock]
                    else:
                        pos.pending_exit_reason = reason  # 不丢失退出事件，顺延

            # ---- Step 2: fill pending orders at today's open (entry = T+1) ----
            for order in pending_orders:
                stock = order["stock"]
                signal_date = order["signal_date"]
                if len(open_positions) >= cfg.max_positions:
                    break
                if stock in open_positions:
                    continue
                if stock in cooldown_until and today < cooldown_until[stock]:
                    continue
                # 买入必须同时满足 buyable 且 tradable（且非停牌）
                if ((stock, today) in not_buyable or (stock, today) in not_tradable
                        or (stock, today) in suspended):
                    continue
                open_t = _get_price(prices, stock, today, "open")
                if open_t is None:
                    continue
                if cash < slot_capital:
                    continue
                if cfg.stock_trend_filter:
                    ma_w = 20 if cfg.stock_trend_filter == "ma20" else 60
                    if not _check_stock_trend(prices, stock, signal_date, ma_w):
                        continue
                entry_idx = date_to_idx[today] + h
                if entry_idx >= len(all_dates):
                    continue  # 持有期越出数据范围，放弃
                exit_date = all_dates[entry_idx]

                exec_buy = open_t * (1 + slip_frac)
                buy_fee = slot_capital * cost_frac
                notional = slot_capital - buy_fee
                shares = notional / exec_buy
                cash -= slot_capital

                sl_price = exec_buy * (1 + cfg.stop_loss) if cfg.stop_loss is not None else None
                tp_price = exec_buy * (1 + cfg.take_profit) if cfg.take_profit is not None else None
                open_positions[stock] = _Position(
                    stock=stock, signal_date=signal_date, entry_date=today,
                    entry_exec_price=exec_buy, shares=shares, gross_alloc=slot_capital,
                    exit_date=exit_date, stop_loss_price=sl_price, take_profit_price=tp_price,
                    last_close=open_t,
                )

            # ---- Step 3: today's signals → pending orders for tomorrow ----
            pending_orders = [{"stock": s, "signal_date": today} for s in sig_by_date.get(today, [])]

            # ---- Step 4: mark-to-market (entry_date <= today) ----
            positions_value = 0.0
            for stock, pos in open_positions.items():
                c = _get_price(prices, stock, today, "close")
                if c is not None:
                    pos.last_close = c
                positions_value += pos.shares * pos.last_close

            equity.append({
                "date": today,
                "nav": round(cash + positions_value, 6),
                "cash": round(cash, 6),
                "positions_value": round(positions_value, 6),
                "n_positions": len(open_positions),
            })

        # ---- end-of-period unrealized (positions never closed, e.g. stuck in pending_exit) ----
        end_cost = sum(p.gross_alloc for p in open_positions.values())
        end_value = sum(p.shares * p.last_close for p in open_positions.values())
        final_nav = equity[-1]["nav"] if equity else cfg.initial_capital
        unrealized = {
            "open_positions_at_end": len(open_positions),
            "unrealized_position_value": round(end_value, 6),
            "unrealized_pnl_pct": round((end_value / end_cost - 1) * 100, 4) if end_cost else 0.0,
            "unrealized_nav_contribution": round(end_value / final_nav, 4) if final_nav else 0.0,
        }

        trades_df = pd.DataFrame(trades)
        equity_df = pd.DataFrame(equity)
        summary_df = self._build_summary(trades_df, equity_df, unrealized)
        return {"trades": trades_df, "equity_curve": equity_df, "summary": summary_df}

    def _build_summary(self, trades_df: pd.DataFrame, equity_df: pd.DataFrame,
                       unrealized: dict | None = None) -> pd.DataFrame:
        cfg = self.config
        unrealized = unrealized or {
            "open_positions_at_end": 0, "unrealized_position_value": 0.0,
            "unrealized_pnl_pct": 0.0, "unrealized_nav_contribution": 0.0,
        }

        # portfolio return from NAV (authoritative)
        if not equity_df.empty:
            nav = equity_df["nav"].to_numpy(float)
            portfolio_total_return = float(nav[-1] / nav[0] - 1) if nav[0] else 0.0
            peak = np.maximum.accumulate(nav)
            max_dd = float(((nav - peak) / peak).min())
        else:
            portfolio_total_return = 0.0
            max_dd = 0.0

        if trades_df.empty:
            row = {
                "config_name": cfg.name, "holding_days": cfg.holding_days,
                "n_trades": 0, "n_stocks": 0, "portfolio_total_return": round(portfolio_total_return, 4),
                "trade_return_sum": 0.0, "max_drawdown": round(max_dd, 4),
            }
            row.update(unrealized)
            return pd.DataFrame([row])

        rets = trades_df["net_return_pct"].dropna()
        stock_contrib = trades_df.groupby("stock")["net_return_pct"].sum().sort_values(ascending=False)
        trades_copy = trades_df.copy()
        trades_copy["ym"] = pd.to_datetime(trades_copy["entry_date"]).dt.strftime("%Y-%m")
        monthly = trades_copy.groupby("ym")["net_return_pct"].sum()

        row = {
            "config_name": cfg.name,
            "holding_days": cfg.holding_days,
            "n_trades": len(rets),
            "n_stocks": trades_df["stock"].nunique(),
            "portfolio_total_return": round(portfolio_total_return, 4),  # NAV 口径（权威组合收益）
            "trade_return_sum": round(float(rets.sum()), 3),             # 单笔 net_return_pct 求和，非组合口径
            "avg_ret": round(float(rets.mean()), 3),
            "median_ret": round(float(rets.median()), 3),
            "win_rate": round(float((rets > 0).mean()), 3),
            "max_drawdown": round(max_dd, 4),
            "avg_holding_days": round(float(trades_df["holding_days"].mean()), 1),
            "best_stock": stock_contrib.index[0] if len(stock_contrib) else "",
            "best_stock_contribution": round(float(stock_contrib.iloc[0]), 1) if len(stock_contrib) else 0.0,
            "worst_stock": stock_contrib.index[-1] if len(stock_contrib) else "",
            "worst_stock_contribution": round(float(stock_contrib.iloc[-1]), 1) if len(stock_contrib) > 1 else 0.0,
            "monthly_positive_rate": round(float((monthly > 0).mean()), 3) if len(monthly) else 0.0,
            "deferred_exits": int(trades_df["deferred"].sum()) if "deferred" in trades_df.columns else 0,
            "stop_loss_exits": int((trades_df["exit_reason"] == "stop_loss").sum()),
            "take_profit_exits": int((trades_df["exit_reason"] == "take_profit").sum()),
        }
        row.update(unrealized)
        return pd.DataFrame([row])
