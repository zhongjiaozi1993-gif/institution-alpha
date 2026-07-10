"""
Unified signal backtest engine.

All risk configs call the same run() function — no per-config branching.
Assumptions:
  - Signal generated after T close, using only T and prior information
  - Entry at T+1 open
  - Exit at T+N close (or stop_loss/take_profit intraday)
  - Equal weight per stock, no leverage
  - Portfolio equity curve for max drawdown

Trading-constraint integration (optional `tradable_flags`):
  - 涨停(limit_up)/停牌(suspend) on T+1: 不可买入 → 放弃该信号
  - 跌停(limit_down)/停牌(suspend) on exit day: 不可卖出 → 顺延到下一可卖日
  Flags come from data/processed/tradable/tradable_flags.parquet
  (columns: trade_date, symbol, buyable_flag, sellable_flag, suspend_flag, ...).
  When flags is None, engine behaves as before (no constraint).
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class BacktestConfig:
    """Single source of truth for all backtest parameters."""
    holding_days: int = 5                # single horizon per run
    stop_loss: float | None = None       # e.g. -0.10
    take_profit: float | None = None     # e.g. 0.30
    cooldown_days: int = 0               # from exit_date
    max_positions: int = 999
    cost_bps: float = 20                 # 20 bps = 0.20%
    slippage_bps: float = 10             # 10 bps = 0.10%
    stock_trend_filter: str | None = None  # "ma20" or "ma60"
    name: str = "default"


@dataclass
class _Position:
    stock: str
    entry_date: str
    entry_price: float
    exit_date: str
    signal_date: str
    stop_loss_price: float | None = None
    take_profit_price: float | None = None


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
    ma = float(window.mean())
    close_t = float(pdf.iloc[ti]["close_yuan"])
    return close_t > ma


class SignalBacktester:
    """Unified backtest engine for institution BUY signals.

    Each run uses a single holding period. Loop over holding_days
    externally for multi-horizon comparison.
    """

    def __init__(self, config: BacktestConfig):
        self.config = config

    @staticmethod
    def _build_flag_lookups(
        flags: pd.DataFrame | None,
    ) -> tuple[set, set, set]:
        """把 tradable_flags 转为 O(1) 查询集合，键 (symbol, 'YYYY-MM-DD')。"""
        not_buyable: set = set()
        not_sellable: set = set()
        suspended: set = set()
        if flags is None or len(flags) == 0:
            return not_buyable, not_sellable, suspended
        f = flags.copy()
        f["date_str"] = pd.to_datetime(f["trade_date"]).dt.strftime("%Y-%m-%d")
        f["symbol"] = f["symbol"].astype(str).str.zfill(6)
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
        return not_buyable, not_sellable, suspended

    def run(
        self,
        signals: pd.DataFrame,
        prices: dict[str, pd.DataFrame],
        tradable_flags: pd.DataFrame | None = None,
    ) -> dict[str, pd.DataFrame]:
        """
        Execute walk-forward backtest.

        Parameters
        ----------
        signals : DataFrame with columns [stock_code, signal_date, ...]
            One row per stock+date signal. signal_date is T (signal
            detected after T close). Duplicates per stock+date deduped.
        prices : dict[stock_code -> DataFrame]
            Each DataFrame must have columns:
              date_str, open_yuan, high_yuan, low_yuan, close_yuan
        tradable_flags : optional DataFrame
            Columns [trade_date, symbol, buyable_flag, sellable_flag,
            suspend_flag]. When provided, 涨停/停牌不可买、跌停/停牌不可卖
            约束被强制执行。None → 无约束（旧行为）。

        Returns
        -------
        dict with keys: trades, equity_curve, summary
        """
        cfg = self.config
        h = cfg.holding_days

        # ---- tradable-flag lookups (empty sets when flags absent) ----
        not_buyable, not_sellable, suspended = self._build_flag_lookups(tradable_flags)

        # ---- deduplicate signals: one signal per stock per day ----
        sig = signals[["stock_code", "signal_date"]].drop_duplicates().copy()
        sig = sig.sort_values("signal_date").reset_index(drop=True)

        # ---- unified trading calendar ----
        all_dates = sorted(set(
            d for pdf in prices.values()
            for d in pdf["date_str"].values
        ))
        date_to_idx = {d: i for i, d in enumerate(all_dates)}

        # ---- state ----
        open_positions: dict[str, _Position] = {}  # stock -> position
        cooldown_until: dict[str, str] = {}         # stock -> first allowed date
        trades: list[dict] = []
        equity: list[dict] = []
        cash = 1.0  # normalized portfolio

        # Group signals by signal_date
        sig_by_date: dict[str, list[str]] = defaultdict(list)
        for _, row in sig.iterrows():
            sig_by_date[row["signal_date"]].append(row["stock_code"])

        for day_idx, today in enumerate(all_dates):
            # ---- Step 1: check exits ----
            for stock, pos in list(open_positions.items()):
                # 停牌: 今日不可交易，持仓顺延
                if (stock, today) in suspended:
                    continue
                high_t = _get_price(prices, stock, today, "high")
                low_t = _get_price(prices, stock, today, "low")
                close_t = _get_price(prices, stock, today, "close")
                if close_t is None:
                    continue

                sellable_today = (stock, today) not in not_sellable

                exit_reason = None
                exit_price = None

                if pos.take_profit_price is not None and high_t is not None and high_t >= pos.take_profit_price:
                    exit_reason = "take_profit"
                    exit_price = pos.take_profit_price
                elif pos.stop_loss_price is not None and low_t is not None and low_t <= pos.stop_loss_price:
                    exit_reason = "stop_loss"
                    exit_price = pos.stop_loss_price
                elif day_idx >= date_to_idx.get(pos.exit_date, day_idx):
                    exit_reason = "maturity"
                    exit_price = close_t

                # 跌停/不可卖: 触发卖出但今日不可卖 → 顺延到下一可卖日
                if exit_reason is not None and not sellable_today:
                    continue

                if exit_reason is not None:
                    gross_ret = (exit_price / pos.entry_price - 1)
                    cost_pct = (cfg.cost_bps + cfg.slippage_bps) * 2 / 10000
                    net_ret = gross_ret - cost_pct
                    pos_weight = 1.0 / max(cfg.max_positions, 1)
                    cash += pos_weight * (1 + net_ret)

                    actual_hold = (date_to_idx.get(today, 0)
                                   - date_to_idx.get(pos.entry_date, 0))

                    trades.append({
                        "stock": stock,
                        "signal_date": pos.signal_date,
                        "entry_date": pos.entry_date,
                        "exit_date": today,
                        "entry_price": round(pos.entry_price, 4),
                        "exit_price": round(exit_price, 4),
                        "holding_days": actual_hold,
                        "gross_return_pct": round(gross_ret * 100, 3),
                        "net_return_pct": round(net_ret * 100, 3),
                        "exit_reason": exit_reason,
                    })
                    del open_positions[stock]

                    if cfg.cooldown_days > 0:
                        exit_idx = date_to_idx.get(today, day_idx)
                        cd_end_idx = exit_idx + cfg.cooldown_days
                        if cd_end_idx < len(all_dates):
                            cooldown_until[stock] = all_dates[cd_end_idx]

            # ---- Step 2: process new signals → enter T+1 ----
            if day_idx + 1 < len(all_dates):
                next_date = all_dates[day_idx + 1]
                for stock in sig_by_date.get(today, []):
                    if len(open_positions) >= cfg.max_positions:
                        break
                    if stock in open_positions:
                        continue
                    if stock in cooldown_until and today < cooldown_until[stock]:
                        continue

                    # 涨停/停牌: T+1 开盘不可买 → 放弃该信号
                    if (stock, next_date) in not_buyable or (stock, next_date) in suspended:
                        continue

                    entry_price = _get_price(prices, stock, next_date, "open")
                    if entry_price is None:
                        continue

                    if cfg.stock_trend_filter:
                        ma_w = 20 if cfg.stock_trend_filter == "ma20" else 60
                        if not _check_stock_trend(prices, stock, today, ma_w):
                            continue

                    entry_with_slip = entry_price * (1 + cfg.slippage_bps / 10000)

                    exit_idx = date_to_idx[next_date] + h
                    if exit_idx >= len(all_dates):
                        continue
                    exit_date = all_dates[exit_idx]

                    sl_price = None
                    tp_price = None
                    if cfg.stop_loss is not None:
                        sl_price = entry_with_slip * (1 + cfg.stop_loss)
                    if cfg.take_profit is not None:
                        tp_price = entry_with_slip * (1 + cfg.take_profit)

                    open_positions[stock] = _Position(
                        stock=stock,
                        entry_date=next_date,
                        entry_price=entry_with_slip,
                        exit_date=exit_date,
                        signal_date=today,
                        stop_loss_price=sl_price,
                        take_profit_price=tp_price,
                    )

                    pos_weight = 1.0 / max(cfg.max_positions, 1)
                    entry_cost_pct = cfg.cost_bps / 10000
                    cash -= pos_weight * (1 + entry_cost_pct)

            # ---- Step 3: mark-to-market ----
            positions_value = 0.0
            for stock, pos in open_positions.items():
                close_t = _get_price(prices, stock, today, "close")
                if close_t is None:
                    continue
                pos_weight = 1.0 / max(cfg.max_positions, 1)
                positions_value += pos_weight * (close_t / pos.entry_price)

            total_value = cash + positions_value
            equity.append({
                "date": today,
                "nav": round(total_value, 6),
                "cash": round(cash, 6),
                "positions_value": round(positions_value, 6),
                "n_positions": len(open_positions),
            })

        # ---- Build outputs ----
        trades_df = pd.DataFrame(trades)
        equity_df = pd.DataFrame(equity)
        summary_df = self._build_summary(trades_df, equity_df)

        return {
            "trades": trades_df,
            "equity_curve": equity_df,
            "summary": summary_df,
        }

    def _build_summary(
        self, trades_df: pd.DataFrame, equity_df: pd.DataFrame
    ) -> pd.DataFrame:
        cfg = self.config

        if trades_df.empty:
            return pd.DataFrame()

        rets = trades_df["net_return_pct"].dropna()
        if len(rets) == 0:
            return pd.DataFrame()

        n_trades = len(rets)
        avg_ret = round(float(rets.mean()), 3)
        median_ret = round(float(rets.median()), 3)
        win_rate = round(float((rets > 0).mean()), 3)
        total_return = round(float(rets.sum()), 3)

        stock_contrib = trades_df.groupby("stock")["net_return_pct"].sum().sort_values(ascending=False)
        best_stock = stock_contrib.index[0] if len(stock_contrib) > 0 else ""
        best_stock_pct = round(float(stock_contrib.iloc[0]), 1) if len(stock_contrib) > 0 else 0.0
        worst_stock = stock_contrib.index[-1] if len(stock_contrib) > 0 else ""
        worst_stock_pct = round(float(stock_contrib.iloc[-1]), 1) if len(stock_contrib) > 1 else 0.0

        if not equity_df.empty:
            nav = equity_df["nav"].values
            peak = np.maximum.accumulate(nav)
            dd = (nav - peak) / peak
            max_dd = round(float(dd.min()), 4)
        else:
            max_dd = 0.0

        trades_copy = trades_df.copy()
        trades_copy["year_month"] = pd.to_datetime(trades_copy["entry_date"]).dt.strftime("%Y-%m")
        monthly = trades_copy.groupby("year_month")["net_return_pct"].sum()
        monthly_positive_rate = round(float((monthly > 0).mean()), 3) if len(monthly) > 0 else 0.0

        sl_exits = int((trades_df["exit_reason"] == "stop_loss").sum())
        tp_exits = int((trades_df["exit_reason"] == "take_profit").sum())
        avg_hold = round(float(trades_df["holding_days"].mean()), 1)

        return pd.DataFrame([{
            "config_name": cfg.name,
            "holding_days": cfg.holding_days,
            "n_trades": n_trades,
            "n_stocks": trades_df["stock"].nunique(),
            "avg_ret": avg_ret,
            "median_ret": median_ret,
            "win_rate": win_rate,
            "total_return": total_return,
            "max_drawdown": max_dd,
            "avg_holding_days": avg_hold,
            "best_stock": best_stock,
            "best_stock_contribution": best_stock_pct,
            "worst_stock": worst_stock,
            "worst_stock_contribution": worst_stock_pct,
            "monthly_positive_rate": monthly_positive_rate,
            "stop_loss_exits": sl_exits,
            "take_profit_exits": tp_exits,
        }])
