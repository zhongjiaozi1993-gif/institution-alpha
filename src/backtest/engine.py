"""
回测引擎
T+1开盘建仓、固定持有期、止损止盈、手续费+滑点
"""
from __future__ import annotations
import numpy as np
import pandas as pd
try:
    from loguru import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)


class BacktestEngine:
    """回测引擎"""

    def __init__(
        self,
        initial_capital: float = 1_000_000,
        commission: float = 0.0003,
        slippage: float = 0.001,
        stop_loss: float = 0.08,
        take_profit: float = 0.15,
        max_holding_days: int = 20,
        max_positions: int = 10,
    ):
        self.initial_capital = initial_capital
        self.commission = commission
        self.slippage = slippage
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.max_holding_days = max_holding_days
        self.max_positions = max_positions

    def run(
        self,
        signals: pd.DataFrame,
        price_data: dict[str, pd.DataFrame],
        start_date: str,
        end_date: str,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        执行回测

        signals: [stock_code, lhb_date, strength, direction, ...]
        price_data: {stock_code: DataFrame[date, open, close, high, low]}

        Returns: (nav_history, trades)
        """
        if signals.empty:
            return pd.DataFrame(), pd.DataFrame()

        signals = signals.copy()
        signals["lhb_date"] = pd.to_datetime(signals["lhb_date"])

        positions = {}  # {stock: {entry_date, entry_price, shares, signal_strength, ...}}
        cash = self.initial_capital
        nav_history = []
        trades = []

        trading_days = self._get_trading_days(price_data, start_date, end_date)

        for today in trading_days:
            # 1. 检查持仓止损/止盈/到期
            for stock, pos in list(positions.items()):
                exit_reason = None
                exit_price = None

                if stock not in price_data:
                    continue
                prices = price_data[stock]
                today_prices = prices[prices["date"] == today]
                if today_prices.empty:
                    continue
                row = today_prices.iloc[0]

                days_held = len(prices[(prices["date"] > pos["entry_date"]) & (prices["date"] <= today)])

                if row["low"] <= pos["stop_loss_price"]:
                    exit_reason = "stop_loss"
                    exit_price = pos["stop_loss_price"]
                elif row["high"] >= pos["take_profit_price"]:
                    exit_reason = "take_profit"
                    exit_price = pos["take_profit_price"]
                elif days_held >= self.max_holding_days:
                    exit_reason = "time_exit"
                    exit_price = row["close"]

                if exit_reason:
                    # 卖出
                    sell_amt = exit_price * pos["shares"] * (1 - self.commission - self.slippage)
                    pnl = sell_amt - pos["cost"]
                    pnl_pct = pnl / pos["cost"]
                    cash += sell_amt

                    trades.append({
                        "stock": stock,
                        "entry_date": pos["entry_date"],
                        "exit_date": today,
                        "entry_price": pos["entry_price"],
                        "exit_price": exit_price,
                        "shares": pos["shares"],
                        "pnl": pnl,
                        "pnl_pct": pnl_pct,
                        "exit_reason": exit_reason,
                        "holding_days": days_held,
                        "signal_strength": pos["signal_strength"],
                    })
                    del positions[stock]

            # 2. 检查新信号（T日信号 → T+1日建仓）
            # 实际回测中，今天的date对应的是T日收盘后出的龙虎榜，
            # T+1（下一个交易日）开盘买入
            day_signals = signals[signals["lhb_date"] == today]

            if not day_signals.empty:
                next_day_idx = trading_days.get_loc(today) + 1
                if next_day_idx < len(trading_days):
                    next_day = trading_days[next_day_idx]

                    for _, sig in day_signals.iterrows():
                        if len(positions) >= self.max_positions:
                            break
                        if cash <= 0:
                            break

                        stock = sig["stock_code"]
                        if stock not in price_data:
                            continue

                        next_prices = price_data[stock][price_data[stock]["date"] == next_day]
                        if next_prices.empty:
                            continue
                        next_row = next_prices.iloc[0]
                        entry_price = next_row["open"] * (1 + self.slippage)

                        position_ratio = min(sig.get("strength", 0.3), 0.3)
                        allocation = cash * position_ratio / max(len(day_signals), 1)
                        shares = int(allocation / entry_price / 100) * 100
                        if shares < 100:
                            continue

                        cost = entry_price * shares * (1 + self.commission)
                        if cost > cash * 0.3:  # 单票不超过30%资金
                            cost = cash * 0.3
                            shares = int(cost / entry_price / (1 + self.commission) / 100) * 100
                            cost = entry_price * shares * (1 + self.commission)

                        if shares < 100:
                            continue

                        cash -= cost
                        positions[stock] = {
                            "entry_date": next_day,
                            "entry_price": entry_price,
                            "shares": shares,
                            "cost": cost,
                            "signal_strength": sig.get("strength", 0.5),
                            "seat_name": sig.get("seat_name", ""),
                            "tier": sig.get("tier", "B"),
                            "stop_loss_price": entry_price * (1 - self.stop_loss),
                            "take_profit_price": entry_price * (1 + self.take_profit),
                        }

            # 3. 计算当日净值
            position_value = 0
            for stock, pos in positions.items():
                if stock in price_data:
                    prices = price_data[stock]
                    today_row = prices[prices["date"] == today]
                    if not today_row.empty:
                        position_value += pos["shares"] * today_row.iloc[0]["close"]

            total_value = cash + position_value
            nav_history.append({
                "date": today,
                "nav": total_value,
                "cash": cash,
                "positions_value": position_value,
                "num_positions": len(positions),
            })

        nav_df = pd.DataFrame(nav_history)
        trades_df = pd.DataFrame(trades)
        return nav_df, trades_df

    def _get_trading_days(
        self,
        price_data: dict[str, pd.DataFrame],
        start_date: str,
        end_date: str,
    ) -> pd.DatetimeIndex:
        """获取交易日历"""
        for df in price_data.values():
            if not df.empty:
                all_dates = pd.to_datetime(df["date"])
                mask = (all_dates >= start_date) & (all_dates <= end_date)
                return pd.DatetimeIndex(sorted(all_dates[mask].unique()))
        return pd.DatetimeIndex([])


def calculate_nav_metrics(nav_df: pd.DataFrame) -> dict:
    """从净值曲线计算关键回测指标"""
    if nav_df.empty:
        return {}

    nav = nav_df.set_index("date")["nav"]
    daily_returns = nav.pct_change().dropna()

    if len(daily_returns) < 2:
        return {"total_return": 0}

    total_ret = nav.iloc[-1] / nav.iloc[0] - 1
    ann_ret = (1 + total_ret) ** (252 / len(daily_returns)) - 1
    ann_vol = daily_returns.std() * np.sqrt(252)
    sharpe = (daily_returns.mean() / daily_returns.std() * np.sqrt(252)) if daily_returns.std() > 0 else 0

    # Max drawdown
    cum = nav / nav.iloc[0]
    peak = cum.expanding().max()
    dd = ((cum - peak) / peak)
    max_dd = dd.min()

    # Calmar
    calmar = ann_ret / abs(max_dd) if max_dd != 0 else 0

    return {
        "total_return": round(total_ret, 4),
        "annualized_return": round(ann_ret, 4),
        "annualized_volatility": round(ann_vol, 4),
        "sharpe_ratio": round(sharpe, 4),
        "max_drawdown": round(max_dd, 4),
        "calmar_ratio": round(calmar, 4),
        "total_days": len(daily_returns),
    }
