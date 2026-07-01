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

        positions = {}
        cash = self.initial_capital
        nav_history = []
        trades = []
        pending_signals = []  # 等待T+1开盘建仓的信号

        trading_days = self._get_trading_days(price_data, start_date, end_date)
        if len(trading_days) == 0:
            return pd.DataFrame(), pd.DataFrame()

        # 构建每只股票 date → row 的查找表，加速回测
        price_lookup = {}
        for stock, df in price_data.items():
            if df.empty:
                continue
            df = df.copy()
            df["date"] = pd.to_datetime(df["date"])
            price_lookup[stock] = df.set_index("date")

        for day_idx, today in enumerate(trading_days):
            # ==== Step 1: 执行前一交易日信号的T+1建仓 ====
            for sig in pending_signals:
                if len(positions) >= self.max_positions:
                    break
                if cash <= 0:
                    break

                stock = sig["stock_code"]
                if stock not in price_lookup:
                    continue

                px = price_lookup[stock]
                if today not in px.index:
                    continue

                row = px.loc[today]
                entry_price = float(row["open"]) * (1 + self.slippage)

                position_ratio = min(sig.get("strength", 0.3), 0.3)
                allocation = cash * position_ratio / max(len(pending_signals), 1)
                shares = int(allocation / entry_price / 100) * 100
                if shares < 100:
                    continue

                cost = entry_price * shares * (1 + self.commission)
                if cost > cash * 0.3:
                    cost = cash * 0.3
                    shares = int(cost / entry_price / (1 + self.commission) / 100) * 100
                    cost = entry_price * shares * (1 + self.commission)

                if shares < 100:
                    continue

                cash -= cost
                positions[stock] = {
                    "entry_date": today,
                    "entry_price": entry_price,
                    "shares": shares,
                    "cost": cost,
                    "signal_strength": sig.get("strength", 0.5),
                    "seat_name": sig.get("seat_name", ""),
                    "tier": sig.get("tier", "B"),
                    "stop_loss_price": entry_price * (1 - self.stop_loss),
                    "take_profit_price": entry_price * (1 + self.take_profit),
                }

            pending_signals = []

            # ==== Step 2: 检查持仓止损/止盈/到期 ====
            for stock, pos in list(positions.items()):
                exit_reason = None
                exit_price = None

                if stock not in price_lookup:
                    continue

                px = price_lookup[stock]
                if today not in px.index:
                    continue

                row = px.loc[today]
                days_held = len(px.loc[pos["entry_date"]:today].index) - 1

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
                        "holding_days": max(days_held, 1),
                        "signal_strength": pos["signal_strength"],
                    })
                    del positions[stock]

            # ==== Step 3: 收集今日龙虎榜信号 → T+1开盘建仓 ====
            day_signals = signals[signals["lhb_date"] == today]
            if not day_signals.empty:
                # 检查是否有下一个交易日
                if day_idx + 1 < len(trading_days):
                    for _, sig in day_signals.iterrows():
                        pending_signals.append({
                            "stock_code": sig["stock_code"],
                            "strength": sig.get("strength", 0.3),
                            "seat_name": sig.get("seat_name", ""),
                            "tier": sig.get("tier", "B"),
                        })

            # ==== Step 4: 计算当日净值 ====
            position_value = 0.0
            for stock, pos in positions.items():
                if stock in price_lookup:
                    px = price_lookup[stock]
                    if today in px.index:
                        position_value += pos["shares"] * float(px.loc[today, "close"])
                    else:
                        # 缺行情时用最近5日内的收盘价，避免估值归零
                        nearby = px.index[px.index <= today]
                        if len(nearby) > 0 and (today - nearby[-1]).days <= 5:
                            position_value += pos["shares"] * float(px.loc[nearby[-1], "close"])

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
        """获取所有股票交易日的并集"""
        all_dates = set()
        for df in price_data.values():
            if df.empty or "date" not in df.columns:
                continue
            dates = pd.to_datetime(df["date"])
            mask = (dates >= start_date) & (dates <= end_date)
            all_dates.update(dates[mask])
        return pd.DatetimeIndex(sorted(all_dates))


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

    cum = nav / nav.iloc[0]
    peak = cum.expanding().max()
    dd = ((cum - peak) / peak)
    max_dd = dd.min()

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
