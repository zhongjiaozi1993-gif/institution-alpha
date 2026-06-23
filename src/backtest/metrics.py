"""
回测绩效评估指标
年化收益、夏普比率、最大回撤、胜率、盈亏比、Calmar比率
"""
from __future__ import annotations
import numpy as np
import pandas as pd


def compute_full_metrics(nav_df: pd.DataFrame, trades_df: pd.DataFrame, rf: float = 0.03) -> dict:
    """计算完整的回测绩效指标"""
    metrics = {}

    if not nav_df.empty:
        nav = nav_df.set_index("date")["nav"]
        daily_returns = nav.pct_change().dropna()

        if len(daily_returns) >= 2:
            total_ret = nav.iloc[-1] / nav.iloc[0] - 1
            years = len(daily_returns) / 252
            ann_ret = (1 + total_ret) ** (1 / years) - 1 if years > 0 else 0
            ann_vol = daily_returns.std() * np.sqrt(252)
            sharpe = ((daily_returns.mean() - rf / 252) / daily_returns.std() * np.sqrt(252)) if daily_returns.std() > 0 else 0

            cum = nav / nav.iloc[0]
            peak = cum.expanding().max()
            dd = (cum - peak) / peak
            max_dd = dd.min()

            # Sortino ratio (downside deviation only)
            downside = daily_returns[daily_returns < 0]
            downside_std = downside.std() * np.sqrt(252) if len(downside) > 1 else 0
            sortino = ((daily_returns.mean() - rf / 252) * 252 / downside_std) if downside_std > 0 else 0

            calmar = ann_ret / abs(max_dd) if max_dd != 0 else 0

            metrics.update({
                "total_return": round(total_ret, 4),
                "annualized_return": round(ann_ret, 4),
                "annualized_volatility": round(ann_vol, 4),
                "sharpe_ratio": round(sharpe, 4),
                "sortino_ratio": round(sortino, 4),
                "max_drawdown": round(max_dd, 4),
                "calmar_ratio": round(calmar, 4),
                "return_skew": round(daily_returns.skew(), 4),
                "return_kurt": round(daily_returns.kurtosis(), 4),
                "positive_days_ratio": round((daily_returns > 0).mean(), 4),
                "total_trading_days": len(daily_returns),
            })

    if not trades_df.empty:
        pnl = trades_df["pnl"]
        pnl_pct = trades_df["pnl_pct"]
        wins = pnl[pnl > 0]
        losses = pnl[pnl < 0]

        metrics.update({
            "total_trades": len(trades_df),
            "win_rate": round((pnl > 0).mean(), 4),
            "avg_win": round(wins.mean(), 2) if len(wins) > 0 else 0,
            "avg_loss": round(abs(losses.mean()), 2) if len(losses) > 0 else 0,
            "profit_loss_ratio": round(wins.mean() / abs(losses.mean()), 4) if len(losses) > 0 else 0,
            "total_pnl": round(pnl.sum(), 2),
            "avg_holding_days": round(trades_df["holding_days"].mean(), 1),
            "max_win": round(pnl.max(), 2),
            "max_loss": round(pnl.min(), 2),
        })

        # 按退出原因统计
        if "exit_reason" in trades_df.columns:
            for reason in ["stop_loss", "take_profit", "time_exit"]:
                subset = trades_df[trades_df["exit_reason"] == reason]
                if not subset.empty:
                    metrics[f"exit_{reason}_count"] = len(subset)
                    metrics[f"exit_{reason}_avg_pnl_pct"] = round(subset["pnl_pct"].mean(), 4)

    return metrics


def print_metrics(metrics: dict) -> None:
    """打印回测指标（分组展示）"""
    print("=" * 50)
    print("回测绩效报告")
    print("=" * 50)

    print("\n收益指标:")
    for k in ["total_return", "annualized_return", "total_pnl"]:
        if k in metrics:
            v = metrics[k]
            if "return" in k and isinstance(v, float):
                print(f"  {k}: {v*100:.2f}%")
            else:
                print(f"  {k}: {v}")

    print("\n风险指标:")
    for k in ["annualized_volatility", "max_drawdown"]:
        if k in metrics:
            v = metrics[k]
            print(f"  {k}: {v*100:.2f}%" if isinstance(v, float) else f"  {k}: {v}")

    print("\n风险调整收益:")
    for k in ["sharpe_ratio", "sortino_ratio", "calmar_ratio"]:
        if k in metrics:
            print(f"  {k}: {metrics[k]}")

    print("\n交易统计:")
    for k in ["total_trades", "win_rate", "profit_loss_ratio", "avg_holding_days"]:
        if k in metrics:
            v = metrics[k]
            if k == "win_rate" and isinstance(v, float):
                print(f"  {k}: {v*100:.1f}%")
            else:
                print(f"  {k}: {v}")

    print("=" * 50)
