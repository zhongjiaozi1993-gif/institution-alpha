"""
回测绩效评估指标（Phase 4.6：兼容 signal_backtester 当前 trades schema）。

- NAV 口径（组合）: portfolio_total_return, annualized, sharpe, max_dd ...
- 交易口径: 支持 net_return_pct（不再要求 pnl / pnl_pct）。
- 明确区分:
    portfolio_total_return  = 组合 NAV 首末比（权威组合收益）
    trade_return_sum        = 单笔 net_return_pct 求和（活跃度，非组合收益）
"""
from __future__ import annotations
import numpy as np
import pandas as pd


def compute_full_metrics(nav_df: pd.DataFrame, trades_df: pd.DataFrame, rf: float = 0.03) -> dict:
    """计算完整绩效指标。nav_df: [date, nav]; trades_df: 含 net_return_pct（或旧 pnl）。"""
    metrics: dict = {}

    # ---- 组合 NAV 口径 ----
    if nav_df is not None and not nav_df.empty:
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

            downside = daily_returns[daily_returns < 0]
            downside_std = downside.std() * np.sqrt(252) if len(downside) > 1 else 0
            sortino = ((daily_returns.mean() - rf / 252) * 252 / downside_std) if downside_std > 0 else 0
            calmar = ann_ret / abs(max_dd) if max_dd != 0 else 0

            metrics.update({
                "portfolio_total_return": round(total_ret, 4),   # 权威组合收益（NAV）
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

    # ---- 交易口径 ----
    if trades_df is not None and not trades_df.empty:
        if "net_return_pct" in trades_df.columns:
            r = trades_df["net_return_pct"].dropna()
            wins = r[r > 0]
            losses = r[r < 0]
            metrics.update({
                "total_trades": len(r),
                "win_rate": round(float((r > 0).mean()), 4),
                "avg_win_pct": round(float(wins.mean()), 4) if len(wins) else 0.0,
                "avg_loss_pct": round(float(abs(losses.mean())), 4) if len(losses) else 0.0,
                "profit_loss_ratio": round(float(wins.mean() / abs(losses.mean())), 4) if len(losses) and losses.mean() != 0 else 0.0,
                "trade_return_sum": round(float(r.sum()), 4),   # 非组合口径
            })
            if "holding_days" in trades_df.columns:
                metrics["avg_holding_days"] = round(float(trades_df["holding_days"].mean()), 1)
            if "exit_reason" in trades_df.columns:
                for reason in ["stop_loss", "take_profit", "maturity"]:
                    sub = trades_df[trades_df["exit_reason"] == reason]
                    if not sub.empty:
                        metrics[f"exit_{reason}_count"] = len(sub)
                        metrics[f"exit_{reason}_avg_ret_pct"] = round(float(sub["net_return_pct"].mean()), 4)
            if "deferred" in trades_df.columns:
                metrics["deferred_exits"] = int(trades_df["deferred"].sum())

        elif "pnl" in trades_df.columns:  # 旧 schema 兼容
            pnl = trades_df["pnl"]
            wins = pnl[pnl > 0]
            losses = pnl[pnl < 0]
            metrics.update({
                "total_trades": len(trades_df),
                "win_rate": round(float((pnl > 0).mean()), 4),
                "total_pnl": round(float(pnl.sum()), 2),
                "profit_loss_ratio": round(float(wins.mean() / abs(losses.mean())), 4) if len(losses) else 0.0,
            })

    return metrics


def print_metrics(metrics: dict) -> None:
    """打印回测指标（分组展示）。"""
    print("=" * 50)
    print("回测绩效报告")
    print("=" * 50)

    print("\n组合收益（NAV 口径）:")
    for k in ["portfolio_total_return", "annualized_return"]:
        if k in metrics:
            print(f"  {k}: {metrics[k]*100:.2f}%")
    if "trade_return_sum" in metrics:
        print(f"  trade_return_sum(非组合口径): {metrics['trade_return_sum']:.2f}%")

    print("\n风险指标:")
    for k in ["annualized_volatility", "max_drawdown"]:
        if k in metrics:
            print(f"  {k}: {metrics[k]*100:.2f}%")

    print("\n风险调整收益:")
    for k in ["sharpe_ratio", "sortino_ratio", "calmar_ratio"]:
        if k in metrics:
            print(f"  {k}: {metrics[k]}")

    print("\n交易统计:")
    for k in ["total_trades", "win_rate", "profit_loss_ratio", "avg_holding_days", "deferred_exits"]:
        if k in metrics:
            v = metrics[k]
            print(f"  {k}: {v*100:.1f}%" if k == "win_rate" else f"  {k}: {v}")
    print("=" * 50)
