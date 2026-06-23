"""
机构Alpha画像
基于历史交易记录计算机构的核心能力指标：胜率、盈亏比、夏普、最大回撤
"""
from __future__ import annotations
import numpy as np
import pandas as pd
try:
    from loguru import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)


def calculate_pl_ratio(returns: pd.Series) -> float:
    """盈亏比 = 平均盈利 / 平均亏损"""
    gains = returns[returns > 0]
    losses = returns[returns < 0]
    if losses.empty or gains.empty:
        return 0.0
    return gains.mean() / abs(losses.mean())


def calculate_max_drawdown(returns: pd.Series) -> float:
    """最大回撤（基于累计收益曲线）"""
    if returns.empty:
        return 0.0
    cum = (1 + returns).cumprod()
    peak = cum.expanding().max()
    return ((cum - peak) / peak).min()


def calculate_sharpe(returns: pd.Series, rf: float = 0.03) -> float:
    """年化夏普比率"""
    if len(returns) < 2:
        return 0.0
    excess = returns.mean() - rf / 252
    vol = returns.std()
    if vol == 0 or np.isnan(vol):
        return 0.0
    return excess / vol * np.sqrt(252)


def calculate_info_ratio(returns: pd.Series, benchmark_returns: pd.Series | None = None) -> float:
    """信息比率（相对基准的超额收益/跟踪误差）"""
    if benchmark_returns is None or len(returns) < 2:
        return 0.0
    if len(returns) != len(benchmark_returns):
        return 0.0
    excess = returns - benchmark_returns
    if excess.std() == 0 or np.isnan(excess.std()):
        return 0.0
    return excess.mean() / excess.std() * np.sqrt(252)


def profile_seat(
    seat_name: str,
    trade_history: pd.DataFrame,
    horizon: int = 20,
    benchmark_data: pd.DataFrame | None = None,
) -> dict:
    """
    计算单个营业部的Alpha画像

    trade_history: [seat_name, stock_code, lhb_date, buy_amount, ret_Xd, excess_Xd, ...]

    Returns: 机构能力画像dict
    """
    if trade_history.empty:
        return {"seat_name": seat_name, "total_trades": 0}

    ret_col = f"ret_{horizon}d"
    excess_col = f"excess_{horizon}d"

    if ret_col not in trade_history.columns:
        logger.warning(f"{seat_name} 缺少 {ret_col} 列")
        return {"seat_name": seat_name, "total_trades": len(trade_history)}

    returns = trade_history[ret_col].dropna()

    profile = {
        "seat_name": seat_name,
        "total_trades": len(trade_history),
        "valid_trades": len(returns),
        f"win_rate_{horizon}d": (returns > 0).mean() if len(returns) > 0 else 0,
        f"avg_ret_{horizon}d": returns.mean() if len(returns) > 0 else 0,
        f"median_ret_{horizon}d": returns.median() if len(returns) > 0 else 0,
        f"profit_loss_ratio_{horizon}d": calculate_pl_ratio(returns),
        f"max_drawdown_{horizon}d": calculate_max_drawdown(returns),
        f"sharpe_{horizon}d": calculate_sharpe(returns),
        f"ret_skew_{horizon}d": returns.skew() if len(returns) > 2 else 0,
    }

    # 多周期收益（如果可用）
    for h in [1, 5, 10, 20, 60]:
        col = f"ret_{h}d"
        if col in trade_history.columns:
            r = trade_history[col].dropna()
            profile[f"win_rate_{h}d"] = (r > 0).mean() if len(r) > 0 else 0
            profile[f"avg_ret_{h}d"] = r.mean() if len(r) > 0 else 0

    if excess_col in trade_history.columns:
        excess = trade_history[excess_col].dropna()
        profile[f"avg_excess_{horizon}d"] = excess.mean() if len(excess) > 0 else 0
        if benchmark_data is not None and len(returns) >= 2:
            profile[f"info_ratio_{horizon}d"] = calculate_info_ratio(returns, excess)

    return profile


def build_alpha_registry(
    seat_histories: pd.DataFrame,
    horizon: int = 20,
    min_trades: int = 3,
) -> dict[str, dict]:
    """
    构建所有营业部的Alpha画像注册表

    Returns: {seat_name: alpha_profile_dict}
    """
    registry = {}
    for seat_name, group in seat_histories.groupby("seat_name"):
        if len(group) < min_trades:
            continue
        profile = profile_seat(seat_name, group, horizon=horizon)
        registry[seat_name] = profile

    logger.info(f"Alpha注册表构建完成，{len(registry)} 个营业部（min_trades >= {min_trades}）")
    return registry


def rank_seats(registry: dict[str, dict], metric: str = "avg_ret_20d") -> pd.DataFrame:
    """按指定指标排名所有营业部"""
    rows = []
    for seat_name, profile in registry.items():
        rows.append({
            "seat_name": seat_name,
            "total_trades": profile.get("total_trades", 0),
            metric: profile.get(metric, 0),
            "win_rate_20d": profile.get("win_rate_20d", 0),
            "sharpe_20d": profile.get("sharpe_20d", 0),
        })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df = df.sort_values(metric, ascending=False).reset_index(drop=True)
    df["rank"] = range(1, len(df) + 1)
    return df
