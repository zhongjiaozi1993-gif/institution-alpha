"""
机构动态评分器
时间衰减加权（半衰期30天），综合评分0-100，S/A/B/C/D五级分类
"""
from __future__ import annotations
import numpy as np
import pandas as pd
try:
    from loguru import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)


def calculate_dynamic_score(
    seat_name: str,
    trade_history: pd.DataFrame,
    today: str | pd.Timestamp,
    half_life: int = 30,
    lookback_days: int = 90,
) -> dict | None:
    """
    计算机构的动态Alpha评分（近期表现加权）

    评分公式：加权收益(40分) + 加权胜率(30分) + 回撤控制(20分) + 交易经验(10分)

    Returns: {seat_name, dynamic_score, tier, recent_trades, ...} or None
    """
    df = trade_history[trade_history["seat_name"] == seat_name].copy()
    if df.empty:
        return None

    df["lhb_date"] = pd.to_datetime(df["lhb_date"])
    today = pd.to_datetime(today)

    cutoff = today - pd.Timedelta(days=lookback_days)
    recent = df[df["lhb_date"] >= cutoff]

    if len(recent) < 3:
        return None

    # 时间衰减权重（指数衰减，半衰期=half_life天）
    days_ago = (today - recent["lhb_date"]).dt.days
    weights = np.exp(-days_ago * np.log(2) / half_life)
    weights = weights.values
    weights = weights / weights.sum()

    # 自动选择可用的最佳收益周期：优先20d → 10d → 5d → 1d
    ret = None
    for h in [20, 10, 5, 1]:
        col = f"ret_{h}d"
        if col in recent.columns:
            vals = recent[col].values
            if (~np.isnan(vals)).sum() >= 2:
                ret = vals
                break
    if ret is None:
        return None

    valid = ~np.isnan(ret)
    weights = weights[valid]
    ret = ret[valid]

    weighted_ret = np.dot(weights, ret)
    weighted_win_rate = np.dot(weights, (ret > 0).astype(float))

    # 回撤控制（基于累计收益）
    cum_ret = (1 + ret).cumprod()
    peak = cum_ret.max()
    dd = (cum_ret[-1] - peak) / peak if peak > 0 else -1

    # 综合评分 0-100
    score_ret = min(max(weighted_ret * 40 / 0.15, -20), 60)  # 15%收益=40分
    score_wr = min(weighted_win_rate * 30 / 0.7, 30)          # 70%胜率=30分
    score_dd = min(max((1 + dd) * 20, 0), 20)                  # 无回撤=20分
    score_exp = min(np.log1p(len(recent)) / np.log1p(20) * 10, 10)  # 交易经验

    score = min(max(
        score_ret + score_wr + score_dd + score_exp,
        0
    ), 100)

    tier = classify_tier(score)

    return {
        "seat_name": seat_name,
        "dynamic_score": round(score, 1),
        "tier": tier,
        "tier_desc": TIER_DESC[tier],
        "recent_trades": len(recent),
        "weighted_ret_20d": round(weighted_ret, 4),
        "weighted_win_rate": round(weighted_win_rate, 4),
        "ret_score": round(score_ret, 1),
        "win_rate_score": round(score_wr, 1),
        "dd_score": round(score_dd, 1),
        "exp_score": round(score_exp, 1),
    }


TIER_DESC = {
    "S": "顶级Alpha机构",
    "A": "高Alpha机构",
    "B": "中等Alpha机构",
    "C": "低Alpha机构",
    "D": "负Alpha机构",
}


def classify_tier(score: float) -> str:
    """根据动态评分分类机构等级"""
    if score >= 80:
        return "S"
    elif score >= 60:
        return "A"
    elif score >= 40:
        return "B"
    elif score >= 20:
        return "C"
    else:
        return "D"


def score_all_seats(
    registry: dict[str, dict],
    trade_history: pd.DataFrame,
    today: str | pd.Timestamp,
    half_life: int = 30,
    lookback_days: int = 90,
) -> pd.DataFrame:
    """
    对所有注册营业部计算动态评分，返回排名DataFrame
    """
    scores = []
    for seat_name in registry:
        result = calculate_dynamic_score(
            seat_name, trade_history, today,
            half_life=half_life, lookback_days=lookback_days
        )
        if result is not None:
            scores.append(result)

    df = pd.DataFrame(scores)
    if df.empty:
        return df
    df = df.sort_values("dynamic_score", ascending=False).reset_index(drop=True)
    return df


def get_seat_multiplier(tier: str, tier_weights: dict[str, float] | None = None) -> float:
    """获取机构等级对应的仓位乘数"""
    if tier_weights is None:
        tier_weights = {"S": 1.5, "A": 1.0, "B": 0.5, "C": 0.0, "D": -0.5}
    return tier_weights.get(tier, 0.0)
