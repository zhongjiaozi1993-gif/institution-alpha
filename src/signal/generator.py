"""
交易信号生成器
高Alpha营业部买入 → 做多信号，等级加权，多机构共振
"""
from __future__ import annotations
import numpy as np
import pandas as pd
try:
    from loguru import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)

from ..alpha.dynamic_scorer import get_seat_multiplier

DEFAULT_TIER_WEIGHTS = {"S": 1.5, "A": 1.0, "B": 0.5, "C": 0.0, "D": -0.5}


def generate_signals(
    lhb_daily: pd.DataFrame,
    seat_scores: pd.DataFrame,
    min_signal_strength: float = 0.3,
    min_confidence: float = 0.6,
    tier_weights: dict[str, float] | None = None,
) -> pd.DataFrame:
    """
    根据当日龙虎榜+机构Alpha评分生成交易信号

    lhb_daily: 当日龙虎榜数据 [代码, 上榜日, 营业部名称, 买入金额, ...]
    seat_scores: 机构动态评分 [seat_name, dynamic_score, tier, ...]

    Returns: [stock_code, lhb_date, seat_name, tier, direction, strength, ...]
    """
    if lhb_daily.empty or seat_scores.empty:
        return pd.DataFrame()

    tier_weights = tier_weights or DEFAULT_TIER_WEIGHTS

    score_map = dict(zip(seat_scores["seat_name"], seat_scores["dynamic_score"]))
    tier_map = dict(zip(seat_scores["seat_name"], seat_scores["tier"]))
    wret_map = dict(zip(seat_scores["seat_name"], seat_scores["weighted_ret_20d"]))
    wr_map = dict(zip(seat_scores["seat_name"], seat_scores["weighted_win_rate"]))

    signals = []
    seat_col = _find_seat_column(lhb_daily)

    for _, row in lhb_daily.iterrows():
        seat_name = row[seat_col]
        if seat_name not in score_map:
            continue

        score = score_map[seat_name] / 100
        tier = tier_map[seat_name]
        multiplier = get_seat_multiplier(tier, tier_weights)
        buy_amt = _safe_float(row.get("买入金额", row.get("buy_amt", row.get("buy_amount", 0))))

        # 信号强度 = Alpha评分 * 金额权重 * 等级乘数
        amount_weight = min(buy_amt / 50_000_000, 1.0)  # 5000万封顶
        strength = score * (0.3 + 0.7 * amount_weight) * abs(multiplier)
        strength = min(1.0, strength)

        if multiplier <= 0 or strength < min_signal_strength:
            continue

        signals.append({
            "stock_code": str(row.get("代码", row.get("stock_code", ""))),
            "lhb_date": row.get("上榜日", row.get("lhb_date", "")),
            "seat_name": seat_name,
            "tier": tier,
            "direction": "LONG" if multiplier > 0 else "SHORT",
            "strength": round(strength, 4),
            "confidence": min(1.0, score * 1.5),  # 置信度
            "expected_ret_20d": wret_map.get(seat_name, 0),
            "weighted_win_rate": wr_map.get(seat_name, 0),
            "buy_amount": buy_amt,
            "position_multiplier": multiplier,
        })

    if not signals:
        return pd.DataFrame()

    signals_df = pd.DataFrame(signals)
    signals_df = signals_df.sort_values("strength", ascending=False).reset_index(drop=True)
    return signals_df


def generate_composite_signals(
    signals: pd.DataFrame,
    resonance_bonus_max: float = 0.3,
    min_stocks_for_composite: int = 2,
) -> pd.DataFrame:
    """
    多机构共振信号合成
    同股票多高Alpha机构买入 → 增强信号
    """
    if signals.empty:
        return signals

    long_signals = signals[signals["direction"] == "LONG"]
    composites = []

    for stock, group in long_signals.groupby("stock_code"):
        if len(group) < min_stocks_for_composite:
            continue

        top_seats = group[group["tier"].isin(["S", "A"])]
        if len(top_seats) < 2:
            continue

        avg_strength = top_seats["strength"].mean()
        resonance_bonus = min(0.2 * (len(top_seats) - 1), resonance_bonus_max)

        composites.append({
            "stock_code": stock,
            "lhb_date": group.iloc[0]["lhb_date"],
            "seat_name": f"COMPOSITE({len(top_seats)}seats)",
            "tier": "S+",
            "direction": "LONG",
            "strength": min(1.0, avg_strength + resonance_bonus),
            "confidence": 1.0,
            "expected_ret_20d": top_seats["expected_ret_20d"].mean(),
            "weighted_win_rate": top_seats["weighted_win_rate"].mean(),
            "buy_amount": top_seats["buy_amount"].sum(),
            "position_multiplier": 2.0,
            "participating_seats": ",".join(top_seats["seat_name"].tolist()),
            "resonance_count": len(top_seats),
        })

    if composites:
        composite_df = pd.DataFrame(composites)
        return pd.concat([signals, composite_df], ignore_index=True)

    return signals


def _find_seat_column(df: pd.DataFrame) -> str:
    for col in ["营业部名称", "seat_name"]:
        if col in df.columns:
            return col
    return df.columns[0]


def _safe_float(val) -> float:
    if pd.isna(val):
        return 0.0
    try:
        return float(val)
    except (ValueError, TypeError):
        return 0.0
