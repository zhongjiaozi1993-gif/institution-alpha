"""
机构收益计算器
计算每次龙虎榜买入后N日的绝对收益和相对基准超额收益
"""
from __future__ import annotations
import numpy as np
import pandas as pd
try:
    from loguru import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)


def calculate_future_returns(
    lhb_records: pd.DataFrame,
    price_data: dict[str, pd.DataFrame],
    horizons: list[int] | None = None,
    benchmark_data: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    计算龙虎榜买入后N日收益率

    lhb_records: [stock_code, lhb_date, seat_name, buy_amount]
    price_data: {stock_code: DataFrame[date, close, open]}
    benchmark_data: DataFrame[date, close] 基准指数日线

    Returns: [stock_code, lhb_date, seat_name, ret_1d, ret_5d, ..., excess_20d, ...]
    """
    horizons = horizons or [1, 5, 10, 20, 60]
    records = lhb_records.copy()
    records["lhb_date"] = pd.to_datetime(records["lhb_date"])

    for h in horizons:
        records[f"ret_{h}d"] = np.nan
        records[f"excess_{h}d"] = np.nan

    for idx, row in records.iterrows():
        stock = str(row.get("stock_code", row.get("代码", "")))
        date = row["lhb_date"]
        buy_amt = row.get("buy_amount", row.get("buy_amt", 0))
        if pd.isna(buy_amt) or buy_amt <= 0:
            continue

        if stock not in price_data:
            continue

        prices = price_data[stock].copy()
        prices = prices[prices["date"] >= date].reset_index(drop=True)
        if prices.empty:
            continue

        entry_close = prices.iloc[0]["close"]
        entry_open = prices.iloc[0].get("open", entry_close)

        for h in horizons:
            if h >= len(prices):
                continue
            exit_price = prices.iloc[h]["close"]
            ret = (exit_price - entry_open) / entry_open
            records.at[idx, f"ret_{h}d"] = ret

            if benchmark_data is not None:
                bench = benchmark_data[benchmark_data["date"] >= date]
                if h < len(bench):
                    bench_ret = (bench.iloc[h]["close"] - bench.iloc[0]["close"]) / bench.iloc[0]["close"]
                    records.at[idx, f"excess_{h}d"] = ret - bench_ret

    existing_cols = [f"ret_{h}d" for h in horizons if f"ret_{h}d" in records.columns]
    if existing_cols:
        records = records.dropna(subset=existing_cols, how="all")
    return records


def build_seat_trade_history(
    lhb_df: pd.DataFrame,
    price_data: dict[str, pd.DataFrame],
    horizons: list[int] | None = None,
    benchmark_data: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    从龙虎榜数据构建每个营业部的交易历史+收益
    lhb_df 需包含: 代码, 上榜日, 营业部名称, 买入金额
    """
    horizons = horizons or [1, 5, 10, 20, 60]
    seat_col = None
    for col in ["营业部名称", "seat_name"]:
        if col in lhb_df.columns:
            seat_col = col
            break
    if seat_col is None:
        logger.error("龙虎榜数据缺少营业部名称列")
        return pd.DataFrame()

    records = lhb_df[[seat_col, "代码", "上榜日", "买入金额"]].copy()
    records.columns = ["seat_name", "stock_code", "lhb_date", "buy_amount"]

    return calculate_future_returns(records, price_data, horizons, benchmark_data)


def calculate_batch_returns(
    seat_histories: pd.DataFrame,
    price_data: dict[str, pd.DataFrame],
    horizons: list[int] | None = None,
) -> pd.DataFrame:
    """批量计算多个营业部的买入后收益（优化版）"""
    horizons = horizons or [1, 5, 10, 20, 60]
    results = []

    for seat_name, group in seat_histories.groupby("seat_name"):
        group = group.sort_values("lhb_date")
        returns = calculate_future_returns(group, price_data, horizons)
        returns["seat_name"] = seat_name
        results.append(returns)

    if not results:
        return pd.DataFrame()
    return pd.concat(results, ignore_index=True)
