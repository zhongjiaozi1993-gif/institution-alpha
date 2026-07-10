"""可交易标记生成器（每日每股）。

生成的 flag（状态描述 AT 日期 T）:
    suspend_flag      停牌（volume==0/amount==0）
    limit_up_flag     涨停（按板块涨跌幅上限，含容忍度）
    limit_down_flag   跌停
    st_flag           ST（无名单数据源 → 恒 False，见 project_audit §9）
    new_stock_flag    新股（首个交易日后不足 60 个交易日）
    low_liquidity_flag 低流动性（近 20 日均额 < 阈值）
    buyable_flag      可买 = 非停牌 且 非涨停
    sellable_flag     可卖 = 非停牌 且 非跌停
    tradable_flag     纳池 = 非停牌 且 非ST 且 非新股 且 非低流动性

口径:
- 涨跌幅用 hfq close 的 pct_change 近似（除权日略有偏差，加容忍度）。
- 板块涨跌幅上限: 创业板(300)/科创板(688)=20%, 北交所(8/4)=30%, 其他=10%。ST(5%) 无法区分。
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parent.parent.parent
DAILY_DIR = PROJECT / "data" / "daily"

LIMIT_TOL = 0.003            # 涨跌停容忍度
NEW_STOCK_DAYS = 60          # 新股交易日阈值
LIQ_WINDOW = 20              # 流动性回看窗口
MIN_MEDIAN_AMOUNT = 10_000_000  # 近 20 日均额阈值(元)
LISTED_BEFORE_TOL_DAYS = 10  # 首交易日在窗口起点 +N 日内 → 视为窗口前已上市


def _price_limit(code: str) -> float:
    code = str(code).zfill(6)
    if code.startswith(("300", "688")):
        return 0.20
    if code.startswith(("8", "4")):
        return 0.30
    return 0.10


def build_flags(
    codes: list[str], start_date: str, end_date: str,
) -> pd.DataFrame:
    """为给定股票池构建每日 tradable flags。"""
    start_ts = pd.Timestamp(start_date)
    frames = []
    for code in codes:
        code = str(code).zfill(6)
        p = DAILY_DIR / f"{code}.parquet"
        if not p.exists():
            continue
        df = pd.read_parquet(p)
        df["date"] = pd.to_datetime(df["date"])
        df = df[(df["date"] >= start_date) & (df["date"] <= end_date)].sort_values("date").reset_index(drop=True)
        if df.empty:
            continue

        limit = _price_limit(code)
        prev_close = df["close"].shift(1)
        pct = df["close"] / prev_close - 1.0
        vol = df["volume"].fillna(0)

        suspend = (vol <= 0) | (df["amount"].fillna(0) <= 0)
        limit_up = (pct >= limit - LIMIT_TOL) & (~suspend)
        limit_down = (pct <= -(limit - LIMIT_TOL)) & (~suspend)

        # 新股: 仅当首交易日晚于窗口起点（窗口内 IPO），其首 60 个交易日为新股
        traded = df[~suspend]
        first_trade = traded["date"].min() if len(traded) else df["date"].min()
        if first_trade > start_ts + pd.Timedelta(days=LISTED_BEFORE_TOL_DAYS):
            trade_rank = (~suspend).cumsum()  # 第几个交易日
            new_stock = trade_rank <= NEW_STOCK_DAYS
        else:
            new_stock = pd.Series(False, index=df.index)

        # 低流动性: 近 20 日均额 < 阈值
        avg_amt = df["amount"].rolling(LIQ_WINDOW, min_periods=5).mean()
        low_liq = avg_amt < MIN_MEDIAN_AMOUNT

        st = pd.Series(False, index=df.index)  # 无 ST 数据源

        suspend = suspend.fillna(False)
        buyable = (~suspend) & (~limit_up)
        sellable = (~suspend) & (~limit_down)
        tradable = (~suspend) & (~st) & (~new_stock) & (~low_liq)

        frames.append(pd.DataFrame({
            "trade_date": df["date"].to_numpy(),
            "symbol": code,
            "suspend_flag": suspend.to_numpy(bool),
            "limit_up_flag": limit_up.fillna(False).to_numpy(bool),
            "limit_down_flag": limit_down.fillna(False).to_numpy(bool),
            "st_flag": st.to_numpy(bool),
            "new_stock_flag": new_stock.to_numpy(bool),
            "low_liquidity_flag": low_liq.fillna(True).to_numpy(bool),
            "buyable_flag": buyable.fillna(False).to_numpy(bool),
            "sellable_flag": sellable.fillna(False).to_numpy(bool),
            "tradable_flag": tradable.fillna(False).to_numpy(bool),
        }))
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)
