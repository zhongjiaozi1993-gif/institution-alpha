"""Local market context for evidence-chain reports."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


PRICE_COLUMNS = {
    "日期": "date",
    "开盘": "open",
    "收盘": "close",
    "最高": "high",
    "最低": "low",
    "成交量": "volume",
    "成交额": "amount",
    "涨跌幅": "pct_chg",
    "换手率": "turnover",
}


def load_price(single_stock_dir: Path) -> pd.DataFrame:
    path = single_stock_dir / "price_daily.csv"
    if not path.exists():
        raise FileNotFoundError(f"price file not found: {path}")

    price = pd.read_csv(path)
    missing = [c for c in ["日期", "开盘", "收盘"] if c not in price.columns]
    if missing:
        raise ValueError(f"price file missing columns: {missing}")

    price = price.rename(columns={k: v for k, v in PRICE_COLUMNS.items() if k in price.columns})
    price["date_ts"] = pd.to_datetime(price["date"])
    price["date"] = price["date_ts"].dt.strftime("%Y%m%d")
    price = price.sort_values("date_ts").reset_index(drop=True)
    for col in ["open", "close", "high", "low", "volume", "amount", "pct_chg", "turnover"]:
        if col in price.columns:
            price[col] = pd.to_numeric(price[col], errors="coerce")
    return price


def attach_market_context(daily: pd.DataFrame, price: pd.DataFrame) -> pd.DataFrame:
    """Attach price action and forward returns to daily behavior evidence."""
    merged = daily.merge(
        price[["date", "open", "close", "high", "low", "amount", "pct_chg", "turnover"]],
        on="date",
        how="left",
    )

    price_by_date = price.set_index("date")
    close = price["close"].to_numpy()
    open_ = price["open"].to_numpy()
    date_to_i = {d: i for i, d in enumerate(price["date"])}

    for horizon in [1, 3, 5, 10, 20]:
        close_ret = []
        t1_open_ret = []
        for date in merged["date"]:
            i = date_to_i.get(date)
            if i is None or i + horizon >= len(close):
                close_ret.append(np.nan)
            else:
                close_ret.append((close[i + horizon] / close[i] - 1) * 100)

            if i is None or i + 1 >= len(open_) or i + horizon >= len(close):
                t1_open_ret.append(np.nan)
            else:
                t1_open_ret.append((close[i + horizon] / open_[i + 1] - 1) * 100)

        merged[f"fwd_{horizon}d_close_pct"] = close_ret
        merged[f"fwd_{horizon}d_t1open_pct"] = t1_open_ret

    merged["price_unit_warning"] = _detect_price_unit_warning(merged)
    return merged


def _detect_price_unit_warning(frame: pd.DataFrame) -> str:
    """Flag the known adjusted-price vs Level-2 price mismatch."""
    valid = frame.dropna(subset=["close", "max_op_price"])
    if valid.empty:
        return ""
    ratio = (valid["close"] / valid["max_op_price"].replace(0, np.nan)).median()
    if pd.notna(ratio) and (ratio >= 3 or ratio <= 0.33):
        return f"日线价格与Level-2价格疑似非同口径，median(close/l2_price)={ratio:.2f}"
    return ""
