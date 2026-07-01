"""
价格口径统一验证脚本

Alpha验证的头号阻塞项：price_daily.csv (HFQ后复权) 与 Level-2 (实际成交价)
不在同一价格口径。脚本对每只股票逐日比较两者 OHLC，输出:
- price_unit_report.csv: 逐日明细 + 转换因子
- 股票级摘要: ratio稳定性、除权日检测、是否可安全计算收益率

Level-2 价格单位: 元×10000 → /10000 = 元
price_daily.csv: Sina HFQ后复权 (单位元)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT / "data" / "single_stock"
PRICE_SCALE = 10000


def l2_daily_ohlc(stock: str, date_str: str) -> dict | None:
    cj_path = DATA_DIR / stock / "raw" / date_str / f"{stock}.SZ" / "逐笔成交.csv"
    if not cj_path.exists():
        return None
    for enc in ["gb18030", "gbk", "utf-8"]:
        try:
            df = pd.read_csv(cj_path, encoding=enc, low_memory=False)
            break
        except (UnicodeDecodeError, UnicodeError):
            continue
    else:
        return None
    if "成交代码" in df.columns:
        df = df[df["成交代码"] != "C"]
    if df.empty:
        return None

    df["price_yuan"] = df["成交价格"].astype(float) / PRICE_SCALE
    df = df.sort_values("时间")
    return {
        "open": float(df["price_yuan"].iloc[0]),
        "high": float(df["price_yuan"].max()),
        "low": float(df["price_yuan"].min()),
        "close": float(df["price_yuan"].iloc[-1]),
    }


def validate_stock(stock: str) -> tuple[pd.DataFrame, dict]:
    stock_dir = DATA_DIR / stock
    raw_dir = stock_dir / "raw"
    price_path = stock_dir / "price_daily.csv"

    if not price_path.exists():
        return pd.DataFrame(), {}

    price = pd.read_csv(price_path)
    price = price.rename(columns={"日期": "date", "开盘": "open_d", "最高": "high_d",
                                   "最低": "low_d", "收盘": "close_d"})
    price["date"] = price["date"].astype(str).str.replace("-", "")
    price = price.set_index("date")

    rows = []
    for day_dir in sorted(raw_dir.iterdir()):
        if not day_dir.is_dir():
            continue
        date_str = day_dir.name
        if len(date_str) != 8:
            continue

        ohlc = l2_daily_ohlc(stock, date_str)
        if ohlc is None:
            continue
        if date_str not in price.index:
            continue

        dp = price.loc[date_str]
        # Use close/close ratio as the primary conversion factor
        ratio = float(dp["close_d"] / ohlc["close"]) if ohlc["close"] else np.nan
        # OHLC agreement: std of 4 ratios should be small
        ratios_4 = [
            dp["open_d"] / ohlc["open"] if ohlc["open"] else np.nan,
            dp["high_d"] / ohlc["high"] if ohlc["high"] else np.nan,
            dp["low_d"] / ohlc["low"] if ohlc["low"] else np.nan,
            ratio,
        ]
        ratio_vals = [v for v in ratios_4 if not np.isnan(v)]
        ratio_std = float(np.std(ratio_vals))

        rows.append({
            "stock": stock,
            "date": date_str,
            "l2_open": round(ohlc["open"], 3),
            "l2_high": round(ohlc["high"], 3),
            "l2_low": round(ohlc["low"], 3),
            "l2_close": round(ohlc["close"], 3),
            "daily_open": round(float(dp["open_d"]), 2),
            "daily_high": round(float(dp["high_d"]), 2),
            "daily_low": round(float(dp["low_d"]), 2),
            "daily_close": round(float(dp["close_d"]), 2),
            "ratio": round(ratio, 4),
            "ratio_ohlc_std": round(ratio_std, 4),
        })

    df = pd.DataFrame(rows)
    if df.empty:
        return df, {}

    # Stock-level summary
    ratios = df["ratio"].dropna()
    median_ratio = float(ratios.median())
    cv = float(ratios.std() / ratios.mean()) if ratios.mean() else 0

    # Detect real corporate action jumps: ratio changes by >5% day-over-day
    df_sorted = df.sort_values("date").copy()
    df_sorted["ratio_pct_change"] = df_sorted["ratio"].pct_change().abs()
    jumps = df_sorted[df_sorted["ratio_pct_change"] > 0.02]  # >2% day-over-day
    jump_dates = jumps["date"].tolist()

    # Verdict
    if cv < 0.01:
        stability = "极稳定(CV<1%)"
        can_compute_returns = True
        note = "各自内部收益率计算正确; 跨L2/日线混算时乘以{:.3f}转换".format(median_ratio)
    elif cv < 0.03:
        stability = "较稳定(CV<3%)"
        can_compute_returns = True
        note = "推荐用L2价格算收益率; 或日线价格除以ratio转L2口径"
    else:
        stability = f"不稳定(CV={cv:.1%})"
        can_compute_returns = False
        note = "不可混算收益率; 需先统一价格口径"

    summary = {
        "stock": stock,
        "n_days": len(df),
        "median_ratio": round(median_ratio, 3),
        "ratio_cv": round(cv, 4),
        "ratio_min": round(float(ratios.min()), 3),
        "ratio_max": round(float(ratios.max()), 3),
        "stability": stability,
        "jump_dates": jump_dates,
        "can_compute_returns": can_compute_returns,
        "note": note,
    }

    return df, summary


def main():
    stocks = [d.name for d in DATA_DIR.iterdir()
              if d.is_dir() and (d / "raw").exists() and (d / "price_daily.csv").exists()]

    if not stocks:
        print("未找到同时有 raw/ 和 price_daily.csv 的股票目录")
        sys.exit(1)

    print(f"验证 {len(stocks)} 只股票\n")

    all_dfs = []
    summaries = []
    for stock in stocks:
        df, s = validate_stock(stock)
        if s:
            all_dfs.append(df)
            summaries.append(s)

            status = "✓ 可计算收益" if s["can_compute_returns"] else "✗ 需先统一口径"
            print(f"  {stock}: ratio={s['median_ratio']:.3f}, CV={s['ratio_cv']:.4f}, "
                  f"{s['stability']}, {s['n_days']}天 — {status}")
            if s["jump_dates"]:
                print(f"    除权跳变日 ({len(s['jump_dates'])}天): {', '.join(s['jump_dates'][:10])}"
                      + ("..." if len(s['jump_dates']) > 10 else ""))
            print(f"    → {s['note']}")
        else:
            print(f"  {stock}: 无数据")

    if all_dfs:
        report = pd.concat(all_dfs, ignore_index=True)
        out_path = PROJECT / "data" / "processed" / "price_unit_report.csv"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        report.to_csv(out_path, index=False)

        summary_df = pd.DataFrame(summaries)
        summary_path = PROJECT / "data" / "processed" / "price_unit_summary.csv"
        summary_df.to_csv(summary_path, index=False)

        print(f"\n明细: {out_path} ({len(report)} 行)")
        print(f"摘要: {summary_path}")


if __name__ == "__main__":
    main()
