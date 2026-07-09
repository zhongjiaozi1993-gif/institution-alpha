#!/usr/bin/env python3
"""
Screen ZZ1000 stocks: ST, 次新, 停牌, 低成交额, 低流动性, 低价股, L2缺失.

Outputs:
  data/processed/stock_universe/zz1000_liquid_selected.csv
  data/processed/stock_universe/zz1000_liquid_selected.txt
  data/processed/stock_universe/zz1000_liquid_dropped.csv
  data/processed/stock_universe/zz1000_liquid_screen_report.md
"""
import sys, time
from pathlib import Path
import pandas as pd
import numpy as np
import akshare as ak

PROJECT = Path(__file__).resolve().parent.parent
DAILY_DIR = PROJECT / "data/daily"
UNIVERSE_DIR = PROJECT / "data/processed/stock_universe"
DAILY_DIR.mkdir(parents=True, exist_ok=True)
UNIVERSE_DIR.mkdir(parents=True, exist_ok=True)

LOOKBACK = 60  # trading days (~3 months)

def symbol(code):
    return f"sh{code}" if code.startswith("6") else f"sz{code}"

# Load ZZ1000 codes + names
universe = pd.read_csv(UNIVERSE_DIR / "index_universe.csv")
zz1000 = universe[universe["index"].str.contains("ZZ1000")].copy()
# Ensure zero-filled 6-digit codes
zz1000["stock_code"] = zz1000["stock_code"].astype(str).str.zfill(6)
codes = sorted(zz1000["stock_code"].tolist())
names = dict(zip(zz1000["stock_code"], zz1000["stock_name"]))
print(f"ZZ1000: {len(codes)} stocks")

# ============================================================
# 1. ST check (from name)
# ============================================================
st_codes = {c for c in codes if "ST" in names.get(c, "").upper()}
print(f"ST: {len(st_codes)}")

# ============================================================
# 2. Fetch daily data (Sina, cached)
# ============================================================
results = []
errors = []

for i, code in enumerate(codes):
    cache_file = DAILY_DIR / f"{code}.parquet"
    df = None

    if cache_file.exists():
        try:
            df = pd.read_parquet(cache_file)
            if "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"])
        except Exception:
            pass

    if df is None or df.empty:
        try:
            df = ak.stock_zh_a_daily(
                symbol=symbol(code),
                start_date="20250101",
                end_date="20260707",
                adjust="qfq"
            )
            if df is not None and not df.empty:
                df["date"] = pd.to_datetime(df["date"])
                df.to_parquet(cache_file, index=False)
        except Exception as e:
            errors.append((code, str(e)[:80]))
        time.sleep(0.5)

    has_daily_error = False
    if df is None or df.empty:
        has_daily_error = True
        errors.append((code, "no data"))

    if has_daily_error:
        results.append({
            "stock_code": code,
            "stock_name": names.get(code, ""),
            "avg_amount_yuan": np.nan,
            "avg_close_yuan": np.nan,
            "n_trade_days": 0,
            "first_date": "",
            "is_st": code in st_codes,
            "is_cixin": False,
            "is_low_turnover": False,
            "is_low_liquidity": False,
            "is_low_price": False,
            "has_daily_error": True,
            "has_level2_error": "Unknown",
        })
        continue

    # first_date from FULL history (earliest date), not just tail
    df_sorted = df.sort_values("date")
    first_date = str(df_sorted["date"].iloc[0])[:10]

    # Last LOOKBACK days for liquidity metrics
    df_recent = df_sorted.tail(LOOKBACK)
    if len(df_recent) < 5:
        errors.append((code, f"only {len(df_recent)} days"))
        has_daily_error = True

    avg_amount = df_recent["amount"].mean()
    avg_close = df_recent["close"].mean()
    n_days = len(df_recent)

    results.append({
        "stock_code": code,
        "stock_name": names.get(code, ""),
        "avg_amount_yuan": avg_amount,
        "avg_close_yuan": avg_close,
        "n_trade_days": n_days,
        "first_date": first_date,
        "is_st": code in st_codes,
        "is_cixin": False,       # filled below
        "is_low_turnover": False, # filled below
        "is_low_liquidity": False, # filled below
        "is_low_price": False,    # filled below
        "has_daily_error": has_daily_error,
        "has_level2_error": "Unknown",
    })

    if (i + 1) % 100 == 0:
        print(f"  [{i+1}/{len(codes)}] {len(results)} ok, {len(errors)} err")

print(f"Fetched: {len(results)} stocks, errors: {len(errors)}")

df_all = pd.DataFrame(results)

# ============================================================
# Apply filters and set flags
# ============================================================

# is_cixin: first date >= 2025-07 (~1 year or less since listing)
cixin_mask = df_all["first_date"] >= "2025-07-01"
df_all.loc[cixin_mask, "is_cixin"] = True

# is_low_turnover: < 50 trading days out of ~60
tingpai_mask = df_all["n_trade_days"] < 50
df_all.loc[tingpai_mask, "is_low_turnover"] = True

# is_low_liquidity: avg amount < 1亿
low_amt_mask = df_all["avg_amount_yuan"] < 1e8
df_all.loc[low_amt_mask, "is_low_liquidity"] = True

# Also liquidity bottom 30%
pct = df_all["avg_amount_yuan"].rank(pct=True)
low_liq_mask = pct <= 0.30
df_all.loc[low_liq_mask, "is_low_liquidity"] = True

# is_low_price: avg close < 3 yuan
low_price_mask = df_all["avg_close_yuan"] < 3.0
df_all.loc[low_price_mask, "is_low_price"] = True

# ============================================================
# Determine selected_flag and exclude_reason (sequential, non-overlapping)
# ============================================================
def determine_exclusion(row):
    reasons = []
    if row["is_st"]:
        reasons.append("ST")
    if row["has_daily_error"]:
        reasons.append("日线数据异常")
    if row["is_cixin"]:
        reasons.append("次新(上市<1年)")
    if row["is_low_turnover"]:
        reasons.append("停牌多(<50/60天)")
    if row["avg_amount_yuan"] < 1e8 and not pd.isna(row["avg_amount_yuan"]):
        reasons.append("日均成交额<1亿")
    if row["is_low_price"]:
        reasons.append("低价股(<3元)")
    # Note: low_liquidity (bottom 30%) is entangled with <1亿, pick first
    # For sequential exclusion, add bottom-30%-only (after removing <1亿)
    return reasons

df_all["exclude_reasons"] = df_all.apply(
    lambda r: "; ".join(determine_exclusion(r)), axis=1
)
df_all["selected_flag"] = df_all["exclude_reasons"] == ""

# Count selected/dropped
selected = df_all[df_all["selected_flag"]].copy()
dropped = df_all[~df_all["selected_flag"]].copy()

zz500_count = 500
new_total = zz500_count + len(selected)

print(f"\n{'='*60}")
print(f"SUMMARY")
print(f"{'='*60}")
print(f"  ZZ1000 screened: {len(codes)} -> {len(selected)} selected, {len(dropped)} dropped")
print(f"  ZZ500 (unscreened): {zz500_count}")
print(f"  New universe: {new_total} stocks")

# ============================================================
# Output files
# ============================================================
OUT_COLS = [
    "stock_code", "stock_name", "selected_flag", "exclude_reasons",
    "avg_amount_yuan", "avg_close_yuan", "n_trade_days", "first_date",
    "is_st", "is_cixin", "is_low_turnover", "is_low_liquidity", "is_low_price",
    "has_daily_error", "has_level2_error",
]

# zz1000_liquid_selected.csv
sel_path = UNIVERSE_DIR / "zz1000_liquid_selected.csv"
selected[OUT_COLS].to_csv(sel_path, index=False)
print(f"\n  Selected: {sel_path} ({len(selected)} stocks)")

# zz1000_liquid_selected.txt
txt_path = UNIVERSE_DIR / "zz1000_liquid_selected.txt"
with open(txt_path, "w") as f:
    for c in sorted(selected["stock_code"]):
        f.write(f"{c}\n")
print(f"  Selected TXT: {txt_path}")

# zz1000_liquid_dropped.csv
drop_path = UNIVERSE_DIR / "zz1000_liquid_dropped.csv"
dropped[OUT_COLS].to_csv(drop_path, index=False)
print(f"  Dropped: {drop_path} ({len(dropped)} stocks)")

# Reason distribution
reason_counts = dropped["exclude_reasons"].value_counts()
print(f"\n  Dropped reasons:")
for r, c in reason_counts.items():
    print(f"    {r}: {c}")

# ============================================================
# Report
# ============================================================
report_path = UNIVERSE_DIR / "zz1000_liquid_screen_report.md"
with open(report_path, "w") as f:
    f.write("# ZZ1000 流动性筛选报告\n\n")
    f.write(f"生成时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}\n\n")

    f.write("## 筛选结果\n\n")
    f.write(f"| 指标 | 数值 |\n")
    f.write(f"|------|------|\n")
    f.write(f"| ZZ1000 原始数量 | {len(codes)} |\n")
    f.write(f"| 通过筛选 | {len(selected)} |\n")
    f.write(f"| 被剔除 | {len(dropped)} |\n")
    f.write(f"| ZZ500（未筛选） | {zz500_count} |\n")
    f.write(f"| 合并 Universe | {new_total} |\n\n")

    f.write("## 剔除原因分布\n\n")
    f.write("| 原因 | 数量 |\n")
    f.write("|------|------|\n")
    for r, c in reason_counts.items():
        f.write(f"| {r} | {c} |\n")

    f.write("\n## 筛选规则\n\n")
    f.write("1. 剔除 ST（从股票名称判断）\n")
    f.write("2. 剔除次新：first_date >= 2025-07（近似上市<1年）\n")
    f.write("3. 剔除停牌多：近60个交易日不足50天\n")
    f.write("4. 剔除日均成交额 < 1亿\n")
    f.write("5. 剔除流动性后30%\n")
    f.write("6. 剔除低价股 < 3元\n")
    f.write("7. 剔除日线数据异常\n")
    f.write("8. Level-2 缺失判断：预留字段，当前值均为 Unknown\n\n")

    f.write("## 注意事项\n\n")
    f.write("- **first_date 非真实上市日**：first_date 来自日线数据（Sina 前复权）的起始日期，")
    f.write("仅反映数据覆盖范围，不等于股票真实上市日期。后续可接入 `list_date` 替代。\n")
    f.write("- **次新判断为近似口径**：基于 first_date >= 2025-07-01 近似判断上市不足1年，")
    f.write("可能存在误判（数据起始日晚于实际上市日）。\n")
    f.write("- **has_level2_error 当前均为 Unknown**：Level-2 数据覆盖情况需后续从 ")
    f.write("`data/single_stock/{code}/raw/` 目录扫描后填入。\n")
    f.write("- **筛选为顺序非重叠**：按上述规则顺序逐一剔除，每个股票只归入第一个命中原因。\n")

    f.write("\n## 抽样检查\n\n")

    if len(selected) > 0:
        f.write("### 通过股票样本（前10只）\n\n")
        f.write("| 代码 | 名称 | 日均成交额(元) | 均价(元) | 交易天数 | first_date |\n")
        f.write("|------|------|---------------|----------|----------|------------|\n")
        for _, r in selected.head(10).iterrows():
            f.write(f"| {r['stock_code']} | {r['stock_name']} | "
                    f"{r['avg_amount_yuan']:,.0f} | {r['avg_close_yuan']:.2f} | "
                    f"{int(r['n_trade_days'])} | {r['first_date']} |\n")

    if len(dropped) > 0:
        f.write("\n### 被剔除股票样本（按原因各取3只）\n\n")
        for reason, grp in dropped.groupby("exclude_reasons"):
            f.write(f"**{reason}**:\n")
            for _, r in grp.head(3).iterrows():
                f.write(f"- {r['stock_code']} {r['stock_name']}\n")

print(f"  Report: {report_path}")
print(f"\nDone.")
