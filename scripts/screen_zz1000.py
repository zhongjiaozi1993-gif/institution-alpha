#!/usr/bin/env python3
"""Screen ZZ1000 stocks: ST, 次新, 停牌, 低成交额, 低流动性, 低价股, L2缺失."""
import sys, time
from pathlib import Path
import pandas as pd
import numpy as np
import akshare as ak

PROJECT = Path(__file__).resolve().parent.parent
DAILY_DIR = PROJECT / "data/daily"
DAILY_DIR.mkdir(parents=True, exist_ok=True)

# Load ZZ1000 codes + names
universe = pd.read_csv(PROJECT / "data/processed/stock_universe/index_universe.csv")
zz1000 = universe[universe["index"].str.contains("ZZ1000")].copy()
codes = sorted(zz1000["stock_code"].tolist())
names = dict(zip(zz1000["stock_code"], zz1000["stock_name"]))
print(f"ZZ1000: {len(codes)} stocks")
print(f"ZZ500:  500 stocks (not screened)")
print(f"Total universe before screening: {500 + len(codes)}")

# ============================================================
# 1. ST check (from name)
# ============================================================
st_codes = {c for c in codes if "ST" in names.get(c, "").upper()}
print(f"\n1. ST: {len(st_codes)}")

# ============================================================
# 2-6. Fetch daily data (Sina, cached)
# ============================================================
LOOKBACK = 60  # trading days (~3 months)

def symbol(code):
    return f"sh{code}" if code.startswith("6") else f"sz{code}"

results = []
errors = []

for i, code in enumerate(codes):
    cache_file = DAILY_DIR / f"{code}.parquet"
    df = None

    # Read cache
    if cache_file.exists():
        try:
            df = pd.read_parquet(cache_file)
            if "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"])
        except Exception:
            pass

    # Fetch if needed
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

        time.sleep(0.5)  # Rate limit

    if df is None or df.empty:
        errors.append((code, "no data"))
        continue

    # Last LOOKBACK trading days
    df = df.sort_values("date").tail(LOOKBACK)
    if len(df) < 5:
        errors.append((code, f"only {len(df)} days"))
        continue

    avg_amount = df["amount"].mean()  # yuan
    avg_close = df["close"].mean()    # yuan
    n_days = len(df)
    first_date = str(df["date"].iloc[0])[:10]

    results.append({
        "stock_code": code,
        "stock_name": names.get(code, ""),
        "avg_amount_yuan": avg_amount,
        "avg_close_yuan": avg_close,
        "n_trade_days": n_days,
        "first_date": first_date,
    })

    if (i + 1) % 100 == 0:
        print(f"  [{i+1}/{len(codes)}] fetched {len(results)} ok, {len(errors)} err")

print(f"\nFetched: {len(results)} stocks, errors: {len(errors)}")

if len(results) < 500:
    print("ERROR: too few stocks, aborting")
    sys.exit(1)

df_all = pd.DataFrame(results)
df_all["avg_amount_wan"] = df_all["avg_amount_yuan"] / 1e4  # for display

# ============================================================
# Apply filters
# ============================================================
elim = {}  # reason -> set of codes

# 2. 次新: first date >= 2025-07 (listed ~1 year or less)
cixin = set(df_all[df_all["first_date"] >= "2025-07-01"]["stock_code"])
elim["2. 次新(上市<1年)"] = cixin

# 3. 停牌多: < 50 trading days out of ~60
tingpai = set(df_all[df_all["n_trade_days"] < 50]["stock_code"])
elim["3. 停牌多(<50/60天)"] = tingpai

# 4. 日均成交额 < 1亿 (100,000,000 yuan)
low_amt = set(df_all[df_all["avg_amount_yuan"] < 1e8]["stock_code"])
elim["4. 日均成交额<1亿"] = low_amt

# 5. 流动性后30% (by avg turnover amount)
pct = df_all["avg_amount_yuan"].rank(pct=True)
low_liq = set(df_all[pct <= 0.30]["stock_code"])
elim["5. 流动性后30%"] = low_liq

# 6. 低价股: avg close < 3 yuan
low_price = set(df_all[df_all["avg_close_yuan"] < 3.0]["stock_code"])
elim["6. 低价股(<3元)"] = low_price

# ============================================================
# Report: per-filter, non-overlapping
# ============================================================
print(f"\n{'='*60}")
print("PER-FILTER BREAKDOWN (sequential, non-overlapping)")
print(f"{'='*60}")

all_elim = set(st_codes) | {c for c, _ in errors}
for reason in sorted(elim.keys()):
    c = elim[reason]
    new = c - all_elim
    all_elim |= c
    print(f"  {reason:30s} {len(c):4d} total, {len(new):4d} new unique")

print(f"  {'1. ST':30s} {len(st_codes):4d} total")
print(f"  {'7. L2缺失/数据异常':30s} {len(errors):4d} total")

# ============================================================
# Summary
# ============================================================
remaining = set(codes) - all_elim
zz500_count = 500
new_total = zz500_count + len(remaining)
est_daily_gb = new_total * 5.6 / 1024

print(f"\n{'='*60}")
print("SUMMARY")
print(f"{'='*60}")
print(f"  ZZ1000 screening: {len(codes)} -> {len(remaining)} ({len(all_elim)} eliminated)")
print(f"  ZZ500 (unscreened): {zz500_count}")
print(f"  New universe: {new_total} stocks")
print(f"  Est daily CSV: {est_daily_gb:.1f} GB (was 8.4 GB, saves {8.4-est_daily_gb:.1f} GB/day)")

# Show some eliminated examples
print(f"\n  Sample eliminated stocks:")
for reason in sorted(elim.keys()):
    c = list(elim[reason])[:3]
    print(f"  {reason}: {', '.join(c)}")
if st_codes:
    print(f"  ST: {', '.join(sorted(st_codes)[:5])}")
