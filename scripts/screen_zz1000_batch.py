#!/usr/bin/env python3
"""Screen ZZ1000 using Sina batch real-time quotes + cached daily data."""
import re, time, urllib.request
from pathlib import Path
import pandas as pd
import numpy as np

PROJECT = Path(__file__).resolve().parent.parent

# Load ZZ1000 codes
universe = pd.read_csv(PROJECT / "data/processed/stock_universe/index_universe.csv")
zz1000 = universe[universe["index"].str.contains("ZZ1000")].copy()
codes = sorted(zz1000["stock_code"].astype(str).str.zfill(6).tolist())
names = dict(zip(zz1000["stock_code"], zz1000["stock_name"]))

# ============================================================
# 1. ST check
# ============================================================
st_codes = {c for c in codes if "ST" in names.get(c, "").upper()}
print(f"1. ST: {len(st_codes)}")

# ============================================================
# 2. Sina batch real-time quotes (price, turnover, etc.)
# ============================================================
def sina_symbol(code):
    return f"sh{code}" if code.startswith("6") else f"sz{code}"

def fetch_batch_quotes(codes_chunk):
    """Fetch real-time quotes for up to ~100 stocks from Sina."""
    symbols = ",".join(sina_symbol(c) for c in codes_chunk)
    url = f"http://hq.sinajs.cn/list={symbols}"
    req = urllib.request.Request(url, headers={
        "Referer": "http://finance.sina.com.cn",
        "User-Agent": "Mozilla/5.0"
    })
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            raw = resp.read().decode("gbk")
        return raw.split("\n")
    except Exception as e:
        print(f"  Batch fetch error: {e}")
        return []

# Parse Sina quote line
# Fields: name, open, yest_close, price, high, low, volume(shares), amount(yuan), ...
def parse_quote(line):
    m = re.match(r'var hq_str_s[hz](\d+)="(.*)"', line)
    if not m:
        return None
    code = m.group(1)
    fields = m.group(2).split(",")
    if len(fields) < 10 or fields[3] == "":
        return None
    try:
        name = fields[0]
        price = float(fields[3])      # current price (yuan)
        volume = float(fields[7])     # volume (shares) - cumulative today
        amount = float(fields[8])     # amount (yuan) - cumulative today
        yest_close = float(fields[2])
        change = (price / yest_close - 1) * 100 if yest_close > 0 else 0
        return {
            "stock_code": code,
            "stock_name": name,
            "price": price,
            "volume": volume,
            "amount": amount,
            "change_pct": change,
        }
    except (ValueError, IndexError):
        return None

print("Fetching batch quotes...")
all_quotes = {}
for i in range(0, len(codes), 80):
    chunk = codes[i:i+80]
    lines = fetch_batch_quotes(chunk)
    for line in lines:
        q = parse_quote(line)
        if q:
            all_quotes[q["stock_code"]] = q
    print(f"  {i+1}-{min(i+80, len(codes))}/{len(codes)}: got {len(lines)} lines, {len(all_quotes)} valid so far")
    time.sleep(0.5)

print(f"\nTotal quotes fetched: {len(all_quotes)}/{len(codes)}")

# ============================================================
# 3. Check cached daily data for more accurate metrics
# ============================================================
DAILY_DIR = PROJECT / "data/daily"
cached_stats = {}
if DAILY_DIR.exists():
    for f in DAILY_DIR.glob("*.parquet"):
        code = f.stem
        if code not in codes:
            continue
        try:
            df = pd.read_parquet(f)
            if "date" not in df.columns or len(df) < 10:
                continue
            df["date"] = pd.to_datetime(df["date"])
            df = df.sort_values("date").tail(60)
            cached_stats[code] = {
                "avg_amount": df["amount"].mean() if "amount" in df.columns else 0,
                "avg_close": df["close"].mean() if "close" in df.columns else 0,
                "n_days": len(df),
                "first_date": str(df["date"].iloc[0])[:10],
            }
        except Exception:
            pass

print(f"Cached daily data: {len(cached_stats)} stocks")

# ============================================================
# Build combined dataset for screening
# ============================================================
rows = []
missing = []
for code in codes:
    name = names.get(code, "")
    q = all_quotes.get(code)
    c = cached_stats.get(code)

    if q is None and c is None:
        missing.append(code)
        continue

    # Use cached averages when available, otherwise fall back to real-time
    if c and c["avg_amount"] > 0:
        avg_amount = c["avg_amount"]
        avg_close = c["avg_close"]
        n_days = c["n_days"]
        first_date = c["first_date"]
    elif q:
        avg_amount = q["amount"] * 20  # rough: daily ≈ real-time * multiplier
        avg_close = q["price"]
        n_days = 60  # assume full trading
        first_date = "2000-01-01"  # assume old stock
    else:
        missing.append(code)
        continue

    rows.append({
        "stock_code": code,
        "stock_name": name,
        "avg_amount": avg_amount,
        "avg_close": avg_close,
        "n_days": n_days,
        "first_date": first_date,
        "source": "cached" if c else "realtime",
    })

df_all = pd.DataFrame(rows)
print(f"Combined data: {len(df_all)} stocks (cached: {len(cached_stats)}, realtime: {len(df_all)-len(cached_stats)})\n")

# ============================================================
# Apply filters
# ============================================================
elim = {}
all_elim = set(st_codes) | set(missing)

# 2. 次新: first_date > 2025-06 (later than fetch start, truly new listing)
# cached stocks with first_date ~2025-01 are old (just our fetch boundary)
cixin = set(df_all[df_all["first_date"] > "2025-06-01"]["stock_code"])
all_elim |= cixin
elim["2. 次新(<1年)"] = cixin

# 3. 停牌多
tingpai = set(df_all[df_all["n_days"] < 50]["stock_code"])
all_elim |= tingpai
elim["3. 停牌多(<50/60天)"] = tingpai

# 4. 日均成交额 < 1亿
low_amt = set(df_all[df_all["avg_amount"] < 1e8]["stock_code"])
all_elim |= low_amt
elim["4. 日均成交额<1亿"] = low_amt

# 5. 流动性后30%
pct = df_all["avg_amount"].rank(pct=True)
low_liq = set(df_all[pct <= 0.30]["stock_code"])
all_elim |= low_liq
elim["5. 流动性后30%"] = low_liq

# 6. 低价股
low_price = set(df_all[df_all["avg_close"] < 3.0]["stock_code"])
all_elim |= low_price
elim["6. 低价股(<3元)"] = low_price

# 7. Missing
elim["7. 数据缺失"] = set(missing)

# ============================================================
# Report
# ============================================================
print(f"{'='*60}")
print("PER-FILTER BREAKDOWN (may overlap)")
print(f"{'='*60}")
for reason in sorted(elim.keys()):
    print(f"  {reason:30s} {len(elim[reason]):4d}")

# Non-overlapping sequential
print(f"\n{'='*60}")
print("NON-OVERLAPPING (sequential order)")
print(f"{'='*60}")
seen = set(st_codes) | set(missing)
print(f"  {'0. ST':30s} {len(st_codes):4d}")
print(f"  {'7. 数据缺失':30s} {len(missing):4d}")
for reason in sorted([k for k in elim.keys() if k not in ("7. 数据缺失",)]):
    new = elim[reason] - seen
    seen |= elim[reason]
    print(f"  {reason:30s} {len(elim[reason]):4d} total, {len(new):4d} new unique")

remaining = set(codes) - seen
zz500 = 500
new_total = zz500 + len(remaining)
est_gb = new_total * 5.6 / 1024

print(f"\n{'='*60}")
print("SUMMARY")
print(f"{'='*60}")
print(f"  ZZ1000: {len(codes)} -> {len(remaining)} ({len(seen)} eliminated)")
print(f"  ZZ500 (kept all): {zz500}")
print(f"  New universe: {new_total}")
print(f"  Est daily CSV: {est_gb:.1f} GB (from 8.4 GB, saves {8.4-est_gb:.1f} GB/day)")
print(f"  Monthly net: ~{est_gb * 21:.0f} GB (21 trading days)")
