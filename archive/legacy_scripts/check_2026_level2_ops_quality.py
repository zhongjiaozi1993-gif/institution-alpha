"""
Check 2026 level2_ops output quality before OOT backtest.

Checks:
  1. File count vs expected trading days
  2. Month coverage
  3. Trading day coverage (dates)
  4. Empty files
  5. Field consistency with 2025 level2_ops
  6. stock_code format (6-digit + suffix)
  7. date format
  8. Key fields missing rate (price, amount, order_count, etc.)
  9. Duplicate records
  10. Daily ops count distribution
  11. Date matching with 2026 price data

Usage:
  python scripts/check_2026_level2_ops_quality.py \
    --ops-dir data/processed/level2_ops/2026 \
    --ref-dir data/processed/level2_ops/2025 \
    --price-dir data/daily \
    --selected-stocks data/processed/stock_universe/selected_stocks.csv

Output:
  data/processed/level2_ops/2026/quality_report_2026.csv
  data/processed/level2_ops/2026/quality_report_2026.txt
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

# Expected fields from 2025 level2_ops
EXPECTED_COLUMNS = [
    "cluster_id", "direction", "total_amount_wan", "avg_price",
    "order_count", "time_span_min", "start_time", "end_time",
    "buy_volume_wan", "price_min", "price_max", "vwap_deviation_pct",
    "avg_order_size_wan", "median_order_qty", "qty_cv", "mid_time_sec",
    "order_interval_std", "order_hhi", "date", "stock_code",
    "matched_orders", "match_key",
]

STOCK_CODE_PATTERN = re.compile(r"^\d{6}\.(SZ|SH|BJ)$")

FLOAT_COLS = [
    "total_amount_wan", "avg_price", "time_span_min", "buy_volume_wan",
    "price_min", "price_max", "vwap_deviation_pct", "avg_order_size_wan",
    "median_order_qty", "qty_cv", "order_interval_std", "order_hhi",
]

INT_COLS = [
    "cluster_id", "order_count", "start_time", "end_time",
    "mid_time_sec", "matched_orders",
]

STR_COLS = ["direction", "date", "stock_code", "match_key"]


def check_file(path: Path, expected_cols: list[str]) -> dict:
    """Check a single level2_ops file, return quality metrics."""
    result = {
        "file": path.name,
        "size_mb": round(path.stat().st_size / 1_048_576, 2),
        "n_rows": 0,
        "n_stocks": 0,
        "date": "",
        "n_missing_fields": 0,
        "missing_fields": "",
        "extra_fields": "",
        "n_bad_stock_code": 0,
        "n_bad_date": 0,
        "n_duplicates": 0,
        "n_missing_key_fields": 0,
        "status": "ok",
    }

    try:
        df = pd.read_csv(path, low_memory=False)
    except Exception as e:
        result["status"] = f"read_error: {e}"
        return result

    if df.empty:
        result["status"] = "empty"
        return result

    result["n_rows"] = len(df)

    # -- date extraction --
    if "date" in df.columns:
        dates = df["date"].dropna().unique()
        result["date"] = str(dates[0]) if len(dates) > 0 else "missing"

    # -- stock_code count --
    if "stock_code" in df.columns:
        result["n_stocks"] = int(df["stock_code"].nunique())

    # -- field consistency --
    actual_cols = list(df.columns)
    missing = [c for c in expected_cols if c not in actual_cols]
    extra = [c for c in actual_cols if c not in expected_cols]
    result["n_missing_fields"] = len(missing)
    result["missing_fields"] = ",".join(missing)
    result["extra_fields"] = ",".join(extra)

    # -- stock_code format --
    if "stock_code" in df.columns:
        codes = df["stock_code"].dropna().astype(str)
        bad = codes[~codes.str.match(STOCK_CODE_PATTERN)]
        result["n_bad_stock_code"] = len(bad)

    # -- date format (YYYYMMDD, 8 digits) --
    if "date" in df.columns:
        dates_str = df["date"].dropna().astype(str)
        bad_dates = dates_str[~dates_str.str.match(r"^\d{8}$")]
        result["n_bad_date"] = len(bad_dates)

    # -- duplicates --
    result["n_duplicates"] = int(df.duplicated().sum())

    # -- key field missing rate --
    key_cols = ["total_amount_wan", "avg_price", "order_count"]
    key_missing = 0
    for c in key_cols:
        if c in df.columns:
            key_missing += int(df[c].isna().sum())
    result["n_missing_key_fields"] = key_missing

    # -- composite status --
    issues = []
    if result["n_rows"] == 0:
        issues.append("empty")
    if result["n_missing_fields"] > 0:
        issues.append(f"missing_{result['n_missing_fields']}fields")
    if result["n_bad_stock_code"] > 0:
        issues.append("bad_stock_code")
    if result["n_duplicates"] > 0:
        issues.append("duplicates")
    if result["n_missing_key_fields"] > result["n_rows"] * 0.05:
        issues.append("high_key_missing")
    if result["n_bad_date"] > 0:
        issues.append("bad_date")

    if issues:
        result["status"] = ";".join(issues)

    return result


def check_price_matching(ops_dates: set[str], price_dir: Path) -> dict:
    """Check which ops dates have matching price data."""
    price_files = sorted(price_dir.glob("*.parquet"))
    price_dates_by_stock = {}

    for pf in price_files:
        stock = pf.stem  # e.g. "000001"
        try:
            pdf = pd.read_parquet(pf)
            pdf["date_str"] = pd.to_datetime(pdf["date"]).dt.strftime("%Y%m%d")
            price_dates_by_stock[stock] = set(pdf["date_str"].values)
        except Exception:
            continue

    if not price_dates_by_stock:
        return {
            "n_stocks_with_price": 0,
            "price_date_coverage": {},
            "ops_dates_without_price": sorted(ops_dates),
        }

    # Collect all price dates across all stocks
    all_price_dates = set()
    for dates in price_dates_by_stock.values():
        all_price_dates |= dates

    missing_price = sorted(ops_dates - all_price_dates)

    # Coverage: for each ops date, how many stocks have price
    coverage = {}
    for d in sorted(ops_dates):
        n_with_price = sum(1 for dates in price_dates_by_stock.values() if d in dates)
        coverage[d] = n_with_price

    return {
        "n_stocks_with_price": len(price_files),
        "price_date_coverage": coverage,
        "ops_dates_without_price": missing_price,
    }


def main():
    ap = argparse.ArgumentParser(description="Check 2026 level2_ops quality")
    ap.add_argument("--ops-dir", required=True,
                    help="2026 level2_ops directory")
    ap.add_argument("--ref-dir", default=None,
                    help="2025 level2_ops directory (for field comparison)")
    ap.add_argument("--price-dir", default="data/daily",
                    help="Daily price parquet directory")
    ap.add_argument("--selected-stocks", default=None,
                    help="selected_stocks.csv path")
    args = ap.parse_args()

    ops_dir = Path(args.ops_dir)
    if not ops_dir.is_absolute():
        ops_dir = PROJECT / ops_dir

    if not ops_dir.exists():
        print(f"[ERROR] ops-dir not found: {ops_dir}")
        print("Run the DBSCAN pipeline on Windows first, then sync to Mac.")
        sys.exit(1)

    price_dir = Path(args.price_dir)
    if not price_dir.is_absolute():
        price_dir = PROJECT / price_dir

    # ----- 1. Collect all files -----
    csv_files = sorted(ops_dir.glob("level2_ops_*.csv"))
    if not csv_files:
        print(f"No level2_ops_*.csv found in {ops_dir}")
        sys.exit(1)

    print(f"{'='*70}")
    print(f"2026 LEVEL2_OPS QUALITY CHECK")
    print(f"{'='*70}")
    print(f"Ops dir:    {ops_dir}")
    print(f"Files found: {len(csv_files)}")

    # ----- 2. Per-file checks -----
    rows = []
    all_dates = set()
    all_stocks = set()
    bad_files = []

    for fp in csv_files:
        row = check_file(fp, EXPECTED_COLUMNS)
        rows.append(row)
        if row["date"] and row["date"] != "missing":
            all_dates.add(row["date"])
        if row["status"] != "ok":
            bad_files.append(row)

    report = pd.DataFrame(rows)
    if "date" in report.columns:
        report = report.sort_values("date")

    # ----- 3. Summary stats -----
    total_rows = report["n_rows"].sum()
    total_bad = len(bad_files)
    n_empty = int((report["n_rows"] == 0).sum())

    # Month coverage from date strings
    months = set()
    for d in all_dates:
        if len(d) >= 6:
            months.add(d[:6])
    months = sorted(months)

    print(f"\n{'='*70}")
    print(f"SUMMARY")
    print(f"{'='*70}")
    print(f"  Files:            {len(csv_files)}")
    print(f"  Months covered:   {len(months)} — {months}")
    print(f"  Trading days:     {len(all_dates)}")
    print(f"  Total ops:        {total_rows:,}")
    print(f"  Empty files:      {n_empty}")
    print(f"  Problem files:    {total_bad}")
    if all_dates:
        print(f"  Date range:       {min(all_dates)} — {max(all_dates)}")

    # ----- 4. Field consistency -----
    print(f"\n{'='*70}")
    print(f"FIELD CONSISTENCY (vs 2025 reference)")
    print(f"{'='*70}")
    print(f"  Expected fields:  {len(EXPECTED_COLUMNS)}")
    print(f"  Fields: {', '.join(EXPECTED_COLUMNS[:8])}...")

    missing_fields_count = Counter()
    extra_fields_count = Counter()
    for r in rows:
        if r["missing_fields"]:
            for f in r["missing_fields"].split(","):
                missing_fields_count[f] += 1
        if r["extra_fields"]:
            for f in r["extra_fields"].split(","):
                extra_fields_count[f] += 1

    if missing_fields_count:
        print(f"  MISSING fields:")
        for f, c in missing_fields_count.most_common():
            print(f"    {f}: {c}/{len(csv_files)} files")
    else:
        print(f"  All 22 expected fields present in all files.")

    if extra_fields_count:
        print(f"  UNEXPECTED fields:")
        for f, c in extra_fields_count.most_common():
            print(f"    {f}: {c}/{len(csv_files)} files")

    # ----- 5. stock_code format -----
    n_bad_codes = report["n_bad_stock_code"].sum()
    print(f"\n{'='*70}")
    print(f"STOCK_CODE FORMAT")
    print(f"{'='*70}")
    print(f"  Bad stock codes:  {n_bad_codes}")
    if n_bad_codes > 0:
        print(f"  FAIL: non-standard stock codes found")
    else:
        print(f"  PASS: all stock codes match XXXXXX.SZ/SH/BJ")

    # Collect all stocks
    print(f"  Reading stock universe from files...")
    all_stocks_set = set()
    stock_days = Counter()
    for fp in csv_files[:10]:  # sample first 10 for stock list
        try:
            df = pd.read_csv(fp, low_memory=False)
            if "stock_code" in df.columns:
                codes = df["stock_code"].dropna().astype(str)
                all_stocks_set |= set(codes.unique())
        except Exception:
            pass
    print(f"  Unique stocks (first 10 files): {len(all_stocks_set)}")

    # Check against selected_stocks if provided
    if args.selected_stocks:
        ss_path = Path(args.selected_stocks)
        if not ss_path.is_absolute():
            ss_path = PROJECT / ss_path
        if ss_path.exists():
            sel = pd.read_csv(ss_path)
            sel_codes = set(sel["stock"].astype(str).str.zfill(6))
            # Check first file's stocks
            if csv_files:
                df0 = pd.read_csv(csv_files[0], low_memory=False)
                actual_codes = set(
                    df0["stock_code"].dropna().astype(str)
                    .str.replace(r"\.(SZ|SH|BJ)$", "", regex=True)
                )
                coverage = len(actual_codes & sel_codes) / len(sel_codes) * 100 if sel_codes else 0
                print(f"  selected_stocks coverage: {coverage:.1f}% "
                      f"({len(actual_codes & sel_codes)}/{len(sel_codes)})")

    # ----- 6. Duplicates -----
    n_dups = report["n_duplicates"].sum()
    print(f"\n{'='*70}")
    print(f"DUPLICATE RECORDS")
    print(f"{'='*70}")
    print(f"  Total duplicates: {n_dups}")
    if n_dups > 0:
        print(f"  WARNING: duplicates found")
        dup_files = report[report["n_duplicates"] > 0]
        for _, r in dup_files.iterrows():
            print(f"    {r['file']}: {r['n_duplicates']} dup rows")

    # ----- 7. Key field missing rate -----
    n_key_missing = report["n_missing_key_fields"].sum()
    pct_key_missing = n_key_missing / max(total_rows, 1) * 100
    print(f"\n{'='*70}")
    print(f"KEY FIELD MISSING RATE")
    print(f"{'='*70}")
    print(f"  Missing (amount/price/orders): {n_key_missing} ({pct_key_missing:.2f}%)")
    if pct_key_missing > 5:
        print(f"  FAIL: >5% key fields missing")
    else:
        print(f"  PASS")

    # ----- 8. Daily ops distribution -----
    print(f"\n{'='*70}")
    print(f"DAILY OPS DISTRIBUTION")
    print(f"{'='*70}")
    n_rows_series = report[report["n_rows"] > 0]["n_rows"]
    if len(n_rows_series) > 0:
        print(f"  Mean ops/day:   {n_rows_series.mean():.0f}")
        print(f"  Median ops/day: {n_rows_series.median():.0f}")
        print(f"  Std ops/day:    {n_rows_series.std():.0f}")
        print(f"  Min ops/day:    {n_rows_series.min()}")
        print(f"  Max ops/day:    {n_rows_series.max()}")
        print(f"  Total BUY ops:  {report['n_rows'].sum():,}")

        # Flag outliers
        mean_val = n_rows_series.mean()
        std_val = n_rows_series.std()
        low = report[(report["n_rows"] > 0) & (report["n_rows"] < mean_val - 2 * std_val)]
        high = report[report["n_rows"] > mean_val + 2 * std_val]
        if len(low) > 0:
            print(f"  Low-outlier days (< {mean_val - 2*std_val:.0f}): {len(low)}")
            for _, r in low.iterrows():
                print(f"    {r['file']}: {r['n_rows']} ops")
        if len(high) > 0:
            print(f"  High-outlier days (>{mean_val + 2*std_val:.0f}): {len(high)}")
            for _, r in high.iterrows():
                print(f"    {r['file']}: {r['n_rows']} ops")

    # ----- 9. Date matching with price data -----
    print(f"\n{'='*70}")
    print(f"PRICE DATE MATCHING")
    print(f"{'='*70}")
    if price_dir.exists():
        price_info = check_price_matching(all_dates, price_dir)
        n_price_stocks = price_info["n_stocks_with_price"]
        missing_dates = price_info["ops_dates_without_price"]

        print(f"  Stocks with price data: {n_price_stocks}")
        if n_price_stocks == 0:
            print(f"  WARNING: No 2026 price data found in {price_dir}")
            print(f"  Run price prefetch: python scripts/prefetch_prices.py --year 2026")
            print(f"  Or check: akshare may not have 2026 data for these dates yet")

        if missing_dates:
            print(f"  Ops dates without price: {len(missing_dates)}")
            if len(missing_dates) <= 20:
                print(f"    {missing_dates}")
            else:
                print(f"    {missing_dates[:10]} ... {missing_dates[-5:]}")
        else:
            print(f"  All ops dates have price data in at least 1 stock")

        # Coverage summary
        coverage = price_info.get("price_date_coverage", {})
        if coverage:
            vals = list(coverage.values())
            print(f"  Mean stocks with price/date: {sum(vals)/len(vals):.0f}")
            print(f"  Min stocks with price/date:  {min(vals)}")
    else:
        print(f"  Price dir not found: {price_dir}")
        print(f"  Skip price matching check")

    # ----- 10. Final verdict -----
    print(f"\n{'='*70}")
    print(f"VERDICT")
    print(f"{'='*70}")

    failures = []
    if total_bad > 0:
        failures.append(f"{total_bad} problem files")
    if missing_fields_count:
        failures.append("field mismatch vs 2025")
    if n_bad_codes > 0:
        failures.append("bad stock codes")
    if pct_key_missing > 5:
        failures.append("high key field missing")
    if n_dups > 0:
        failures.append(f"{n_dups} duplicate rows")

    if failures:
        print(f"  ISSUES: {'; '.join(failures)}")
        print(f"  Review and fix before proceeding to backtest.")
    else:
        print(f"  ALL CHECKS PASSED — ready for InstitutionTracker + backtest.")

    # ----- Save reports -----
    report_path = ops_dir / "quality_report_2026.csv"
    report.to_csv(report_path, index=False)
    print(f"\n  CSV report: {report_path}")

    # Text summary
    txt_path = ops_dir / "quality_report_2026.txt"
    with open(txt_path, "w") as f:
        f.write(f"2026 level2_ops quality check\n")
        f.write(f"{'='*60}\n")
        f.write(f"Files: {len(csv_files)}\n")
        f.write(f"Months: {months}\n")
        f.write(f"Trading days: {len(all_dates)}\n")
        f.write(f"Total ops: {total_rows:,}\n")
        f.write(f"Empty files: {n_empty}\n")
        f.write(f"Bad stock codes: {n_bad_codes}\n")
        f.write(f"Duplicate rows: {n_dups}\n")
        f.write(f"Key field missing: {n_key_missing} ({pct_key_missing:.2f}%)\n")
        f.write(f"Status: {'ISSUES' if failures else 'PASS'}\n")
        if failures:
            f.write(f"Issues: {'; '.join(failures)}\n")
    print(f"  Text report: {txt_path}")

    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
