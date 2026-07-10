"""
Extract raw Level-2 CSVs for v6 priority stocks from 7z archives

For each trading day, extracts only 逐笔委托.csv and 逐笔成交.csv
for the target stocks, organized into data/single_stock/{stock}/raw/{date}/.

Runs on the machine with 7z archives (typically Windows training box).
Then sync single_stock/ directory to Mac for v4/v6 pipeline.

用法:
  python scripts/extract_raw_for_v6.py \
    --stocks config/v6_priority_stocks.txt \
    --archive-root "C:/Users/1/Desktop/2025" \
    --out-dir data/single_stock \
    --year 2025
"""

from __future__ import annotations

import argparse
import io
import re
import sys
import time
from pathlib import Path

import py7zr

PROJECT = Path(__file__).resolve().parent.parent


def wind_code(stock: str) -> str:
    suffix = "SH" if stock.startswith(("6", "9")) else "SZ"
    return f"{stock}.{suffix}"


def find_archives(archive_root: Path, year: str) -> dict[str, Path]:
    """Scan archive_root for YYYYMMDD.7z files, return {date: path}"""
    by_date: dict[str, Path] = {}
    patterns = [f"**/{year}*/{year}*.7z", f"**/{year}*.7z",
                f"**/{year}*/{year}*.7Z", f"**/{year}*.7Z"]
    seen = set()
    for pat in patterns:
        for p in archive_root.rglob(pat.split("/")[-1]):
            m = re.search(r"(\d{8})", p.name)
            if m and m.group(1)[:4] == year:
                d = m.group(1)
                if d not in seen:
                    by_date[d] = p
                    seen.add(d)
    return by_date


def extract_day(archive_path: Path, date: str, stocks: list[str],
                 out_base: Path) -> tuple[int, int]:
    """Extract all target stocks' raw CSVs from one 7z archive in a single open.

    Returns (extracted, attempted) counts.
    """
    wcodes = {wind_code(s): s for s in stocks}

    # Pre-filter: which stocks are already done?
    needed = {}
    for wcode, stock in wcodes.items():
        out_dir = out_base / stock / "raw" / date / wcode
        if not (out_dir.exists() and (out_dir / "逐笔成交.csv").exists()):
            needed[wcode] = stock

    if not needed:
        return 0, 0

    extracted = 0
    try:
        with py7zr.SevenZipFile(archive_path, "r") as z:
            all_files = z.getnames()

            # Group targets by wcode
            by_wcode: dict[str, list[str]] = {}
            for f in all_files:
                for wcode in needed:
                    if wcode in f and ("逐笔委托.csv" in f or "逐笔成交.csv" in f):
                        by_wcode.setdefault(wcode, []).append(f)

            if not by_wcode:
                return 0, 0

            # Extract to temp dir, then copy to final location
            all_targets = [f for files in by_wcode.values() for f in files]
            import tempfile
            with tempfile.TemporaryDirectory() as tmp:
                z.extract(targets=all_targets, path=tmp)
                tmp_path = Path(tmp)

                for wcode, files in by_wcode.items():
                    stock = needed[wcode]
                    out_dir = out_base / stock / "raw" / date / wcode
                    out_dir.mkdir(parents=True, exist_ok=True)
                    for f in files:
                        src = tmp_path / f
                        if src.exists():
                            dst = out_dir / Path(f).name
                            dst.write_bytes(src.read_bytes())
                    extracted += 1

    except Exception as e:
        print(f"    [WARN] {date}: {e}")

    return extracted, len(needed)


def main():
    ap = argparse.ArgumentParser(description="Extract raw L2 CSVs for v6 stocks")
    ap.add_argument("--stocks", required=True,
                    help="File with one stock code per line, or comma-separated list")
    ap.add_argument("--archive-root", required=True,
                    help="Root directory containing YYYYMM/YYYYMMDD.7z archives")
    ap.add_argument("--out-dir", default="data/single_stock",
                    help="Output root for single_stock data")
    ap.add_argument("--year", default="2025", help="Year to process")
    ap.add_argument("--limit", type=int, default=0, help="Process first N days")
    args = ap.parse_args()

    # Parse stocks
    stocks_path = Path(args.stocks)
    if stocks_path.exists():
        stocks = [line.strip() for line in open(stocks_path)
                  if line.strip() and not line.strip().startswith("#")]
    else:
        stocks = [s.strip().zfill(6) for s in args.stocks.split(",") if s.strip()]

    print(f"Target stocks: {len(stocks)}")
    print(f"Archive root: {args.archive_root}")
    print(f"Year: {args.year}")

    archive_root = Path(args.archive_root)
    out_base = Path(args.out_dir) if Path(args.out_dir).is_absolute() else PROJECT / args.out_dir
    out_base.mkdir(parents=True, exist_ok=True)

    archives = find_archives(archive_root, args.year)
    dates = sorted(archives.keys())
    if args.limit:
        dates = dates[:args.limit]

    print(f"Archives found: {len(archives)}, processing {len(dates)} days")

    t0 = time.time()
    total_extracted = 0
    total_attempts = 0

    for i, date in enumerate(dates):
        archive = archives[date]
        ex, at = extract_day(archive, date, stocks, out_base)
        total_extracted += ex
        total_attempts += at

        if (i + 1) % 20 == 0 or i == 0:
            elapsed = time.time() - t0
            print(f"  [{i+1}/{len(dates)}] {date}  "
                  f"extracted={total_extracted}/{total_attempts}  ({elapsed:.0f}s)")

    elapsed = time.time() - t0
    print(f"\nDone: {total_extracted}/{total_attempts} stock-days extracted "
          f"({elapsed:.0f}s, {elapsed/60:.1f} min)")

    # Verify
    for stock in stocks:
        raw_dir = out_base / stock / "raw"
        if raw_dir.exists():
            ndays = len(list(raw_dir.iterdir()))
            print(f"  {stock}: {ndays} days")
        else:
            print(f"  {stock}: 0 days (MISSING)")


if __name__ == "__main__":
    main()
