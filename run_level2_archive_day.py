"""
Process one Level-2 .7z archive and extract institution-like operations.

This script is intentionally conservative: start with --stocks or --max-stocks
before processing a full trading day.
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import pandas as pd
import py7zr

from src.cluster.split_detector import detect_institution_operations
from src.data.level2_reader import match_orders_to_trades, read_level2_stock_dir


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run Level-2 operation detection from one .7z day archive.")
    p.add_argument("--archive", required=True, help="Path to YYYYMMDD.7z")
    p.add_argument("--date", required=True, help="Trading date, e.g. 20250102")
    p.add_argument("--stocks", default="", help="Comma-separated Wind codes, e.g. 000001.SZ,000002.SZ")
    p.add_argument("--max-stocks", type=int, default=0, help="Process first N stocks if --stocks is empty")
    p.add_argument("--extract-dir", default="data/tmp_archive_extract", help="Temporary extraction directory")
    p.add_argument("--output", default="", help="Output parquet path")
    p.add_argument("--min-amount", type=float, default=100.0, help="Min cluster amount in 万元")
    p.add_argument("--eps", type=float, default=0.15)
    p.add_argument("--min-samples", type=int, default=5)
    p.add_argument("--keep-extracted", action="store_true")
    return p.parse_args()


def list_stock_dirs(archive: Path, date: str) -> list[str]:
    with py7zr.SevenZipFile(archive, "r") as z:
        stocks = {
            name.split("/")[1]
            for name in z.getnames()
            if name.startswith(f"{date}/") and len(name.split("/")) >= 2 and "." in name.split("/")[1]
        }
    return sorted(stocks)


def extract_stock(archive: Path, date: str, stock: str, extract_root: Path) -> Path:
    targets = [
        f"{date}/{stock}/行情.csv",
        f"{date}/{stock}/逐笔委托.csv",
        f"{date}/{stock}/逐笔成交.csv",
    ]
    with py7zr.SevenZipFile(archive, "r") as z:
        z.extract(path=extract_root, targets=targets)
    return extract_root / date / stock


def main() -> None:
    args = parse_args()
    archive = Path(args.archive)
    date = args.date
    extract_root = Path(args.extract_dir) / date
    output = Path(args.output) if args.output else Path(f"data/processed/level2_ops_{date}.parquet")
    output.parent.mkdir(parents=True, exist_ok=True)

    if args.stocks.strip():
        stocks = [s.strip() for s in args.stocks.split(",") if s.strip()]
    else:
        stocks = list_stock_dirs(archive, date)
        if args.max_stocks > 0:
            stocks = stocks[: args.max_stocks]

    print(f"archive={archive}")
    print(f"date={date}")
    print(f"stocks={len(stocks)}")

    rows: list[dict] = []
    failures: list[tuple[str, str]] = []

    shutil.rmtree(extract_root, ignore_errors=True)
    extract_root.mkdir(parents=True, exist_ok=True)

    for i, stock in enumerate(stocks, 1):
        try:
            stock_dir = extract_stock(archive, date, stock, extract_root)
            data = read_level2_stock_dir(stock_dir)
            if "逐笔委托" not in data or "逐笔成交" not in data:
                failures.append((stock, "missing required csv"))
                continue
            matched = match_orders_to_trades(data["逐笔委托"], data["逐笔成交"])
            ops = detect_institution_operations(
                matched,
                eps=args.eps,
                min_samples=args.min_samples,
                min_total_amount_wan=args.min_amount,
            )
            for op in ops:
                op["date"] = date
                op["stock_code"] = stock
                op["matched_orders"] = len(matched)
                op["match_key"] = matched["match_key"].iloc[0] if not matched.empty and "match_key" in matched else ""
            rows.extend(ops)
            print(f"[{i}/{len(stocks)}] {stock}: matched={len(matched)} ops={len(ops)}")
        except Exception as exc:
            failures.append((stock, repr(exc)))
            print(f"[{i}/{len(stocks)}] {stock}: ERROR {exc!r}")
        finally:
            if not args.keep_extracted:
                shutil.rmtree(extract_root / date / stock, ignore_errors=True)

    df = pd.DataFrame(rows)
    df.to_parquet(output, index=False)
    print(f"output={output.resolve()}")
    print(f"ops={len(df)}")
    print(f"failures={len(failures)}")
    if failures:
        print("first_failures=", failures[:10])


if __name__ == "__main__":
    main()
