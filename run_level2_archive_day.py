"""
Process one Level-2 .7z archive and extract institution-like operations.

The runner checkpoints after every stock so interrupted day jobs can resume
without reprocessing completed symbols.
"""
from __future__ import annotations

import argparse
import csv
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
    p.add_argument("--output", default="", help="Output CSV path")
    p.add_argument("--checkpoint-dir", default="", help="Directory for per-stock checkpoints")
    p.add_argument("--no-resume", action="store_true", help="Ignore existing checkpoint progress")
    p.add_argument(
        "--asset-type",
        choices=["a-share", "all"],
        default="a-share",
        help="Filter listed symbols. Default keeps A-share stocks and excludes bonds/funds.",
    )
    p.add_argument("--min-amount", type=float, default=100.0, help="Min cluster amount in wan")
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


def is_a_share_code(stock: str) -> bool:
    code, _, exchange = stock.partition(".")
    if exchange == "SZ":
        return code.startswith(("000", "001", "002", "003", "300", "301"))
    if exchange == "SH":
        return code.startswith(("600", "601", "603", "605", "688", "689"))
    if exchange == "BJ":
        return code.startswith(
            (
                "430", "830", "831", "832", "833", "834", "835", "836", "837", "838", "839",
                "870", "871", "872", "873", "874", "875", "876", "877", "878", "879", "920",
            )
        )
    return False


def extract_stock(archive: Path, date: str, stock: str, extract_root: Path) -> Path:
    targets = [
        f"{date}/{stock}/行情.csv",
        f"{date}/{stock}/逐笔委托.csv",
        f"{date}/{stock}/逐笔成交.csv",
    ]
    with py7zr.SevenZipFile(archive, "r") as z:
        z.extract(path=extract_root, targets=targets)
    return extract_root / date / stock


def progress_path(checkpoint_dir: Path) -> Path:
    return checkpoint_dir / "progress.csv"


def stock_part_path(checkpoint_dir: Path, stock: str) -> Path:
    return checkpoint_dir / f"ops_{stock.replace('.', '_')}.csv"


def load_completed(checkpoint_dir: Path) -> set[str]:
    path = progress_path(checkpoint_dir)
    if not path.exists():
        return set()
    completed: set[str] = set()
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("status") == "ok" and row.get("stock"):
                completed.add(row["stock"])
    return completed


def append_progress(checkpoint_dir: Path, row: dict[str, object]) -> None:
    path = progress_path(checkpoint_dir)
    exists = path.exists()
    fieldnames = ["stock", "status", "matched", "ops", "error"]
    with path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in fieldnames})


def write_stock_ops(checkpoint_dir: Path, stock: str, ops: list[dict]) -> None:
    if not ops:
        return
    pd.DataFrame(ops).to_csv(stock_part_path(checkpoint_dir, stock), index=False)


def combine_checkpoints(checkpoint_dir: Path, output: Path, stocks: list[str]) -> pd.DataFrame:
    parts = [stock_part_path(checkpoint_dir, stock) for stock in stocks]
    parts = [p for p in parts if p.exists()]
    if parts:
        df = pd.concat((pd.read_csv(p) for p in parts), ignore_index=True)
    else:
        df = pd.DataFrame()
    df.to_csv(output, index=False)
    return df


def main() -> None:
    args = parse_args()
    archive = Path(args.archive)
    date = args.date
    extract_root = Path(args.extract_dir) / date
    output = Path(args.output) if args.output else Path(f"data/processed/level2_ops_{date}.csv")
    if output.suffix == ".parquet":
        output = output.with_suffix(".csv")
    checkpoint_dir = Path(args.checkpoint_dir) if args.checkpoint_dir else output.with_suffix("")
    output.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    if args.stocks.strip():
        stocks = [s.strip() for s in args.stocks.split(",") if s.strip()]
    else:
        stocks = list_stock_dirs(archive, date)
        if args.asset_type == "a-share":
            stocks = [s for s in stocks if is_a_share_code(s)]
        if args.max_stocks > 0:
            stocks = stocks[: args.max_stocks]

    completed = set() if args.no_resume else load_completed(checkpoint_dir)
    pending = [s for s in stocks if s not in completed]

    print(f"archive={archive}", flush=True)
    print(f"date={date}", flush=True)
    print(f"asset_type={args.asset_type}", flush=True)
    print(f"stocks={len(stocks)} completed={len(completed)} pending={len(pending)}", flush=True)
    print(f"checkpoint_dir={checkpoint_dir.resolve()}", flush=True)

    failures: list[tuple[str, str]] = []

    shutil.rmtree(extract_root, ignore_errors=True)
    extract_root.mkdir(parents=True, exist_ok=True)

    for i, stock in enumerate(stocks, 1):
        if stock in completed:
            if i == 1 or i == len(stocks) or i % 100 == 0:
                print(f"[{i}/{len(stocks)}] {stock}: SKIP checkpoint", flush=True)
            continue

        try:
            stock_dir = extract_stock(archive, date, stock, extract_root)
            data = read_level2_stock_dir(stock_dir)
            if "逐笔委托" not in data or "逐笔成交" not in data:
                append_progress(
                    checkpoint_dir,
                    {"stock": stock, "status": "error", "matched": 0, "ops": 0, "error": "missing required csv"},
                )
                failures.append((stock, "missing required csv"))
                print(f"[{i}/{len(stocks)}] {stock}: ERROR missing required csv", flush=True)
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

            write_stock_ops(checkpoint_dir, stock, ops)
            append_progress(checkpoint_dir, {"stock": stock, "status": "ok", "matched": len(matched), "ops": len(ops)})
            print(f"[{i}/{len(stocks)}] {stock}: matched={len(matched)} ops={len(ops)}", flush=True)
        except Exception as exc:
            failures.append((stock, repr(exc)))
            append_progress(checkpoint_dir, {"stock": stock, "status": "error", "matched": 0, "ops": 0, "error": repr(exc)})
            print(f"[{i}/{len(stocks)}] {stock}: ERROR {exc!r}", flush=True)
        finally:
            if not args.keep_extracted:
                shutil.rmtree(extract_root / date / stock, ignore_errors=True)

    df = combine_checkpoints(checkpoint_dir, output, stocks)
    print(f"output={output.resolve()}", flush=True)
    print(f"ops={len(df)}", flush=True)
    print(f"failures={len(failures)}", flush=True)
    if failures:
        print("first_failures=", failures[:10], flush=True)


if __name__ == "__main__":
    main()
