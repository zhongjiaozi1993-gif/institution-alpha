"""
Batch run sofia_v4_hunter + sofia_v6_enhanced for priority stocks

用法:
  python scripts/batch_run_v4_v6.py --stocks config/v6_priority_stocks.txt
  python scripts/batch_run_v4_v6.py --stocks 000547,000510,000688
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
SCRIPTS = PROJECT / "scripts"


def run_stock(stock: str, year: str = "2025") -> dict:
    """Run v4 + v6 for a single stock, return timing info"""
    t0 = time.time()
    result = {"stock": stock, "v4_ok": False, "v6_ok": False, "v4_time": 0, "v6_time": 0}

    # Step 1: v4
    t1 = time.time()
    r = subprocess.run(
        [sys.executable, str(SCRIPTS / "sofia_v4_hunter.py"),
         "--stock", stock, "--year", year],
        capture_output=True, text=True, timeout=1800,
        cwd=str(PROJECT),
    )
    result["v4_time"] = round(time.time() - t1, 1)
    result["v4_ok"] = r.returncode == 0
    if not result["v4_ok"]:
        result["v4_error"] = r.stderr[-200:] if r.stderr else f"exit={r.returncode}"
        return result

    # Step 2: v6
    t2 = time.time()
    r = subprocess.run(
        [sys.executable, str(SCRIPTS / "sofia_v6_enhanced.py"),
         "--stock", stock],
        capture_output=True, text=True, timeout=300,
        cwd=str(PROJECT),
    )
    result["v6_time"] = round(time.time() - t2, 1)
    result["v6_ok"] = r.returncode == 0
    if not result["v6_ok"]:
        result["v6_error"] = r.stderr[-200:] if r.stderr else f"exit={r.returncode}"

    result["total_time"] = round(time.time() - t0, 1)
    return result


def main():
    ap = argparse.ArgumentParser(description="Batch run v4+v6 pipeline")
    ap.add_argument("--stocks", required=True,
                    help="File with one stock per line, or comma-separated")
    ap.add_argument("--year", default="2025")
    ap.add_argument("--skip-existing", action="store_true",
                    help="Skip stocks with existing v6 institution_registry.json")
    args = ap.parse_args()

    stocks_path = Path(args.stocks)
    if stocks_path.exists() and not stocks_path.is_dir():
        stocks = [line.strip() for line in open(stocks_path)
                  if line.strip() and not line.strip().startswith("#")]
    else:
        stocks = [s.strip().zfill(6) for s in args.stocks.split(",") if s.strip()]

    # Sort by data size (lightest first, so heavy ones run later)
    def _data_size(stock):
        d = PROJECT / "data" / "single_stock" / stock
        if d.exists():
            return sum(f.stat().st_size for f in d.rglob("*") if f.is_file())
        return 0
    stocks.sort(key=_data_size)
    sizes = [(s, _data_size(s) / 1_048_576) for s in stocks]
    for s, mb in sizes[:5]:
        print(f"  {s}: {mb:.0f} MB")
    print(f"  ... {len(stocks)} stocks total")

    print(f"\nStocks: {len(stocks)}")
    print(f"Year: {args.year}")
    print(f"Skip existing: {args.skip_existing}")
    print()

    ok, fail = 0, 0
    failures = []
    t_start = time.time()

    for i, stock in enumerate(stocks):
        # Check if v6 output already exists
        v6_path = PROJECT / "data" / "single_stock" / stock / "sofia_v6" / "institution_registry.json"
        if args.skip_existing and v6_path.exists():
            print(f"[{i+1}/{len(stocks)}] {stock} - v6 exists, skip")
            ok += 1
            continue

        print(f"[{i+1}/{len(stocks)}] {stock} ...", end=" ", flush=True)
        r = run_stock(stock, args.year)

        status = []
        if r["v4_ok"]:
            status.append(f"v4={r['v4_time']}s")
        else:
            status.append(f"v4=FAIL({r.get('v4_error','?')})")
        if r["v6_ok"]:
            status.append(f"v6={r['v6_time']}s")
        else:
            status.append(f"v6=FAIL({r.get('v6_error','?')})")

        print(" | ".join(status))

        if r["v4_ok"] and r["v6_ok"]:
            ok += 1
        else:
            fail += 1
            failures.append(r)

    elapsed = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"Done: OK={ok} FAIL={fail} / {len(stocks)}")
    print(f"Total: {elapsed/60:.0f} min")

    if failures:
        print(f"\nFailures:")
        for f in failures:
            print(f"  {f['stock']}: v4={f.get('v4_error','ok')} v6={f.get('v6_error','ok')}")


if __name__ == "__main__":
    main()
