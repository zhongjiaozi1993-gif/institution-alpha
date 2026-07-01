"""
Level-2 处理计划执行器 — 按 process_plan.csv 逐日处理 archive

用法:
  # Dry-run 检查
  python scripts/run_level2_process_plan.py \
    --plan data/processed/stock_universe/process_plan.csv \
    --start-year 2025 --end-year 2025 --limit 10 --dry-run

  # 试跑 10 天
  python scripts/run_level2_process_plan.py \
    --plan data/processed/stock_universe/process_plan.csv \
    --start-year 2025 --end-year 2025 --limit 10

  # 跑 2025 全年
  python scripts/run_level2_process_plan.py \
    --plan data/processed/stock_universe/process_plan.csv \
    --start-year 2025 --end-year 2025
"""

from __future__ import annotations

import argparse
import csv
import shutil
import subprocess
import sys
import time
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent

ARCHIVE_DAY_SCRIPT = PROJECT / "run_level2_archive_day.py"


def wind_code(stock: str) -> str:
    """补全交易所后缀: 002516→002516.SZ, 600519→600519.SH"""
    suffix = "SH" if stock.startswith(("6", "9")) else "SZ"
    return f"{stock}.{suffix}"


def read_plan(plan_path: Path, start_year: str | None, end_year: str | None,
              limit: int) -> list[dict]:
    """读取 process_plan.csv, 按年份和 limit 筛选"""
    import csv
    rows = []
    with open(plan_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            year = row["year"]
            if start_year and year < start_year:
                continue
            if end_year and year > end_year:
                continue
            rows.append(row)
            if limit and len(rows) >= limit:
                break
    return rows


def run_day(archive: str, date: str, stocks_plain: list[str],
            extract_dir: str, output_dir: str,
            dry_run: bool = False,
            python: str = "python") -> bool:
    """对单日 archive 运行 run_level2_archive_day.py"""

    stocks_wind = ",".join(wind_code(s) for s in stocks_plain)
    output = str(Path(output_dir) / f"level2_ops_{date}.csv")

    # 如果输出已存在则跳过
    if Path(output).exists():
        print(f"  -> exists, skip")
        return True

    cmd = [
        python, str(ARCHIVE_DAY_SCRIPT),
        "--archive", archive,
        "--date", date,
        "--stocks", stocks_wind,
        "--extract-dir", extract_dir,
        "--output", output,
    ]

    if dry_run:
        print(f"  [DRY-RUN] {' '.join(cmd)}")
        return True

    print(f"  -> processing {len(stocks_plain)} stocks...")
    result = subprocess.run(cmd, capture_output=False, timeout=1800)
    return result.returncode == 0


def main():
    ap = argparse.ArgumentParser(description="Level-2 处理计划执行器")
    ap.add_argument("--plan", default="data/processed/stock_universe/process_plan.csv",
                    help="process_plan.csv 路径")
    ap.add_argument("--start-year", default="", help="起始年份, e.g. 2025")
    ap.add_argument("--end-year", default="", help="结束年份, e.g. 2025")
    ap.add_argument("--limit", type=int, default=0,
                    help="限制处理天数 (0=全部)")
    ap.add_argument("--temp-dir", default="temp_extract",
                    help="临时解压目录 (默认 temp_extract)")
    ap.add_argument("--output-dir", default="data/processed/level2_ops",
                    help="输出 parquet 目录")
    ap.add_argument("--python", default="python",
                    help="Python 解释器 (默认 python)")
    ap.add_argument("--dry-run", action="store_true",
                    help="只打印命令, 不执行")
    args = ap.parse_args()

    plan_path = PROJECT / args.plan if not Path(args.plan).is_absolute() else Path(args.plan)
    if not plan_path.exists():
        print(f"[ERROR] plan file not found: {plan_path}")
        sys.exit(1)

    start_year = args.start_year or None
    end_year = args.end_year or None

    rows = read_plan(plan_path, start_year, end_year, args.limit)

    if not rows:
        print("No matching plan entries")
        sys.exit(0)

    print(f"Plan: {len(rows)} days")
    print(f"Years: {start_year or 'all'} -> {end_year or 'all'}")
    print(f"Limit: {args.limit or 'none'}")
    print(f"Mode: {'DRY-RUN' if args.dry_run else 'EXEC'}")
    print(f"Temp dir: {args.temp_dir}")
    print(f"Output dir: {args.output_dir}")
    print()

    # organize output by year
    output_base = PROJECT / args.output_dir
    output_base.mkdir(parents=True, exist_ok=True)

    ok = 0
    skip = 0
    fail = 0
    failures: list[dict] = []
    t_start = time.time()

    for i, row in enumerate(rows):
        date = row["date"]
        year = row["year"]
        archive = row["archive"]
        stocks = [s.strip() for s in row["stocks"].split(",") if s.strip()]

        year_dir = output_base / year
        year_dir.mkdir(parents=True, exist_ok=True)

        extract_dir = str(PROJECT / args.temp_dir / date)

        t0 = time.time()
        print(f"[{i+1}/{len(rows)}] {date} ({year}) "
              f"archive={Path(archive).name} stocks={len(stocks)}")

        output_path = year_dir / f"level2_ops_{date}.csv"
        if output_path.exists():
            elapsed = time.time() - t0
            print(f"  -> exists, skip ({elapsed:.0f}s)")
            skip += 1
            continue

        success = run_day(
            archive=archive,
            date=date,
            stocks_plain=stocks,
            extract_dir=extract_dir,
            output_dir=str(year_dir),
            dry_run=args.dry_run,
            python=args.python,
        )

        elapsed = time.time() - t0
        if success:
            ok += 1
            print(f"  -> done ({elapsed:.0f}s)")
        else:
            fail += 1
            failures.append({"date": date, "archive": archive, "elapsed_s": round(elapsed, 0)})
            print(f"  [FAIL] ({elapsed:.0f}s)")

        # clean temp dir (unless dry-run)
        if not args.dry_run:
            temp_path = PROJECT / args.temp_dir / date
            if temp_path.exists():
                shutil.rmtree(temp_path, ignore_errors=True)

    # final cleanup temp_extract
    temp_root = PROJECT / args.temp_dir
    if not args.dry_run and temp_root.exists():
        remaining = list(temp_root.iterdir())
        if not remaining:
            shutil.rmtree(temp_root, ignore_errors=True)
        else:
            print(f"\n[WARN] temp_extract has {len(remaining)} leftover dirs")

    # write failure log
    if failures:
        fail_path = output_base / "failed_dates.csv"
        with open(fail_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["date", "archive", "elapsed_s"])
            w.writeheader()
            w.writerows(failures)
        print(f"\nFailure log: {fail_path} ({len(failures)} dates)")

    total_elapsed = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"Done: OK={ok} SKIP={skip} FAIL={fail} / {len(rows)}")
    print(f"Total time: {total_elapsed/60:.0f} min ({total_elapsed/3600:.1f} hr)")
    print(f"Output: {output_base}")


if __name__ == "__main__":
    main()
