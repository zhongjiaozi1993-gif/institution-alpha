r"""
Level-2 股票池规划 — 扫描 7z 归档，统计覆盖，筛选股票池

用法:
  python scripts/plan_level2_universe.py \
    --archive-root "C:/Users/1/Desktop/2025" \
    --years 2024,2025,2026 \
    --max-stocks 300 \
    --whitelist config/stock_whitelist.txt

输出:
  - data/processed/stock_universe/stock_coverage.csv
  - data/processed/stock_universe/selected_stocks.csv
  - data/processed/stock_universe/selected_stocks.txt
  - data/processed/stock_universe/dropped_stocks.csv
  - data/processed/stock_universe/process_plan.csv
"""

from __future__ import annotations

import argparse
import csv
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent

# ─── 代码前缀规则 ──────────────────────────────────────────────────

# 第一阶段保留的代码前缀
KEEP_PREFIXES = {"000", "001", "002", "003", "300", "301",
                 "600", "601", "603", "605"}

# 默认排除的前缀
DROP_PREFIXES = {"200", "900", "4", "8"}  # B股, 北交所/新三板

# 科创板需 --include-star 才纳入
STAR_PREFIX = "688"


def stock_code_from_path(path: str) -> str | None:
    """从 7z 内部路径提取纯数字股票代码, e.g. 20250102\\002516.SZ → 002516"""
    m = re.search(r'[/\\\\](\d{6})\.(SZ|SH)', path)
    return m.group(1) if m else None


def should_keep(stock: str, include_star: bool = False) -> bool:
    """代码前缀过滤"""
    if stock[:3] in KEEP_PREFIXES:
        return True
    if include_star and stock.startswith(STAR_PREFIX):
        return True
    return False


# ─── Archive 扫描 ──────────────────────────────────────────────────


def find_archives(archive_root: Path, years: list[str] | None) -> dict[str, list[Path]]:
    """扫描 archive_root 下所有 7z 文件, 按年份分组.
    同日有 .7z 和 .zip 时优先 .7z.

    Returns {year: [archive_path, ...]}
    """
    by_date: dict[str, Path] = {}

    for ext in ["*.7z", "*.7Z", "*.zip"]:
        for p in archive_root.rglob(ext):
            date_match = re.search(r'(\d{8})', p.name)
            if not date_match:
                continue
            date_str = date_match.group(1)
            year = date_str[:4]
            if years and year not in years:
                continue
            if date_str not in by_date or p.suffix.lower() == '.7z':
                by_date[date_str] = p

    archives: dict[str, list[Path]] = defaultdict(list)
    for date_str, p in sorted(by_date.items()):
        year = date_str[:4]
        archives[year].append(p)
    return dict(archives)


def scan_archives(archive_root: Path, years: list[str] | None,
                  seven_zip: str = "7z") -> dict[str, dict[str, int]]:
    """扫描所有 7z 文件, 统计 stock × year 出现天数.

    Returns {stock: {year: days}}
    """
    archives = find_archives(archive_root, years)
    coverage: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    for year, files in sorted(archives.items()):
        print(f"\nScanning {year}: {len(files)} archives")
        for i, archive in enumerate(files):
            if (i + 1) % 50 == 0 or i == 0:
                print(f"  [{i+1}/{len(files)}] {archive.name}...")

            try:
                result = subprocess.run(
                    [seven_zip, "l", str(archive), "-ba"],
                    capture_output=True, text=True, timeout=120,
                )
                if result.returncode != 0:
                    continue
                stocks_seen = set()
                for line in result.stdout.splitlines():
                    stock = stock_code_from_path(line.strip())
                    if stock and stock not in stocks_seen:
                        stocks_seen.add(stock)
                        coverage[stock][year] += 1
            except (subprocess.TimeoutExpired, FileNotFoundError) as e:
                print(f"    [WARN] skip {archive.name}: {e}")
                continue

    return dict(coverage)


# ─── 筛选逻辑 ──────────────────────────────────────────────────────


def apply_filters(coverage: dict[str, dict[str, int]],
                  years: list[str],
                  whitelist: set[str] | None = None,
                  blacklist: set[str] | None = None,
                  min_days_per_year: int = 120,
                  min_total_days: int = 180,
                  max_stocks: int = 300,
                  include_star: bool = False,
                  ) -> tuple[list[dict], list[dict]]:
    """筛选股票, 返回 (selected, dropped)"""

    if whitelist is None:
        whitelist = set()
    if blacklist is None:
        blacklist = set()

    def _with_days(stock, total_days, reason, year_days):
        row = {"stock": stock, "total_days": total_days, "reason": reason}
        for y, d in year_days.items():
            row[f"days_{y}"] = d
        return row

    selected = []
    dropped = []

    for stock, year_days in sorted(coverage.items()):
        total_days = sum(year_days.values())

        # 黑名单优先
        if stock in blacklist:
            dropped.append(_with_days(stock, total_days, "blacklist", year_days))
            continue

        # 白名单强制保留
        if stock in whitelist:
            selected.append(_with_days(stock, total_days, "whitelist", year_days))
            continue

        # 代码前缀过滤
        if not should_keep(stock, include_star):
            dropped.append(_with_days(stock, total_days, "bad_prefix", year_days))
            continue

        # 总天数不够
        if total_days < min_total_days:
            dropped.append(_with_days(stock, total_days, "low_total_days", year_days))
            continue

        # 单年天数不够 (在要求的年份中)
        low_year = False
        for y in years:
            if year_days.get(y, 0) < min_days_per_year:
                low_year = True
                break
        if low_year:
            dropped.append(_with_days(stock, total_days, "low_year_days", year_days))
            continue

        selected.append(_with_days(stock, total_days, "ok", year_days))

    # 按总覆盖天数排序
    selected.sort(key=lambda x: -x["total_days"])
    dropped.sort(key=lambda x: -x["total_days"])

    # 截断到 max_stocks (白名单不受容量限制)
    if len(selected) > max_stocks:
        wl = [s for s in selected if s["reason"] == "whitelist"]
        non_wl = [s for s in selected if s["reason"] != "whitelist"]
        keep_non_wl = max(0, max_stocks - len(wl))
        overflow = non_wl[keep_non_wl:]
        selected = wl + non_wl[:keep_non_wl]
        for s in overflow:
            s["reason"] = "over_capacity"
        dropped.extend(overflow)

    return selected, dropped


# ─── 处理计划生成 ──────────────────────────────────────────────────


def build_process_plan(selected_stocks: list[dict],
                       archive_root: Path,
                       years: list[str],
                       seven_zip: str = "7z") -> list[dict]:
    """生成逐日处理计划: 每天一个 archive + 该日需要的股票列表"""
    stock_set = {s["stock"] for s in selected_stocks}
    archives = find_archives(archive_root, years)

    rows = []
    for year, files in sorted(archives.items()):
        for archive in files:
            date_match = re.search(r'(\d{8})', archive.name)
            if not date_match:
                continue
            date_str = date_match.group(1)

            # 列出该 archive 中有哪些 selected stocks
            try:
                result = subprocess.run(
                    [seven_zip, "l", str(archive), "-ba"],
                    capture_output=True, text=True, timeout=120,
                )
                if result.returncode != 0:
                    continue
                stocks_in_archive = set()
                for line in result.stdout.splitlines():
                    code = stock_code_from_path(line.strip())
                    if code:
                        stocks_in_archive.add(code)
            except Exception:
                continue

            needed = sorted(stocks_in_archive & stock_set)
            if not needed:
                continue

            rows.append({
                "date": date_str,
                "year": year,
                "archive": str(archive),
                "n_stocks": len(needed),
                "stocks": ",".join(needed),
            })

    return rows


# ─── CSV 输出 ──────────────────────────────────────────────────────


def coverage_to_rows(coverage: dict[str, dict[str, int]],
                     years: list[str]) -> list[dict]:
    rows = []
    for stock, yd in sorted(coverage.items()):
        row = {"stock": stock, "total_days": sum(yd.values())}
        for y in years:
            row[f"days_{y}"] = yd.get(y, 0)
        rows.append(row)
    rows.sort(key=lambda r: -r["total_days"])
    return rows


# ─── Main ───────────────────────────────────────────────────────────


def main():
    ap = argparse.ArgumentParser(description="Level-2 股票池规划")
    ap.add_argument("--archive-root", required=True,
                    help="Level-2 7z 根目录, e.g. D:\\level2_archives")
    ap.add_argument("--out-dir", default="data/processed/stock_universe",
                    help="输出目录 (默认 data/processed/stock_universe)")
    ap.add_argument("--years", default="",
                    help="逗号分隔年份, 空=全部, e.g. 2024,2025,2026")
    ap.add_argument("--min-days-per-year", type=int, default=120,
                    help="单年最低覆盖天数 (默认 120)")
    ap.add_argument("--min-total-days", type=int, default=180,
                    help="总最低覆盖天数 (默认 180)")
    ap.add_argument("--max-stocks", type=int, default=300,
                    help="最大保留股票数 (默认 300)")
    ap.add_argument("--include-star", action="store_true",
                    help="纳入 688 科创板")
    ap.add_argument("--whitelist", default="",
                    help="白名单 txt/csv, 一行一个股票代码")
    ap.add_argument("--blacklist", default="",
                    help="黑名单 txt/csv, 一行一个股票代码")
    ap.add_argument("--seven-zip", default="7z",
                    help="7z CLI 路径 (默认 7z)")
    ap.add_argument("--cache-scan", default="",
                    help="扫描缓存文件, 跳过重新扫描")
    args = ap.parse_args()

    years = [y.strip() for y in args.years.split(",") if y.strip()] or None
    archive_root = Path(args.archive_root)
    out_dir = PROJECT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    if not archive_root.exists():
        print(f"[ERROR] Archive root not found: {archive_root}")
        sys.exit(1)

    # 加载白名单/黑名单
    def _load_list(path: str) -> set[str]:
        if not path:
            return set()
        p = Path(path)
        if not p.exists():
            p = PROJECT / path
        if not p.exists():
            print(f"  [WARN] file not found: {path}, skipping")
            return set()
        codes = set()
        with open(p) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    codes.add(line.replace(".SZ", "").replace(".SH", ""))
        return codes

    whitelist = _load_list(args.whitelist)
    blacklist = _load_list(args.blacklist)

    print(f"Archive root: {archive_root}")
    print(f"Years: {years or 'all'}")
    print(f"STAR market: {'include' if args.include_star else 'exclude'}")
    print(f"Whitelist: {len(whitelist)} stocks, Blacklist: {len(blacklist)} stocks")
    print(f"Thresholds: per-year>={args.min_days_per_year}d, total>={args.min_total_days}d, max={args.max_stocks}")

    # Step 1: 扫描
    if args.cache_scan and Path(args.cache_scan).exists():
        print(f"\nLoading scan cache: {args.cache_scan}")
        import json
        with open(args.cache_scan) as f:
            raw = json.load(f)
        coverage = {stock: {str(y): d for y, d in years_dict.items()}
                    for stock, years_dict in raw.items()}
    else:
        print("\nScanning 7z archives (may be slow)...")
        coverage = scan_archives(archive_root, years, args.seven_zip)

    if not coverage:
        print("[ERROR] No stock data found in archives")
        sys.exit(1)

    print(f"\nFound {len(coverage)} stocks total")

    # Step 2: 筛选
    year_list = years or sorted({y for yd in coverage.values() for y in yd})
    selected, dropped = apply_filters(
        coverage, year_list,
        whitelist=whitelist, blacklist=blacklist,
        min_days_per_year=args.min_days_per_year,
        min_total_days=args.min_total_days,
        max_stocks=args.max_stocks,
        include_star=args.include_star,
    )

    # Step 3: 输出
    # stock_coverage.csv (全部股票)
    cov_rows = coverage_to_rows(coverage, year_list)
    cov_path = out_dir / "stock_coverage.csv"
    _write_csv(cov_path, cov_rows)
    print(f"\nstock_coverage: {cov_path} ({len(cov_rows)} stocks)")

    # selected_stocks.csv
    sel_path = out_dir / "selected_stocks.csv"
    _write_csv(sel_path, selected)
    print(f"selected_stocks: {sel_path} ({len(selected)} stocks)")

    # selected_stocks.txt (纯代码, 一行一个)
    txt_path = out_dir / "selected_stocks.txt"
    with open(txt_path, "w") as f:
        for s in selected:
            f.write(f"{s['stock']}\n")
    print(f"selected_stocks: {txt_path}")

    # dropped_stocks.csv
    drop_path = out_dir / "dropped_stocks.csv"
    _write_csv(drop_path, dropped)
    print(f"dropped_stocks: {drop_path} ({len(dropped)} stocks)")

    # 原因分布
    from collections import Counter
    reasons = Counter(d["reason"] for d in dropped)
    for r, c in reasons.most_common():
        print(f"  {r}: {c}")

    # process_plan.csv
    plan_rows = build_process_plan(selected, archive_root, year_list, args.seven_zip)
    plan_path = out_dir / "process_plan.csv"
    _write_csv(plan_path, plan_rows)
    print(f"\nprocess_plan: {plan_path} ({len(plan_rows)} days, "
          f"{sum(r['n_stocks'] for r in plan_rows)} stock-days)")

    # 打印摘要
    print(f"\n{'='*60}")
    print(f"Summary: {len(selected)} selected, {len(dropped)} dropped")
    print(f"Trading days to process: {len(plan_rows)}")
    for s in selected[:20]:
        ydays = " ".join(f"{y}:{s.get(f'days_{y}', '?')}d" for y in year_list[:4])
        print(f"  {s['stock']}  {ydays}  ({s['reason']})")
    if len(selected) > 20:
        print(f"  ... {len(selected)} stocks total")


def _write_csv(path: Path, rows: list[dict]):
    if not rows:
        path.write_text("")
        return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
