"""
Level-2 ops 输出质量校验 — 扫描 parquet 目录，统计每日期货量和异常

用法:
  python scripts/check_level2_ops_outputs.py --ops-dir data/processed/level2_ops/2025

输出:
  <ops-dir>/ops_quality_report.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT = Path(__file__).resolve().parent.parent


def read_ops_file(path: Path) -> pd.DataFrame | None:
    """读取 CSV 或 parquet，返回 DataFrame 或 None"""
    if path.suffix == ".csv":
        try:
            return pd.read_csv(path, low_memory=False)
        except Exception:
            return None
    else:
        try:
            return pd.read_parquet(path)
        except Exception:
            return None


def check_ops_file(path: Path) -> dict:
    """检查单个输出文件的质量，返回一行报告"""
    info: dict = {
        "date": "",
        "file": path.name,
        "file_size_mb": round(path.stat().st_size / 1_048_576, 2),
        "n_rows": 0,
        "n_stocks": 0,
        "n_buy_ops": 0,
        "n_sell_ops": 0,
        "total_buy_wan": 0.0,
        "total_sell_wan": 0.0,
        "n_missing_stock_code": 0,
        "n_missing_direction": 0,
        "n_missing_amount": 0,
        "status": "ok",
    }

    df = read_ops_file(path)
    if df is None:
        info["status"] = "read_error"
        return info

    if df.empty:
        info["status"] = "empty"
        return info

    info["n_rows"] = len(df)

    # date
    if "date" in df.columns:
        dates = df["date"].dropna().unique()
        info["date"] = str(dates[0]) if len(dates) > 0 else "missing"
    else:
        info["n_missing_stock_code"] = info["n_rows"]
        info["status"] = "no_date_col"

    # stock_code
    if "stock_code" in df.columns:
        info["n_missing_stock_code"] = int(df["stock_code"].isna().sum())
        info["n_stocks"] = int(df["stock_code"].nunique())
    else:
        info["n_missing_stock_code"] = info["n_rows"]

    # direction
    if "direction" in df.columns:
        info["n_missing_direction"] = int(df["direction"].isna().sum())
        info["n_buy_ops"] = int((df["direction"] == "BUY").sum())
        info["n_sell_ops"] = int((df["direction"] == "SELL").sum())
    else:
        info["n_missing_direction"] = info["n_rows"]

    # amount — try common column names
    amt_col = None
    for c in ["amount_wan", "total_amount_wan"]:
        if c in df.columns:
            amt_col = c
            break
    if amt_col:
        info["n_missing_amount"] = int(df[amt_col].isna().sum())
        buy_mask = df["direction"] == "BUY" if "direction" in df.columns else pd.Series(False, index=df.index)
        sell_mask = df["direction"] == "SELL" if "direction" in df.columns else pd.Series(False, index=df.index)
        info["total_buy_wan"] = round(float(df.loc[buy_mask, amt_col].sum()), 1)
        info["total_sell_wan"] = round(float(df.loc[sell_mask, amt_col].sum()), 1)

    # 综合判断 status
    if info["status"] == "ok":
        if info["n_rows"] == 0:
            info["status"] = "empty"
        elif info["n_stocks"] == 0:
            info["status"] = "no_stocks"
        elif info["n_buy_ops"] == 0 and info["n_sell_ops"] == 0:
            info["status"] = "no_ops"
        elif info["n_missing_stock_code"] > info["n_rows"] * 0.1:
            info["status"] = "high_missing_stock"

    return info


def main():
    ap = argparse.ArgumentParser(description="Level-2 ops 输出质量校验")
    ap.add_argument("--ops-dir", required=True,
                    help="parquet 目录, e.g. data/processed/level2_ops/2025")
    args = ap.parse_args()

    ops_dir = Path(args.ops_dir)
    if not ops_dir.is_absolute():
        ops_dir = PROJECT / ops_dir

    if not ops_dir.exists():
        print(f"[ERROR] ops-dir not found: {ops_dir}")
        sys.exit(1)

    parquets = sorted(ops_dir.glob("level2_ops_*.csv")) or sorted(ops_dir.glob("level2_ops_*.parquet"))
    if not parquets:
        print(f"No output files found in {ops_dir}")
        sys.exit(0)

    print(f"Checking {len(parquets)} output files in {ops_dir}")

    rows = []
    for pq in parquets:
        row = check_ops_file(pq)
        rows.append(row)

    report = pd.DataFrame(rows)

    # 按日期排序
    if "date" in report.columns:
        report = report.sort_values("date")

    # 汇总统计
    print(f"\n{'='*70}")
    print(f"Total files:   {len(report)}")
    print(f"Total rows:    {report['n_rows'].sum()}")
    print(f"Total stocks:  {report['n_stocks'].sum()} (unique per day)")
    print(f"Total BUY ops: {report['n_buy_ops'].sum()}")
    print(f"Total SELL ops:{report['n_sell_ops'].sum()}")
    total_buy = report["total_buy_wan"].sum()
    total_sell = report["total_sell_wan"].sum()
    print(f"Total buy wan: {total_buy:,.0f}")
    print(f"Total sell wan:{total_sell:,.0f}")
    total_size = report["file_size_mb"].sum()
    print(f"Total size:    {total_size:.1f} MB ({total_size/1024:.2f} GB)")
    print(f"Avg size/day:  {report['file_size_mb'].mean():.2f} MB")

    # 状态分布
    status_counts = report["status"].value_counts()
    print(f"\nStatus distribution:")
    for s, c in status_counts.items():
        flag = " [BAD]" if s not in ("ok",) else ""
        print(f"  {s}: {c}{flag}")

    # 异常汇总
    bad = report[report["status"] != "ok"]
    if len(bad) > 0:
        print(f"\nAbnormal files ({len(bad)}):")
        for _, row in bad.iterrows():
            print(f"  {row['date']} {row['file']}: {row['status']}")

    # 空文件
    empty = report[report["n_rows"] == 0]
    if len(empty) > 0:
        print(f"\nEmpty files: {len(empty)}")
        for _, row in empty.iterrows():
            print(f"  {row['date']} {row['file']}")

    # 输出
    out_path = ops_dir / "ops_quality_report.csv"
    report.to_csv(out_path, index=False)
    print(f"\nReport saved: {out_path}")

    # Exit code: non-zero if any bad files
    if len(bad) > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
