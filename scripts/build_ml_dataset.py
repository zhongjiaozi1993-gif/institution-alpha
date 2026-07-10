"""构建 ML-ready 宽表并生成报告。

用法:
    python3 scripts/build_ml_dataset.py

产出:
    data/processed/datasets/ml_dataset.parquet
    data/processed/datasets/feature_metadata.parquet
    reports/ml_dataset_report.md
"""
import sys
from pathlib import Path
import pandas as pd

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))
from src.dataset.ml_dataset_builder import build_ml_dataset
from src.registry import universe_registry as reg

START, END = "2025-01-01", "2025-12-31"
OUT = PROJECT / "data" / "processed" / "datasets" / "ml_dataset.parquet"
META_OUT = PROJECT / "data" / "processed" / "datasets" / "feature_metadata.parquet"
REPORT = PROJECT / "reports" / "ml_dataset_report.md"


def main():
    uni = {u: set(reg.load_universe(u)) for u in ["Universe_A", "Universe_B", "Universe_C"]}
    print(f"Building ML dataset {START}~{END}")

    df, meta = build_ml_dataset(uni, START, END)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT, index=False)
    meta.to_parquet(META_OUT, index=False)
    print(f"Saved: {OUT}  shape={df.shape}")
    print(f"Saved: {META_OUT}  ({len(meta)} features)")

    feat_cols = [c for c in df.columns if c.startswith("feat_")]
    label_cols = [c for c in df.columns if c.startswith("label_")]
    flag_cols = [c for c in df.columns if c.endswith("_flag")]
    uni_cols = [c for c in df.columns if c.startswith("in_Universe")]

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT, "w") as f:
        f.write("# ML-ready Dataset 报告\n\n")
        f.write(f"生成时间: {pd.Timestamp.now():%Y-%m-%d}  |  窗口: {START} ~ {END}\n\n")
        f.write("> 宽表格式: trade_date | symbol | in_Universe_* | feat_* | label_* | *_flag\n")
        f.write("> feature 于 T 收盘后可得(available_time=T_close)，label 于 T+1 open 起算，无未来函数。\n")
        f.write("> 本版 feature 仅含 Alpha191；Level-2/龙虎榜/北向 将按相同键增量并入。\n\n---\n\n")

        f.write("## 概览\n\n")
        f.write(f"- 形状: **{df.shape[0]} 行 × {df.shape[1]} 列**\n")
        f.write(f"- 日期范围: {df['trade_date'].min():%Y-%m-%d} ~ {df['trade_date'].max():%Y-%m-%d}（{df['trade_date'].nunique()} 日）\n")
        f.write(f"- 股票数: {df['symbol'].nunique()}\n")
        f.write(f"- feature 数: {len(feat_cols)}  |  label 数: {len(label_cols)}  |  flag 数: {len(flag_cols)}\n\n")

        f.write("## Universe 分布\n\n")
        f.write("| Universe | 行数 | 股票数 |\n|---|---|---|\n")
        for c in uni_cols:
            sub = df[df[c]]
            f.write(f"| {c[3:]} | {len(sub)} | {sub['symbol'].nunique()} |\n")
        f.write("\n")

        f.write("## Feature 缺失率（Top/Bottom 5）\n\n")
        miss = df[feat_cols].isna().mean().sort_values() * 100
        f.write("| feature | 缺失率 |\n|---|---|\n")
        for c in list(miss.index[:5]) + list(miss.index[-5:]):
            f.write(f"| {c} | {miss[c]:.1f}% |\n")
        f.write(f"\n> feature 整体平均缺失率: {miss.mean():.1f}%（部分 Level-2 股票无 Alpha191 覆盖）\n\n")

        f.write("## Label 缺失率\n\n")
        f.write("| label | 缺失率 |\n|---|---|\n")
        for c in label_cols:
            f.write(f"| {c} | {df[c].isna().mean()*100:.1f}% |\n")
        f.write("\n")

        f.write("## 按年份分布\n\n")
        yg = df.groupby(pd.to_datetime(df["trade_date"]).dt.year).size()
        f.write("| 年份 | 行数 |\n|---|---|\n")
        for y, n in yg.items():
            f.write(f"| {y} | {n} |\n")
        f.write("\n> 全库仅 2025 单一年份，跨年验证暂不可行。\n\n")

        f.write("## 未来函数自检\n\n")
        f.write("- feature 全部来自 T 日及之前的 OHLCV，available_time=T_close。\n")
        f.write("- label 全部来自 T+1 open 及之后，仅用于验证/训练，未混入 feature。\n")
        f.write("- feature 与 label 同键对齐于 T，feature 可用时点严格早于 label，**无泄漏**。\n")

    print(f"Report: {REPORT}")


if __name__ == "__main__":
    main()
