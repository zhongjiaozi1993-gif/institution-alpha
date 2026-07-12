"""构建统一 label 表并生成报告。

用法:
    python3 scripts/build_labels.py

产出:
    data/processed/labels/labels.parquet
    reports/label_report.md
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))
from src.features.label_builder import build_labels, DEFAULT_HORIZONS
from src.registry import universe_registry as reg

START, END = "2025-01-01", "2025-12-31"
OUT = PROJECT / "data" / "processed" / "labels" / "labels.parquet"
REPORT = PROJECT / "reports" / "label_report.md"
L2_FEATURES = PROJECT / "data" / "processed" / "level2" / "level2_daily_features.parquet"


def l2_symbols() -> set:
    """Level-2 特征宽表的股票集合（含未进 universe 过滤的 L2 股票）。"""
    if not L2_FEATURES.exists():
        return set()
    s = pd.read_parquet(L2_FEATURES, columns=["symbol"])["symbol"]
    return set(s.astype(str).str.zfill(6).unique())


def universe_symbols() -> tuple[list[str], dict[str, set]]:
    """所有 universe 成员的并集 + 每个 universe 的成员集合。"""
    uni_sets = {}
    for uid in ["Universe_A", "Universe_B", "Universe_C"]:
        try:
            uni_sets[uid] = set(reg.load_universe(uid))
        except FileNotFoundError:
            uni_sets[uid] = set()
    union = sorted(set().union(*uni_sets.values()))
    return union, uni_sets


def main():
    codes, uni_sets = universe_symbols()
    codes = sorted(set(codes) | l2_symbols())   # 并入全部 L2 池（含未进 universe 的补数股）
    print(f"Building labels for {len(codes)} stocks (universe union ∪ L2 pool), {START}~{END}")

    labels = build_labels(codes, START, END, DEFAULT_HORIZONS)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    labels.to_parquet(OUT, index=False)
    print(f"Saved: {OUT} ({len(labels)} rows)")

    label_cols = [f"label_{h}d" for h in DEFAULT_HORIZONS]
    excess_cols = [f"label_{h}d_excess_index" for h in DEFAULT_HORIZONS]
    all_label_cols = label_cols + excess_cols

    # ---- Report ----
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT, "w") as f:
        f.write("# Label 报告\n\n")
        f.write(f"生成时间: {pd.Timestamp.now():%Y-%m-%d}  |  窗口: {START} ~ {END}\n\n")
        f.write("> label 定义: `label_hd = open[T+1+h] / open[T+1] - 1`（T+1 开盘入场，无未来函数）。\n")
        f.write("> 超额 = 个股 - 中证1000(idx_000852) 同窗口收益。单位为小数收益。\n")
        f.write("> label 仅用于验证/训练，**不可作为 feature**。行业超额因无行业数据未生成。\n")
        f.write("> **远期覆盖**: 计算远期收益时保留 end_date 之后的行情再按信号日过滤输出，\n")
        f.write("> 故年末信号日的 label_10d/20d 不再被 end_date 截断为 NaN（残留缺失来自退市/停牌等无远期价个股）。\n\n---\n\n")

        f.write("## 概览\n\n")
        f.write(f"- 样本行数: **{len(labels)}**\n")
        f.write(f"- 股票数: {labels['symbol'].nunique()}  |  日期数: {labels['trade_date'].nunique()}\n")
        f.write(f"- 日期范围: {labels['trade_date'].min():%Y-%m-%d} ~ {labels['trade_date'].max():%Y-%m-%d}\n\n")

        f.write("## 各 label 统计\n\n")
        f.write("| label | 缺失率 | 均值 | 标准差 | p1 | p25 | 中位 | p75 | p99 |\n")
        f.write("|---|---|---|---|---|---|---|---|---|\n")
        for c in all_label_cols:
            s = labels[c]
            miss = s.isna().mean() * 100
            v = s.dropna()
            q = v.quantile([0.01, 0.25, 0.5, 0.75, 0.99]) if len(v) else {k: np.nan for k in [0.01,0.25,0.5,0.75,0.99]}
            f.write(f"| {c} | {miss:.1f}% | {v.mean():+.4f} | {v.std():.4f} | "
                    f"{q[0.01]:+.3f} | {q[0.25]:+.3f} | {q[0.5]:+.3f} | {q[0.75]:+.3f} | {q[0.99]:+.3f} |\n")
        f.write("\n> 缺失率随 horizon 增大而上升（末端 T+1+h 越界），属正常。\n\n")

        f.write("## 极端值检查\n\n")
        f.write("| label | <-50% | >+50% | max绝对值 |\n|---|---|---|---|\n")
        for c in label_cols:
            v = labels[c].dropna()
            f.write(f"| {c} | {(v<-0.5).sum()} | {(v>0.5).sum()} | {v.abs().max():.3f} |\n")
        f.write("\n")

        f.write("## 按月分布（label_5d 均值 / 样本数）\n\n")
        labels["month"] = pd.to_datetime(labels["trade_date"]).dt.to_period("M").astype(str)
        mg = labels.groupby("month")["label_5d"].agg(["mean", "count"])
        f.write("| 月份 | label_5d 均值 | 样本数 |\n|---|---|---|\n")
        for m, r in mg.iterrows():
            f.write(f"| {m} | {r['mean']:+.4f} | {int(r['count'])} |\n")
        f.write("\n")

        f.write("## 按 universe 分布（label_5d_excess_index 均值 / 样本数）\n\n")
        f.write("| Universe | 超额均值 | 样本数 |\n|---|---|---|\n")
        for uid, syms in uni_sets.items():
            sub = labels[labels["symbol"].isin(syms)]["label_5d_excess_index"].dropna()
            if len(sub):
                f.write(f"| {uid} | {sub.mean():+.4f} | {len(sub)} |\n")
        f.write("\n")

    print(f"Report: {REPORT}")


if __name__ == "__main__":
    main()
