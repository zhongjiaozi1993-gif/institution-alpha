"""构建 Level-2 日频特征宽表（Phase 5）。

用法:
    python3 scripts/build_level2_features.py                 # 全 Universe_C
    python3 scripts/build_level2_features.py --limit 20      # 只跑前 20 只（冒烟）

产出:
    data/processed/level2/level2_daily_features.parquet      # trade_date|symbol|l2_*
    data/processed/level2/level2_feature_metadata.parquet    # 特征元信息
    reports/level2_feature_report.md                         # 覆盖率/分布/样本
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))
from src.features import level2_feature_builder as fb
from src.registry import universe_registry as reg

OUT_DIR = PROJECT / "data" / "processed" / "level2"
FEATURES_PATH = OUT_DIR / "level2_daily_features.parquet"
META_PATH = OUT_DIR / "level2_feature_metadata.parquet"
SKIPPED_PATH = OUT_DIR / "level2_skipped_stock_days.csv"
REPORT = PROJECT / "reports" / "level2_feature_report.md"


def _skipped_frame(audit_df: pd.DataFrame) -> pd.DataFrame:
    """覆盖审计表 → 非 ok 的 stock-day（symbol, day, trade_date, month, reason）。"""
    cols = ["symbol", "day", "trade_date", "month", "reason"]
    if audit_df.empty:
        return pd.DataFrame(columns=cols)
    sk = audit_df[audit_df["status"] != "ok"].copy()
    if sk.empty:
        return pd.DataFrame(columns=cols)
    sk["reason"] = sk["status"] + ": " + sk["reason"].fillna("")
    return sk[cols].sort_values(["symbol", "day"]).reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser(description="构建 Level-2 日频特征宽表")
    ap.add_argument("--universe", default="Universe_C")
    ap.add_argument("--limit", type=int, default=None, help="只处理前 N 只股票（调试）")
    args = ap.parse_args()

    codes = sorted(set(reg.load_universe(args.universe)))
    if args.limit:
        codes = codes[: args.limit]
    print(f"{args.universe}: {len(codes)} stocks → building Level-2 daily features")

    df, audit_df = fb.build_all_features(codes)
    meta = fb.feature_metadata()
    if df.empty:
        print("无特征产出（检查 data/single_stock/*/raw）")
        return

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(FEATURES_PATH, index=False)
    meta.to_parquet(META_PATH, index=False)
    audit_df.to_csv(OUT_DIR / "level2_coverage_audit.csv", index=False)
    print(f"Saved {len(df)} rows × {len(fb.FEATURE_NAMES)} feats → {FEATURES_PATH}")

    skipped_df = _skipped_frame(audit_df)
    skipped_df.to_csv(SKIPPED_PATH, index=False)
    print(f"Skipped {len(skipped_df)} non-ok stock-days → {SKIPPED_PATH}")

    _write_report(df, meta, args.universe, skipped_df)
    print(f"Report → {REPORT}")


def _write_skipped_section(f, skipped_df: pd.DataFrame):
    """报告中的跳过样本分布（按原因 / 按月 / 按股票 top）。"""
    f.write("## 跳过样本分布（损坏/解码失败）\n\n")
    if skipped_df.empty:
        f.write("无跳过样本。\n\n")
        return
    f.write(f"共跳过 **{len(skipped_df)}** 个 stock-day，明细见 "
            "`data/processed/level2/level2_skipped_stock_days.csv`。\n\n")
    by_reason = skipped_df["reason"].value_counts()
    f.write("**按原因**：" + "，".join(f"{k}={v}" for k, v in by_reason.items()) + "\n\n")
    by_month = skipped_df["month"].value_counts().sort_index()
    if len(by_month):
        f.write("**按月**：" + "，".join(f"{k}={v}" for k, v in by_month.items()) + "\n\n")
    by_sym = skipped_df["symbol"].value_counts().head(10)
    f.write("**按股票(top10)**：" + "，".join(f"{k}={v}" for k, v in by_sym.items()) + "\n\n")


def _write_report(df: pd.DataFrame, meta: pd.DataFrame, universe: str, skipped_df: pd.DataFrame):
    feat_cols = fb.FEATURE_NAMES
    n_rows = len(df)
    n_stocks = df["symbol"].nunique()
    n_dates = df["trade_date"].nunique()
    per_date = df.groupby("trade_date")["symbol"].nunique()
    dense_dates = int((per_date >= 5).sum())
    dense_rows = int(per_date[per_date >= 5].sum())

    # 非零率（多数 Level-2 特征在小单日为 0，非零率反映信号密度）
    nonzero = {c: float((df[c] != 0).mean()) for c in feat_cols}
    desc = df[feat_cols].describe().T

    with open(REPORT, "w") as f:
        f.write("# Level-2 日频特征报告（Phase 5, v1）\n\n")
        f.write(f"生成时间: {pd.Timestamp.now():%Y-%m-%d %H:%M}  |  universe: {universe}\n\n")
        f.write("> 每股每日一行；所有特征仅用 **T 日逐笔**，available_time = **T_close**，"
                "无跨日窗口、无未来函数。label 从 T+1 开盘起算，错开可用时点。\n\n---\n\n")

        f.write("## 覆盖\n\n")
        f.write("| 指标 | 数值 |\n|---|---|\n")
        f.write(f"| 特征数 | {len(feat_cols)} |\n")
        f.write(f"| 行数(stock-day) | {n_rows} |\n")
        f.write(f"| 股票数 | {n_stocks} |\n")
        f.write(f"| 交易日数 | {n_dates} |\n")
        f.write(f"| 单日≥5股的日数 | {dense_dates}（保留 {dense_rows} 行，可做截面 RankIC） |\n")
        f.write(f"| 平均股票/日 | {n_rows / max(n_dates,1):.1f} |\n")
        f.write(f"| 跳过(损坏CSV)stock-day | {len(skipped_df)} |\n\n")
        f.write("> Level-2 覆盖**时间高度不均**：2025-01 的约 17 个交易日为近全池宽截面（~175 只），"
                "其余交易日多为 ~27 只深度股（全年逐笔跟踪）。截面验证在 ≥5 股的日子上进行；"
                "宽截面集中在年初，需注意跨时段可比性。\n\n")

        _write_skipped_section(f, skipped_df)

        f.write("## 特征清单（分组）\n\n")
        f.write("| 特征 | 分组 | 说明 | 非零率 | 均值 | 标准差 |\n|---|---|---|---|---|---|\n")
        for _, m in meta.iterrows():
            c = m["feature"]
            f.write(f"| {c} | {m['group']} | {m['description']} | "
                    f"{nonzero[c]*100:.0f}% | {desc.loc[c,'mean']:.4f} | {desc.loc[c,'std']:.4f} |\n")

        f.write("\n## 关键特征分位（p10/p50/p90）\n\n")
        key = ["l2_net_active_ratio", "l2_big_net_ratio", "l2_cluster_buy_intensity",
               "l2_cluster_net_wan", "l2_avg_cluster_hhi", "l2_late_net_ratio", "l2_intraday_ret"]
        f.write("| 特征 | p10 | p50 | p90 |\n|---|---|---|---|\n")
        for c in key:
            q = df[c].quantile([0.1, 0.5, 0.9])
            f.write(f"| {c} | {q.iloc[0]:.4f} | {q.iloc[1]:.4f} | {q.iloc[2]:.4f} |\n")

        f.write("\n## 样本（最活跃 5 行，按成交额）\n\n")
        top = df.nlargest(5, "l2_amount_yi")[
            ["trade_date", "symbol", "l2_amount_yi", "l2_net_active_ratio",
             "l2_cluster_count", "l2_cluster_buy_intensity", "l2_big_net_ratio"]]
        f.write(top.to_markdown(index=False))
        f.write("\n\n## 已知限制\n\n")
        f.write("1. 覆盖稀疏且不均，跨日滚动特征暂缺（避免缺口泄漏），v1 全为单日 T 特征。\n")
        f.write("2. 撤单率因沪深口径不一致暂未纳入（SH 委托类型=D，SZ 撤单在成交流），后续统一。\n")
        f.write("3. 小盘股多数日 DBSCAN 无集群（cluster 特征为 0），零值本身即“无机构拆单”的信息。\n")
        f.write("4. DBSCAN 参数沿用 eps=0.15/min_samples=5/min_total=100万，未逐股调参。\n")


if __name__ == "__main__":
    main()
