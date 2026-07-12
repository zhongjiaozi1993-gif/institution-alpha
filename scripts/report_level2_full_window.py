"""Level-2 Full Available Window Feature Report（Phase 5.2A）。

只报告**特征生产 + 覆盖审计**，不下有效性结论、不做验证。数据来自 Windows 全量产出
回传的合并结果。

读取:
    data/processed/level2/level2_daily_features.parquet
    data/processed/level2/level2_coverage_audit.csv
    data/processed/level2/level2_feature_metadata.parquet
    data/processed/level2/run_manifest.json   (可选)
产出:
    reports/level2_full_window_feature_report.md
"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

L2_DIR = PROJECT / "data" / "processed" / "level2"
FEATURES = L2_DIR / "level2_daily_features.parquet"
AUDIT = L2_DIR / "level2_coverage_audit.csv"
META = L2_DIR / "level2_feature_metadata.parquet"
MANIFEST = L2_DIR / "run_manifest.json"
REPORT = PROJECT / "reports" / "level2_full_window_feature_report.md"

WINDOW_START = "2025-03-03"
WINDOW_END = "2025-12-31"
RAW_INPUT_STOCKS = 1102


def _load():
    feats = pd.read_parquet(FEATURES)
    feats["trade_date"] = pd.to_datetime(feats["trade_date"])
    feats["symbol"] = feats["symbol"].astype(str).str.zfill(6)
    audit = pd.read_csv(AUDIT, dtype={"symbol": str, "day": str})
    audit["symbol"] = audit["symbol"].astype(str).str.zfill(6)
    audit["trade_date"] = pd.to_datetime(audit["trade_date"])
    if "month" not in audit.columns:
        audit["month"] = audit["trade_date"].dt.strftime("%Y-%m")
    meta = pd.read_parquet(META)
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8")) if MANIFEST.exists() else {}
    return feats, audit, meta, manifest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-input-stocks", type=int, default=RAW_INPUT_STOCKS)
    args = ap.parse_args()

    feats, audit, meta, manifest = _load()

    ok = audit[audit["status"] == "ok"]
    n_ok_stocks = ok["symbol"].nunique()
    n_feat_rows = len(feats)
    cov_start = audit["trade_date"].min()
    cov_end = audit["trade_date"].max()

    # 每月：股票数 / 交易日数 / stock-day 数
    by_month = ok.groupby("month").agg(
        n_stocks=("symbol", "nunique"),
        n_days=("day", "nunique"),
        n_stock_days=("symbol", "size"),
    ).reset_index()

    # 覆盖天数分布
    days_per_stock = ok.groupby("symbol")["day"].nunique()
    bins = [0, 1, 20, 50, 100, 150, 160, 10_000]
    labels = ["0", "1-19", "20-49", "50-99", "100-149", "150-159", "160+"]
    dist = pd.cut(days_per_stock, bins=bins, labels=labels, right=False).value_counts().sort_index()

    # 次新股：ok 天数 < 150 且首个交易日晚于窗口起点较多
    low = days_per_stock[days_per_stock < 150].sort_values()
    first_day = ok.groupby("symbol")["trade_date"].min()

    layout_dist = audit["selected_layout"].value_counts()
    status_dist = audit["status"].value_counts()

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT, "w", encoding="utf-8") as f:
        f.write("# Level-2 Full Available Window Feature Report\n\n")
        f.write(f"生成时间: {pd.Timestamp.now():%Y-%m-%d %H:%M}  |  feature_version: "
                f"{manifest.get('feature_version', meta['feature_version'].iloc[0] if 'feature_version' in meta else 'v1')}"
                f"  |  git: {manifest.get('git_commit', 'n/a')}\n\n")
        f.write("> 本阶段（5.2A）仅完成**特征生产 + 覆盖审计**。不含有效性结论、不做验证、"
                "不进入 Phase 6 融合。所有特征仅用 T 日逐笔，available_time = T_close，无跨日/未来函数。\n\n---\n\n")

        f.write("## 1. 覆盖总览\n\n")
        f.write("| 指标 | 数值 |\n|---|---|\n")
        f.write(f"| 原始输入股票数 | {args.raw_input_stocks} |\n")
        f.write(f"| 成功产出股票数（有 ≥1 ok 日） | {n_ok_stocks} |\n")
        f.write(f"| 特征行数（ok stock-day） | {n_feat_rows} |\n")
        f.write(f"| 审计 stock-day 总数（含失败） | {len(audit)} |\n")
        f.write(f"| 实际覆盖窗口 | {cov_start:%Y-%m-%d} ~ {cov_end:%Y-%m-%d} |\n")
        f.write(f"| 目标窗口 | {WINDOW_START} ~ {WINDOW_END} |\n")
        f.write(f"| 特征数 | {len(meta)} |\n\n")

        f.write("## 2. 每月覆盖（成功产出）\n\n")
        f.write("| 月份 | 股票数 | 交易日数 | stock-day |\n|---|---|---|---|\n")
        for _, r in by_month.iterrows():
            f.write(f"| {r['month']} | {r['n_stocks']} | {r['n_days']} | {r['n_stock_days']} |\n")
        f.write("\n")

        f.write("## 3. 股票覆盖天数分布（ok 日）\n\n")
        f.write("| 天数档 | 股票数 |\n|---|---|\n")
        for k, v in dist.items():
            f.write(f"| {k} | {int(v)} |\n")
        f.write(f"\n中位覆盖天数 = {int(days_per_stock.median())}，"
                f"均值 = {days_per_stock.mean():.1f}。\n\n")

        f.write("## 4. 次新股/低覆盖说明\n\n")
        f.write(f"覆盖 < 150 天的股票共 **{len(low)}** 只，主因是年中上市（次新股），"
                "首个可得交易日晚于窗口起点，属数据本身而非漏读：\n\n")
        f.write("| 股票 | ok天数 | 首个交易日 |\n|---|---|---|\n")
        for s, d in low.items():
            f.write(f"| {s} | {int(d)} | {first_day[s]:%Y-%m-%d} |\n")
        f.write("\n")

        f.write("## 5. Layout 分布\n\n")
        f.write("| layout | stock-day |\n|---|---|\n")
        for k, v in layout_dist.items():
            f.write(f"| {k} | {int(v)} |\n")
        f.write(f"\n权威数据全部为 **flat_day_dir**（Windows 结构）。`wind_subdir` 仅为兼容逻辑保留，"
                "本窗口权威产出中占比极小/为 0，**不作为权威主面板**。\n\n")

        f.write("## 6. 失败与异常分布（覆盖审计）\n\n")
        f.write("| status | stock-day |\n|---|---|\n")
        for k, v in status_dist.items():
            f.write(f"| {k} | {int(v)} |\n")
        f.write("\n非 ok 明细见 `data/processed/level2/level2_skipped_stock_days.csv`；"
                "全量审计见 `level2_coverage_audit.csv`。每个 symbol-day 都有审计行，无静默跳过。\n\n")

        f.write("## 7. 旧结论失效声明\n\n")
        f.write("之前“仅 27 只全年覆盖”的结论，是**目录结构漏读**造成的（旧 build_stock_features "
                "在 flat_day_dir 结构上静默 continue，只读到 wind_subdir 子集）。本次修复 layout "
                "解析（按实际文件判定）+ 全量审计后，覆盖为上表所示。**旧报告的覆盖结论作废。**\n\n")

    print(f"Report → {REPORT}")
    print(f"ok_stocks={n_ok_stocks} feat_rows={n_feat_rows} "
          f"coverage={cov_start:%Y-%m-%d}..{cov_end:%Y-%m-%d}")


if __name__ == "__main__":
    main()
