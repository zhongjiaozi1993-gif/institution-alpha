"""构建每日 tradable flags 并生成报告。

用法:
    python3 scripts/build_tradable_flags.py

产出:
    data/processed/tradable/tradable_flags.parquet
    reports/tradable_report.md
"""
import sys
from pathlib import Path
import pandas as pd

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))
from src.features.tradable_flag_builder import build_flags
from src.registry import universe_registry as reg

START, END = "2025-01-01", "2025-12-31"
OUT = PROJECT / "data" / "processed" / "tradable" / "tradable_flags.parquet"
REPORT = PROJECT / "reports" / "tradable_report.md"
L2_FEATURES = PROJECT / "data" / "processed" / "level2" / "level2_daily_features.parquet"


def l2_symbols() -> set:
    """Level-2 特征宽表的股票集合（含未进 universe 过滤的 L2 股票）。"""
    if not L2_FEATURES.exists():
        return set()
    s = pd.read_parquet(L2_FEATURES, columns=["symbol"])["symbol"]
    return set(s.astype(str).str.zfill(6).unique())

FLAG_COLS = ["suspend_flag", "limit_up_flag", "limit_down_flag", "st_flag",
             "new_stock_flag", "low_liquidity_flag", "buyable_flag",
             "sellable_flag", "tradable_flag"]


def main():
    codes = sorted(set().union(*[
        set(reg.load_universe(u)) for u in ["Universe_A", "Universe_B", "Universe_C"]
    ]) | l2_symbols())   # 并入全部 L2 池（含未进 universe 的补数股）
    print(f"Building tradable flags for {len(codes)} stocks (universe union ∪ L2 pool), {START}~{END}")

    flags = build_flags(codes, START, END)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    flags.to_parquet(OUT, index=False)
    print(f"Saved: {OUT} ({len(flags)} rows)")

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    total = len(flags)
    with open(REPORT, "w") as f:
        f.write("# Tradable Flag 报告\n\n")
        f.write(f"生成时间: {pd.Timestamp.now():%Y-%m-%d}  |  窗口: {START} ~ {END}\n\n")
        f.write("> flag 描述日期 T 的状态。涨跌停用 hfq close pct_change 近似（含 0.3% 容忍度）。\n")
        f.write("> st_flag 恒 False（无 ST 名单数据源）；new_stock_flag 仅对窗口内 IPO 生效。\n\n---\n\n")

        f.write("## 概览\n\n")
        f.write(f"- 样本行数: **{total}**  |  股票数: {flags['symbol'].nunique()}  |  日期数: {flags['trade_date'].nunique()}\n\n")

        f.write("## 各 flag 触发占比\n\n")
        f.write("| flag | True 数量 | 占比 |\n|---|---|---|\n")
        for c in FLAG_COLS:
            n = int(flags[c].sum())
            f.write(f"| {c} | {n} | {n/total*100:.2f}% |\n")
        f.write("\n")

        f.write("## 按月：停牌 / 涨停 / 跌停 / 不可交易 占比\n\n")
        flags["month"] = pd.to_datetime(flags["trade_date"]).dt.to_period("M").astype(str)
        mg = flags.groupby("month").agg(
            suspend=("suspend_flag", "mean"),
            limit_up=("limit_up_flag", "mean"),
            limit_down=("limit_down_flag", "mean"),
            not_tradable=("tradable_flag", lambda s: 1 - s.mean()),
        )
        f.write("| 月份 | 停牌% | 涨停% | 跌停% | 不可纳池% |\n|---|---|---|---|---|\n")
        for m, r in mg.iterrows():
            f.write(f"| {m} | {r['suspend']*100:.2f} | {r['limit_up']*100:.2f} | "
                    f"{r['limit_down']*100:.2f} | {r['not_tradable']*100:.2f} |\n")
        f.write("\n")

        f.write("## 已知缺口\n\n")
        f.write("- **ST**: 无名单数据源，st_flag 恒 False，可能高估可交易范围。\n")
        f.write("- **停牌**: 仅以 volume==0 判定；整段缺行（长期停牌）未通过全市场日历补齐。\n")
        f.write("- **涨跌停**: 后复权价在除权日 pct_change 略偏，已加容忍度缓解。\n")

    print(f"Report: {REPORT}")


if __name__ == "__main__":
    main()
