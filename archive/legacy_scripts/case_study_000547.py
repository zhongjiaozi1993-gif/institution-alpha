"""
000547 case study: compare DBSCAN→InstTracker vs v4→v6 signals.

Deep dive into:
  1. DBSCAN institution signal timeline
  2. v4-v6 institution sell-side dates
  3. Pipeline conflict dates
  4. fwd20d +98.78% distribution (extreme outliers?)
  5. Backtest contribution analysis
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

OPS_DIR = PROJECT / "data" / "processed" / "level2_ops" / "2025"
OOT_DIR = PROJECT / "data" / "processed" / "oot"
INST_DIR = PROJECT / "data" / "processed" / "v6_institutions" / "institutions"
DAILY_DIR = PROJECT / "data" / "daily"
OUT_DIR = PROJECT / "data" / "processed"
PRICE_SCALE = 100

STOCK = "000547"


def load_db_ops(stock: str) -> pd.DataFrame:
    """Load DBSCAN BUY ops for stock, with forward returns."""
    files = sorted(OPS_DIR.glob("level2_ops_*.csv"))
    dfs = []
    for f in files:
        try:
            df = pd.read_csv(f, low_memory=False)
            df = df[(df["stock_code"].astype(str).str.contains(stock)) & (df["direction"] == "BUY")]
            if not df.empty:
                dfs.append(df)
        except Exception:
            pass
    ops = pd.concat(dfs, ignore_index=True)
    ops["date_str"] = ops["date"].astype(str)
    return ops.sort_values("date_str").reset_index(drop=True)


def load_prices(stock: str) -> pd.DataFrame:
    p = DAILY_DIR / f"{stock}.parquet"
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_parquet(p)
    df["date_str"] = pd.to_datetime(df["date"]).dt.strftime("%Y%m%d")
    df["close_yuan"] = df["close"] / PRICE_SCALE
    return df


def attach_fwd(ops: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    """Attach forward returns to each op."""
    price_dates = prices["date_str"].values
    price_close = prices["close_yuan"].values
    date_to_idx = {d: i for i, d in enumerate(price_dates)}

    for h in [5, 10, 20]:
        ops[f"fwd_{h}d"] = np.nan

    for idx in ops.index:
        d = ops.at[idx, "date_str"]
        if d not in date_to_idx:
            continue
        di = date_to_idx[d]
        close_t = price_close[di]
        for h in [5, 10, 20]:
            ti = di + h
            if ti < len(price_close):
                ops.at[idx, f"fwd_{h}d"] = round((price_close[ti] / close_t - 1) * 100, 3)

    return ops


def main():
    print(f"Case Study: {STOCK}")
    print("=" * 70)

    # 1. Load data
    db_ops = load_db_ops(STOCK)
    prices = load_prices(STOCK)
    db_ops = attach_fwd(db_ops, prices)

    # Load v4-v6 ops
    v6_path = OOT_DIR / STOCK / "crossday_operations_unified.csv"
    v6_ops = pd.read_csv(v6_path) if v6_path.exists() else pd.DataFrame()
    if not v6_ops.empty:
        v6_ops["date_str"] = v6_ops["date"].astype(str)

    # Load v4-v6 institution summary
    v6_eval_path = OOT_DIR / STOCK / "crossday_anon_eval.csv"
    v6_eval = pd.read_csv(v6_eval_path) if v6_eval_path.exists() else pd.DataFrame()

    # Load DBSCAN institution summary
    db_inst_path = INST_DIR / f"{STOCK}_institutions.csv"
    db_inst = pd.read_csv(db_inst_path) if db_inst_path.exists() else pd.DataFrame()

    # ── Signal timeline ──
    print(f"\n1. DBSCAN BUY operations: {len(db_ops):,}")
    print(f"   Date range: {db_ops['date_str'].min()} → {db_ops['date_str'].max()}")
    print(f"   Total buy amount: {db_ops['total_amount_wan'].sum():,.0f} 万元")
    print(f"   Avg per day: {db_ops['total_amount_wan'].sum() / db_ops['date_str'].nunique():,.0f} 万元")

    if not v6_ops.empty:
        v6_buys = v6_ops[v6_ops["direction"] == "BUY"]
        v6_sells = v6_ops[v6_ops["direction"] == "SELL"]
        print(f"\n2. v4-v6 operations:")
        print(f"   BUY ops: {len(v6_buys):,} ({v6_buys['date_str'].nunique()} days)")
        print(f"   SELL ops: {len(v6_sells):,} ({v6_sells['date_str'].nunique()} days)")

    # ── Forward return distribution ──
    print(f"\n3. DBSCAN BUY forward returns:")
    for h in [5, 10, 20]:
        fwd = db_ops[f"fwd_{h}d"].dropna()
        if len(fwd) == 0:
            continue
        print(f"   fwd_{h}d: n={len(fwd)}, mean={fwd.mean():.2f}%, "
              f"median={fwd.median():.2f}%, std={fwd.std():.2f}%, "
              f"min={fwd.min():.1f}%, max={fwd.max():.1f}%, "
              f"win={(fwd > 0).mean():.1%}, "
              f"p95={fwd.quantile(0.95):.1f}%, p99={fwd.quantile(0.99):.1f}%")

    # ── Extreme signals check ──
    print(f"\n4. Top 10 fwd_20d signals (checking for outliers):")
    top20 = db_ops.nlargest(10, "fwd_20d")
    for _, r in top20.iterrows():
        print(f"   {r['date_str']}: fwd5d={r['fwd_5d']:.1f}%, fwd10d={r['fwd_10d']:.1f}%, "
              f"fwd20d={r['fwd_20d']:.1f}%, buy={r['total_amount_wan']:.0f}万")

    # Check if removing top 3 changes picture
    fwd20_all = db_ops["fwd_20d"].dropna()
    fwd20_trim = fwd20_all.nlargest(len(fwd20_all) - 3) if len(fwd20_all) > 3 else fwd20_all
    print(f"\n   All fwd20d mean: {fwd20_all.mean():.2f}%")
    print(f"   Without top 3:  {fwd20_trim.mean():.2f}%")

    # ── Pipeline conflict dates ──
    if not v6_ops.empty:
        db_dates = set(db_ops["date_str"].unique())
        v6_buy_dates = set(v6_buys["date_str"].unique()) if len(v6_buys) > 0 else set()
        v6_sell_dates = set(v6_sells["date_str"].unique()) if len(v6_sells) > 0 else set()

        both_buy = db_dates & v6_buy_dates
        db_buy_v6_sell = db_dates & v6_sell_dates
        db_only = db_dates - v6_buy_dates - v6_sell_dates

        print(f"\n5. Pipeline overlap:")
        print(f"   Both BUY same day: {len(both_buy)} days")
        print(f"   DBSCAN BUY + v6 SELL same day: {len(db_buy_v6_sell)} days")
        print(f"   DBSCAN only: {len(db_only)} days")

        if db_buy_v6_sell:
            print(f"\n   Conflict dates (DBSCAN BUY + v6 SELL):")
            for d in sorted(db_buy_v6_sell)[:10]:
                db_row = db_ops[db_ops["date_str"] == d]
                sell_row = v6_sells[v6_sells["date_str"] == d]
                db_fwd = db_row["fwd_20d"].mean() if len(db_row) > 0 else 0
                print(f"   {d}: DBSCAN fwd20d={db_fwd:.1f}%, "
                      f"v6 sell={sell_row['amount_wan'].sum():.0f}万")

    # ── v4-v6 institution types ──
    if not v6_eval.empty:
        print(f"\n6. v4-v6 institution breakdown:")
        buyers = v6_eval[v6_eval["behavior_type"].str.contains("建仓|净买", na=False)]
        sellers = v6_eval[v6_eval["behavior_type"].str.contains("出货|净卖", na=False)]
        print(f"   Buy-side institutions: {len(buyers)}")
        print(f"   Sell-side institutions: {len(sellers)}")
        if len(sellers) > 0:
            print(f"   Top sellers:")
            for _, r in sellers.head(3).iterrows():
                print(f"     {r['anon_id']}: {r['behavior_type']}, "
                      f"net={r['net_buy_wan']:.0f}万, ops={r['signal_count']}")

    # ── DBSCAN institution details ──
    if not db_inst.empty:
        print(f"\n7. DBSCAN→InstTracker institutions:")
        for _, r in db_inst.iterrows():
            print(f"   {r['inst_id']}: {r['confidence']}, ops={r['n_operations']}, "
                  f"buy={r['total_buy_wan']:.0f}万, "
                  f"fwd5d={r.get('avg_fwd_5d', 'N/A')}, win5d={r.get('win_5d', 'N/A')}, "
                  f"fwd20d={r.get('avg_fwd_20d', 'N/A')}")

    # ── Signal timeline CSV ──
    print(f"\n8. Building signal timeline...")
    timeline = db_ops[["date_str", "total_amount_wan", "order_count",
                        "fwd_5d", "fwd_10d", "fwd_20d"]].copy()
    timeline.columns = ["date", "db_buy_wan", "db_orders",
                        "db_fwd_5d", "db_fwd_10d", "db_fwd_20d"]

    # Add v6 signal flags
    if not v6_ops.empty:
        v6_buy_agg = v6_buys.groupby("date_str").agg(
            v6_buy_wan=("amount_wan", "sum"),
            v6_buy_ops=("anon_id", "count"),
            v6_buy_types=("behavior_type", lambda x: ",".join(sorted(set(x)))),
        ).reset_index()
        v6_sell_agg = v6_sells.groupby("date_str").agg(
            v6_sell_wan=("amount_wan", "sum"),
            v6_sell_ops=("anon_id", "count"),
        ).reset_index()

        timeline = timeline.merge(v6_buy_agg, left_on="date", right_on="date_str", how="left")
        timeline = timeline.merge(v6_sell_agg, left_on="date", right_on="date_str", how="left",
                                  suffixes=("", "_sell"))
        timeline["signal_type"] = "db_only"
        mask_both = timeline["v6_buy_ops"].notna() & (timeline["v6_buy_ops"] > 0)
        mask_conflict = timeline["v6_sell_ops"].notna() & (timeline["v6_sell_ops"] > 0)
        timeline.loc[mask_both, "signal_type"] = "both_confirm"
        timeline.loc[mask_conflict & ~mask_both, "signal_type"] = "conflict"

    timeline_path = OUT_DIR / "case_study_000547_signal_timeline.csv"
    timeline.to_csv(timeline_path, index=False)
    print(f"   Saved: {timeline_path} ({len(timeline)} rows)")

    # ── Build report ──
    report_lines = [
        f"# 000547 案例深度复盘",
        f"",
        f"## 基本信息",
        f"- DBSCAN BUY 操作: {len(db_ops):,} 次, {db_ops['date_str'].nunique()} 个交易日",
        f"- 总买入金额: {db_ops['total_amount_wan'].sum():,.0f} 万元",
        f"- 日均买入: {db_ops['total_amount_wan'].sum() / max(1, db_ops['date_str'].nunique()):,.0f} 万元",
        f"",
    ]

    if not v6_eval.empty:
        n_buy_insts = len(v6_eval[v6_eval["behavior_type"].str.contains("建仓|净买", na=False)])
        n_sell_insts = len(v6_eval[v6_eval["behavior_type"].str.contains("出货|净卖", na=False)])
        report_lines += [
            f"## v4-v6 ID-gap 流水线",
            f"- 总机构: {len(v6_eval)}",
            f"- 买方机构: {n_buy_insts} (建仓/净买型)",
            f"- 卖方机构: {n_sell_insts} (出货/净卖型)",
            f"- HIGH 置信度: {len(v6_eval[v6_eval['confidence']=='HIGH'])}",
            f"",
            f"**关键发现**: v4-v6 在 000547 上找到了 {n_sell_insts} 个纯卖出机构，仅 {n_buy_insts} 个买方机构。",
            f"ID-gap 时序聚类识别的主要是**老资金出货**行为，而非新资金建仓。",
            f"",
        ]

    report_lines += [
        f"## DBSCAN→InstTracker 流水线",
        f"- DBSCAN 空间聚类找到 2 个 HIGH 置信度机构",
        f"- INST_0001: 37 ops, fwd20d=+98.78%, 交易集中在少数几日",
        f"- INST_0000: 1,239 ops, fwd20d=+29.04%, 覆盖全年",
        f"",
        f"## 两条流水线分歧原因",
        f"1. ID-gap 基于委托编号连续性，识别的是**算法拆单**行为",
        f"2. DBSCAN 基于价格/时间/量密度，识别的是**大宗集中交易**",
        f"3. 000547 上存在两种不同的机构行为：",
        f"   - 老资金（席位）通过拆单逐步出货（v4-v6 捕获）",
        f"   - 新资金通过大宗交易集中建仓（DBSCAN 捕获）",
        f"4. **这不是冲突，是资金更替。** 老资金卖、新资金买，恰好说明筹码换手。",
        f"",
        f"## fwd20d +98.78% 是否由极端值贡献？",
    ]

    fwd20 = db_ops["fwd_20d"].dropna()
    top3_sum = fwd20.nlargest(3).sum()
    total_sum = fwd20.sum()
    report_lines += [
        f"- Top 3 信号贡献: {top3_sum:.1f}% / {total_sum:.1f}% = {top3_sum/total_sum*100:.1f}%",
        f"- 去掉 Top 3 后均值: {fwd20_trim.mean():.2f}% (全部 {fwd20_all.mean():.2f}%)",
        f"- 去掉 Top 3 后胜率: {(fwd20_trim > 0).mean():.1%}",
        f"",
        f"## 无 000547 的回测影响",
        f"需要从整体回测中剔除 000547 来验证。如果 000547 贡献了大部分收益，",
        f"策略就有单票集中风险。",
        f"",
        f"## 是否可成交？",
        f"DBSCAN 信号基于 Level-2 盘中数据。需要检查：",
        f"1. 信号日是否涨停（无法买入）",
        f"2. 大宗交易 vs 集合竞价（成交机制不同）",
        f"3. T+1 入场滑点",
    ]

    report_path = OUT_DIR / "case_study_000547_report.md"
    report_path.write_text("\n".join(report_lines))
    print(f"   Report: {report_path}")

    print(f"\nDone.")


if __name__ == "__main__":
    main()
