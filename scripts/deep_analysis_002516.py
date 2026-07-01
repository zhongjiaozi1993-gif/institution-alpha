"""
Deep analysis of 002516 旷达科技 — factor efficacy, player identification, timing.
"""
from __future__ import annotations

import sys
from pathlib import Path
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

PROJECT = Path(__file__).parent.parent
STOCK = "002516"
DATA_DIR = PROJECT / "data" / "single_stock" / STOCK

FEATURE_COLS = [
    "dynamic_big_threshold_yuan", "big_order_count", "big_order_ratio",
    "big_order_volume_ratio",
    "aggressive_order_ratio", "avg_price_aggressiveness",
    "buy_order_count", "sell_order_count", "buy_sell_order_ratio",
    "fill_rate", "avg_fill_ratio", "partial_fill_ratio",
    "avg_buy_slippage_bps", "slippage_std_bps",
    "morning_volume_ratio", "opening_30min_ratio", "closing_30min_ratio",
    "trade_buy_volume_ratio", "trade_imbalance",
    "vpin_mean", "vpin_std", "vpin_max",
    "intraday_range_pct",
    "median_trades_per_order", "long_order_ratio",
    "n_clusters", "total_cluster_amount_wan", "total_cluster_orders",
    "avg_cluster_span_min", "max_cluster_amount_wan",
    "clusters_per_hour", "buy_cluster_ratio",
]


def load_data():
    """Load features + prices, compute all forward returns."""
    feat_path = DATA_DIR / "features" / "all_features.csv"
    price_path = DATA_DIR / "price_daily.csv"

    features = pd.read_csv(feat_path)
    features["date"] = pd.to_datetime(features["date"].astype(str), format="%Y%m%d")

    prices = pd.read_csv(price_path)
    prices["日期"] = pd.to_datetime(prices["日期"])
    prices = prices.sort_values("日期").reset_index(drop=True)
    price_close = prices.set_index("日期")["收盘"]

    df = features.merge(prices, left_on="date", right_on="日期", how="left")

    # Forward returns for many horizons
    for h in [1, 3, 5, 10, 15, 20, 30, 40, 60]:
        fut = price_close.shift(-h)
        df[f"fwd_{h}d"] = (fut.loc[df["date"]].values / df["收盘"].values - 1) * 100

    df = df.dropna(subset=["fwd_5d"])
    return df, prices


# ═══════════════════════════════════════════════════════════════
# 1. FULL FACTOR EFFICACY ANALYSIS
# ═══════════════════════════════════════════════════════════════

def factor_efficacy(df: pd.DataFrame):
    """Compute IC (rank correlation) and quantile spread for each factor × horizon."""
    print("=" * 80)
    print("1. FACTOR EFFICACY — IC (Spearman rank correlation with forward returns)")
    print("=" * 80)

    horizons = [5, 10, 20, 40]
    available = [c for c in FEATURE_COLS if c in df.columns and df[c].std() > 0]

    results = []
    for feat in available:
        row = {"factor": feat}
        for h in horizons:
            valid = df[[feat, f"fwd_{h}d"]].dropna()
            if len(valid) < 20:
                row[f"IC_{h}d"] = np.nan
                row[f"QS_{h}d"] = np.nan
                continue
            ic = valid[feat].corr(valid[f"fwd_{h}d"], method="spearman")
            row[f"IC_{h}d"] = round(ic, 4)

            valid["q"] = pd.qcut(valid[feat], 5, labels=False, duplicates="drop")
            top = valid[valid["q"] == 4][f"fwd_{h}d"].mean()
            bot = valid[valid["q"] == 0][f"fwd_{h}d"].mean()
            row[f"QS_{h}d"] = round(top - bot, 2)

        results.append(row)

    res_df = pd.DataFrame(results)

    # Sort by abs IC_10d
    res_df["abs_IC_10d"] = res_df["IC_10d"].abs()
    res_df = res_df.sort_values("abs_IC_10d", ascending=False)

    print(f"\n{'Factor':<35} {'IC_5d':>8} {'IC_10d':>8} {'IC_20d':>8} {'IC_40d':>8} {'QS_10d':>8} {'QS_20d':>8}")
    print("-" * 90)
    for _, row in res_df.iterrows():
        name = row["factor"][:34]
        print(f"{name:<35} {row['IC_5d']:>8.4f} {row['IC_10d']:>8.4f} {row['IC_20d']:>8.4f} {row['IC_40d']:>8.4f} "
              f"{row['QS_10d']:>7.1f}% {row['QS_20d']:>7.1f}%")

    # Summary
    sig_5 = (res_df["IC_5d"].abs() > 0.05).sum()
    sig_10 = (res_df["IC_10d"].abs() > 0.05).sum()
    sig_20 = (res_df["IC_20d"].abs() > 0.05).sum()
    print(f"\nFactors with |IC|>0.05: 5d={sig_5}, 10d={sig_10}, 20d={sig_20}")

    top5 = res_df.head(5)["factor"].tolist()
    bot5 = res_df.tail(5)["factor"].tolist()
    print(f"Top 5: {top5}")
    print(f"Bottom 5: {bot5}")

    return res_df


# ═══════════════════════════════════════════════════════════════
# 2. EXTENDED HORIZON ANALYSIS
# ═══════════════════════════════════════════════════════════════

def horizon_analysis(df: pd.DataFrame):
    """Test all horizons from 1d to 60d with LightGBM."""
    print("\n" + "=" * 80)
    print("2. EXTENDED HORIZON — 持仓期1天~60天 模型表现")
    print("=" * 80)

    available = [c for c in FEATURE_COLS if c in df.columns]
    X_all = df[available].fillna(df[available].median()).values
    n = len(df)
    split = int(n * 0.8)

    try:
        import lightgbm as lgb
        has_lgb = True
    except ImportError:
        has_lgb = False

    print(f"\n{'Horizon':<10} {'DirAcc':>8} {'Top25%':>8} {'Bot25%':>8} {'Spread':>8} "
          f"{'Always':>8} {'Ann.Top':>8} {'WinRate':>8}")
    print("-" * 75)

    for h in [1, 3, 5, 10, 15, 20, 30, 40, 60]:
        col = f"fwd_{h}d"
        if col not in df.columns:
            continue
        y = df[col].values
        valid = ~np.isnan(y)
        X, y = X_all[valid], y[valid]
        if len(y) < 30:
            continue
        split_h = int(len(y) * 0.8)

        if has_lgb:
            model = lgb.LGBMRegressor(
                n_estimators=100, max_depth=4, learning_rate=0.05,
                min_child_samples=5, subsample=0.8, random_state=42, verbose=-1,
            )
        else:
            from sklearn.linear_model import Ridge
            model = Ridge(alpha=1.0)

        model.fit(X[:split_h], y[:split_h])
        y_pred = model.predict(X[split_h:])
        y_test = y[split_h:]

        # Top 25% vs bottom 25%
        p75 = np.percentile(y_pred, 75)
        p25 = np.percentile(y_pred, 25)
        top_ret = y_test[y_pred >= p75].mean() if (y_pred >= p75).sum() > 0 else 0
        bot_ret = y_test[y_pred <= p25].mean() if (y_pred <= p25).sum() > 0 else 0
        always = y_test.mean()
        dir_acc = np.mean((y_test > 0) == (y_pred > 0))
        win_rate = np.mean(y_test[y_pred >= p75] > 0) if (y_pred >= p75).sum() > 0 else 0

        # Annualized top-quartile return
        n_signals = (y_pred >= p75).sum()
        ann_ret = top_ret / h * 250 if h > 0 and n_signals > 0 else 0

        print(f"{h}d        {dir_acc:>7.1%}  {top_ret:>7.2f}%  {bot_ret:>7.2f}%  "
              f"{top_ret-bot_ret:>7.2f}%  {always:>7.2f}%  {ann_ret:>7.1f}%  {win_rate:>7.1%}")

    # Also compute: what if we just hold for N days regardless?
    print(f"\n--- Passive holding (no model) ---")
    for h in [1, 3, 5, 10, 20, 40, 60]:
        rets = df[f"fwd_{h}d"].dropna().values
        pos = np.mean(rets > 0)
        avg = np.mean(rets)
        ann = avg / h * 250
        print(f"  {h}d: avg={avg:.2f}%, win={pos:.1%}, ann={ann:.1f}%")


# ═══════════════════════════════════════════════════════════════
# 3. INSTITUTIONAL PLAYER TRACKING
# ═══════════════════════════════════════════════════════════════

def player_analysis():
    """Analyze cluster patterns to identify big players and their behaviors."""
    print("\n" + "=" * 80)
    print("3. PLAYER TRACKING — 机构聚类行为画像")
    print("=" * 80)

    ops_path = DATA_DIR / "ops" / "all_ops.csv"
    if not ops_path.exists():
        print("No ops data found")
        return

    ops = pd.read_csv(ops_path)
    ops["date"] = pd.to_datetime(ops["date"].astype(str), format="%Y%m%d")

    print(f"\nTotal operations detected: {len(ops)}")
    print(f"Date range: {ops['date'].min()} to {ops['date'].max()}")

    # ---- Size classification ----
    ops["size_tier"] = pd.cut(
        ops["total_amount_wan"],
        bins=[0, 100, 500, 2000, 100000],
        labels=["小(<100万)", "中(100-500万)", "大(500-2000万)", "超大(>2000万)"],
    )

    print(f"\n--- Size distribution ---")
    size_dist = ops.groupby("size_tier", observed=False).agg(
        count=("total_amount_wan", "count"),
        total_wan=("total_amount_wan", "sum"),
        avg_wan=("total_amount_wan", "mean"),
        avg_span=("time_span_min", "mean"),
        avg_orders=("order_count", "mean"),
    )
    print(size_dist.to_string())

    # ---- Buy vs Sell ----
    print(f"\n--- Buy vs Sell ---")
    for direction in ["BUY", "SELL"]:
        sub = ops[ops["direction"] == direction]
        print(f"\n{direction}:")
        print(f"  Count: {len(sub)}")
        print(f"  Total amount: {sub['total_amount_wan'].sum():.0f}万 = {sub['total_amount_wan'].sum()/10000:.1f}亿")
        print(f"  Avg amount: {sub['total_amount_wan'].mean():.0f}万")
        print(f"  Avg orders per op: {sub['order_count'].mean():.1f}")
        print(f"  Avg time span: {sub['time_span_min'].mean():.1f}min")
        print(f"  HHI (concentration): {sub['order_hhi'].mean():.4f}")
        print(f"  Interval std: {sub['order_interval_std'].mean():.1f}s")

    # ---- Top 20 largest operations ----
    print(f"\n--- Top 20 largest institution operations ---")
    top20 = ops.nlargest(20, "total_amount_wan")
    print(f"{'Date':<12} {'Dir':>4} {'Amount(万)':>10} {'Price':>7} {'Orders':>7} {'Span(min)':>8} {'HHI':>7}")
    print("-" * 65)
    for _, op in top20.iterrows():
        d = pd.Timestamp(op["date"]).strftime("%Y-%m-%d")
        print(f"{d:<12} {op['direction']:>4} {op['total_amount_wan']:>10.0f} "
              f"{op['avg_price']:>7.2f} {int(op['order_count']):>7} "
              f"{op['time_span_min']:>8.1f} {op['order_hhi']:>7.4f}")

    # ---- Temporal patterns of big players ----
    print(f"\n--- When do big players operate? (>500万) ---")
    big_ops = ops[ops["total_amount_wan"] >= 500].copy()
    big_ops["hour"] = big_ops["mid_time_sec"] / 3600

    time_bins = [(9.5, 10.0, "开盘冲刺"), (10.0, 11.0, "早盘"),
                 (11.0, 11.5, "早盘尾段"), (13.0, 14.0, "午盘"),
                 (14.0, 14.5, "午后"), (14.5, 15.0, "收盘冲刺")]
    for lo, hi, label in time_bins:
        n = ((big_ops["hour"] >= lo) & (big_ops["hour"] < hi)).sum()
        amt = big_ops[((big_ops["hour"] >= lo) & (big_ops["hour"] < hi))]["total_amount_wan"].sum()
        print(f"  {label} ({lo:.1f}-{hi:.1f}h): {n} ops, {amt:.0f}万")

    # ---- Cross-day pattern: clusters on consecutive days ----
    print(f"\n--- Cluster concentration ---")
    daily = ops.groupby("date").agg(
        n_ops=("total_amount_wan", "count"),
        total_wan=("total_amount_wan", "sum"),
        max_wan=("total_amount_wan", "max"),
    ).sort_values("total_wan", ascending=False)

    print(f"Days with most cluster activity:")
    for (d, row) in daily.head(10).iterrows():
        dt = pd.Timestamp(d).strftime("%Y-%m-%d")
        print(f"  {dt}: {int(row['n_ops'])} ops, {row['total_wan']:.0f}万 total, "
              f"max {row['max_wan']:.0f}万")

    return ops


# ═══════════════════════════════════════════════════════════════
# 4. BIG PLAYER vs BIG RETAIL IDENTIFICATION
# ═══════════════════════════════════════════════════════════════

def big_vs_retail_analysis():
    """
    Look at raw order data to distinguish big institutions from big retail.
    Key differentiators:
      - Institution: split large order into many small trades (长单), mechanical timing
      - Big retail: single large order, manual timing
    """
    print("\n" + "=" * 80)
    print("4. BIG INSTITUTION vs BIG RETAIL — 大机构 vs 大散户画像")
    print("=" * 80)

    raw_dir = DATA_DIR / "raw"

    # Sample a high-activity day
    ops = pd.read_csv(DATA_DIR / "ops" / "all_ops.csv")
    ops["date"] = pd.to_datetime(ops["date"].astype(str), format="%Y%m%d")
    busy_days = ops.groupby("date").size().sort_values(ascending=False)

    print(f"\nTop 5 busiest days (most clusters):")
    for d, n in busy_days.head(5).items():
        print(f"  {pd.Timestamp(d).strftime('%Y-%m-%d')}: {n} clusters")

    # Pick the busiest day and analyze raw data
    top_date = busy_days.index[0]
    date_str = pd.Timestamp(top_date).strftime("%Y%m%d")

    from src.data.level2_reader import read_level2_stock_dir, match_orders_to_trades

    stock_dir = raw_dir / date_str / "002516.SZ"
    if not stock_dir.exists():
        print(f"\nRaw data not available for {date_str}")
        return

    data = read_level2_stock_dir(stock_dir)
    orders = data.get("逐笔委托", pd.DataFrame())
    trades = data.get("逐笔成交", pd.DataFrame())

    if orders.empty or trades.empty:
        print("No data loaded")
        return

    # Convert numeric
    orders["委托价格"] = pd.to_numeric(orders["委托价格"], errors="coerce")
    orders["委托数量"] = pd.to_numeric(orders["委托数量"], errors="coerce")
    orders = orders[orders["委托价格"] > 0]

    # Compute order amount in 元
    orders["委托金额_元"] = orders["委托价格"] / 10000 * orders["委托数量"]

    # ---- Classification ----
    # 超大单: >100万元 (institution must split)
    # 大单: 20-100万 (could be either)
    # 中单: 5-20万 (big retail territory)
    # 小单: <5万 (retail)

    orders["size_class"] = pd.cut(
        orders["委托金额_元"],
        bins=[0, 50000, 200000, 1000000, float("inf")],
        labels=["小单(<5万)", "中单(5-20万)", "大单(20-100万)", "超大单(>100万)"],
    )

    print(f"\n--- Order size distribution on {pd.Timestamp(top_date).strftime('%Y-%m-%d')} ---")
    for label in ["小单(<5万)", "中单(5-20万)", "大单(20-100万)", "超大单(>100万)"]:
        sub = orders[orders["size_class"] == label]
        if len(sub) == 0:
            continue
        buy_sub = sub[sub["委托代码"] == "B"]
        sell_sub = sub[sub["委托代码"] == "S"]
        print(f"\n  {label}:")
        print(f"    Count: {len(sub)} ({len(sub)/len(orders)*100:.1f}% of all)")
        print(f"    Buy/Sell: {len(buy_sub)}/{len(sell_sub)}")
        print(f"    Avg amount: {sub['委托金额_元'].mean():.0f}元")
        print(f"    Avg qty: {sub['委托数量'].mean():.0f}股")
        print(f"    Median amount: {sub['委托金额_元'].median():.0f}元")

    # ---- Who are the "big players"? ----
    # Show top individual orders (potential big players)
    print(f"\n--- Top 15 individual BUY orders (potential institution/大户 entries) ---")
    buy_orders = orders[orders["委托代码"] == "B"].nlargest(15, "委托金额_元")
    print(f"{'Time':<12} {'Amount(元)':>12} {'Qty(股)':>10} {'Price(元)':>10} {'OrderID':>15}")
    print("-" * 65)
    for _, o in buy_orders.iterrows():
        print(f"{str(o['时间']):<12} {o['委托金额_元']:>12.0f} {o['委托数量']:>10.0f} "
              f"{o['委托价格']/10000:>10.2f} {str(o.get('交易所委托号', o.get('委托编号', 'N/A'))):>15}")


# ═══════════════════════════════════════════════════════════════
# 5. MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    print("=" * 80)
    print("旷达科技 002516 — 深度因子与机构行为分析")
    print("=" * 80)

    df, prices = load_data()
    print(f"Dataset: {len(df)} days, {len(df.columns)} columns")
    print(f"Price range: {df['收盘'].min():.2f} - {df['收盘'].max():.2f} 元")
    print(f"Stock return (full period): {(df['收盘'].iloc[-1]/df['收盘'].iloc[0]-1)*100:.1f}%")

    # 1. Factor efficacy
    factor_efficacy(df)

    # 2. Horizon analysis
    horizon_analysis(df)

    # 3. Player tracking
    player_analysis()

    # 4. Big vs retail
    big_vs_retail_analysis()

    print("\n" + "=" * 80)
    print("分析完成")
    print("=" * 80)


if __name__ == "__main__":
    main()
