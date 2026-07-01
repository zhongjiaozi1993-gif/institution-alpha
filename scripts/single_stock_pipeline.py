"""
Single-stock Level-2 pipeline — 旷达科技 002516
Extracts, clusters, and computes enriched features for one stock.

Uses py7zr.read() (NOT extract()) to reliably get single-stock CSV files
from multi-GB archives without extracting everything.
"""
from __future__ import annotations

import sys, io, shutil
from pathlib import Path
import py7zr
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.level2_reader import (
    read_level2_stock_dir, match_orders_to_trades,
)
from src.cluster.split_detector import (
    detect_institution_operations,
    _time_to_seconds, OPEN_TIME, CLOSE_TIME, TRADING_SECONDS,
)

STOCK = "002516"
STOCK_WIND = "002516.SZ"
STOCK_NAME = "旷达科技"

# Data paths (Mac)
ARCHIVE_ROOT = Path("/tmp/level2_test")          # 7z archives
OUTPUT_ROOT = Path(__file__).parent.parent / "data" / "single_stock" / STOCK


# ============================================================
# 1. Extract one stock from 7z via z.read()
# ============================================================
def extract_one_stock(archive_path: Path, date: str, output_dir: Path) -> Path:
    """Extract one stock's CSV files using py7zr.read(). Keeps raw data."""
    stock_dir = output_dir / date / STOCK_WIND
    if stock_dir.exists() and list(stock_dir.glob("*.csv")):
        print(f"  [{date}] Already extracted, skip")
        return stock_dir

    stock_dir.mkdir(parents=True, exist_ok=True)

    targets = [
        f"{date}/{STOCK_WIND}/逐笔委托.csv",
        f"{date}/{STOCK_WIND}/逐笔成交.csv",
    ]

    try:
        with py7zr.SevenZipFile(archive_path, "r") as z:
            # z.read() returns {archive_name: BytesIO} — reliable, no filesystem issues
            data = z.read(targets)
            for arcname, bio in data.items():
                fname = Path(arcname).name
                dest = stock_dir / fname
                dest.write_bytes(bio.read())
        n_files = len(list(stock_dir.glob("*.csv")))
        print(f"  [{date}] Extracted {n_files} files OK")
    except Exception as e:
        print(f"  [{date}] Extract FAILED: {e}")
        # Clean up partial
        shutil.rmtree(stock_dir, ignore_errors=True)
        return stock_dir

    return stock_dir


# ============================================================
# 2. Enriched feature computation
# ============================================================
def compute_enriched_features(
    orders: pd.DataFrame,
    trades: pd.DataFrame,
    matched: pd.DataFrame,
    clusters: list[dict],
    date: str,
) -> dict:
    """
    A. 动态大小单（广发风格）: mean+N*std threshold
    B. 订单侵略性: price vs VWAP
    C. 执行质量: fill rates, slippage
    D. 时间模式: morning/opening/closing concentration
    E. VPIN简化: equal-volume bucket buy/sell imbalance
    F. 市场冲击: intraday range, large trade impact
    G. 长短单: trades per order distribution
    """
    features = {"date": date, "stock": STOCK}

    if orders.empty or trades.empty:
        return features

    orders = orders.copy()
    trades = trades.copy()

    orders["委托价格"] = pd.to_numeric(orders["委托价格"], errors="coerce")
    orders["委托数量"] = pd.to_numeric(orders["委托数量"], errors="coerce")
    trades["成交价格"] = pd.to_numeric(trades["成交价格"], errors="coerce")
    trades["成交数量"] = pd.to_numeric(trades["成交数量"], errors="coerce")

    orders = orders[orders["委托价格"] > 0]
    trades = trades[trades["成交价格"] > 0]

    orders["time_sec"] = orders["时间"].astype(str).apply(_time_to_seconds)
    trades["time_sec"] = trades["时间"].astype(str).apply(_time_to_seconds)

    # ---- A. Dynamic big/small order (广发风格) ----
    order_amt_raw = orders["委托价格"] * orders["委托数量"]  # price*10000 * qty
    order_amt_yuan = order_amt_raw / 10000  # → 元

    amt_mean = order_amt_yuan.mean()
    amt_std = order_amt_yuan.std()
    big_threshold = amt_mean + 2 * amt_std if amt_std > 0 else amt_mean * 3
    is_big = order_amt_yuan >= big_threshold

    features["dynamic_big_threshold_yuan"] = round(float(big_threshold), 0)
    features["big_order_count"] = int(is_big.sum())
    features["big_order_ratio"] = round(float(is_big.mean()), 4)
    if order_amt_yuan.sum() > 0:
        features["big_order_volume_ratio"] = round(
            float(order_amt_yuan[is_big].sum() / order_amt_yuan.sum()), 4
        )

    # ---- B. Order aggressiveness ----
    if not trades.empty:
        vwap = (trades["成交价格"] * trades["成交数量"]).sum() / trades["成交数量"].sum()
    else:
        vwap = orders["委托价格"].mean()
    vwap_yuan = vwap / 10000
    features["daily_vwap_yuan"] = round(float(vwap_yuan), 2)

    price_vs_vwap = (orders["委托价格"] / 10000) / vwap_yuan
    features["aggressive_order_ratio"] = round(float((price_vs_vwap >= 1.0).mean()), 4)
    features["avg_price_aggressiveness"] = round(float(price_vs_vwap.mean() - 1), 6)

    buy_orders = orders[orders["委托代码"] == "B"]
    sell_orders = orders[orders["委托代码"] == "S"]
    features["buy_order_count"] = len(buy_orders)
    features["sell_order_count"] = len(sell_orders)
    features["buy_sell_order_ratio"] = round(
        len(buy_orders) / max(len(sell_orders), 1), 4
    )

    # ---- C. Execution quality ----
    if not matched.empty:
        features["matched_order_count"] = len(matched)
        features["fill_rate"] = round(len(matched) / max(len(orders), 1), 4)

        if "委托数量" in matched.columns and "成交数量" in matched.columns:
            fill_ratios = matched["成交数量"] / matched["委托数量"].replace(0, np.nan)
            fill_ratios = fill_ratios.dropna().clip(0, 2)  # cap outliers
            if len(fill_ratios) > 0:
                features["avg_fill_ratio"] = round(float(fill_ratios.mean()), 4)
                features["partial_fill_ratio"] = round(float((fill_ratios < 0.95).mean()), 4)

    # Slippage for buy orders
    if not matched.empty and "成交价格" in matched.columns and "委托价格" in matched.columns:
        buy_matched = matched[matched["委托代码"] == "B"]
        if len(buy_matched) > 0:
            p_order = buy_matched["委托价格"] / 10000
            p_trade = buy_matched["成交价格"] / 10000
            slippages = (p_trade - p_order) / p_order.replace(0, np.nan)
            slippages = slippages.dropna()
            if len(slippages) > 0:
                features["avg_buy_slippage_bps"] = round(float(slippages.mean() * 10000), 1)
                features["slippage_std_bps"] = round(float(slippages.std() * 10000), 1)

    # ---- D. Temporal patterns ----
    morning_open = 9.5 * 3600
    morning_close = 11.5 * 3600

    trade_sec = trades["time_sec"]
    morning_mask = (trade_sec >= morning_open) & (trade_sec <= morning_close)
    opening_mask = (trade_sec >= morning_open) & (trade_sec <= morning_open + 1800)
    closing_mask = (trade_sec >= 14.5 * 3600) & (trade_sec <= CLOSE_TIME)

    total_vol = trades["成交数量"].sum()
    if total_vol > 0:
        features["morning_volume_ratio"] = round(
            float(trades.loc[morning_mask, "成交数量"].sum() / total_vol), 4
        )
        features["opening_30min_ratio"] = round(
            float(trades.loc[opening_mask, "成交数量"].sum() / total_vol), 4
        )
        features["closing_30min_ratio"] = round(
            float(trades.loc[closing_mask, "成交数量"].sum() / total_vol), 4
        )

    # ---- E. VPIN-style imbalance ----
    if "BS标志" in trades.columns:
        buys_v = trades[trades["BS标志"] == "B"]["成交数量"].sum()
        sells_v = trades[trades["BS标志"] == "S"]["成交数量"].sum()
        total_v = buys_v + sells_v
        if total_v > 0:
            features["trade_buy_volume_ratio"] = round(float(buys_v / total_v), 4)
            features["trade_imbalance"] = round(float((buys_v - sells_v) / total_v), 4)

        # VPIN buckets
        n_buckets = 50
        if len(trades) > n_buckets:
            cumvol = trades["成交数量"].cumsum()
            bucket_vol = cumvol.iloc[-1] / n_buckets
            imbalances = []
            for i in range(n_buckets):
                lo, hi = i * bucket_vol, (i + 1) * bucket_vol
                bucket = trades[(cumvol >= lo) & (cumvol < hi)]
                b = bucket[bucket["BS标志"] == "B"]["成交数量"].sum()
                s = bucket[bucket["BS标志"] == "S"]["成交数量"].sum()
                if b + s > 0:
                    imbalances.append(abs(b - s) / (b + s))
            if imbalances:
                features["vpin_mean"] = round(float(np.mean(imbalances)), 4)
                features["vpin_std"] = round(float(np.std(imbalances)), 4)
                features["vpin_max"] = round(float(np.max(imbalances)), 4)

    # ---- F. Market impact ----
    if not trades.empty and vwap_yuan > 0:
        prices_yuan = trades["成交价格"] / 10000
        features["intraday_range_pct"] = round(
            float((prices_yuan.max() - prices_yuan.min()) / vwap_yuan * 100), 4
        )

        if len(trades) > 100:
            large_cutoff = trades["成交数量"].quantile(0.95)
            large_trades = trades[trades["成交数量"] >= large_cutoff]
            if len(large_trades) > 0:
                pre = trades[trades["time_sec"] < large_trades["time_sec"].min()]["成交价格"]
                post = trades[trades["time_sec"] > large_trades["time_sec"].max()]["成交价格"]
                if len(pre) > 0 and len(post) > 0:
                    features["large_trade_impact_bps"] = round(
                        float((post.mean() - pre.mean()) / pre.mean() * 10000), 1
                    )

    # ---- G. Long/short order (广发长短单) ----
    # Use the match key column (may be 交易所委托号 or 委托编号 depending on data)
    match_col = matched.get("match_key", [None])[0] if "match_key" in matched.columns else None
    group_col = match_col if match_col and match_col in matched.columns else (
        "交易所委托号" if "交易所委托号" in matched.columns else "委托编号"
    )
    if group_col in matched.columns:
        tpo = matched.groupby(group_col).size()
        if len(tpo) > 0:
            features["median_trades_per_order"] = round(float(tpo.median()), 1)
            features["long_order_ratio"] = round(float((tpo > tpo.median()).mean()), 4)

    # ---- Cluster aggregates ----
    if clusters:
        features["n_clusters"] = len(clusters)
        features["total_cluster_amount_wan"] = round(
            sum(c["total_amount_wan"] for c in clusters), 1
        )
        features["total_cluster_orders"] = sum(c["order_count"] for c in clusters)
        features["avg_cluster_span_min"] = round(
            float(np.mean([c["time_span_min"] for c in clusters])), 1
        )
        features["max_cluster_amount_wan"] = round(
            max(c["total_amount_wan"] for c in clusters), 1
        )
        features["clusters_per_hour"] = round(len(clusters) / 4, 1)
        buy_clusters = [c for c in clusters if c["direction"] == "BUY"]
        features["buy_cluster_ratio"] = round(
            len(buy_clusters) / max(len(clusters), 1), 4
        )

    return features


# ============================================================
# 3. Main
# ============================================================
def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    raw_dir = OUTPUT_ROOT / "raw"
    feat_dir = OUTPUT_ROOT / "features"
    ops_dir = OUTPUT_ROOT / "ops"
    for d in [raw_dir, feat_dir, ops_dir]:
        d.mkdir(exist_ok=True)

    # Find already-extracted stock directories
    stock_dirs = sorted(raw_dir.glob(f"*/{STOCK_WIND}"))
    if not stock_dirs:
        # Try extraction from archives if available
        archives = sorted(ARCHIVE_ROOT.glob("*.7z"))
        print(f"No pre-extracted data. Found {len(archives)} archive(s), extracting...")
        for archive in archives:
            date = archive.stem
            extract_one_stock(archive, date, raw_dir)
        stock_dirs = sorted(raw_dir.glob(f"*/{STOCK_WIND}"))

    print(f"Found {len(stock_dirs)} stock directories")

    all_features = []
    all_ops = []

    for stock_dir in stock_dirs:
        date = stock_dir.parent.name
        print(f"\n{'='*60}")
        print(f"[{date}] Processing {STOCK_NAME} ({STOCK})")
        print(f"{'='*60}")

        try:
            data = read_level2_stock_dir(stock_dir)
        except Exception as e:
            print(f"  ERROR loading: {e}")
            continue

        orders = data.get("逐笔委托", pd.DataFrame())
        trades = data.get("逐笔成交", pd.DataFrame())
        print(f"  Orders: {len(orders)}, Trades: {len(trades)}")

        if orders.empty or trades.empty:
            continue

        matched = match_orders_to_trades(orders, trades)
        print(f"  Matched: {len(matched)}")

        # Try eps sweep for best clustering (like Windows version)
        clusters = []
        for eps in [0.05, 0.10, 0.15, 0.25]:
            for min_samp in [3, 5]:
                clusters = detect_institution_operations(
                    matched, eps=eps, min_samples=min_samp,
                    min_total_amount_wan=50,
                )
                if clusters:
                    break
            if clusters:
                break
        print(f"  Clusters: {len(clusters)} (eps={eps if clusters else 'N/A'})")

        features = compute_enriched_features(orders, trades, matched, clusters, date)
        all_features.append(features)

        for c in clusters:
            c["date"] = date
            c["stock_code"] = STOCK
        all_ops.extend(clusters)

        feat_df = pd.DataFrame([features])
        feat_df.to_csv(feat_dir / f"features_{date}.csv", index=False)
        print(f"  Features: {len(features)} metrics")

        for k, v in features.items():
            if isinstance(v, float) and k not in ("date",):
                print(f"    {k}: {v}")
        if clusters:
            print(f"  Top clusters:")
            for c in clusters[:3]:
                print(f"    {c['direction']} {c['total_amount_wan']:.0f}万 "
                      f"@{c['avg_price']:.2f} x{c['order_count']} "
                      f"span={c['time_span_min']:.0f}min")

    # ---- Final summary ----
    print(f"\n{'='*60}")
    print(f"SUMMARY: {len(all_features)} days")
    print(f"{'='*60}")

    if all_ops:
        ops_df = pd.DataFrame(all_ops)
        ops_df.to_csv(ops_dir / "all_ops.csv", index=False)
        print(f"Total ops: {len(all_ops)}")

    if all_features:
        summary = pd.DataFrame(all_features)
        summary.to_csv(feat_dir / "all_features.csv", index=False)
        print(f"Feature columns ({len(summary.columns)}):")
        for c in summary.columns:
            if c not in ("date", "stock"):
                vals = summary[c].dropna()
                if len(vals) > 0:
                    print(f"  {c}: {vals.mean():.4f} +/- {vals.std():.4f}")


if __name__ == "__main__":
    main()
