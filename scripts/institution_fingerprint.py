"""
Precise institution identification via sequential order IDs + behavior fingerprinting.
No LHB needed — we identify anonymous institutions by their algorithmic footprint.

Method:
  1. Find orders with sequential IDs, same time (±1s), same price → one algo
  2. Profile each "algo-cluster": size, timing, price aggressiveness
  3. Cross-date matching: same fingerprint on different days → same institution
  4. Track accumulation/distribution cycles
"""
from __future__ import annotations

import sys
from pathlib import Path
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.level2_reader import read_level2_stock_dir, match_orders_to_trades
from src.cluster.split_detector import _time_to_seconds

PROJECT = Path(__file__).parent.parent
STOCK = "002516"
RAW_DIR = PROJECT / "data" / "single_stock" / STOCK / "raw"


def load_day(date_str: str) -> tuple:
    """Load raw data for one day."""
    stock_dir = RAW_DIR / date_str / "002516.SZ"
    if not stock_dir.exists():
        return pd.DataFrame(), pd.DataFrame()
    data = read_level2_stock_dir(stock_dir)
    orders = data.get("逐笔委托", pd.DataFrame())
    trades = data.get("逐笔成交", pd.DataFrame())
    return orders, trades


def detect_algo_clusters(orders: pd.DataFrame, id_gap_max: int = 500,
                         time_window_sec: float = 5.0,
                         price_pct_tol: float = 0.005) -> list[dict]:
    """
    Find institution algos by detecting sequential exchange order IDs
    submitted at nearly the same time and price.

    Exchange assigns sequential IDs to orders. An algo submitting N orders
    in rapid succession will get N consecutive (or near-consecutive) IDs.

    id_gap_max: max gap between consecutive order IDs to consider same algo
    time_window_sec: max time difference between first and last order
    price_pct_tol: max price deviation as fraction of mean price
    """
    if orders.empty:
        return []

    orders = orders.copy()
    orders["委托价格"] = pd.to_numeric(orders["委托价格"], errors="coerce")
    orders["委托数量"] = pd.to_numeric(orders["委托数量"], errors="coerce")
    orders = orders[orders["委托价格"] > 0]

    # Use 交易所委托号 (exchange order ID) — sequential for algos
    id_col = "交易所委托号" if "交易所委托号" in orders.columns else "委托编号"
    orders["_id_num"] = pd.to_numeric(orders[id_col], errors="coerce")
    orders = orders.dropna(subset=["_id_num"]).sort_values("_id_num")
    orders["_time_sec"] = orders["时间"].astype(str).apply(_time_to_seconds)

    if len(orders) < 3:
        return []

    clusters = []
    visited = set()
    ids = orders["_id_num"].values
    times = orders["_time_sec"].values
    prices_raw = orders["委托价格"].values
    qtys = orders["委托数量"].values
    directions = orders["委托代码"].values if "委托代码" in orders.columns else np.array(["?"] * len(orders))

    i = 0
    while i < len(ids) - 2:
        if i in visited:
            i += 1
            continue

        # Look ahead for sequential IDs within gap and time window
        cluster_idx = [i]
        j = i + 1
        while j < len(ids) and (ids[j] - ids[cluster_idx[-1]]) <= id_gap_max:
            if times[j] - times[i] <= time_window_sec:
                cluster_idx.append(j)
            j += 1

        if len(cluster_idx) >= 3:
            cluster_ids = set(cluster_idx)
            c_prices = prices_raw[cluster_idx] / 10000  # to yuan
            c_qtys = qtys[cluster_idx]
            c_dirs = directions[cluster_idx]
            c_ids = ids[cluster_idx]

            # Price consistency check
            p_mean = np.mean(c_prices)
            if p_mean > 0 and np.max(np.abs(c_prices - p_mean)) / p_mean <= price_pct_tol:
                total_amount = np.sum(c_prices * c_qtys)
                buy_count = np.sum(c_dirs == "B")
                sell_count = np.sum(c_dirs == "S")

                clusters.append({
                    "order_ids": c_ids.tolist(),
                    "n_orders": len(cluster_idx),
                    "start_id": int(c_ids.min()),
                    "end_id": int(c_ids.max()),
                    "id_gap": int(c_ids.max() - c_ids.min()) - len(cluster_idx) + 1,
                    "time_start": float(times[i]),
                    "time_end": float(times[cluster_idx[-1]]),
                    "avg_price_yuan": round(float(p_mean), 2),
                    "total_amount_wan": round(float(total_amount) / 10000, 0),
                    "avg_qty": round(float(np.mean(c_qtys)), 0),
                    "direction": "BUY" if buy_count > sell_count else "SELL" if sell_count > buy_count else "MIXED",
                    "buy_count": int(buy_count),
                    "sell_count": int(sell_count),
                    "qty_cv": round(float(np.std(c_qtys) / np.mean(c_qtys)), 3) if np.mean(c_qtys) > 0 else 0,
                })
                visited.update(cluster_idx)

        i += 1

    # Sort by amount desc
    clusters.sort(key=lambda x: x["total_amount_wan"], reverse=True)
    return clusters


def profile_institution(clusters: list[dict], date: str) -> dict:
    """Create behavior fingerprint from algo clusters for one day."""
    if not clusters:
        return {"date": date, "n_algos": 0}

    amounts = [c["total_amount_wan"] for c in clusters]
    n_orders = [c["n_orders"] for c in clusters]
    times = [c["time_start"] for c in clusters]
    prices = [c["avg_price_yuan"] for c in clusters]

    # Time of day preference
    morning = sum(1 for t in times if 34200 <= t <= 41400)  # 09:30-11:30
    afternoon = sum(1 for t in times if 46800 <= t)  # after 13:00

    # Size preference
    big = sum(1 for a in amounts if a >= 500)

    buy_clusters = [c for c in clusters if c["direction"] == "BUY"]
    sell_clusters = [c for c in clusters if c["direction"] == "SELL"]

    return {
        "date": date,
        "n_algos": len(clusters),
        "total_buy_wan": round(sum(c["total_amount_wan"] for c in buy_clusters), 0),
        "total_sell_wan": round(sum(c["total_amount_wan"] for c in sell_clusters), 0),
        "net_flow_wan": round(sum(c["total_amount_wan"] for c in buy_clusters) -
                               sum(c["total_amount_wan"] for c in sell_clusters), 0),
        "avg_algo_size_wan": round(np.mean(amounts), 0) if amounts else 0,
        "max_algo_size_wan": max(amounts) if amounts else 0,
        "avg_n_orders_per_algo": round(np.mean(n_orders), 1) if n_orders else 0,
        "morning_ratio": round(morning / len(clusters), 2) if clusters else 0,
        "buy_count": len(buy_clusters),
        "sell_count": len(sell_clusters),
    }


def fingerprint_match(clusters_a: list[dict], clusters_b: list[dict]) -> float:
    """
    Compute similarity score between two days' algo clusters.
    Returns 0-1 score where 1 = identical behavior.
    """
    if not clusters_a or not clusters_b:
        return 0.0

    def _features(clusters):
        if not clusters:
            return np.zeros(6)
        amounts = [c["total_amount_wan"] for c in clusters]
        n_orders = [c["n_orders"] for c in clusters]
        buy_ratio = sum(1 for c in clusters if c["direction"] == "BUY") / len(clusters)
        times = [c["time_start"] for c in clusters]
        morning_ratio = sum(1 for t in times if 34200 <= t <= 41400) / len(times)
        return np.array([
            np.log1p(np.mean(amounts)), np.log1p(np.std(amounts)),
            np.mean(n_orders), buy_ratio, morning_ratio, len(clusters),
        ])

    fa = _features(clusters_a)
    fb = _features(clusters_b)

    # Cosine similarity
    denom = np.linalg.norm(fa) * np.linalg.norm(fb)
    if denom == 0:
        return 0.0
    return float(np.dot(fa, fb) / denom)


def main():
    print("=" * 80)
    print("机构算法拆单精确识别 — 基于委托编号连续性")
    print("=" * 80)

    # Focus on September 8-11 cluster
    dates = ["20250908", "20250909", "20250910", "20250911",
             "20250813", "20250814", "20250818"]  # Aug comparator days
    # Also check nearby dates for continuation
    extra_dates = ["20250905", "20250912", "20250915", "20250916", "20250917"]
    all_dates = dates + extra_dates

    all_clusters = {}
    all_profiles = {}

    for date_str in all_dates:
        orders, trades = load_day(date_str)
        if orders.empty:
            continue

        clusters = detect_algo_clusters(orders, id_gap_max=500, time_window_sec=5.0)
        profile = profile_institution(clusters, date_str)

        all_clusters[date_str] = clusters
        all_profiles[date_str] = profile

        # Show large clusters
        big = [c for c in clusters if c["total_amount_wan"] >= 200]
        if big:
            print(f"\n{'='*60}")
            print(f"[{date_str}] {len(clusters)} algo clusters, {len(big)} large (>200万)")
            print(f"{'='*60}")
            for c in big:
                time_str = f"{c['time_start']/3600:.2f}h"
                ids = c["order_ids"]
                print(f"  {c['direction']:>5} {c['total_amount_wan']:>8.0f}万 "
                      f"@{c['avg_price_yuan']:.2f}元 "
                      f"x{c['n_orders']}笔 "
                      f"IDs=[{ids[0]}..{ids[-1]}] gap={c['id_gap']} "
                      f"@{time_str} qty_cv={c['qty_cv']:.3f}")

    # Cross-date similarity matrix
    print(f"\n{'='*80}")
    print("跨日期行为指纹相似度矩阵")
    print(f"{'='*80}")

    active_dates = sorted(all_profiles.keys())
    print(f"\n{'':>10}", end="")
    for d in active_dates:
        print(f"{d[-4:]:>8}", end="")
    print(f"\n{'-'*(10+8*len(active_dates))}")

    for da in active_dates:
        print(f"{da:<10}", end="")
        for db in active_dates:
            score = fingerprint_match(all_clusters.get(da, []), all_clusters.get(db, []))
            marker = "█" if score > 0.95 else "▓" if score > 0.85 else "░" if score > 0.7 else " "
            print(f"  {marker}{score:.2f}", end="")
        print()

    # Summary table
    print(f"\n{'='*80}")
    print("每日机构活动总结")
    print(f"{'='*80}")
    print(f"{'Date':<12} {'Algos':>6} {'Buy万':>10} {'Sell万':>10} {'Net万':>10} "
          f"{'MaxSize万':>10} {'Avg订单':>8}")
    print("-" * 70)
    for d in active_dates:
        p = all_profiles[d]
        print(f"{d:<12} {p['n_algos']:>6} {p['total_buy_wan']:>10.0f} "
              f"{p['total_sell_wan']:>10.0f} {p['net_flow_wan']:>10.0f} "
              f"{p['max_algo_size_wan']:>10.0f} {p['avg_n_orders_per_algo']:>8.1f}")

    # Focus: the September 4-day cycle
    print(f"\n{'='*80}")
    print("9月建仓-出货周期分析")
    print(f"{'='*80}")
    sep_dates = ["20250905", "20250908", "20250909", "20250910", "20250911",
                 "20250912", "20250915", "20250916", "20250917"]

    cum_buy = 0
    cum_sell = 0
    for d in sep_dates:
        if d in all_profiles:
            p = all_profiles[d]
            cum_buy += p["total_buy_wan"]
            cum_sell += p["total_sell_wan"]
            bar_buy = "█" * int(p["total_buy_wan"] / 200) if p["total_buy_wan"] > 0 else ""
            bar_sell = "░" * int(p["total_sell_wan"] / 200) if p["total_sell_wan"] > 0 else ""
            print(f"  {d}: BUY {p['total_buy_wan']:>8.0f}万 {bar_buy}")
            print(f"         SELL {p['total_sell_wan']:>7.0f}万 {bar_sell}")
            print(f"         NET  {p['net_flow_wan']:>7.0f}万")

    print(f"\n  累计: BUY {cum_buy:.0f}万, SELL {cum_sell:.0f}万, NET {cum_buy-cum_sell:.0f}万")
    print(f"  净买入 = {(cum_buy-cum_sell)/10000:.1f}亿")

    if cum_buy > cum_sell:
        print(f"  >>> 机构在此期间净建仓{(cum_buy-cum_sell)/10000:.1f}亿 <<<")
    else:
        print(f"  >>> 机构在此期间净出货{(cum_sell-cum_buy)/10000:.1f}亿 <<<")

    # Check price impact
    price_path = PROJECT / "data" / "single_stock" / STOCK / "price_daily.csv"
    if price_path.exists():
        prices = pd.read_csv(price_path)
        prices["日期"] = pd.to_datetime(prices["日期"])
        print(f"\n--- 同期股价走势 ---")
        for d in sep_dates:
            dt = pd.Timestamp(f"{d[:4]}-{d[4:6]}-{d[6:8]}")
            row = prices[prices["日期"] == dt]
            if len(row) > 0:
                close = row["收盘"].values[0]
                change = row["涨跌幅"].values[0] if "涨跌幅" in prices.columns else 0
                print(f"  {d}: close={close:.2f}元, chg={change:.2f}%")


if __name__ == "__main__":
    main()
