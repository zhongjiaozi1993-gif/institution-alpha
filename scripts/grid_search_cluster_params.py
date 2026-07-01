"""
DBSCAN聚类参数网格搜索

对 eps × min_samples × min_amount 组合逐日运行拆单检测，
记录每组参数下的操作数量、金额分布、未来收益、胜率、证据命中率，
输出 grid_search_result.csv 和 grid_search_summary.csv。

用法:
  python3 scripts/grid_search_cluster_params.py --stock 002516 --sample-days 50
"""

from __future__ import annotations

import argparse
import json
import sys
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))
DATA_DIR = PROJECT / "data" / "single_stock"
PRICE_SCALE = 10000  # L2价格 元×10000 → 元
FWD_DAYS = [5, 10, 20]  # 未来收益窗口

# 已知重点事件日（用于证据命中率评估）
KNOWN_EVENT_DATES = {
    "002516": {
        "super_buy":  ["20250813", "20250908", "20250910"],
        "super_sell": ["20250911"],
    }
}



def preload_wtcj_cache(stock: str, dates: list[str]) -> dict[str, pd.DataFrame]:
    """预加载所有日期的委托-成交匹配数据，返回 {date_str: wtcj_df}"""
    from src.data.level2_reader import read_level2_day, match_orders_to_trades

    raw_base = str(DATA_DIR / stock / "raw")
    cache = {}
    for i, d in enumerate(dates):
        date_with_dash = f"{d[:4]}-{d[4:6]}-{d[6:8]}"
        try:
            data = read_level2_day(raw_base, date_with_dash, stock)
            if data and "逐笔委托" in data and "逐笔成交" in data:
                wtcj = match_orders_to_trades(data["逐笔委托"], data["逐笔成交"])
                if not wtcj.empty:
                    cache[d] = wtcj
        except Exception:
            continue
        if (i + 1) % 20 == 0:
            print(f"    预加载: {i+1}/{len(dates)}")
    return cache


def compute_fwd_return_l2(stock: str, current_date: str, horizon: int) -> float | None:
    """
    用 L2 成交数据计算 horizon 日后的 close-to-close 收益。
    避免 HFQ/日线价格口径问题。
    Returns: pct return (0.05 = +5%), or None if data unavailable
    """
    # Collect L2 close prices for current_date through current_date + N trading days
    raw_dir = DATA_DIR / stock / "raw"
    all_dates = sorted([d.name for d in raw_dir.iterdir()
                        if d.is_dir() and len(d.name) == 8])

    try:
        idx = all_dates.index(current_date)
    except ValueError:
        return None

    target_idx = idx + horizon
    if target_idx >= len(all_dates):
        return None

    # L2 close for current day
    cur_close = _l2_daily_close(stock, current_date)
    fwd_close = _l2_daily_close(stock, all_dates[target_idx])
    if cur_close is None or fwd_close is None or cur_close == 0:
        return None
    return float((fwd_close / cur_close) - 1) * 100


def _l2_daily_close(stock: str, date_str: str) -> float | None:
    """从逐笔成交提取当日最后一笔成交价（元）"""
    cj_path = DATA_DIR / stock / "raw" / date_str / f"{stock}.SZ" / "逐笔成交.csv"
    if not cj_path.exists():
        return None
    for enc in ["gb18030", "gbk", "utf-8"]:
        try:
            df = pd.read_csv(cj_path, encoding=enc, low_memory=False)
            break
        except (UnicodeDecodeError, UnicodeError):
            continue
    else:
        return None
    if "成交代码" in df.columns:
        df = df[df["成交代码"] != "C"]
    if df.empty:
        return None
    df = df.sort_values("时间")
    return float(df["成交价格"].iloc[-1]) / PRICE_SCALE


def evaluate_params(stock: str, dates: list[str],
                    params: dict[str, float],
                    wtcj_cache: dict[str, pd.DataFrame]) -> dict:
    """对一组参数在所有日期上评估（使用预加载的 wtcj 缓存）"""
    from src.cluster.split_detector import detect_institution_operations

    daily_ops = {}
    all_buy_ops = []
    all_sell_ops = []

    for d in dates:
        wtcj = wtcj_cache.get(d)
        if wtcj is None:
            continue
        try:
            ops = detect_institution_operations(
                wtcj,
                eps=params["eps"],
                min_samples=int(params["min_samples"]),
                min_total_amount_wan=params["min_amount_wan"],
            )
        except Exception:
            continue

        # Compute forward returns for BUY ops
        for op in ops:
            op["_date"] = d
            if op["direction"] == "BUY":
                for horizon in FWD_DAYS:
                    ret = compute_fwd_return_l2(stock, d, horizon)
                    op[f"fwd_{horizon}d"] = ret
                all_buy_ops.append(op)
            else:
                all_sell_ops.append(op)

        if ops:
            daily_ops[d] = ops

    if not all_buy_ops:
        return {"params": params, "n_days_with_ops": 0, "n_ops_total": 0}

    buys = pd.DataFrame(all_buy_ops)
    sells_df = pd.DataFrame(all_sell_ops) if all_sell_ops else pd.DataFrame()

    # 基础统计
    n_days = len(daily_ops)
    n_buys = len(buys)
    n_sells = len(all_sell_ops)
    avg_amount = float(buys["total_amount_wan"].mean()) if n_buys > 0 else 0
    median_amount = float(buys["total_amount_wan"].median()) if n_buys > 0 else 0
    total_amount = float(buys["total_amount_wan"].sum()) if n_buys > 0 else 0

    # BUY/SELL 比例
    buy_sell_ratio = n_buys / max(1, n_sells)

    # Notable event: single op > 5000万 或 day net > 10000万
    n_super_ops = int((buys["total_amount_wan"] >= 5000).sum()) if n_buys > 0 else 0
    # Per-day net
    day_nets = {}
    for d, ops in daily_ops.items():
        b = sum(o["total_amount_wan"] for o in ops if o["direction"] == "BUY")
        s = sum(o["total_amount_wan"] for o in ops if o["direction"] == "SELL")
        day_nets[d] = b - s
    n_notable_days = sum(1 for v in day_nets.values() if abs(v) >= 5000)

    # 未来收益统计 (BUY ops)
    ret_stats = {}
    for horizon in FWD_DAYS:
        col = f"fwd_{horizon}d"
        if col in buys.columns:
            rets = buys[col].dropna()
            if len(rets) > 0:
                ret_stats[f"fwd_{horizon}d_mean"] = round(float(rets.mean()), 2)
                ret_stats[f"fwd_{horizon}d_win_rate"] = round(
                    float((rets > 0).sum() / len(rets)), 3)
                ret_stats[f"fwd_{horizon}d_n"] = len(rets)

    # 证据命中率
    events = KNOWN_EVENT_DATES.get(stock, {})
    super_buy_dates = set(events.get("super_buy", []))
    super_sell_dates = set(events.get("super_sell", []))
    notable_set = set(d for d, v in day_nets.items() if v >= 5000)
    sell_set = set(d for d, v in day_nets.items() if v <= -5000)

    hit_buy = len(notable_set & super_buy_dates)
    hit_sell = len(sell_set & super_sell_dates)
    total_known = len(super_buy_dates) + len(super_sell_dates)
    hit_rate = (hit_buy + hit_sell) / max(1, total_known)

    # 操作金额分布
    amount_bins = [0, 200, 500, 1000, 5000, 1e9]
    amount_dist = {}
    if n_buys > 0:
        for i in range(len(amount_bins) - 1):
            lo, hi = amount_bins[i], amount_bins[i+1]
            cnt = int(((buys["total_amount_wan"] >= lo) &
                       (buys["total_amount_wan"] < hi)).sum())
            amount_dist[f"amt_{lo}_{hi}"] = cnt

    return {
        "eps": params["eps"],
        "min_samples": int(params["min_samples"]),
        "min_amount_wan": params["min_amount_wan"],
        "n_days_total": len(dates),
        "n_days_with_ops": n_days,
        "n_ops_total": n_buys + n_sells,
        "n_buys": n_buys,
        "n_sells": n_sells,
        "buy_sell_ratio": round(buy_sell_ratio, 2),
        "avg_amount_wan": round(avg_amount, 1),
        "median_amount_wan": round(median_amount, 1),
        "total_amount_wan": round(total_amount, 1),
        "n_super_ops": n_super_ops,
        "n_notable_days": n_notable_days,
        "hit_rate": round(hit_rate, 3),
        "hit_detail": f"buy:{hit_buy}/{len(super_buy_dates)} sell:{hit_sell}/{len(super_sell_dates)}",
        **ret_stats,
        **amount_dist,
    }


def main():
    ap = argparse.ArgumentParser(description="DBSCAN聚类参数网格搜索")
    ap.add_argument("--stock", default="002516")
    ap.add_argument("--sample-days", type=int, default=50,
                    help="采样交易日数量 (默认50, 每N天取1天)")
    ap.add_argument("--eps-list", default="0.08,0.10,0.12,0.15,0.20,0.25")
    ap.add_argument("--min-samples-list", default="3,5,8,10")
    ap.add_argument("--min-amount-list", default="100,300,500,1000")
    args = ap.parse_args()

    stock = args.stock

    # 采样交易日（等距 + 强制包含已知事件日）
    raw_dir = DATA_DIR / stock / "raw"
    all_dates = sorted([d.name for d in raw_dir.iterdir()
                        if d.is_dir() and len(d.name) == 8 and
                        (d / f"{stock}.SZ").exists()])
    all_2025 = [d for d in all_dates if d.startswith("2025")]
    step = max(1, len(all_2025) // args.sample_days)
    sample_dates = set(all_2025[::step][:args.sample_days])

    # 强制包含已知事件日
    events = KNOWN_EVENT_DATES.get(stock, {})
    for event_list in events.values():
        for d in event_list:
            if d in all_dates:
                sample_dates.add(d)

    sample_dates = sorted(sample_dates)
    print(f"股票: {stock}, 采样 {len(sample_dates)}/{len(all_2025)} 天")
    known_in = set(sum(KNOWN_EVENT_DATES.get(stock, {}).values(), [])) & set(sample_dates)
    print(f"已知事件日覆盖: {known_in}")

    # 预加载所有日期数据
    print("预加载委托-成交匹配数据...")
    wtcj_cache = preload_wtcj_cache(stock, sample_dates)
    if not wtcj_cache:
        print("  ✗ 无法加载任何数据，终止")
        sys.exit(1)
    print(f"  ✓ 加载了 {len(wtcj_cache)}/{len(sample_dates)} 天的数据")
    test_date = list(wtcj_cache.keys())[0]
    print(f"  ✓ 示例 {test_date}: {len(wtcj_cache[test_date])} 条委托-成交")

    # 网格参数
    eps_list = [float(x) for x in args.eps_list.split(",")]
    min_samples_list = [int(x) for x in args.min_samples_list.split(",")]
    min_amount_list = [int(x) for x in args.min_amount_list.split(",")]

    grid = list(product(eps_list, min_samples_list, min_amount_list))
    print(f"\n网格: {len(eps_list)}×{len(min_samples_list)}×{len(min_amount_list)}"
          f" = {len(grid)} 组参数")
    print(f"eps: {eps_list}")
    print(f"min_samples: {min_samples_list}")
    print(f"min_amount_wan: {min_amount_list}")
    print()

    results = []
    for i, (eps, ms, ma) in enumerate(grid):
        params = {"eps": eps, "min_samples": ms, "min_amount_wan": ma}
        r = evaluate_params(stock, sample_dates, params, wtcj_cache)
        if r is None:
            print(f"  [{i+1:3d}/{len(grid)}] eps={eps:.2f} ms={ms} amt={ma}W "
                  f"→ 跳过(无数据)")
            continue
        results.append(r)

        n_ops = r.get("n_ops_total", 0)
        hit = r.get("hit_rate", 0)
        wr5 = r.get("fwd_5d_win_rate", "-")
        print(f"  [{i+1:3d}/{len(grid)}] eps={eps:.2f} ms={ms} amt={ma}W "
              f"→ ops={n_ops:4d} hit={hit:.2f} wr5={wr5}")

    # 排序列
    col_order = ["eps", "min_samples", "min_amount_wan",
                 "n_days_total", "n_days_with_ops", "n_ops_total",
                 "n_buys", "n_sells", "buy_sell_ratio",
                 "avg_amount_wan", "median_amount_wan", "total_amount_wan",
                 "n_super_ops", "n_notable_days", "hit_rate", "hit_detail"]

    ret_cols = []
    for h in FWD_DAYS:
        ret_cols.extend([f"fwd_{h}d_mean", f"fwd_{h}d_win_rate"])
    col_order.extend(ret_cols)

    amt_cols = sorted([c for c in results[0] if c.startswith("amt_")]) if results else []
    col_order.extend(amt_cols)

    df = pd.DataFrame(results)[col_order]
    df = df.sort_values("hit_rate", ascending=False)

    out_dir = PROJECT / "data" / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 完整结果
    full_path = out_dir / "grid_search_result.csv"
    df.to_csv(full_path, index=False)
    print(f"\n完整结果: {full_path} ({len(df)} 行)")

    # Top 10
    print(f"\n=== Top 10 (按证据命中率) ===")
    top = df.head(10)[["eps", "min_samples", "min_amount_wan",
                        "n_ops_total", "n_buys", "n_sells",
                        "buy_sell_ratio", "avg_amount_wan",
                        "fwd_5d_mean", "fwd_5d_win_rate", "hit_rate"]]
    print(top.to_string(index=False))

    # 如果完全没命中已知事件，按 ops 数量和 ret 排序展示
    if df["hit_rate"].max() == 0:
        print(f"\n=== Top 10 (按 fwd_5d_win_rate) ===")
        top2 = df[df["n_ops_total"] >= 10].sort_values(
            "fwd_5d_win_rate", ascending=False).head(10)
        if len(top2) == 0:
            top2 = df.head(10)
        print(top2[["eps", "min_samples", "min_amount_wan",
                     "n_ops_total", "fwd_5d_mean", "fwd_5d_win_rate",
                     "buy_sell_ratio"]].to_string(index=False))


if __name__ == "__main__":
    main()
