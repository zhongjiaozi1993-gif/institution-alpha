"""
SOFIA v4 — 全年度匿名机构猎手 + 行为指纹识别系统

核心能力:
  1. 全2025年逐日算法拆单聚类
  2. 跨日行为指纹匹配 → 匿名机构注册表
  3. 识别各机构拆单手法、时段偏好、价格特征
  4. 输出: TOP10交易日、TOP10机构、行为规律汇总

用法:
  python3 scripts/sofia_v4_hunter.py --stock 002516 --year 2025
  python3 scripts/sofia_v4_hunter.py --stock 002516 --year 2025 --min-amount 200
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.level2_reader import read_level2_stock_dir
from src.cluster.split_detector import _time_to_seconds

PROJECT = Path(__file__).parent.parent
STOCK_DIR = PROJECT / "data" / "single_stock"


def load_day(stock: str, date_str: str):
    """Load Level-2 orders and trades for a single stock-day."""
    stock_dir = STOCK_DIR / stock / "raw" / date_str / f"{stock}.SZ"
    if not stock_dir.exists():
        return pd.DataFrame(), pd.DataFrame()
    try:
        data = read_level2_stock_dir(stock_dir)
    except Exception:
        return pd.DataFrame(), pd.DataFrame()
    return data.get("逐笔委托", pd.DataFrame()), data.get("逐笔成交", pd.DataFrame())


# ═══════════════════════════════════════════════════════
# 算法拆单聚类 (同 sofia_trace.py 核心逻辑)
# ═══════════════════════════════════════════════════════

def detect_algo_clusters(orders: pd.DataFrame,
                         id_gap_max: int = 500,
                         time_window_sec: float = 5.0,
                         price_pct_tol: float = 0.005,
                         min_orders: int = 5,
                         min_amount_wan: float = 50.0) -> list[dict]:
    """检测单日内机构算法拆单簇。"""
    if orders.empty:
        return []

    orders = orders.copy()
    orders["委托价格"] = pd.to_numeric(orders["委托价格"], errors="coerce")
    orders["委托数量"] = pd.to_numeric(orders["委托数量"], errors="coerce")
    orders = orders[orders["委托价格"] > 0]

    id_col = "交易所委托号" if "交易所委托号" in orders.columns else "委托编号"
    orders["_id_num"] = pd.to_numeric(orders[id_col], errors="coerce")
    orders = orders.dropna(subset=["_id_num"]).sort_values("_id_num")
    orders["_time_sec"] = orders["时间"].astype(str).apply(_time_to_seconds)

    if len(orders) < min_orders:
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

        cluster_idx = [i]
        j = i + 1
        while j < len(ids) and (ids[j] - ids[cluster_idx[-1]]) <= id_gap_max:
            if times[j] - times[i] <= time_window_sec:
                cluster_idx.append(j)
            j += 1

        if len(cluster_idx) >= min_orders:
            c_prices = prices_raw[cluster_idx] / 10000
            c_qtys = qtys[cluster_idx]
            c_dirs = directions[cluster_idx]
            c_ids = ids[cluster_idx]
            c_times = times[cluster_idx]

            p_mean = np.mean(c_prices)
            if p_mean > 0 and np.max(np.abs(c_prices - p_mean)) / p_mean <= price_pct_tol:
                total_amount = np.sum(c_prices * c_qtys)
                total_amount_wan = float(total_amount) / 10000
                if total_amount_wan < min_amount_wan:
                    i += 1
                    continue
                buy_count = int(np.sum(c_dirs == "B"))
                sell_count = int(np.sum(c_dirs == "S"))

                # 拆单均匀度: 每笔数量的CV
                qty_cv = float(np.std(c_qtys) / np.mean(c_qtys)) if np.mean(c_qtys) > 0 else 0

                # 拆单节奏: 相邻笔的时间间隔分布
                if len(c_times) > 1:
                    time_diffs = np.diff(np.sort(c_times))
                    avg_time_gap = float(np.mean(time_diffs))
                    time_gap_std = float(np.std(time_diffs))
                else:
                    avg_time_gap = 0.0
                    time_gap_std = 0.0

                # 价格激进程度: 偏离当日均价方向
                # 暂存，后续统一计算

                clusters.append({
                    "n_orders": len(cluster_idx),
                    "start_id": int(c_ids.min()),
                    "end_id": int(c_ids.max()),
                    "id_span": int(c_ids.max() - c_ids.min()),
                    "avg_id_gap": round(float(np.mean(np.diff(c_ids))), 1) if len(c_ids) > 1 else 0,
                    "time_start": float(times[i]),
                    "time_end": float(times[cluster_idx[-1]]),
                    "avg_price_yuan": round(float(p_mean), 2),
                    "total_amount_wan": round(total_amount_wan, 0),
                    "avg_qty": round(float(np.mean(c_qtys)), 0),
                    "direction": "BUY" if buy_count > sell_count else "SELL" if sell_count > buy_count else "MIXED",
                    "buy_count": int(buy_count),
                    "sell_count": int(sell_count),
                    "qty_cv": round(qty_cv, 3),
                    "avg_time_gap_sec": round(avg_time_gap, 2),
                    "time_gap_std": round(time_gap_std, 2),
                })
                visited.update(cluster_idx)
        i += 1

    clusters.sort(key=lambda x: x["total_amount_wan"], reverse=True)
    return clusters


# ═══════════════════════════════════════════════════════
# 行为指纹提取 & 跨日匹配
# ═══════════════════════════════════════════════════════

def extract_fingerprint(c: dict) -> dict:
    """从算法簇提取行为指纹特征向量。"""
    fp = {
        # 规模特征
        "log_amount": np.log10(c["total_amount_wan"] + 1),
        "n_orders": c["n_orders"],
        "avg_qty": c["avg_qty"],
        # 拆单特征
        "qty_cv": c["qty_cv"],
        "avg_id_gap": c["avg_id_gap"],
        "avg_time_gap_sec": c.get("avg_time_gap_sec", 0),
        # 价格特征
        "avg_price_yuan": c["avg_price_yuan"],
        # 时间特征
        "time_start_hour": c["time_start"] / 3600,
        "session": _classify_session(c["time_start"]),
        # 方向
        "direction": c["direction"],
    }
    return fp


def _classify_session(time_sec: float) -> str:
    """分类交易时段。"""
    h = time_sec / 3600
    if h < 9.30:
        return "AUCTION"       # 集合竞价 (09:15-09:25)
    elif h < 10.00:
        return "OPEN"           # 开盘 (09:30-10:00)
    elif h < 11.00:
        return "MORNING"        # 早盘 (10:00-11:00)
    elif h < 11.30:
        return "LATE_MORNING"   # 午前 (11:00-11:30)
    elif h < 13.30:
        return "EARLY_AFTER"    # 午后 (13:00-13:30)
    elif h < 14.30:
        return "AFTERNOON"      # 下午 (13:30-14:30)
    elif h < 15.00:
        return "CLOSE"          # 尾盘 (14:30-15:00)
    else:
        return "POST_CLOSE"     # 盘后


def match_score_fp(fp1: dict, fp2: dict) -> float:
    """
    多维加权行为指纹相似度评分。

    权重设计逻辑:
    - 规模权重高(0.20): 同一机构的交易规模通常稳定
    - 拆单手法权重最高(0.35): CV+IDgap+时间节奏 = 算法参数指纹
    - 时段偏好(0.15): 某些机构固定时段操作
    - 价格偏好(0.10): 弱特征，因市场价变化
    - 方向(0.10): 同向加分
    - 笔数规模(0.10): 拆单粒度
    """
    score = 0.0

    # 1. 规模相似度 (0.20)
    if fp1["log_amount"] > 0 and fp2["log_amount"] > 0:
        ratio = min(fp1["log_amount"], fp2["log_amount"]) / max(fp1["log_amount"], fp2["log_amount"])
        score += 0.20 * ratio

    # 2. 拆单CV相似度 (0.15) — 核心算法指纹
    if fp1["qty_cv"] > 0 and fp2["qty_cv"] > 0:
        cv_ratio = min(fp1["qty_cv"], fp2["qty_cv"]) / max(fp1["qty_cv"], fp2["qty_cv"])
        score += 0.15 * cv_ratio

    # 3. ID间隔相似度 (0.10) — 拆单密度指纹
    if fp1["avg_id_gap"] > 0 and fp2["avg_id_gap"] > 0:
        gap_ratio = min(fp1["avg_id_gap"], fp2["avg_id_gap"]) / max(fp1["avg_id_gap"], fp2["avg_id_gap"])
        score += 0.10 * gap_ratio

    # 4. 时间节奏相似度 (0.10) — 相邻笔时间间隔
    if fp1["avg_time_gap_sec"] > 0 and fp2["avg_time_gap_sec"] > 0:
        tg_ratio = min(fp1["avg_time_gap_sec"], fp2["avg_time_gap_sec"]) / max(fp1["avg_time_gap_sec"], fp2["avg_time_gap_sec"])
        score += 0.10 * tg_ratio

    # 5. 笔数规模相似度 (0.10)
    if fp1["n_orders"] > 0 and fp2["n_orders"] > 0:
        n_ratio = min(fp1["n_orders"], fp2["n_orders"]) / max(fp1["n_orders"], fp2["n_orders"])
        score += 0.10 * n_ratio

    # 6. 时段偏好 (0.15)
    if fp1["session"] == fp2["session"]:
        score += 0.15
    elif _session_adjacent(fp1["session"], fp2["session"]):
        score += 0.07

    # 7. 方向一致 (0.10)
    if fp1["direction"] == fp2["direction"]:
        score += 0.10

    # 8. 均价偏好 (0.05) — 弱特征，仅同向时有效
    if fp1["direction"] == fp2["direction"] and fp1["avg_price_yuan"] > 0 and fp2["avg_price_yuan"] > 0:
        px_ratio = min(fp1["avg_price_yuan"], fp2["avg_price_yuan"]) / max(fp1["avg_price_yuan"], fp2["avg_price_yuan"])
        score += 0.05 * px_ratio

    # 9. 平均每笔数量相似度 (0.05)
    if fp1["avg_qty"] > 0 and fp2["avg_qty"] > 0:
        qty_ratio = min(fp1["avg_qty"], fp2["avg_qty"]) / max(fp1["avg_qty"], fp2["avg_qty"])
        score += 0.05 * qty_ratio

    return score


def _session_adjacent(s1: str, s2: str) -> bool:
    """检查两个时段是否相邻。"""
    order = ["AUCTION", "OPEN", "MORNING", "LATE_MORNING",
             "EARLY_AFTER", "AFTERNOON", "CLOSE", "POST_CLOSE"]
    try:
        return abs(order.index(s1) - order.index(s2)) == 1
    except ValueError:
        return False


# ═══════════════════════════════════════════════════════
# 匿名机构注册表
# ═══════════════════════════════════════════════════════

def build_institution_registry(all_clusters: dict,
                               min_amount_wan: float = 200,
                               similarity_threshold: float = 0.60) -> list[dict]:
    """
    跨日匹配构建匿名机构注册表。

    算法: 贪婪分组
    1. 所有≥min_amount的簇按金额降序排列
    2. 从最大簇开始，与所有其他簇比较指纹相似度
    3. 相似度≥threshold的归入同一机构
    4. 已分配的簇不再参与新组
    """
    all_big = []
    for d, cs in all_clusters.items():
        for c in cs:
            if c["total_amount_wan"] >= min_amount_wan:
                fp = extract_fingerprint(c)
                all_big.append({**c, "date": d, "fingerprint": fp})

    assigned = set()
    groups = []

    for i, c1 in enumerate(all_big):
        if i in assigned:
            continue
        group = [c1]
        assigned.add(i)
        for j, c2 in enumerate(all_big):
            if j in assigned or c1["date"] == c2["date"]:
                continue
            score = match_score_fp(c1["fingerprint"], c2["fingerprint"])
            if score >= similarity_threshold:
                group.append(c2)
                assigned.add(j)
        groups.append(group)

    groups.sort(key=lambda g: sum(c["total_amount_wan"] for c in g), reverse=True)

    # 构建机构摘要
    registry = []
    for idx, group in enumerate(groups):
        total_amt = sum(c["total_amount_wan"] for c in group)
        buy_amt = sum(c["total_amount_wan"] for c in group if c["direction"] == "BUY")
        sell_amt = sum(c["total_amount_wan"] for c in group if c["direction"] == "SELL")
        days = sorted(set(c["date"] for c in group))
        rep = max(group, key=lambda c: c["total_amount_wan"])

        # 汇总行为指纹
        all_sessions = [c["fingerprint"]["session"] for c in group]
        session_counts = pd.Series(all_sessions).value_counts().to_dict()
        top_session = max(session_counts, key=session_counts.get) if session_counts else "?"

        all_dirs = [c["direction"] for c in group]
        buy_pct = all_dirs.count("BUY") / len(all_dirs) * 100

        qty_cvs = [c["qty_cv"] for c in group]
        id_gaps = [c["avg_id_gap"] for c in group]
        prices = [c["avg_price_yuan"] for c in group]
        time_starts = [c["time_start"] / 3600 for c in group]
        time_gaps = [c.get("avg_time_gap_sec", 0) for c in group]

        sz = "巨鲸" if total_amt >= 50000 else "大型" if total_amt >= 10000 else "中型" if total_amt >= 3000 else "小型"

        # 行为模式定性
        behavior_patterns = _classify_behavior(group)

        registry.append({
            "anon_id": f"ANON-{idx+1:03d}",
            "size_label": sz,
            "total_footprint_wan": round(total_amt, 0),
            "n_days": len(days),
            "n_clusters": len(group),
            "date_range": f"{min(days)} - {max(days)}",
            "buy_wan": round(buy_amt, 0),
            "sell_wan": round(sell_amt, 0),
            "net_wan": round(buy_amt - sell_amt, 0),
            "buy_pct": round(buy_pct, 1),
            "representative": {
                "date": rep["date"],
                "direction": rep["direction"],
                "amount_wan": rep["total_amount_wan"],
                "price_yuan": rep["avg_price_yuan"],
                "n_orders": rep["n_orders"],
                "avg_id_gap": rep["avg_id_gap"],
            },
            "behavior": behavior_patterns,
            "fingerprint_summary": {
                "top_session": top_session,
                "session_distribution": {k: round(v / len(group) * 100, 1) for k, v in session_counts.items()},
                "avg_qty_cv": round(float(np.mean(qty_cvs)), 3),
                "avg_id_gap": round(float(np.mean(id_gaps)), 1),
                "avg_price_yuan": round(float(np.mean(prices)), 2),
                "price_range": f"{min(prices):.2f}-{max(prices):.2f}",
                "avg_time_hour": round(float(np.mean(time_starts)), 2),
                "avg_time_gap_sec": round(float(np.mean(time_gaps)), 2),
            },
            "all_clusters": [{
                "date": c["date"],
                "direction": c["direction"],
                "amount_wan": c["total_amount_wan"],
                "price_yuan": c["avg_price_yuan"],
                "n_orders": c["n_orders"],
                "avg_id_gap": c["avg_id_gap"],
                "qty_cv": c["qty_cv"],
                "session": c["fingerprint"]["session"],
            } for c in sorted(group, key=lambda x: x["date"])],
        })

    return registry


def _classify_behavior(group: list[dict]) -> dict:
    """从一组操作中提取行为模式定性描述。"""
    n = len(group)
    sessions = [c["fingerprint"]["session"] for c in group]
    dirs = [c["direction"] for c in group]
    qty_cvs = [c["qty_cv"] for c in group]
    id_gaps = [c["avg_id_gap"] for c in group]
    time_gaps = [c.get("avg_time_gap_sec", 0) for c in group]

    # 拆单手法分类
    avg_cv = np.mean(qty_cvs)
    avg_gap = np.mean(id_gaps)
    avg_time_gap = np.mean(time_gaps)

    if avg_cv < 0.3 and avg_gap < 50:
        split_style = "精密等量拆单 (CV<0.3, IDgap<50) — 典型量化算法"
    elif avg_cv < 0.5 and avg_gap < 100:
        split_style = "均匀拆单 (CV<0.5, IDgap<100) — 半算法化交易"
    elif avg_cv < 1.0 and avg_gap < 300:
        split_style = "松散拆单 (CV<1.0, IDgap<300) — 人工辅助拆单"
    else:
        split_style = "不规则拆单 (高CV/大IDgap) — 可能非算法交易"

    # 时段偏好
    session_mode = max(set(sessions), key=sessions.count)
    session_labels = {
        "AUCTION": "集合竞价偏好 — 通过竞价建立仓位，追求确定价格",
        "OPEN": "开盘偏好 — 开盘抢筹/出货",
        "MORNING": "早盘偏好 — 利用早盘流动性",
        "CLOSE": "尾盘偏好 — 收盘前操作，避免日内波动",
        "AFTERNOON": "午后偏好 — 避开早盘噪音",
    }
    time_preference = session_labels.get(session_mode, f"{session_mode}时段偏好")

    # 操作风格
    buy_pct = dirs.count("BUY") / n * 100
    if buy_pct >= 80:
        op_style = "单向买入型 — 极少卖出，建仓/增持目的"
    elif buy_pct <= 20:
        op_style = "单向卖出型 — 极少买入，减持/出货目的"
    elif 40 <= buy_pct <= 60:
        op_style = "双边交易型 — 买卖均衡，做市或波段交易"
    elif buy_pct > 60:
        op_style = "偏多交易型 — 买多卖少，净增仓"
    else:
        op_style = "偏空交易型 — 卖多买少，净减仓"

    # 规模稳定性
    amounts = [c["total_amount_wan"] for c in group]
    amt_cv = float(np.std(amounts) / np.mean(amounts)) if np.mean(amounts) > 0 else 0
    if amt_cv < 0.5:
        size_stability = "规模稳定 (CV<0.5) — 固定仓位管理"
    elif amt_cv < 1.0:
        size_stability = "规模适中 (CV<1.0) — 灵活仓位调整"
    else:
        size_stability = "规模多变 (CV≥1.0) — 机会主义交易"

    return {
        "split_style": split_style,
        "time_preference": time_preference,
        "operation_style": op_style,
        "size_stability": size_stability,
        "avg_qty_cv": round(avg_cv, 3),
        "avg_id_gap": round(avg_gap, 1),
        "avg_time_gap_sec": round(avg_time_gap, 2),
        "typical_session": session_mode,
        "buy_pct": round(buy_pct, 1),
        "amount_cv": round(amt_cv, 3),
    }


# ═══════════════════════════════════════════════════════
# TOP N 识别
# ═══════════════════════════════════════════════════════

def find_top_trading_days(all_clusters: dict, top_n: int = 10) -> list[dict]:
    """识别TOP N交易事件日。"""
    day_summaries = []
    for d, cs in all_clusters.items():
        buy = sum(c["total_amount_wan"] for c in cs if c["direction"] == "BUY")
        sell = sum(c["total_amount_wan"] for c in cs if c["direction"] == "SELL")
        net = buy - sell
        total_flow = buy + sell
        large = [c for c in cs if c["total_amount_wan"] >= 500]
        top = cs[0] if cs else None

        day_summaries.append({
            "date": d,
            "n_algos": len(cs),
            "n_large_algos": len(large),
            "buy_wan": round(buy, 0),
            "sell_wan": round(sell, 0),
            "net_wan": round(net, 0),
            "total_flow_wan": round(total_flow, 0),
            "abs_net": abs(net),
            "top_cluster_wan": top["total_amount_wan"] if top else 0,
            "top_cluster_price": top["avg_price_yuan"] if top else 0,
            "top_cluster_orders": top["n_orders"] if top else 0,
            "top_cluster_gap": top["avg_id_gap"] if top else 0,
            "top_cluster_dir": top["direction"] if top else "",
        })

    # 按绝对净流入排序
    day_summaries.sort(key=lambda x: x["abs_net"], reverse=True)
    return day_summaries[:top_n]


# ═══════════════════════════════════════════════════════
# 输出
# ═══════════════════════════════════════════════════════

def print_top10_days(top_days: list[dict]):
    """打印TOP10交易事件日。"""
    print(f"\n{'='*120}")
    print("TOP 10 交易事件日 (按机构净流向绝对值)")
    print(f"{'='*120}")
    print(f"  {'Rank':<6} {'Date':<12} {'净流向(万)':>12} {'买入(万)':>12} {'卖出(万)':>12} "
          f"{'总流量(万)':>12} {'算法数':>7} {'最大簇(万)':>12} {'方向':>5} {'笔数':>6} {'IDgap':>7}")
    print(f"  {'-'*115}")
    for rank, d in enumerate(top_days, 1):
        arrow = "▲▲" if d["net_wan"] > 5000 else "▲" if d["net_wan"] > 1000 else \
                "▼▼" if d["net_wan"] < -5000 else "▼" if d["net_wan"] < -1000 else "─"
        print(f"  {rank:<6} {d['date']:<12} {arrow}{d['net_wan']:>10.0f} {d['buy_wan']:>12.0f} "
              f"{d['sell_wan']:>12.0f} {d['total_flow_wan']:>12.0f} {d['n_algos']:>7} "
              f"{d['top_cluster_wan']:>12.0f} {d['top_cluster_dir']:>5} {d['top_cluster_orders']:>6} "
              f"{d['top_cluster_gap']:>7.0f}")


def print_top10_institutions(registry: list[dict]):
    """打印TOP10机构画像。"""
    print(f"\n{'='*120}")
    print("TOP 10 匿名机构 (按总足迹金额)")
    print(f"{'='*120}")

    for idx, inst in enumerate(registry[:10], 1):
        rep = inst["representative"]
        bh = inst["behavior"]
        fp = inst["fingerprint_summary"]

        print(f"""
  ╔══════════════════════════════════════════════════════════════╗
  ║  [{idx:2d}] {inst['anon_id']}  [{inst['size_label']}]  总足迹={inst['total_footprint_wan']:.0f}万  覆盖{inst['n_days']}天
  ╠══════════════════════════════════════════════════════════════╣
  ║  资金流向: 买入{inst['buy_wan']:.0f}万  卖出{inst['sell_wan']:.0f}万  净{inst['net_wan']:+.0f}万
  ║  代表操作: {rep['date']} {rep['direction']} {rep['amount_wan']:.0f}万 @{rep['price_yuan']:.2f}元 x{rep['n_orders']}笔
  ║  日期范围: {inst['date_range']}
  ║
  ║  【拆单手法】{bh['split_style']}
  ║  【时段偏好】{bh['time_preference']} (模式: {bh['typical_session']})
  ║  【操作风格】{bh['operation_style']} (买入占比{bh['buy_pct']:.0f}%)
  ║  【规模稳定】{bh['size_stability']}
  ║
  ║  【行为指纹】
  ║    · 拆单CV均值: {fp['avg_qty_cv']:.3f} (低=均匀,高=不规则)
  ║    · ID间隔均值: {fp['avg_id_gap']:.0f} (低=连续下单,高=间隔大)
  ║    · 笔间间隔均值: {fp['avg_time_gap_sec']:.1f}s
  ║    · 均价区间: {fp['price_range']}元
  ║    · 时段分布: {fp['session_distribution']}
  ╚══════════════════════════════════════════════════════════════╝""")

        # 列出具体操作日期
        print(f"    操作日历:")
        for c in inst["all_clusters"]:
            bar = "█" * min(30, int(c["amount_wan"] / 300))
            print(f"      {c['date']} {c['direction']:>4} {c['amount_wan']:>8.0f}万 "
                  f"@{c['price_yuan']:.2f} x{c['n_orders']:>4}笔 gap={c['avg_id_gap']:>5.0f} "
                  f"[{c['session']:<12}] cv={c['qty_cv']:.3f} {bar}")
        print()


def print_behavior_summary(registry: list[dict]):
    """打印机构行为规律汇总。"""
    print(f"\n{'='*120}")
    print("机构行为规律汇总 — 识别各机构独特手法")
    print(f"{'='*120}")

    print(f"""
  ┌──────────────────────────────────────────────────────────────────────────────────────┐
  │  拆单手法谱系 (从精密算法到人工拆单)                                                    │
  ├──────────────────────────────────────────────────────────────────────────────────────┤
  │  精密等量拆单 (CV<0.3, IDgap<50):   量化算法/程序化交易，每笔数量几乎相同，ID连续       │
  │  均匀拆单   (CV<0.5, IDgap<100):   半算法化交易，有规律但不完全均匀                     │
  │  松散拆单   (CV<1.0, IDgap<300):   人工辅助拆单，有一定随意性                            │
  │  不规则拆单 (CV≥1.0, IDgap≥300):   可能非算法交易，大单直接砸或分少数几笔                │
  └──────────────────────────────────────────────────────────────────────────────────────┘

  ┌──────────────────────────────────────────────────────────────────────────────────────┐
  │  时段偏好分类                                                                            │
  ├──────────────────────────────────────────────────────────────────────────────────────┤
  │  集合竞价(AUCTION):   追求确定开盘价，适合建仓/减持不想暴露意图                           │
  │  开盘(OPEN):          抢流动性，适合快速建仓/出货                                          │
  │  早盘(MORNING):       利用流动性充裕时段，大宗交易                                         │
  │  尾盘(CLOSE):         规避日内波动，收盘价附近完成交易                                      │
  └──────────────────────────────────────────────────────────────────────────────────────┘

  ┌──────────────────────────────────────────────────────────────────────────────────────┐
  │  可复现识别规则 (当同一机构再次操作时，以下特征可匹配)                                      │
  ├──────────────────────────────────────────────────────────────────────────────────────┤
  │  1. 拆单CV + ID间隔 = 最强指纹 (算法参数不易改变)                                         │
  │  2. 时段偏好 = 中等指纹 (机构有固定操作习惯)                                               │
  │  3. 每笔规模 + 总规模 = 中等指纹 (受市场状况影响)                                         │
  │  4. 均价偏好 = 弱指纹 (随市场价格变化)                                                    │
  └──────────────────────────────────────────────────────────────────────────────────────┘
""")

    # 对比表: 各机构的拆单手法
    print(f"  {'机构':<12} {'规模':<6} {'天数':>4} {'拆单手法':<45} {'时段偏好':<20} {'操作风格':<20}")
    print(f"  {'-'*115}")
    for inst in registry[:15]:
        bh = inst["behavior"]
        print(f"  {inst['anon_id']:<12} {inst['size_label']:<6} {inst['n_days']:>4} "
              f"{bh['split_style'][:43]:<45} {bh['typical_session']:<20} {bh['operation_style'][:18]:<20}")


def save_outputs(registry: list[dict], top_days: list[dict],
                 all_clusters: dict, stock: str, year: str):
    """保存所有输出到CSV和JSON。"""
    out_dir = STOCK_DIR / stock / "sofia_v4"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. 机构注册表 JSON
    registry_out = []
    for inst in registry:
        r = {k: v for k, v in inst.items() if k != "all_clusters"}
        r["all_clusters"] = inst["all_clusters"]
        registry_out.append(r)

    with open(out_dir / "institution_registry.json", "w") as f:
        json.dump(registry_out, f, ensure_ascii=False, indent=2)

    # 2. 机构注册表 CSV
    reg_rows = []
    for inst in registry:
        bh = inst["behavior"]
        fp = inst["fingerprint_summary"]
        rep = inst["representative"]
        reg_rows.append({
            "anon_id": inst["anon_id"],
            "size_label": inst["size_label"],
            "total_footprint_wan": inst["total_footprint_wan"],
            "n_days": inst["n_days"],
            "n_clusters": inst["n_clusters"],
            "date_range": inst["date_range"],
            "buy_wan": inst["buy_wan"],
            "sell_wan": inst["sell_wan"],
            "net_wan": inst["net_wan"],
            "buy_pct": inst["buy_pct"],
            "split_style": bh["split_style"],
            "time_preference": bh["time_preference"],
            "operation_style": bh["operation_style"],
            "size_stability": bh["size_stability"],
            "avg_qty_cv": bh["avg_qty_cv"],
            "avg_id_gap": bh["avg_id_gap"],
            "typical_session": bh["typical_session"],
            "avg_price_yuan": fp["avg_price_yuan"],
            "price_range": fp["price_range"],
            "rep_date": rep["date"],
            "rep_direction": rep["direction"],
            "rep_amount_wan": rep["amount_wan"],
            "rep_orders": rep["n_orders"],
        })
    pd.DataFrame(reg_rows).to_csv(out_dir / "institution_registry.csv", index=False)

    # 3. TOP交易日
    pd.DataFrame(top_days).to_csv(out_dir / "top_trading_days.csv", index=False)

    # 4. 每日算法簇汇总
    daily_rows = []
    for d, cs in all_clusters.items():
        buy = sum(c["total_amount_wan"] for c in cs if c["direction"] == "BUY")
        sell = sum(c["total_amount_wan"] for c in cs if c["direction"] == "SELL")
        daily_rows.append({
            "date": d,
            "n_algos": len(cs),
            "n_large": sum(1 for c in cs if c["total_amount_wan"] >= 200),
            "buy_wan": round(buy, 0),
            "sell_wan": round(sell, 0),
            "net_wan": round(buy - sell, 0),
        })
    pd.DataFrame(daily_rows).to_csv(out_dir / "daily_algo_summary.csv", index=False)

    # 5. 全量算法簇明细
    cluster_rows = []
    for d, cs in all_clusters.items():
        for c in cs:
            if c["total_amount_wan"] >= 100:
                cluster_rows.append({
                    "date": d,
                    "direction": c["direction"],
                    "amount_wan": c["total_amount_wan"],
                    "price_yuan": c["avg_price_yuan"],
                    "n_orders": c["n_orders"],
                    "avg_id_gap": c["avg_id_gap"],
                    "qty_cv": c["qty_cv"],
                    "time_start_h": round(c["time_start"] / 3600, 2),
                    "session": _classify_session(c["time_start"]),
                    "avg_qty": c["avg_qty"],
                })
    pd.DataFrame(cluster_rows).to_csv(out_dir / "all_algo_clusters.csv", index=False)

    print(f"\n输出文件已保存到: {out_dir}/")
    print(f"  institution_registry.json  — 机构注册表 (JSON, 完整)")
    print(f"  institution_registry.csv   — 机构注册表 (CSV)")
    print(f"  top_trading_days.csv       — TOP10交易日")
    print(f"  daily_algo_summary.csv     — 每日算法簇汇总")
    print(f"  all_algo_clusters.csv      — 全量算法簇明细 (≥100万)")


# ═══════════════════════════════════════════════════════
# 主流程
# ═══════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="SOFIA v4 — 全年度匿名机构猎手")
    parser.add_argument("--stock", default="002516")
    parser.add_argument("--year", default="2025")
    parser.add_argument("--min-amount", type=float, default=200,
                        help="聚类最低金额(万), 默认200")
    parser.add_argument("--similarity", type=float, default=0.60,
                        help="跨日匹配相似度阈值, 默认0.60")
    parser.add_argument("--quick", action="store_true",
                        help="快速模式: 仅前60个交易日")
    args = parser.parse_args()

    stock = args.stock
    year = args.year

    # 获取交易日列表
    raw_dir = STOCK_DIR / stock / "raw"
    all_dates = sorted([d.name for d in raw_dir.iterdir()
                        if d.is_dir() and d.name.startswith(year)])
    dates = all_dates[:60] if args.quick else all_dates

    print(f"SOFIA v4 — 匿名机构猎手")
    print(f"  股票: {stock}")
    print(f"  年份: {year}")
    print(f"  交易日: {len(dates)}天")
    print(f"  最少金额: {args.min_amount}万")
    print(f"  相似度阈值: {args.similarity}")
    print()

    # Phase 1: 逐日算法聚类
    all_clusters = {}
    day_count = 0
    for date_str in dates:
        orders, _ = load_day(stock, date_str)
        if orders.empty:
            continue
        clusters = detect_algo_clusters(orders)
        if clusters:
            all_clusters[date_str] = clusters
        day_count += 1
        if day_count % 30 == 0:
            print(f"  进度: {day_count}/{len(dates)} 天完成, 已检测{sum(len(v) for v in all_clusters.values())}个算法簇")

    print(f"\n  聚类完成: {len(all_clusters)}天有算法簇, 共{sum(len(v) for v in all_clusters.values())}个簇")

    # Phase 2: 匿名机构注册表
    print(f"\n  构建匿名机构注册表...")
    registry = build_institution_registry(all_clusters,
                                          min_amount_wan=args.min_amount,
                                          similarity_threshold=args.similarity)
    print(f"  注册机构: {len(registry)}个")

    # Phase 3: TOP N
    print(f"\n  识别TOP10交易日...")
    top_days = find_top_trading_days(all_clusters, top_n=10)

    # Phase 4: 输出
    print_top10_days(top_days)
    print_top10_institutions(registry)
    print_behavior_summary(registry)

    # Phase 5: 保存
    save_outputs(registry, top_days, all_clusters, stock, year)

    # Phase 6: 关键发现
    print(f"\n{'='*120}")
    print("关键发现")
    print(f"{'='*120}")

    total_buy = sum(sum(c["total_amount_wan"] for c in cs if c["direction"] == "BUY")
                    for cs in all_clusters.values())
    total_sell = sum(sum(c["total_amount_wan"] for c in cs if c["direction"] == "SELL")
                     for cs in all_clusters.values())

    precision_institutions = sum(1 for inst in registry
                                 if inst["behavior"]["split_style"].startswith("精密等量拆单"))
    auction_lovers = sum(1 for inst in registry
                         if inst["behavior"]["typical_session"] == "AUCTION")
    single_direction = sum(1 for inst in registry
                           if inst["buy_pct"] >= 80 or inst["buy_pct"] <= 20)

    print(f"""
  1. 【全年级别】
     全年机构算法总买入: {total_buy:.0f}万 = {total_buy/10000:.1f}亿
     全年机构算法总卖出: {total_sell:.0f}万 = {total_sell/10000:.1f}亿
     全年净流向: {total_buy-total_sell:+.0f}万

  2. 【机构结构】
     共识别 {len(registry)} 个匿名机构
     其中精密算法型(量化): {precision_institutions}个
     竞价偏好型: {auction_lovers}个
     单向操作型: {single_direction}个

  3. 【2025年最大事件日】
     {top_days[0]['date']}: 净{top_days[0]['net_wan']:+.0f}万 ({top_days[0]['n_algos']}个算法)
     最大单簇: {top_days[0]['top_cluster_wan']:.0f}万 x{top_days[0]['top_cluster_orders']}笔

  4. 【可复现的识别规则】
     当同一机构未来再次操作:
     - 精密算法型: 匹配拆单CV+ID间隔 (最可靠)
     - 竞价偏好型: 匹配时段+拆单手法
     - 单向操作型: 匹配方向+规模模式
     匹配阈值≥0.60即可识别为同一机构
""")

    # 加载价格数据做补充分析
    price_path = STOCK_DIR / stock / "price_daily.csv"
    if price_path.exists():
        prices = pd.read_csv(price_path)
        prices["日期"] = pd.to_datetime(prices["日期"])
        first = pd.Timestamp(f"{year}-01-01")
        last = pd.Timestamp(f"{year}-12-31")
        year_prices = prices[(prices["日期"] >= first) & (prices["日期"] <= last)]
        if len(year_prices) > 0:
            ytd_return = (year_prices["收盘"].values[-1] / year_prices["收盘"].values[0] - 1) * 100
            print(f"  5. 【年度背景】{year}年002516涨幅: {ytd_return:+.1f}%")
            print(f"     首日收盘: {year_prices['收盘'].values[0]:.2f}")
            print(f"     末日收盘: {year_prices['收盘'].values[-1]:.2f}")


if __name__ == "__main__":
    main()
