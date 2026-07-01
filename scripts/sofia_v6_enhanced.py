"""
SOFIA v6 — v4骨架 + v5增强 跨日机构追踪
=========================================

方法论总览
---------

### 同日聚类 (来自v4, 确定性高)
基于交易所委托号的物理连续性:
- 同一股票同日内的委托号是全局递增的
- 算法拆单会产生连续的委托号序列（IDgap小）
- 检测条件: IDgap≤500, 时间窗口≤5秒, 价格容差≤0.5%, 最少5笔, 最少50万元
- 输出: 每个簇 = 一个机构的单次操作（日期+方向+金额+均价+笔数+指纹）

### 跨日匹配 (v4贪婪分组 → v6保留骨架)
v4阶段使用0.60相似度阈值进行贪婪分组，建立20个机构的初步身份。
v6不再重新匹配，而是:
  1. 保留v4的机构身份作为锚定（避免身份爆炸）
  2. 计算机构间7维相似度进行去重合并
  3. 单轮配对，不链式传导（避免过度合并）

### Post-Merge去重 (7条件, ≥4触发合并)
  1. 日期重叠率 ≥ 60%
  2. 买入金额相似度 ≥ 85% 或 卖出金额相似度 ≥ 85%
  3. 时段偏好top session一致
  4. 净流向方向一致
  5. IDgap相似度 ≥ 0.65
  6. 拆单CV相似度 ≥ 0.65
  7. 操作频率相似 (ops比例≥0.45 且 days比例≥0.45)

### 5因子置信度评分 (替代v5的纯方向纯度)
| 因子 | 权重 | 含义 |
|------|------|------|
| continuity | 0.25 | 操作次数+活跃天数+时间跨度+交易密度 |
| amount | 0.20 | 总成交额+净流向规模+单笔中位数 |
| position | 0.20 | 持仓曲线的趋势性和单调性 |
| style | 0.15 | 指纹内部一致性(CV/IDgap/时段) |
| crossday | 0.10 | 交易间隔的规律性 |
| direction | 0.10 | abs(net)/gross 方向性强度 |

置信度阈值: ≥0.60=HIGH, ≥0.35=MEDIUM, <0.35=LOW

### 7类行为标签
| 类型 | 判定条件 |
|------|---------|
| 长期纯买建仓型 | buy_pct≥85%, net>0, date_span≥90天, n_days≥30 |
| 纯买建仓型 | buy_pct≥85%, net>0 (不满足长期条件) |
| 长期纯卖出货型 | buy_pct≤15%, net<0, date_span≥90天, n_days≥30 |
| 纯卖出货型 | buy_pct≤15%, net<0 |
| 长期净买调仓型 | net>0, buy_pct∈[55,85), n_days≥30 |
| 净买调仓型 | net>0, buy_pct∈[55,85) |
| 长期净卖出货型 | net<0, buy_pct∈[15,45], n_days≥30 |
| 净卖出货型 | net<0, buy_pct∈[15,45] |
| 波段交易型 | 方向切换率≥25% |
| 短期突击型 | n_days≤5, gross≥5000万 |
| 长期维护/做市型 | n_days≥30, abs(net)/gross<0.15 |
| 低活跃型 | n_ops<10 |
| 双向调仓型 | 其他情况 |

### 持仓曲线构建
- 按时间排序每机构的所有买卖操作
- 累计净持仓 = Σ买入金额 - Σ卖出金额
- 关键节点标注: 首次建仓、方向切换、大额操作、末次操作
- 月度聚合: 每月净买卖节奏（建仓/出货/调整）

输出:
  data/single_stock/{stock}/sofia_v6/
    institution_registry.json   — 增强注册表
    institution_registry.csv    — 汇总CSV
    position_curves.csv         — 每日持仓曲线
    merge_log.csv               — 合并日志
    v6_report.md                — 人读报告
"""
from __future__ import annotations

import json
import sys
import argparse
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT = Path(__file__).parent.parent
STOCK = "002516"
STOCK_DIR = PROJECT / "data" / "single_stock" / STOCK
V4_DIR = STOCK_DIR / "sofia_v4"
V6_DIR = STOCK_DIR / "sofia_v6"
V6_DIR.mkdir(parents=True, exist_ok=True)

SESSION_ORDER = {"AUCTION": 0, "OPEN": 1, "MORNING": 2, "LATE_MORNING": 3,
                 "EARLY_AFTER": 4, "AFTERNOON": 5, "CLOSE": 6}


def configure_stock(stock: str) -> None:
    """Point all v6 outputs at the requested single-stock workspace."""
    global STOCK, STOCK_DIR, V4_DIR, V6_DIR
    STOCK = stock
    STOCK_DIR = PROJECT / "data" / "single_stock" / STOCK
    V4_DIR = STOCK_DIR / "sofia_v4"
    V6_DIR = STOCK_DIR / "sofia_v6"
    V6_DIR.mkdir(parents=True, exist_ok=True)


def load_json_compat(path: Path):
    for encoding in ("utf-8", "utf-8-sig", "gb18030", "gbk"):
        try:
            with open(path, encoding=encoding) as f:
                return json.load(f)
        except UnicodeDecodeError:
            continue
    with open(path, encoding="utf-8", errors="replace") as f:
        return json.load(f)


# ═══════════════════════════════════════════════════
# 1. 加载v4数据
# ═══════════════════════════════════════════════════

def load_v4_institutions():
    """加载v4注册表，提取机构的完整簇列表和指纹。"""
    registry = load_json_compat(V4_DIR / "institution_registry.json")

    institutions = []
    for inst in registry:
        clusters = inst["all_clusters"]
        if not clusters:
            continue

        ops = []
        for c in clusters:
            ops.append({
                "date": c["date"],
                "direction": c["direction"],
                "amount_wan": float(c["amount_wan"]),
                "price_yuan": float(c["price_yuan"]),
                "n_orders": int(c["n_orders"]),
                "avg_id_gap": float(c["avg_id_gap"]),
                "qty_cv": float(c["qty_cv"]),
                "session": c["session"],
            })

        total_buy = sum(op["amount_wan"] for op in ops if op["direction"] == "BUY")
        total_sell = sum(op["amount_wan"] for op in ops if op["direction"] == "SELL")
        dates = sorted(set(op["date"] for op in ops))
        amounts = [op["amount_wan"] for op in ops]
        sessions = [op["session"] for op in ops]
        top_session = Counter(sessions).most_common(1)[0][0] if sessions else "?"

        institutions.append({
            "anon_id": inst["anon_id"],
            "operations": sorted(ops, key=lambda o: o["date"]),
            "n_operations": len(ops),
            "n_days": len(dates),
            "date_range": f"{min(dates)} - {max(dates)}",
            "total_buy_wan": round(total_buy, 0),
            "total_sell_wan": round(total_sell, 0),
            "net_wan": round(total_buy - total_sell, 0),
            "buy_pct": round(total_buy / max(total_buy + total_sell, 1) * 100, 1),
            "avg_amount_wan": round(float(np.mean(amounts)), 0),
            "median_amount_wan": round(float(np.median(amounts)), 0),
            "amount_cv": round(float(np.std(amounts) / np.mean(amounts)), 3) if np.mean(amounts) > 0 else 0,
            "fingerprint": {
                "top_session": top_session,
                "session_dist": {s: round(sessions.count(s) / len(sessions) * 100, 1)
                                 for s in sorted(set(sessions))},
                "avg_id_gap": round(float(np.median([op["avg_id_gap"] for op in ops])), 1),
                "avg_qty_cv": round(float(np.median([op["qty_cv"] for op in ops])), 3),
                "avg_price_yuan": round(float(np.median([op["price_yuan"] for op in ops])), 2),
            },
            "v4_size_label": inst.get("size_label", ""),
            "v4_behavior": inst.get("behavior", {}),
            "v4_fingerprint_summary": inst.get("fingerprint_summary", {}),
        })

    return institutions


# ═══════════════════════════════════════════════════
# 2. 机构间相似度 + 去重合并
# ═══════════════════════════════════════════════════

def institution_similarity(a: dict, b: dict) -> tuple[float, list[str]]:
    """
    计算两个机构的相似度，返回(满足条件数, 详情)。

    7个合并条件:
      1. 日期重叠率 ≥ 70%
      2. 买入金额相似度 ≥ 90% 或 卖出金额相似度 ≥ 90%
      3. 时段分布top session一致
      4. 净流向方向一致
      5. IDgap相似 ≥ 0.7
      6. CV相似 ≥ 0.7
      7. 操作频率相似 (n_ops比例 ≥ 0.5 且 n_days比例 ≥ 0.5)
    """
    conditions = 0
    details = []

    # 1. 日期重叠率
    dates_a = set(op["date"] for op in a["operations"])
    dates_b = set(op["date"] for op in b["operations"])
    union_dates = dates_a | dates_b
    if union_dates:
        overlap = len(dates_a & dates_b) / len(union_dates)
        if overlap >= 0.60:  # 略放宽, 允许部分重叠
            conditions += 1
            details.append(f"日期重叠={overlap:.1%}")

    # 2. 买卖金额相似度
    buy_sim = min(a["total_buy_wan"], b["total_buy_wan"]) / max(a["total_buy_wan"], b["total_buy_wan"], 1)
    sell_sim = min(a["total_sell_wan"], b["total_sell_wan"]) / max(a["total_sell_wan"], b["total_sell_wan"], 1)

    if buy_sim >= 0.85 or sell_sim >= 0.85:
        conditions += 1
        details.append(f"金额相似 buy={buy_sim:.2f} sell={sell_sim:.2f}")

    # 3. 时段一致
    if a["fingerprint"]["top_session"] == b["fingerprint"]["top_session"]:
        conditions += 1
        details.append(f"时段一致={a['fingerprint']['top_session']}")

    # 4. 净流向同向
    if (a["net_wan"] > 0) == (b["net_wan"] > 0):
        conditions += 1
        details.append(f"净流向同向")

    # 5. IDgap相似
    g1, g2 = a["fingerprint"]["avg_id_gap"], b["fingerprint"]["avg_id_gap"]
    if g1 > 0 and g2 > 0:
        gap_ratio = min(g1, g2) / max(g1, g2)
        if gap_ratio >= 0.65:
            conditions += 1
            details.append(f"IDgap相似={gap_ratio:.2f}")

    # 6. CV相似
    cv1, cv2 = a["fingerprint"]["avg_qty_cv"], b["fingerprint"]["avg_qty_cv"]
    if cv1 > 0 and cv2 > 0:
        cv_ratio = min(cv1, cv2) / max(cv1, cv2)
        if cv_ratio >= 0.65:
            conditions += 1
            details.append(f"CV相似={cv_ratio:.2f}")

    # 7. 操作频率相似
    ops_ratio = min(a["n_operations"], b["n_operations"]) / max(a["n_operations"], b["n_operations"])
    days_ratio = min(a["n_days"], b["n_days"]) / max(a["n_days"], b["n_days"])
    if ops_ratio >= 0.45 and days_ratio >= 0.45:
        conditions += 1
        details.append(f"操作频率相似 ops={ops_ratio:.2f} days={days_ratio:.2f}")

    return conditions, details


def dedup_institutions(institutions: list[dict]) -> tuple[list[dict], list[dict]]:
    """
    单轮配对合并: 每对判断相似度，满足≥5/7条件则合并。
    不链式传导: 合并后不再参与后续合并。
    """
    n = len(institutions)
    merged_set = set()  # 被合并的机构ID（不再出现在输出中）
    merge_groups = []   # [(keeper_idx, [merged_idx1, merged_idx2, ...])]

    # 按规模排序：大规模机构优先作为吸收方
    indices = sorted(range(n), key=lambda i: -institutions[i]["n_operations"])

    for i_idx in indices:
        if i_idx in merged_set:
            continue
        for j_idx in indices:
            if j_idx in merged_set or j_idx == i_idx:
                continue

            inst_a = institutions[i_idx]
            inst_b = institutions[j_idx]

            conditions, details = institution_similarity(inst_a, inst_b)

            if conditions >= 4:
                merged_set.add(j_idx)
                # 查找是否已有 i_idx 的合并组
                found = False
                for grp in merge_groups:
                    if grp[0] == i_idx:
                        grp[1].append(j_idx)
                        grp[2].append({
                            "merged": inst_b["anon_id"],
                            "into": inst_a["anon_id"],
                            "conditions": conditions,
                            "details": "; ".join(details),
                        })
                        found = True
                        break
                if not found:
                    merge_groups.append((i_idx, [j_idx], [{
                        "merged": inst_b["anon_id"],
                        "into": inst_a["anon_id"],
                        "conditions": conditions,
                        "details": "; ".join(details),
                    }]))

    # 构建合并后的机构列表
    # 未被合并且也不是吸收方的机构保持不变
    keeper_indices = [i for i in range(n) if i not in merged_set]

    merged_insts = []
    merge_log = []

    for i in keeper_indices:
        inst = institutions[i]
        grp = next((g for g in merge_groups if g[0] == i), None)

        if grp is None:
            # 未被合并的独立机构
            merged_insts.append(inst)
        else:
            # 吸收了其他机构的机构
            merged_indices = grp[1]
            merge_log.extend(grp[2])

            # 合并操作记录
            all_ops = list(inst["operations"])
            for j in merged_indices:
                all_ops.extend(institutions[j]["operations"])

            # 去重（同日同方向同金额的可能是重复记录）
            seen = set()
            deduped_ops = []
            for op in sorted(all_ops, key=lambda o: o["date"]):
                key = (op["date"], op["direction"], op["amount_wan"])
                if key not in seen:
                    seen.add(key)
                    deduped_ops.append(op)

            all_ops = sorted(deduped_ops, key=lambda o: o["date"])

            # 重算统计量
            total_buy = sum(op["amount_wan"] for op in all_ops if op["direction"] == "BUY")
            total_sell = sum(op["amount_wan"] for op in all_ops if op["direction"] == "SELL")
            dates = sorted(set(op["date"] for op in all_ops))
            amounts = [op["amount_wan"] for op in all_ops]
            sessions = [op["session"] for op in all_ops]
            top_session = Counter(sessions).most_common(1)[0][0] if sessions else "?"

            merged_id = inst["anon_id"]  # 保留原ID
            merged_name = "+".join([inst["anon_id"]] +
                                   [institutions[j]["anon_id"] for j in merged_indices])

            merged_insts.append({
                "anon_id": merged_id,
                "merged_from": merged_name,
                "operations": all_ops,
                "n_operations": len(all_ops),
                "n_days": len(dates),
                "date_range": f"{min(dates)} - {max(dates)}",
                "total_buy_wan": round(total_buy, 0),
                "total_sell_wan": round(total_sell, 0),
                "net_wan": round(total_buy - total_sell, 0),
                "buy_pct": round(total_buy / max(total_buy + total_sell, 1) * 100, 1),
                "avg_amount_wan": round(float(np.mean(amounts)), 0),
                "median_amount_wan": round(float(np.median(amounts)), 0),
                "amount_cv": round(float(np.std(amounts) / np.mean(amounts)), 3) if np.mean(amounts) > 0 else 0,
                "fingerprint": {
                    "top_session": top_session,
                    "session_dist": {s: round(sessions.count(s) / len(sessions) * 100, 1)
                                     for s in sorted(set(sessions))},
                    "avg_id_gap": round(float(np.median([op["avg_id_gap"] for op in all_ops])), 1),
                    "avg_qty_cv": round(float(np.median([op["qty_cv"] for op in all_ops])), 3),
                    "avg_price_yuan": round(float(np.median([op["price_yuan"] for op in all_ops])), 2),
                },
                "v4_size_label": inst["v4_size_label"],
                "v4_behavior": inst["v4_behavior"],
                "v4_fingerprint_summary": inst["v4_fingerprint_summary"],
            })

    # 按净流向绝对值排序
    merged_insts.sort(key=lambda x: abs(x["net_wan"]), reverse=True)

    # 重新编号
    for idx, inst in enumerate(merged_insts):
        inst["anon_id"] = f"ANON-{idx+1:03d}"

    # 更新merge_log中的ANON ID映射
    old_to_new = {}
    for inst in merged_insts:
        if "merged_from" in inst:
            for old_id in inst["merged_from"].split("+"):
                old_to_new[old_id.strip()] = inst["anon_id"]
        else:
            old_to_new[inst["anon_id"]] = inst["anon_id"]
    # ...实际上merged_insts中的独立机构也换了ID，先不处理映射

    return merged_insts, merge_log


# ═══════════════════════════════════════════════════
# 3. 持仓曲线
# ═══════════════════════════════════════════════════

def build_position_curves(institutions: list[dict]) -> pd.DataFrame:
    """为每个机构构建每日累计净持仓曲线。"""
    rows = []

    for inst in institutions:
        ops = sorted(inst["operations"], key=lambda o: o["date"])
        cum = 0.0
        for op in ops:
            cum += op["amount_wan"] if op["direction"] == "BUY" else -op["amount_wan"]
            rows.append({
                "anon_id": inst["anon_id"],
                "date": op["date"],
                "direction": op["direction"],
                "amount_wan": op["amount_wan"],
                "price_yuan": op["price_yuan"],
                "cum_position_wan": round(cum, 0),
                "n_orders": op["n_orders"],
                "session": op["session"],
            })

    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════
# 4. 新置信度评分 (5因子)
# ═══════════════════════════════════════════════════

def compute_confidence(inst: dict) -> tuple[str, float, dict]:
    """
    5因子置信度评分:
      continuity  (0.25): 操作次数+活跃天数+时间跨度+密度
      amount      (0.20): 总成交+净流向规模+单笔中位数
      position    (0.20): 持仓曲线趋势性
      style       (0.15): 指纹内部一致性
      crossday    (0.10): 跨日出现规律性
      direction   (0.10): 方向性强度
    """
    ops = inst["operations"]
    n_ops = len(ops)
    dates = sorted(set(op["date"] for op in ops))
    n_days = len(dates)
    amounts = [op["amount_wan"] for op in ops]

    total_buy = inst["total_buy_wan"]
    total_sell = inst["total_sell_wan"]
    gross = total_buy + total_sell
    net = total_buy - total_sell

    # 1. 连续性 (0-1)
    if n_days > 1:
        date_span = (pd.to_datetime(max(dates)) - pd.to_datetime(min(dates))).days
    else:
        date_span = 1

    continuity = min(1.0, (
        0.30 * min(n_ops / 80, 1.0) +
        0.30 * min(n_days / 40, 1.0) +
        0.20 * min(date_span / 200, 1.0) +
        0.20 * min(n_days / max(date_span, 1) * 30, 1.0)  # 密度: 至少每30天出现一次
    ))

    # 2. 金额规模 (0-1)
    amount_score = min(1.0, (
        0.35 * min(gross / 80000, 1.0) +
        0.35 * min(abs(net) / 20000, 1.0) +
        0.30 * min(float(np.median(amounts)) / 500, 1.0)
    ))

    # 3. 持仓曲线逻辑性 (0-1)
    cum = 0.0
    cum_series = []
    for op in sorted(ops, key=lambda o: o["date"]):
        cum += op["amount_wan"] if op["direction"] == "BUY" else -op["amount_wan"]
        cum_series.append(cum)

    if len(cum_series) > 5:
        cum_arr = np.array(cum_series)
        dominant_sign = 1 if net > 0 else -1
        moves = np.diff(cum_arr)
        if len(moves) > 0:
            aligned_moves = np.sum(np.sign(moves) == np.sign(dominant_sign))
            monotonicity = aligned_moves / len(moves)
        else:
            monotonicity = 0.5

        total_swing = float(np.sum(np.abs(moves)))
        net_change = float(abs(cum_arr[-1] - cum_arr[0]))
        trend_strength = net_change / max(total_swing, 1)

        position_score = 0.5 * monotonicity + 0.5 * trend_strength
    else:
        position_score = 0.5

    # 4. 风格一致性 (0-1)
    qty_cvs = [op["qty_cv"] for op in ops]
    id_gaps = [op["avg_id_gap"] for op in ops]
    sessions = [op["session"] for op in ops]

    if len(qty_cvs) > 1:
        cv_cv = float(np.std(qty_cvs) / max(np.mean(qty_cvs), 0.001))
        gap_cv = float(np.std(id_gaps) / max(np.mean(id_gaps), 0.001))
        session_conc = max(Counter(sessions).values()) / len(sessions)

        style_score = (
            0.35 * max(0, 1 - min(cv_cv, 3)) +
            0.35 * max(0, 1 - min(gap_cv, 3)) +
            0.30 * session_conc
        )
    else:
        style_score = 0.5

    # 5. 跨日稳定性 (0-1)
    if n_days > 3:
        date_dt = sorted([pd.to_datetime(d) for d in dates])
        gaps = [(date_dt[i+1] - date_dt[i]).days for i in range(len(date_dt)-1)]
        if np.mean(gaps) > 0:
            gap_cv_val = float(np.std(gaps) / np.mean(gaps))
            crossday_score = max(0, 1 - min(gap_cv_val, 2))
        else:
            crossday_score = 0.5
    else:
        crossday_score = 0.5

    # 6. 方向性强度
    if gross > 0:
        direction_strength = abs(net) / gross
    else:
        direction_strength = 0

    # 加权总分
    total = (
        0.25 * continuity +
        0.20 * amount_score +
        0.20 * position_score +
        0.15 * style_score +
        0.10 * crossday_score +
        0.10 * direction_strength
    )

    if total >= 0.60:
        level = "HIGH"
    elif total >= 0.35:
        level = "MEDIUM"
    else:
        level = "LOW"

    details = {
        "continuity": round(continuity, 3),
        "amount": round(amount_score, 3),
        "position": round(position_score, 3),
        "style": round(style_score, 3),
        "crossday": round(crossday_score, 3),
        "direction_strength": round(direction_strength, 3),
        "total": round(total, 3),
    }

    return level, total, details


# ═══════════════════════════════════════════════════
# 5. 行为类型分类
# ═══════════════════════════════════════════════════

def classify_behavior_type(inst: dict) -> str:
    """7类行为标签。"""
    ops = inst["operations"]
    n_ops = len(ops)
    dates = sorted(set(op["date"] for op in ops))
    n_days = len(dates)

    total_buy = inst["total_buy_wan"]
    total_sell = inst["total_sell_wan"]
    gross = total_buy + total_sell
    net = total_buy - total_sell
    buy_pct = inst["buy_pct"]

    if n_days > 1:
        date_span = (pd.to_datetime(max(dates)) - pd.to_datetime(min(dates))).days
    else:
        date_span = 1

    # 短期突击型: ≤5天但金额≥5000万
    if n_days <= 5 and gross >= 5000:
        return "短期突击型"

    # 纯买建仓型: ≥85%买入
    if buy_pct >= 85 and net > 0:
        if date_span >= 90 and n_days >= 30:
            return "长期纯买建仓型"
        return "纯买建仓型"

    # 纯卖出货型: ≥85%卖出
    if buy_pct <= 15 and net < 0:
        if date_span >= 90 and n_days >= 30:
            return "长期纯卖出货型"
        return "纯卖出货型"

    # 净买调仓型: 55-85%买入
    if net > 0 and buy_pct >= 55:
        if n_days >= 30:
            return "长期净买调仓型"
        return "净买调仓型"

    # 净卖出货型: 15-45%买入
    if net < 0 and buy_pct <= 45:
        if n_days >= 30:
            return "长期净卖出货型"
        return "净卖出货型"

    # 波段交易型: 方向频繁切换
    ops_sorted = sorted(ops, key=lambda o: o["date"])
    dirs = [op["direction"] for op in ops_sorted]
    switches = sum(1 for i in range(1, len(dirs)) if dirs[i] != dirs[i-1])
    switch_rate = switches / max(len(dirs), 1)
    if switch_rate >= 0.25:
        return "波段交易型"

    # 长期维护型: 持续出现但净流向很小
    if n_days >= 30 and abs(net) / max(gross, 1) < 0.15:
        return "长期维护/做市型"

    # 低活跃型
    if n_ops < 10:
        return "低活跃型"

    return "双向调仓型"


# ═══════════════════════════════════════════════════
# 6. 报告生成
# ═══════════════════════════════════════════════════

def generate_report(institutions: list[dict], merge_log: list[dict]) -> Path:
    lines = [
        f"# {STOCK} SOFIA v6 跨日机构追踪报告",
        "",
        "## 方法论",
        "",
        "- **骨架**: 保留v4的20个机构簇分配，不做重新映射",
        "- **去重**: 单轮配对合并，≥4/7条件，无链式传导",
        "- **置信度**: 5因子加权（连续性0.25 + 金额0.20 + 持仓曲线0.20 + 风格0.15 + 跨日0.10 + 方向0.10）",
        "- **行为标签**: 7类（纯买建仓/净买调仓/净卖出货/波段交易/短期突击/长期维护/低活跃）",
        "",
        f"## 概览",
        "",
        f"- 输入机构: 20 (v4注册表)",
        f"- 合并事件: {len(merge_log)}",
        f"- 最终机构: {len(institutions)}",
        "",
    ]

    if merge_log:
        lines.extend([
            "## 合并日志",
            "",
            "| 被合并 | 并入 | 条件 | 详情 |",
            "|--------|------|------|------|",
        ])
        for m in merge_log:
            lines.append(f"| {m['merged']} | {m['into']} | {m['conditions']}/7 | {m['details']} |")
        lines.append("")

    lines.extend([
        "## 机构总览",
        "",
        "| 机构 | 置信度 | 行为类型 | 操作数 | 天数 | 买入(万) | 卖出(万) | 净(万) | 时段 | IDgap | 日期范围 |",
        "|------|--------|---------|--------|------|---------|---------|--------|------|-------|---------|",
    ])

    for inst in institutions:
        fp = inst["fingerprint"]
        conf_emoji = {"HIGH": "🟢", "MEDIUM": "🟡", "LOW": "🔴"}.get(inst["confidence"], "⚪")
        merged_note = f" (←{inst['merged_from']})" if "merged_from" in inst else ""
        lines.append(
            f"| {inst['anon_id']}{merged_note} | {conf_emoji}{inst['confidence']} "
            f"({inst['confidence_score']:.2f}) | {inst['behavior_type']} | "
            f"{inst['n_operations']} | {inst['n_days']} | "
            f"{inst['total_buy_wan']:.0f} | {inst['total_sell_wan']:.0f} | "
            f"{inst['net_wan']:+.0f} | {fp['top_session']} | {fp['avg_id_gap']:.0f} | "
            f"{inst['date_range']} |"
        )

    # 逐机构详解
    for inst in institutions:
        fp = inst["fingerprint"]
        cd = inst["confidence_details"]
        ops = sorted(inst["operations"], key=lambda o: o["date"])

        lines.extend([
            "",
            f"### {inst['anon_id']} [{inst['confidence']}] — {inst['behavior_type']}",
            "",
        ])

        if "merged_from" in inst:
            lines.append(f"- **合并来源**: {inst['merged_from']}")

        lines.extend([
            f"- **总买入**: {inst['total_buy_wan']:.0f}万 | **总卖出**: {inst['total_sell_wan']:.0f}万 | "
            f"**净**: {inst['net_wan']:+.0f}万 | **买入占比**: {inst['buy_pct']:.1f}%",
            f"- **操作次数**: {inst['n_operations']} | **覆盖天数**: {inst['n_days']} | "
            f"**日期范围**: {inst['date_range']}",
            f"- **平均每笔**: {inst['avg_amount_wan']:.0f}万 | **中位数**: {inst['median_amount_wan']:.0f}万 | "
            f"**规模CV**: {inst['amount_cv']:.3f}",
            f"- **时段偏好**: {fp['top_session']} | **时段分布**: {fp['session_dist']}",
            f"- **IDgap均值**: {fp['avg_id_gap']:.0f} | **CV均值**: {fp['avg_qty_cv']:.3f}",
            "",
            "#### 置信度分解",
            "| 连续性 | 金额 | 持仓曲线 | 风格 | 跨日 | 方向强度 | **总分** |",
            "|--------|------|---------|------|------|---------|--------|",
            f"| {cd['continuity']:.3f} | {cd['amount']:.3f} | {cd['position']:.3f} | "
            f"{cd['style']:.3f} | {cd['crossday']:.3f} | {cd['direction_strength']:.3f} | "
            f"**{cd['total']:.3f}** |",
        ])

        # 持仓曲线
        lines.extend([
            "",
            "#### 持仓变化曲线（关键节点）",
            "| 日期 | 方向 | 金额(万) | 价格 | 累计净持仓(万) | 阶段 |",
            "|------|------|---------|------|--------------|------|",
        ])

        cum = 0.0
        step = max(1, len(ops) // 20)
        for i, op in enumerate(ops):
            prev_cum = cum
            cum += op["amount_wan"] if op["direction"] == "BUY" else -op["amount_wan"]

            prev_dir = ops[i-1]["direction"] if i > 0 else None
            is_key = (
                i == 0 or i == len(ops) - 1 or
                op["amount_wan"] >= inst["avg_amount_wan"] * 2 or  # 大单
                (prev_dir and op["direction"] != prev_dir)  # 方向切换
            )

            if is_key or i % step == 0:
                if cum > prev_cum and op["direction"] == "BUY":
                    stage = "建仓"
                elif cum < prev_cum and op["direction"] == "SELL":
                    stage = "减仓"
                elif op["direction"] == "SELL":
                    stage = "了结"
                elif op["direction"] == "BUY":
                    stage = "回补"
                else:
                    stage = "—"

                dir_sym = "买" if op["direction"] == "BUY" else "卖"
                lines.append(
                    f"| {op['date']} | {dir_sym} | {op['amount_wan']:.0f} | "
                    f"{op['price_yuan']:.2f} | {cum:+.0f} | {stage} |"
                )

        # 月度节奏
        if len(ops) >= 10:
            df_ops = pd.DataFrame(ops)
            df_ops["month"] = df_ops["date"].str[:6]
            monthly = df_ops.groupby("month").agg(
                buy_wan=("amount_wan", lambda x: x[df_ops.loc[x.index, "direction"] == "BUY"].sum()),
                sell_wan=("amount_wan", lambda x: x[df_ops.loc[x.index, "direction"] == "SELL"].sum()),
                ops=("date", "count"),
            ).reset_index()
            monthly["net"] = monthly["buy_wan"] - monthly["sell_wan"]

            lines.extend([
                "",
                "#### 月度买卖节奏",
                "| 月份 | 买入(万) | 卖出(万) | 净(万) | 操作数 | 节奏 |",
                "|------|---------|---------|--------|--------|------|",
            ])
            for _, m in monthly.iterrows():
                rhythm = ("←建仓" if m["net"] > 1000 else
                         "出货→" if m["net"] < -1000 else "调整")
                lines.append(
                    f"| {m['month']} | {m['buy_wan']:.0f} | {m['sell_wan']:.0f} | "
                    f"{m['net']:+.0f} | {int(m['ops'])} | {rhythm} |"
                )

    # 跨机构对比
    lines.extend([
        "",
        "## 跨机构对比",
        "",
        "### 买方阵营 (净买入Top 5)",
        "",
    ])
    buyers = sorted(institutions, key=lambda x: x["net_wan"], reverse=True)[:5]
    for inst in buyers:
        lines.append(
            f"- **{inst['anon_id']}** [{inst['confidence']}] {inst['behavior_type']}: "
            f"净+{inst['net_wan']:.0f}万, {inst['n_operations']}次/{inst['n_days']}天, "
            f"时段={inst['fingerprint']['top_session']}, 置信度分={inst['confidence_score']:.2f}"
        )

    lines.extend([
        "",
        "### 卖方阵营 (净卖出Top 5)",
        "",
    ])
    sellers = sorted(institutions, key=lambda x: x["net_wan"])[:5]
    for inst in sellers:
        lines.append(
            f"- **{inst['anon_id']}** [{inst['confidence']}] {inst['behavior_type']}: "
            f"净{inst['net_wan']:.0f}万, {inst['n_operations']}次/{inst['n_days']}天, "
            f"时段={inst['fingerprint']['top_session']}, 置信度分={inst['confidence_score']:.2f}"
        )

    lines.extend([
        "",
        "### 时段-方向矩阵",
        "",
        "| 时段 | 主要买方 | 主要卖方 |",
        "|------|---------|---------|",
    ])
    for sess in ["AUCTION", "OPEN", "MORNING", "LATE_MORNING", "EARLY_AFTER", "AFTERNOON", "CLOSE"]:
        sess_buyers = [i for i in institutions
                       if i["fingerprint"]["top_session"] == sess and i["net_wan"] > 0]
        sess_sellers = [i for i in institutions
                        if i["fingerprint"]["top_session"] == sess and i["net_wan"] < 0]
        buyers_str = ", ".join(f"{i['anon_id']}" for i in sess_buyers[:3]) or "—"
        sellers_str = ", ".join(f"{i['anon_id']}" for i in sess_sellers[:3]) or "—"
        lines.append(f"| {sess} | {buyers_str} | {sellers_str} |")

    report_path = V6_DIR / "v6_report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


# ═══════════════════════════════════════════════════
# 7. 主流程
# ═══════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="SOFIA v6 enhanced — v4 institution merge/profile layer")
    parser.add_argument("--stock", default=STOCK, help="股票代码, 例如 002516/301529/300100")
    args = parser.parse_args()
    configure_stock(args.stock)

    print("SOFIA v6 — v4骨架 + v5增强")
    print(f"股票: {STOCK}")
    print()

    # 加载v4机构
    print("加载v4注册表...")
    v4_insts = load_v4_institutions()
    print(f"  {len(v4_insts)} 个机构")
    for inst in v4_insts[:3]:
        print(f"  {inst['anon_id']}: {inst['n_operations']}次/{inst['n_days']}天 "
              f"买{inst['total_buy_wan']:.0f}/卖{inst['total_sell_wan']:.0f}/净{inst['net_wan']:+.0f}万")

    # 去重合并
    print("\n机构间去重分析...")
    # 先看所有对的相似度
    all_pairs = []
    for i in range(len(v4_insts)):
        for j in range(i + 1, len(v4_insts)):
            conds, details = institution_similarity(v4_insts[i], v4_insts[j])
            all_pairs.append((v4_insts[i]["anon_id"], v4_insts[j]["anon_id"],
                            conds, details))

    # 显示高相似度对 (≥3条件)
    high_sim = [p for p in all_pairs if p[2] >= 3]
    high_sim.sort(key=lambda p: -p[2])
    print(f"  高相似度对 (≥3/7): {len(high_sim)}")
    for p in high_sim[:15]:
        print(f"    {p[0]} ↔ {p[1]}: {p[2]}/7 — {p[3][0] if p[3] else ''}")

    # 执行合并 (≥4条件, 非链式)
    print(f"\n执行合并 (≥4/7条件, 单轮无链式)...")
    merged_insts, merge_log = dedup_institutions(v4_insts)

    if merge_log:
        print(f"  合并: {len(merge_log)} 对")
        for m in merge_log:
            print(f"    {m['merged']} → {m['into']} ({m['conditions']}/7)")
    else:
        print(f"  无需合并 (无机构满足≥4/7条件)")

    # 置信度评分 + 行为分类
    print("\n计算置信度 + 行为类型...")
    for inst in merged_insts:
        level, score, details = compute_confidence(inst)
        inst["confidence"] = level
        inst["confidence_score"] = round(score, 3)
        inst["confidence_details"] = details
        inst["behavior_type"] = classify_behavior_type(inst)

    # 持仓曲线
    print("构建持仓曲线...")
    positions = build_position_curves(merged_insts)
    positions.to_csv(V6_DIR / "position_curves.csv", index=False)

    # 保存注册表
    print("保存输出...")
    registry_out = []
    for inst in merged_insts:
        entry = {k: v for k, v in inst.items()
                 if k not in ("operations", "v4_behavior", "v4_fingerprint_summary")}
        entry["operations"] = inst["operations"]
        registry_out.append(entry)
    with open(V6_DIR / "institution_registry.json", "w", encoding="utf-8") as f:
        json.dump(registry_out, f, ensure_ascii=False, indent=2)

    # 汇总CSV
    summary_rows = []
    for inst in merged_insts:
        fp = inst["fingerprint"]
        cd = inst["confidence_details"]
        row = {
            "anon_id": inst["anon_id"],
            "confidence": inst["confidence"],
            "confidence_score": inst["confidence_score"],
            "behavior_type": inst["behavior_type"],
            "n_ops": inst["n_operations"],
            "n_days": inst["n_days"],
            "buy_wan": inst["total_buy_wan"],
            "sell_wan": inst["total_sell_wan"],
            "net_wan": inst["net_wan"],
            "buy_pct": inst["buy_pct"],
            "avg_amount": inst["avg_amount_wan"],
            "amount_cv": inst["amount_cv"],
            "top_session": fp["top_session"],
            "avg_id_gap": fp["avg_id_gap"],
            "avg_qty_cv": fp["avg_qty_cv"],
            "date_range": inst["date_range"],
        }
        if "merged_from" in inst:
            row["merged_from"] = inst["merged_from"]
        for k, v in cd.items():
            row[f"cont_{k}"] = v
        summary_rows.append(row)
    pd.DataFrame(summary_rows).to_csv(V6_DIR / "institution_registry.csv", index=False)

    # 合并日志
    if merge_log:
        pd.DataFrame(merge_log).to_csv(V6_DIR / "merge_log.csv", index=False)

    # 报告
    print("生成报告...")
    report_path = generate_report(merged_insts, merge_log)

    # 终端输出
    print(f"\n{'='*70}")
    print(f"v6 完成 — {len(merged_insts)} 个机构 (v4原始20个 → 合并后{len(merged_insts)}个)")
    print(f"{'='*70}")

    print(f"\n置信度分布:")
    for level in ["HIGH", "MEDIUM", "LOW"]:
        count = sum(1 for i in merged_insts if i["confidence"] == level)
        names = [i["anon_id"] for i in merged_insts if i["confidence"] == level]
        print(f"  {level}: {count} {names}")

    print(f"\n行为类型分布:")
    type_counts = Counter(i["behavior_type"] for i in merged_insts)
    for bt, cnt in type_counts.most_common():
        names = [i["anon_id"] for i in merged_insts if i["behavior_type"] == bt]
        print(f"  {bt}: {cnt} {names}")

    print(f"\n{'机构':<10} {'合并自':<25} {'置信度':<8} {'行为类型':<16} "
          f"{'操作':<6} {'净(万)':<12} {'买入%':<7} {'时段':<14} {'日期范围'}")
    print("-" * 110)
    for inst in merged_insts:
        merged = inst.get("merged_from", "")[:24]
        print(f"{inst['anon_id']:<10} {merged:<25} {inst['confidence']:<8} "
              f"{inst['behavior_type']:<16} {inst['n_operations']:<6} "
              f"{inst['net_wan']:<+12.0f} {inst['buy_pct']:<6.1f}% "
              f"{inst['fingerprint']['top_session']:<14} {inst['date_range']}")

    print(f"\n输出: {V6_DIR}/")
    print(f"  institution_registry.json  — 机构注册表 ({len(merged_insts)}个)")
    print(f"  institution_registry.csv   — 汇总CSV")
    print(f"  position_curves.csv        — 每日持仓曲线")
    print(f"  v6_report.md               — 人读报告")


if __name__ == "__main__":
    main()
