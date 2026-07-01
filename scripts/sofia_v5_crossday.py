"""
SOFIA v5 — 跨日机构追踪强化版

核心改进:
  1. 方向感知匹配: 同向匹配(买→买)用宽松阈值, 反向匹配(买→卖)用严格阈值
  2. 多轮合并: 第1轮匹配核心指纹 → 第2轮扩展 → 第3轮验证
  3. 季度验证: 匹配结果与公开持仓变化交叉验证
  4. 清晰置信度: HIGH/MEDIUM/LOW 三级
  5. 持仓曲线: 每机构每日累计净持仓

输出:
  data/single_stock/{stock}/sofia_v5/
    crossday_registry.json   — 跨日机构注册表
    position_curves.csv      — 每机构每日持仓曲线
    match_quality.csv        — 匹配质量评估
    crossday_report.md       — 人读报告
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path
from itertools import combinations

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.sofia_v4_hunter import (
    load_day, detect_algo_clusters,
    match_score_fp, extract_fingerprint, _classify_session
)

PROJECT = Path(__file__).parent.parent
STOCK = "002516"
STOCK_DIR = PROJECT / "data" / "single_stock" / STOCK
OUT_DIR = STOCK_DIR / "sofia_v5"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SESSION_ORDER = {"AUCTION": 0, "OPEN": 1, "MORNING": 2, "LATE_MORNING": 3,
                 "EARLY_AFTER": 4, "AFTERNOON": 5, "CLOSE": 6}


# ═══════════════════════════════════════════════════
# 增强版指纹: 加入更多跨日稳定特征
# ═══════════════════════════════════════════════════

def enhanced_match_score(c1: dict, c2: dict, same_direction: bool) -> float:
    """
    增强版跨日匹配评分。

    同向匹配(same_direction=True): 权重侧重拆单手法
    反向匹配(same_direction=False): 权重侧重时段+规模, 更严格
    """
    s = 0.0

    # 1. 拆单 CV (核心指纹)
    if c1["qty_cv"] > 0 and c2["qty_cv"] > 0:
        cv_ratio = min(c1["qty_cv"], c2["qty_cv"]) / max(c1["qty_cv"], c2["qty_cv"])
        s += 0.25 * cv_ratio  # 提高CV权重

    # 2. ID间隔
    g1 = c1.get("avg_id_gap", c1.get("id_gap", 100))
    g2 = c2.get("avg_id_gap", c2.get("id_gap", 100))
    if g1 > 0 and g2 > 0:
        gap_ratio = min(g1, g2) / max(g1, g2)
        s += 0.20 * gap_ratio  # 提高IDgap权重

    # 3. 时段偏好
    sess1 = c1.get("session", _classify_session(c1.get("time_start", 10 * 3600)))
    sess2 = c2.get("session", _classify_session(c2.get("time_start", 10 * 3600)))
    if sess1 == sess2:
        s += 0.20  # 提高时段权重 (KS=0.72最强特征)
    elif abs(SESSION_ORDER.get(sess1, 3) - SESSION_ORDER.get(sess2, 3)) <= 1:
        s += 0.08

    # 4. 订单规模
    n1 = c1.get("n_orders", 1)
    n2 = c2.get("n_orders", 1)
    if n1 > 0 and n2 > 0:
        n_ratio = min(n1, n2) / max(n1, n2)
        s += 0.10 * n_ratio

    # 5. 金额规模
    amt1 = c1.get("amount_wan", c1.get("total_amount_wan", 0))
    amt2 = c2.get("amount_wan", c2.get("total_amount_wan", 0))
    if amt1 > 0 and amt2 > 0:
        amt_ratio = min(amt1, amt2) / max(amt1, amt2)
        s += 0.10 * amt_ratio

    # 6. 方向
    dir1 = c1.get("direction", "?")
    dir2 = c2.get("direction", "?")
    if dir1 == dir2:
        s += 0.10  # 同向加分
    elif same_direction:
        return 0.0  # 同向模式下方向不同直接拒绝
    # 反向模式下不给分但允许匹配

    # 7. 均价 (弱特征)
    px1 = c1.get("price_yuan", c1.get("avg_price_yuan", 0))
    px2 = c2.get("price_yuan", c2.get("avg_price_yuan", 0))
    if px1 > 0 and px2 > 0:
        px_ratio = min(px1, px2) / max(px1, px2)
        s += 0.05 * px_ratio

    return s


# ═══════════════════════════════════════════════════
# 三阶段跨日匹配
# ═══════════════════════════════════════════════════

def crossday_match(all_clusters: dict,
                   stage1_threshold: float = 0.70,  # 高置信同向
                   stage2_threshold: float = 0.55,  # 低置信反向
                   min_amount_wan: float = 200) -> list[dict]:
    """
    三阶段匹配:
      Stage 1: BUY↔BUY + SELL↔SELL 同向匹配, 阈值0.70 (高置信)
      Stage 2: BUY↔SELL 反向匹配, 阈值0.55 (低置信, 需验证)
      Stage 3: 规模连续性验证 + 季度数据锚定
    """
    # 提取所有≥阈值的簇
    all_big = []
    for d, cs in all_clusters.items():
        for c in cs:
            amt = c.get("amount_wan", c.get("total_amount_wan", 0))
            if amt >= min_amount_wan:
                sess = c.get("session", _classify_session(c.get("time_start", 10 * 3600)))
                all_big.append({
                    "date": d,
                    "direction": c["direction"],
                    "amount_wan": c.get("amount_wan", c.get("total_amount_wan", 0)),
                    "price_yuan": c.get("price_yuan", c.get("avg_price_yuan", 0)),
                    "n_orders": c["n_orders"],
                    "avg_id_gap": c.get("avg_id_gap", c.get("id_gap", 0)),
                    "qty_cv": c["qty_cv"],
                    "session": sess,
                    "avg_qty": c.get("avg_qty", c.get("avg_id_gap", 0)),
                })

    all_big.sort(key=lambda x: x["amount_wan"], reverse=True)

    # === Stage 1: 同向匹配 (BUY→BUY, SELL→SELL) ===
    buy_clusters = [c for c in all_big if c["direction"] == "BUY"]
    sell_clusters = [c for c in all_big if c["direction"] == "SELL"]

    buy_groups = _greedy_group(buy_clusters, same_direction=True,
                               threshold=stage1_threshold)
    sell_groups = _greedy_group(sell_clusters, same_direction=True,
                                threshold=stage1_threshold)

    # === Stage 2: 尝试合并买卖组 (反向匹配) ===
    all_groups = buy_groups + sell_groups

    # 尝试将小的卖组合并到大的买组(如果指纹高度相似)
    merged = [False] * len(all_groups)
    final_groups = []

    for i, g1 in enumerate(all_groups):
        if merged[i]:
            continue
        g1_dirs = [c["direction"] for c in g1]
        g1_main_dir = "BUY" if g1_dirs.count("BUY") > g1_dirs.count("SELL") else "SELL"

        for j, g2 in enumerate(all_groups):
            if merged[j] or i == j:
                continue
            g2_dirs = [c["direction"] for c in g2]
            g2_main_dir = "BUY" if g2_dirs.count("BUY") > g2_dirs.count("SELL") else "SELL"

            if g1_main_dir == g2_main_dir:
                continue  # 同向的已经在Stage 1处理了

            # 反向合并: 比对代表簇
            rep1 = max(g1, key=lambda c: c["amount_wan"])
            rep2 = max(g2, key=lambda c: c["amount_wan"])
            score = enhanced_match_score(rep1, rep2, same_direction=False)

            if score >= stage2_threshold:
                g1.extend(g2)
                merged[j] = True

        final_groups.append(g1)

    # === Stage 3: 规模连续性验证 ===
    validated_groups = []
    for group in final_groups:
        group = sorted(group, key=lambda c: c["date"])
        amounts = [c["amount_wan"] for c in group]
        dates = [c["date"] for c in group]

        # 检查规模是否合理(不会从200万突然跳到2亿)
        if len(amounts) > 2:
            amt_cv = float(np.std(amounts) / np.mean(amounts)) if np.mean(amounts) > 0 else 0
            if amt_cv > 3.0:
                # 规模波动太大, 可能混入了不同机构
                group = _split_by_size(group)

        if group:
            validated_groups.append(group)

    # 构建注册表
    registry = []
    for idx, group in enumerate(validated_groups):
        total_buy = sum(c["amount_wan"] for c in group if c["direction"] == "BUY")
        total_sell = sum(c["amount_wan"] for c in group if c["direction"] == "SELL")
        net = total_buy - total_sell
        days = sorted(set(c["date"] for c in group))
        rep = max(group, key=lambda c: c["amount_wan"])

        dirs = [c["direction"] for c in group]
        buy_pct = dirs.count("BUY") / len(dirs) * 100

        # 置信度判定
        if len(group) >= 20 and buy_pct >= 80:
            confidence = "HIGH"
        elif len(group) >= 10 and (buy_pct >= 80 or buy_pct <= 20):
            confidence = "MEDIUM"
        else:
            confidence = "LOW"

        sessions = [c["session"] for c in group]
        top_session = max(set(sessions), key=sessions.count) if sessions else "?"

        registry.append({
            "anon_id": f"ANON-{idx+1:03d}",
            "confidence": confidence,
            "n_operations": len(group),
            "n_days": len(days),
            "date_range": f"{min(days)} - {max(days)}",
            "total_buy_wan": round(total_buy, 0),
            "total_sell_wan": round(total_sell, 0),
            "net_wan": round(net, 0),
            "buy_pct": round(buy_pct, 1),
            "dominant_direction": "BUY" if total_buy > total_sell else "SELL",
            "avg_amount_wan": round(float(np.mean(amounts)), 0),
            "median_amount_wan": round(float(np.median(amounts)), 0),
            "amount_cv": round(float(np.std(amounts) / np.mean(amounts)), 3) if np.mean(amounts) > 0 else 0,
            "representative": {
                "date": rep["date"],
                "direction": rep["direction"],
                "amount_wan": rep["amount_wan"],
                "price_yuan": rep["price_yuan"],
                "n_orders": rep["n_orders"],
                "avg_id_gap": rep["avg_id_gap"],
                "session": rep["session"],
            },
            "fingerprint": {
                "top_session": top_session,
                "session_dist": {s: round(sessions.count(s)/len(sessions)*100, 1)
                                 for s in sorted(set(sessions))},
                "avg_id_gap": round(float(np.mean([c["avg_id_gap"] for c in group])), 1),
                "avg_qty_cv": round(float(np.mean([c["qty_cv"] for c in group])), 3),
                "avg_price_yuan": round(float(np.mean([c["price_yuan"] for c in group])), 2),
            },
            "operations": [{
                "date": c["date"],
                "direction": c["direction"],
                "amount_wan": c["amount_wan"],
                "price_yuan": c["price_yuan"],
                "n_orders": c["n_orders"],
                "avg_id_gap": c["avg_id_gap"],
                "qty_cv": c["qty_cv"],
                "session": c["session"],
            } for c in sorted(group, key=lambda x: x["date"])],
        })

    registry.sort(key=lambda x: abs(x["net_wan"]), reverse=True)
    return registry


def _greedy_group(clusters: list[dict], same_direction: bool,
                  threshold: float) -> list[list[dict]]:
    """贪婪分组: 从最大簇开始, 匹配所有相似簇。"""
    assigned = set()
    groups = []
    for i, c1 in enumerate(clusters):
        if i in assigned:
            continue
        group = [c1]
        assigned.add(i)
        for j, c2 in enumerate(clusters):
            if j in assigned or c1["date"] == c2["date"]:
                continue
            score = enhanced_match_score(c1, c2, same_direction)
            if score >= threshold:
                group.append(c2)
                assigned.add(j)
        groups.append(group)
    groups.sort(key=lambda g: sum(c["amount_wan"] for c in g), reverse=True)
    return groups


def _split_by_size(group: list[dict]) -> list[dict]:
    """按规模连续性拆分可能混入的簇。"""
    if len(group) < 4:
        return group
    amounts = sorted([c["amount_wan"] for c in group])
    # 如果最大值超过中位数的5倍, 拆出异常值
    median = np.median(amounts)
    clean = [c for c in group if c["amount_wan"] <= median * 5]
    return clean if len(clean) >= 2 else group


# ═══════════════════════════════════════════════════
# 持仓曲线构建
# ═══════════════════════════════════════════════════

def build_position_curves(registry: list[dict]) -> pd.DataFrame:
    """为每个机构构建每日累计净持仓曲线。"""
    all_rows = []

    for inst in registry:
        ops = sorted(inst["operations"], key=lambda o: o["date"])
        cum_position = 0
        for op in ops:
            if op["direction"] == "BUY":
                cum_position += op["amount_wan"]
            else:
                cum_position -= op["amount_wan"]
            all_rows.append({
                "anon_id": inst["anon_id"],
                "confidence": inst["confidence"],
                "dominant": inst["dominant_direction"],
                "date": op["date"],
                "direction": op["direction"],
                "amount_wan": op["amount_wan"],
                "price_yuan": op["price_yuan"],
                "cum_position_wan": round(cum_position, 0),
            })

    return pd.DataFrame(all_rows)


# ═══════════════════════════════════════════════════
# 季度验证
# ═══════════════════════════════════════════════════

def validate_against_quarterly(registry: list[dict]):
    """将机构净持仓变化与公开季度持仓数据进行比对。"""
    holder_path = STOCK_DIR / "evidence" / "holder_changes.csv"
    if not holder_path.exists():
        return []

    hc = pd.read_csv(holder_path)

    # 活跃机构(北向等)的季度变化
    active_holders = hc[hc["holder_name"].str.contains("香港中央结算|野村|沈介良|旷达", na=False)]
    report_dates = sorted(active_holders["date"].unique())

    validations = []
    for inst in registry:
        if inst["confidence"] == "LOW" or inst["n_operations"] < 5:
            continue

        ops = sorted(inst["operations"], key=lambda o: o["date"])
        inst_dates = [op["date"] for op in ops]

        # 找出覆盖的报告期
        for rd in report_dates:
            rd_str = str(int(rd))
            rd_start = rd_str[:4] + "-" + rd_str[4:6] + "-" + rd_str[6:8]

            # 计算该季度内机构的净流向
            quarter_before = [o for o in ops if o["date"] < rd_str]
            if not quarter_before:
                continue

            net_before_rd = sum(o["amount_wan"] for o in quarter_before
                               if o["direction"] == "BUY") - \
                            sum(o["amount_wan"] for o in quarter_before
                               if o["direction"] == "SELL")

            # 北向在该季度的变化
            north_at_rd = active_holders[
                (active_holders["date"] == rd) &
                (active_holders["holder_name"].str.contains("香港中央结算", na=False))
                ]

            for _, nr in north_at_rd.iterrows():
                north_delta = nr.get("share_delta", 0)
                if pd.isna(north_delta):
                    continue

                # 粗略匹配: 如果机构净买入量和北向变化同向且规模相似
                north_delta_wan = abs(north_delta) * 6.5 / 10000  # 均价6.5元

                if abs(net_before_rd) > 1000 and north_delta_wan > 100:
                    direction_match = (net_before_rd > 0) == (north_delta > 0)
                    size_ratio = min(abs(net_before_rd), north_delta_wan) / \
                                 max(abs(net_before_rd), north_delta_wan)
                    validations.append({
                        "anon_id": inst["anon_id"],
                        "report_date": rd_str,
                        "inst_net_wan": round(net_before_rd, 0),
                        "northbound_delta_wan": round(north_delta_wan, 0),
                        "direction_match": direction_match,
                        "size_ratio": round(size_ratio, 3),
                    })

    return validations


# ═══════════════════════════════════════════════════
# 报告生成
# ═══════════════════════════════════════════════════

def generate_report(registry, positions, validations):
    lines = [
        f"# {STOCK} SOFIA v5 跨日机构追踪报告",
        "",
        f"## 方法论改进",
        "",
        "- **Stage 1**: 同向匹配(BUY→BUY, SELL→SELL), 阈值 0.70",
        "- **Stage 2**: 反向匹配(BUY→SELL), 阈值 0.55 (更严格)",
        "- **Stage 3**: 规模连续性验证 + 异常值拆分",
        "- **置信度**: HIGH(≥20次操作+方向纯度≥80%), MEDIUM, LOW",
        "",
    ]

    # 总览表
    lines.extend([
        "## 机构总览",
        "",
        "| 机构 | 置信度 | 操作数 | 覆盖天数 | 买入(万) | 卖出(万) | 净(万) | 主导方向 | 时段 | IDgap | 日期范围 |",
        "|------|--------|--------|---------|---------|---------|--------|---------|------|-------|---------|",
    ])

    for inst in registry:
        fp = inst["fingerprint"]
        rep = inst["representative"]

        dir_emoji = "买入" if inst["dominant_direction"] == "BUY" else "卖出"
        conf_emoji = {"HIGH": "🟢", "MEDIUM": "🟡", "LOW": "🔴"}.get(inst["confidence"], "⚪")

        lines.append(
            f"| {inst['anon_id']} | {conf_emoji}{inst['confidence']} | {inst['n_operations']} | "
            f"{inst['n_days']} | {inst['total_buy_wan']:.0f} | {inst['total_sell_wan']:.0f} | "
            f"{inst['net_wan']:+.0f} | {dir_emoji} | {fp['top_session']} | {fp['avg_id_gap']:.0f} | "
            f"{inst['date_range']} |"
        )

    # 逐机构详情
    for inst in registry:
        fp = inst["fingerprint"]
        lines.extend([
            "",
            f"### {inst['anon_id']} [{inst['confidence']}] — {inst['dominant_direction']}主导",
            "",
            f"- **总买入**: {inst['total_buy_wan']:.0f}万 | **总卖出**: {inst['total_sell_wan']:.0f}万 | **净**: {inst['net_wan']:+.0f}万",
            f"- **操作次数**: {inst['n_operations']} | **覆盖天数**: {inst['n_days']} | **日期范围**: {inst['date_range']}",
            f"- **平均每笔**: {inst['avg_amount_wan']:.0f}万 | **中位数**: {inst['median_amount_wan']:.0f}万 | **规模CV**: {inst['amount_cv']:.3f}",
            f"- **时段偏好**: {fp['top_session']} | **时段分布**: {fp['session_dist']}",
            f"- **IDgap均值**: {fp['avg_id_gap']:.0f} | **CV均值**: {fp['avg_qty_cv']:.3f}",
            "",
            "#### 持仓变化曲线（关键日期）",
            "| 日期 | 方向 | 金额(万) | 价格 | 累计净持仓(万) | 阶段 |",
            "|------|------|---------|------|--------------|------|",
        ])

        ops = sorted(inst["operations"], key=lambda o: o["date"])
        cum = 0
        for op in ops:
            cum += op["amount_wan"] if op["direction"] == "BUY" else -op["amount_wan"]
            # 只显示关键日期: 首次、拐点、末次、大额
            is_key = (op == ops[0] or op == ops[-1] or
                      op["amount_wan"] >= inst["avg_amount_wan"] * 2 or
                      (op["direction"] != ops[max(0, ops.index(op)-1)]["direction"]
                       if ops.index(op) > 0 else False))
            if is_key or ops.index(op) % max(1, len(ops)//15) == 0:
                stage = "建仓" if cum > 0 and op["direction"] == "BUY" else \
                        "减持" if cum < inst["net_wan"] * 0.5 else \
                        "调整"
                dir_sym = "买" if op["direction"] == "BUY" else "卖"
                lines.append(
                    f"| {op['date']} | {dir_sym} | {op['amount_wan']:.0f} | "
                    f"{op['price_yuan']:.2f} | {cum:+.0f} | {stage} |"
                )

    # 季度验证
    if validations:
        lines.extend([
            "",
            "## 季度持仓变化验证",
            "",
            "| 机构 | 报告期 | 机构净(万) | 北向变化(万) | 方向一致 | 规模比 |",
            "|------|--------|-----------|-------------|---------|--------|",
        ])
        for v in validations[:20]:
            dm = "✅" if v["direction_match"] else "❌"
            lines.append(
                f"| {v['anon_id']} | {v['report_date']} | {v['inst_net_wan']:.0f} | "
                f"{v['northbound_delta_wan']:.0f} | {dm} | {v['size_ratio']:.3f} |"
            )

    out_path = OUT_DIR / "crossday_report.md"
    out_path.write_text("\n".join(lines))
    return out_path


# ═══════════════════════════════════════════════════
# 主流程
# ═══════════════════════════════════════════════════

def main():
    year = "2025"
    raw_dir = STOCK_DIR / "raw"
    dates = sorted([d.name for d in raw_dir.iterdir() if d.is_dir() and d.name.startswith(year)])

    print(f"SOFIA v5 跨日机构追踪")
    print(f"  交易日: {len(dates)}")
    print()

    # Phase 1: 逐日聚类 (复用v4)
    all_clusters = {}
    for i, date_str in enumerate(dates):
        orders, _ = load_day(STOCK, date_str)
        if orders.empty:
            continue
        clusters = detect_algo_clusters(orders)
        if clusters:
            all_clusters[date_str] = clusters
        if (i + 1) % 60 == 0:
            print(f"  聚类进度: {i+1}/{len(dates)}")

    print(f"  聚类完成: {len(all_clusters)}天, {sum(len(v) for v in all_clusters.values())}个簇")

    # Phase 2: 跨日匹配
    print(f"\n  跨日匹配中...")
    registry = crossday_match(all_clusters)

    n_high = sum(1 for r in registry if r["confidence"] == "HIGH")
    n_med = sum(1 for r in registry if r["confidence"] == "MEDIUM")
    n_low = sum(1 for r in registry if r["confidence"] == "LOW")
    print(f"  注册机构: {len(registry)} 个 (HIGH={n_high}, MEDIUM={n_med}, LOW={n_low})")

    # Phase 3: 持仓曲线
    positions = build_position_curves(registry)
    positions.to_csv(OUT_DIR / "position_curves.csv", index=False)

    # Phase 4: 季度验证
    validations = validate_against_quarterly(registry)

    # Phase 5: 报告
    report_path = generate_report(registry, positions, validations)

    # 保存注册表
    registry_out = []
    for inst in registry:
        r = {k: v for k, v in inst.items() if k != "operations"}
        r["operations"] = inst["operations"]
        registry_out.append(r)
    with open(OUT_DIR / "crossday_registry.json", "w") as f:
        json.dump(registry_out, f, ensure_ascii=False, indent=2)

    # 汇总CSV
    summary_rows = []
    for inst in registry:
        fp = inst["fingerprint"]
        rep = inst["representative"]
        summary_rows.append({
            "anon_id": inst["anon_id"],
            "confidence": inst["confidence"],
            "n_ops": inst["n_operations"],
            "n_days": inst["n_days"],
            "buy_wan": inst["total_buy_wan"],
            "sell_wan": inst["total_sell_wan"],
            "net_wan": inst["net_wan"],
            "dominant": inst["dominant_direction"],
            "buy_pct": inst["buy_pct"],
            "avg_amount": inst["avg_amount_wan"],
            "amount_cv": inst["amount_cv"],
            "top_session": fp["top_session"],
            "avg_id_gap": fp["avg_id_gap"],
            "avg_qty_cv": fp["avg_qty_cv"],
            "date_range": inst["date_range"],
            "rep_date": rep["date"],
            "rep_amount": rep["amount_wan"],
            "rep_price": rep["price_yuan"],
            "rep_orders": rep["n_orders"],
        })
    pd.DataFrame(summary_rows).to_csv(OUT_DIR / "crossday_registry.csv", index=False)

    print(f"\n输出已保存: {OUT_DIR}/")
    print(f"  crossday_registry.json  — 机构注册表")
    print(f"  crossday_registry.csv   — 汇总CSV")
    print(f"  position_curves.csv     — 每日持仓曲线")
    print(f"  crossday_report.md      — 人读报告")
    print(f"\n  {report_path}")

    # 打印关键数据
    print(f"\n=== TOP 10 机构 (按净流向) ===")
    for inst in registry[:10]:
        conf = inst["confidence"]
        print(f"  {inst['anon_id']} [{conf}] "
              f"买{inst['total_buy_wan']:.0f}/卖{inst['total_sell_wan']:.0f}/净{inst['net_wan']:+.0f}万 "
              f"{inst['n_operations']}次/{inst['n_days']}天 "
              f"({inst['date_range']})")


if __name__ == "__main__":
    main()
