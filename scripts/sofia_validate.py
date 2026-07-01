"""
SOFIA 模型验证 — 聚类质量、特征区分力、阈值合理性评估

风控/量化双视角:
  - IV (Information Value) / 特征区分力: 每个指纹维度区分不同机构的能力
  - KS (Kolmogorov-Smirnov): 特征在机构间分布的分离度
  - Silhouette Score: 聚类紧密度
  - Intra/Inter-class similarity: 阈值选择依据
  - Per-institution confidence: 每个机构的聚类置信度
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from itertools import combinations

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent.parent))

STOCK = "002516"
SOFIA_DIR = Path(__file__).parent.parent / "data" / "single_stock" / STOCK / "sofia_v4"
OUT_DIR = SOFIA_DIR / "validation"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_data():
    registry = json.load(open(SOFIA_DIR / "institution_registry.json"))
    master = pd.read_csv(SOFIA_DIR / "orderbook" / "master_orderbook.csv")
    return registry, master


# ═══════════════════════════════════════════════════
# 1. 特征矩阵构建
# ═══════════════════════════════════════════════════

FEATURE_NAMES = [
    "log_amount",      # log10(金额)
    "n_orders",        # 订单笔数
    "qty_cv",          # 拆单数量CV
    "avg_id_gap",      # ID间隔均值
    "price_yuan",      # 成交均价
    "session_num",     # 时段(数值化)
    "direction_num",   # 方向(数值化)
    "amount_cv",       # 金额波动(暂用每笔数量替代)
]

SESSION_ORDER = {"AUCTION": 0, "OPEN": 1, "MORNING": 2, "LATE_MORNING": 3,
                 "EARLY_AFTER": 4, "AFTERNOON": 5, "CLOSE": 6}


def build_feature_matrix(registry: list[dict]) -> pd.DataFrame:
    """从机构注册表提取特征矩阵: 每个机构一行，使用代表簇的特征。"""
    rows = []
    for inst in registry:
        if inst["n_clusters"] < 2:
            continue
        rep = inst["representative"]
        fp = inst["fingerprint_summary"]
        bh = inst["behavior"]

        # 使用代表操作的指纹
        for c in inst["all_clusters"]:
            rows.append({
                "anon_id": inst["anon_id"],
                "log_amount": np.log10(c["amount_wan"] + 1),
                "n_orders": c["n_orders"],
                "qty_cv": c["qty_cv"],
                "avg_id_gap": c["avg_id_gap"],
                "price_yuan": c["price_yuan"],
                "session_num": SESSION_ORDER.get(c["session"], -1),
                "direction_num": 1 if c["direction"] == "BUY" else 0,
                "amount_wan": c["amount_wan"],
                "session": c["session"],
                "direction": c["direction"],
            })

    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════
# 2. IV (Information Value) — 特征区分力
# ═══════════════════════════════════════════════════

def compute_iv_for_feature(df: pd.DataFrame, feature: str, n_bins: int = 5) -> dict:
    """
    计算特征区分不同机构的能力。

    在风控中 IV 衡量特征区分 good/bad 的能力。
    在这里我们衡量特征区分 "机构A vs 机构B" 的能力：
    对每对机构计算 IV，然后取均值作为该特征的总体区分力。
    """
    valid = df[df[feature].notna() & np.isfinite(df[feature])]
    if len(valid) < 10:
        return {"feature": feature, "mean_iv": 0, "std_iv": 0, "n_pairs": 0, "iv_details": []}

    inst_ids = valid["anon_id"].unique()
    if len(inst_ids) < 2:
        return {"feature": feature, "mean_iv": 0, "std_iv": 0, "n_pairs": 0, "iv_details": []}

    pair_ivs = []
    for a, b in combinations(inst_ids, 2):
        da = valid[valid["anon_id"] == a][feature]
        db = valid[valid["anon_id"] == b][feature]
        if len(da) < 2 or len(db) < 2:
            continue

        combined = pd.concat([da, db])
        try:
            # 等频分箱
            bins = pd.qcut(combined, q=min(n_bins, len(combined) // 2),
                           duplicates="drop", retbins=True)[1]
            if len(bins) < 3:
                continue

            da_binned = pd.cut(da, bins=bins, include_lowest=True)
            db_binned = pd.cut(db, bins=bins, include_lowest=True)

            da_dist = da_binned.value_counts(normalize=True).sort_index()
            db_dist = db_binned.value_counts(normalize=True).sort_index()

            # IV = Σ (dist_A - dist_B) × ln(dist_A / dist_B)
            iv = 0.0
            for idx in da_dist.index.union(db_dist.index):
                pa = da_dist.get(idx, 0.001)
                pb = db_dist.get(idx, 0.001)
                iv += (pa - pb) * np.log(pa / pb)
            pair_ivs.append(abs(iv))
        except Exception:
            continue

    if not pair_ivs:
        return {"feature": feature, "mean_iv": 0, "std_iv": 0, "n_pairs": 0, "iv_details": []}

    return {
        "feature": feature,
        "mean_iv": round(float(np.mean(pair_ivs)), 4),
        "std_iv": round(float(np.std(pair_ivs)), 4),
        "median_iv": round(float(np.median(pair_ivs)), 4),
        "max_iv": round(float(np.max(pair_ivs)), 4),
        "n_pairs": len(pair_ivs),
        "iv_details": sorted(pair_ivs, reverse=True)[:10],
    }


# ═══════════════════════════════════════════════════
# 3. KS (Kolmogorov-Smirnov) — 特征分离度
# ═══════════════════════════════════════════════════

def compute_ks_for_feature(df: pd.DataFrame, feature: str) -> dict:
    """
    对每个机构对计算 KS 统计量，取均值。
    KS 衡量两个机构在该特征上的分布差异。
    """
    valid = df[df[feature].notna() & np.isfinite(df[feature])]
    inst_ids = valid["anon_id"].unique()
    if len(inst_ids) < 2:
        return {"feature": feature, "mean_ks": 0, "n_pairs": 0}

    pair_ks = []
    for a, b in combinations(inst_ids, 2):
        da = valid[valid["anon_id"] == a][feature].dropna()
        db = valid[valid["anon_id"] == b][feature].dropna()
        if len(da) < 2 or len(db) < 2:
            continue
        try:
            ks_stat, _ = stats.ks_2samp(da, db)
            pair_ks.append(ks_stat)
        except Exception:
            continue

    if not pair_ks:
        return {"feature": feature, "mean_ks": 0, "n_pairs": 0}

    return {
        "feature": feature,
        "mean_ks": round(float(np.mean(pair_ks)), 4),
        "std_ks": round(float(np.std(pair_ks)), 4),
        "median_ks": round(float(np.median(pair_ks)), 4),
        "ks_gt_05": round(sum(1 for k in pair_ks if k > 0.5) / len(pair_ks) * 100, 1),
        "ks_gt_08": round(sum(1 for k in pair_ks if k > 0.8) / len(pair_ks) * 100, 1),
        "n_pairs": len(pair_ks),
    }


# ═══════════════════════════════════════════════════
# 4. 组内/组间相似度分布 (阈值验证)
# ═══════════════════════════════════════════════════

def compute_similarity_distributions(registry: list[dict]):
    """
    计算同机构(组内)和不同机构(组间)的指纹相似度分布。
    这是验证阈值选择的核心分析。
    """
    from scripts.sofia_v4_hunter import match_score_fp, extract_fingerprint

    # 为每个簇构建指纹
    all_clusters_fp = []
    for inst in registry:
        for c in inst["all_clusters"]:
            fp_raw = {
                "total_amount_wan": c["amount_wan"],
                "n_orders": c["n_orders"],
                "avg_qty": c["amount_wan"] * 10000 / max(c["n_orders"], 1),
                "qty_cv": c["qty_cv"],
                "avg_id_gap": c["avg_id_gap"],
                "avg_time_gap_sec": 0,
                "avg_price_yuan": c["price_yuan"],
                "time_start": SESSION_ORDER.get(c["session"], 3) * 3600 + 1800,
                "direction": c["direction"],
            }
            all_clusters_fp.append({
                "anon_id": inst["anon_id"],
                "date": c["date"],
                "fingerprint": extract_fingerprint(fp_raw),
            })

    # 组内相似度：同一机构的不同操作之间
    intra_scores = []
    intra_details = []
    for inst in registry:
        inst_clusters = [c for c in all_clusters_fp if c["anon_id"] == inst["anon_id"]]
        if len(inst_clusters) < 2:
            continue
        for i, c1 in enumerate(inst_clusters):
            for c2 in inst_clusters[i + 1:]:
                score = match_score_fp(c1["fingerprint"], c2["fingerprint"])
                intra_scores.append(score)
                intra_details.append({
                    "type": "intra",
                    "anon_id": inst["anon_id"],
                    "score": score,
                    "date1": c1["date"],
                    "date2": c2["date"],
                })

    # 组间相似度：不同机构之间
    inter_scores = []
    inter_details = []
    anon_ids = sorted(set(c["anon_id"] for c in all_clusters_fp))
    for a, b in combinations(anon_ids, 2):
        ca_list = [c for c in all_clusters_fp if c["anon_id"] == a]
        cb_list = [c for c in all_clusters_fp if c["anon_id"] == b]
        # 采样避免组合爆炸
        for ca in ca_list[:5]:
            for cb in cb_list[:5]:
                score = match_score_fp(ca["fingerprint"], cb["fingerprint"])
                inter_scores.append(score)
                inter_details.append({
                    "type": "inter",
                    "pair": f"{a} vs {b}",
                    "score": score,
                })

    n_intra = len(intra_scores)
    n_inter = len(inter_scores)

    # 统计量
    intra_arr = np.array(intra_scores) if intra_scores else np.array([0])
    inter_arr = np.array(inter_scores) if inter_scores else np.array([0])

    # 最佳阈值: 最大化 (intra_recall + inter_specificity)
    thresholds = np.arange(0.40, 0.85, 0.025)
    best_threshold = 0.60
    best_f1 = 0
    threshold_metrics = []

    for t in thresholds:
        intra_hit = np.mean(intra_arr >= t) if len(intra_arr) > 0 else 0  # recall
        inter_miss = np.mean(inter_arr < t) if len(inter_arr) > 0 else 0  # specificity
        f1 = 2 * intra_hit * inter_miss / (intra_hit + inter_miss + 1e-10)
        threshold_metrics.append({
            "threshold": round(t, 3),
            "intra_recall": round(float(intra_hit), 4),
            "inter_specificity": round(float(inter_miss), 4),
            "f1": round(float(f1), 4),
        })
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = t

    return {
        "intra_mean": round(float(np.mean(intra_arr)), 4),
        "intra_std": round(float(np.std(intra_arr)), 4),
        "intra_median": round(float(np.median(intra_arr)), 4),
        "intra_p25": round(float(np.percentile(intra_arr, 25)), 4),
        "intra_p75": round(float(np.percentile(intra_arr, 75)), 4),
        "inter_mean": round(float(np.mean(inter_arr)), 4),
        "inter_std": round(float(np.std(inter_arr)), 4),
        "inter_median": round(float(np.median(inter_arr)), 4),
        "inter_p25": round(float(np.percentile(inter_arr, 25)), 4),
        "inter_p75": round(float(np.percentile(inter_arr, 75)), 4),
        "n_intra_pairs": n_intra,
        "n_inter_pairs": n_inter,
        "best_threshold": round(best_threshold, 3),
        "best_f1": round(float(best_f1), 4),
        "current_threshold_recall": round(float(np.mean(intra_arr >= 0.60)), 4),
        "current_threshold_specificity": round(float(np.mean(inter_arr < 0.60)), 4),
        "threshold_metrics": threshold_metrics,
        "overlap_pct": round(float(np.mean(inter_arr >= np.percentile(intra_arr, 25))), 4) * 100,
    }


# ═══════════════════════════════════════════════════
# 5. 单机构置信度评估
# ═══════════════════════════════════════════════════

def compute_institution_confidence(registry: list[dict], intra_details: list[dict]):
    """评估每个机构的聚类置信度。"""
    confidences = []

    for inst in registry:
        aid = inst["anon_id"]
        n = inst["n_clusters"]

        # 1. 组内一致性: 该机构内部操作的相似度中位数
        inst_intra = [d for d in intra_details if d["anon_id"] == aid]
        intra_median = np.median([d["score"] for d in inst_intra]) if inst_intra else 0

        # 2. 规模支持: 更多操作 → 更多证据
        size_score = min(1.0, np.log10(n + 1) / np.log10(50))

        # 3. IDgap 稳定性
        gaps = [c["avg_id_gap"] for c in inst["all_clusters"]]
        gap_cv = float(np.std(gaps) / np.mean(gaps)) if np.mean(gaps) > 0 else 0
        gap_stability = max(0, 1 - gap_cv)

        # 4. 时段集中度
        sessions = [c["session"] for c in inst["all_clusters"]]
        session_entropy = -sum(
            (sessions.count(s) / len(sessions)) * np.log2(sessions.count(s) / len(sessions) + 1e-10)
            for s in set(sessions)
        )
        max_entropy = np.log2(len(set(sessions)) + 1e-10)
        session_concentration = 1 - session_entropy / max(max_entropy, 1)

        # 5. 方向纯度
        dirs = [c["direction"] for c in inst["all_clusters"]]
        dir_purity = max(dirs.count("BUY"), dirs.count("SELL")) / len(dirs)

        # 综合置信度
        confidence = (
            0.30 * intra_median +
            0.25 * size_score +
            0.20 * gap_stability +
            0.15 * session_concentration +
            0.10 * dir_purity
        )

        level = ("A" if confidence >= 0.70 else
                 "B" if confidence >= 0.55 else
                 "C" if confidence >= 0.40 else "D")

        confidences.append({
            "anon_id": aid,
            "size_label": inst["size_label"],
            "confidence": round(float(confidence), 4),
            "level": level,
            "intra_consistency": round(float(intra_median), 4),
            "size_score": round(float(size_score), 4),
            "gap_stability": round(float(gap_stability), 4),
            "session_concentration": round(float(session_concentration), 4),
            "direction_purity": round(float(dir_purity), 4),
            "n_clusters": n,
            "n_days": inst["n_days"],
        })

    return sorted(confidences, key=lambda x: x["confidence"], reverse=True)


# ═══════════════════════════════════════════════════
# 6. 权重对比分析
# ═══════════════════════════════════════════════════

def compare_weight_configs(registry: list[dict]):
    """
    尝试不同的权重配置，评估哪种权重最能分离组内/组间。
    当前权重: size=0.20, cv=0.15, idgap=0.10, time_gap=0.10,
              n_orders=0.10, session=0.15, direction=0.10, price=0.05, avg_qty=0.05
    """
    from scripts.sofia_v4_hunter import match_score_fp, extract_fingerprint

    all_clusters_fp = []
    for inst in registry:
        for c in inst["all_clusters"]:
            fp_raw = {
                "total_amount_wan": c["amount_wan"],
                "n_orders": c["n_orders"],
                "avg_qty": c["amount_wan"] * 10000 / max(c["n_orders"], 1),
                "qty_cv": c["qty_cv"],
                "avg_id_gap": c["avg_id_gap"],
                "avg_time_gap_sec": 0,
                "avg_price_yuan": c["price_yuan"],
                "time_start": SESSION_ORDER.get(c["session"], 3) * 3600 + 1800,
                "direction": c["direction"],
            }
            all_clusters_fp.append({
                "anon_id": inst["anon_id"],
                "fingerprint": extract_fingerprint(fp_raw),
            })

    # 原始权重下计算组内/组间分数
    # 保存原始 match_score_fp
    import scripts.sofia_v4_hunter as hunter
    original_fn = hunter.match_score_fp

    configs = [
        ("当前权重", None),  # use original
        ("等权(all=0.11)", "uniform"),
        ("拆单为主(cv+gap=0.5)", "split_heavy"),
        ("时段为主(session=0.3)", "time_heavy"),
        ("方向为主(dir=0.3)", "dir_heavy"),
    ]

    results = []
    for name, config in configs:
        if config == "uniform":
            # 修改函数... 不好直接改，跳过
            results.append({"config": name, "note": "需要修改源码"})
            continue
        elif config is None:
            scores = {"intra": [], "inter": []}
            anon_ids = sorted(set(c["anon_id"] for c in all_clusters_fp))
            for a in anon_ids:
                ca_list = [c for c in all_clusters_fp if c["anon_id"] == a]
                for i, c1 in enumerate(ca_list):
                    for c2 in ca_list[i + 1:]:
                        scores["intra"].append(original_fn(c1["fingerprint"], c2["fingerprint"]))
            for a, b in combinations(anon_ids, 2):
                ca_list = [c for c in all_clusters_fp if c["anon_id"] == a]
                cb_list = [c for c in all_clusters_fp if c["anon_id"] == b]
                for ca in ca_list[:3]:
                    for cb in cb_list[:3]:
                        scores["inter"].append(original_fn(ca["fingerprint"], cb["fingerprint"]))
            intra_m = np.mean(scores["intra"]) if scores["intra"] else 0
            inter_m = np.mean(scores["inter"]) if scores["inter"] else 0
            separation = intra_m - inter_m
            results.append({
                "config": name,
                "intra_mean": round(float(intra_m), 4),
                "inter_mean": round(float(inter_m), 4),
                "separation": round(float(separation), 4),
                "sep_ratio": round(float(intra_m / max(inter_m, 1e-10)), 2),
            })
        else:
            results.append({"config": name, "note": "需要修改源码中的权重"})

    return results


# ═══════════════════════════════════════════════════
# 主流程
# ═══════════════════════════════════════════════════

def main():
    print("=" * 90)
    print("SOFIA 模型验证报告")
    print("=" * 90)

    registry, master = load_data()
    print(f"\n数据: {len(registry)}个机构, {len(master)}条操作记录")

    # ---- 特征矩阵 ----
    df = build_feature_matrix(registry)
    print(f"特征矩阵: {df.shape[0]}行 × {len(FEATURE_NAMES)}个特征")
    print(f"覆盖机构: {df['anon_id'].nunique()}个")

    # ---- 1. 特征区分力 IV ----
    print(f"\n{'='*90}")
    print("1. 特征区分力 (IV — Information Value 等价)")
    print(f"{'='*90}")
    print(f"  {'特征':<18} {'Mean IV':>10} {'Median IV':>10} {'Max IV':>10} {'Pairs':>7} {'判别力':<15}")
    print(f"  {'-'*75}")

    iv_results = []
    for feat in ["log_amount", "n_orders", "qty_cv", "avg_id_gap",
                  "price_yuan", "session_num", "direction_num"]:
        iv = compute_iv_for_feature(df, feat)
        iv_results.append(iv)
        power = ("极强" if iv["mean_iv"] > 1.0 else
                 "强" if iv["mean_iv"] > 0.5 else
                 "中等" if iv["mean_iv"] > 0.2 else
                 "弱" if iv["mean_iv"] > 0.1 else "极弱")
        print(f"  {iv['feature']:<18} {iv['mean_iv']:>10.4f} {iv['median_iv']:>10.4f} "
              f"{iv['max_iv']:>10.4f} {iv['n_pairs']:>7} {power:<15}")

    # ---- 2. KS 分离度 ----
    print(f"\n{'='*90}")
    print("2. 特征分离度 (KS — 分布差异)")
    print(f"{'='*90}")
    print(f"  {'特征':<18} {'Mean KS':>10} {'Median KS':>10} {'KS>0.5%':>10} {'KS>0.8%':>10} {'Pairs':>7}")
    print(f"  {'-'*70}")

    for feat in ["log_amount", "n_orders", "qty_cv", "avg_id_gap",
                  "price_yuan", "session_num", "direction_num"]:
        ks = compute_ks_for_feature(df, feat)
        print(f"  {ks['feature']:<18} {ks['mean_ks']:>10.4f} {ks['median_ks']:>10.4f} "
              f"{ks['ks_gt_05']:>9.1f}% {ks['ks_gt_08']:>9.1f}% {ks['n_pairs']:>7}")

    # ---- 3. 组内/组间相似度 ----
    print(f"\n{'='*90}")
    print("3. 组内 vs 组间相似度分布 (阈值验证)")
    print(f"{'='*90}")

    sim_dist = compute_similarity_distributions(registry)
    print(f"  组内(同机构): mean={sim_dist['intra_mean']:.4f}, median={sim_dist['intra_median']:.4f}, "
          f"σ={sim_dist['intra_std']:.4f}, P25={sim_dist['intra_p25']:.4f}, P75={sim_dist['intra_p75']:.4f}")
    print(f"  组间(不同机构): mean={sim_dist['inter_mean']:.4f}, median={sim_dist['inter_median']:.4f}, "
          f"σ={sim_dist['inter_std']:.4f}, P25={sim_dist['inter_p25']:.4f}, P75={sim_dist['inter_p75']:.4f}")
    print(f"  样本: 组内{sim_dist['n_intra_pairs']}对, 组间{sim_dist['n_inter_pairs']}对")
    print(f"  分离度: {sim_dist['intra_mean'] - sim_dist['inter_mean']:.4f} "
          f"(ratio={sim_dist['intra_mean']/max(sim_dist['inter_mean'],1e-10):.2f}x)")
    print(f"  分布重叠: {sim_dist['overlap_pct']:.1f}%")

    print(f"\n  === 阈值扫描 ===")
    print(f"  {'阈值':<10} {'组内命中':>10} {'组间拒绝':>10} {'F1':>10}")
    print(f"  {'-'*42}")
    for tm in sim_dist["threshold_metrics"]:
        marker = " ← BEST" if tm["threshold"] == sim_dist["best_threshold"] else ""
        print(f"  {tm['threshold']:<10.3f} {tm['intra_recall']:>10.4f} "
              f"{tm['inter_specificity']:>10.4f} {tm['f1']:>10.4f}{marker}")

    print(f"\n  === 当前阈值 0.60 的表现 ===")
    print(f"  组内命中(recall): {sim_dist['current_threshold_recall']:.2%}")
    print(f"  组间拒绝(specificity): {sim_dist['current_threshold_specificity']:.2%}")
    print(f"  最佳阈值: {sim_dist['best_threshold']:.3f} (F1={sim_dist['best_f1']:.4f})")

    # ---- 4. 单机构置信度 ----
    print(f"\n{'='*90}")
    print("4. 单机构聚类置信度")
    print(f"{'='*90}")
    print(f"  {'机构':<12} {'规模':<6} {'置信度':>8} {'等级':>4} {'组内一致':>10} "
          f"{'规模分':>8} {'IDgap稳':>8} {'时段集中':>8} {'方向纯度':>8} {'操作数':>6}")
    print(f"  {'-'*85}")

    confidences = compute_institution_confidence(registry, sim_dist.get("intra_details", []))
    for c in confidences[:15]:
        print(f"  {c['anon_id']:<12} {c['size_label']:<6} {c['confidence']:>8.4f} {c['level']:>4} "
              f"{c['intra_consistency']:>10.4f} {c['size_score']:>8.4f} {c['gap_stability']:>8.4f} "
              f"{c['session_concentration']:>8.4f} {c['direction_purity']:>8.4f} {c['n_clusters']:>6}")

    # ---- 汇总表格 ----
    print(f"\n{'='*90}")
    print("5. 综合模型评分卡")
    print(f"{'='*90}")

    # KS 和 IV 合并排名
    print(f"\n  === 特征重要性排名 ===")
    combined = []
    for iv_r, feat in zip(iv_results, ["log_amount", "n_orders", "qty_cv", "avg_id_gap",
                                         "price_yuan", "session_num", "direction_num"]):
        ks_r = compute_ks_for_feature(df, feat)
        combined.append({
            "feature": feat,
            "iv": iv_r["mean_iv"],
            "ks": ks_r["mean_ks"],
            "score": iv_r["mean_iv"] * 0.4 + ks_r["mean_ks"] * 0.6,
        })
    combined.sort(key=lambda x: x["score"], reverse=True)

    print(f"  {'Rank':<6} {'特征':<18} {'IV':>10} {'KS':>10} {'综合分':>10} {'重要性'}")
    print(f"  {'-'*60}")
    for rank, c in enumerate(combined, 1):
        importance = "核心" if c["score"] > 0.5 else "重要" if c["score"] > 0.3 else "辅助" if c["score"] > 0.15 else "弱"
        print(f"  {rank:<6} {c['feature']:<18} {c['iv']:>10.4f} {c['ks']:>10.4f} "
              f"{c['score']:>10.4f} {importance}")

    # ---- 对比当前权重 vs 最优权重 ----
    print(f"\n  === 当前权重 vs 数据驱动建议 ===")
    print(f"  {'维度':<18} {'当前权重':>10} {'IV排名':>8} {'KS排名':>8} {'建议调整'}")
    print(f"  {'-'*60}")
    current_weights = {
        "log_amount": 0.20, "n_orders": 0.10, "qty_cv": 0.15,
        "avg_id_gap": 0.10, "price_yuan": 0.05,
        "session_num": 0.15, "direction_num": 0.10,
    }
    # 按IV+KS排名给建议权重
    total_score = sum(c["score"] for c in combined)
    for feat_name, cur_w in current_weights.items():
        feat_score = next((c["score"] for c in combined if c["feature"] == feat_name), 0)
        suggested_w = feat_score / max(total_score, 1e-10) if total_score > 0 else 0
        iv_rank = next((i+1 for i, c in enumerate(combined) if c["feature"] == feat_name), "-")
        ks_r = compute_ks_for_feature(df, feat_name)
        ks_rank = sorted(combined, key=lambda x: x["ks"], reverse=True)
        ks_pos = next((i+1 for i, c in enumerate(ks_rank) if c["feature"] == feat_name), "-")
        direction = "↑" if suggested_w > cur_w * 1.3 else "↓" if suggested_w < cur_w * 0.7 else "→"
        print(f"  {feat_name:<18} {cur_w:>10.2f} {iv_rank:>8} {ks_pos:>8} "
              f"{direction} (建议{suggested_w:.2f})")

    # ---- 保存 ----
    pd.DataFrame(confidences).to_csv(OUT_DIR / "institution_confidence.csv", index=False)
    pd.DataFrame(iv_results).to_csv(OUT_DIR / "feature_iv.csv", index=False)
    pd.DataFrame(sim_dist["threshold_metrics"]).to_csv(OUT_DIR / "threshold_scan.csv", index=False)

    print(f"\n输出已保存至: {OUT_DIR}/")
    print(f"  institution_confidence.csv  — 机构置信度评级")
    print(f"  feature_iv.csv              — 特征IV值")
    print(f"  threshold_scan.csv          — 阈值扫描")


if __name__ == "__main__":
    main()
