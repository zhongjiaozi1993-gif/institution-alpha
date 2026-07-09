"""Sprint 0.7: DBSCAN BUY signal stock profile analysis.

Tasks: group stocks A/B/C/D, compare valid vs invalid, find common traits,
generate next-round selection rules.
"""
import pandas as pd
import numpy as np
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
SUMMARY = PROJECT / "data/processed/oot/oot_top100_summary.csv"
TOP100 = PROJECT / "data/processed/stock_universe/zz1000_liquid_top100.csv"
SELECTED = PROJECT / "data/processed/stock_universe/zz1000_liquid_selected.csv"
CANDIDATES_FILE = PROJECT / "data/processed/oot/production_candidate_stocks_top100.txt"
OUT_DIR = PROJECT / "data/processed/analysis"
OUT_PROFILE = OUT_DIR / "dbscan_stock_profile_top100.csv"
OUT_REPORT = OUT_DIR / "dbscan_valid_vs_invalid_report.md"
OUT_RULES = OUT_DIR / "dbscan_candidate_selection_rules.md"

OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---- Load data ----
summary = pd.read_csv(SUMMARY, dtype={"stock_code": str})
summary["stock_code"] = summary["stock_code"].str.zfill(6)
top100 = pd.read_csv(TOP100, dtype={"stock_code": str})
top100["stock_code"] = top100["stock_code"].str.zfill(6)
sel = pd.read_csv(SELECTED)
sel["stock_code"] = sel["stock_code"].astype(str).str.zfill(6)

# Load production candidates (zero-fill)
candidates = set()
with open(CANDIDATES_FILE) as f:
    for line in f:
        line = line.strip()
        if line:
            candidates.add(line.zfill(6))

# Build stock name lookup (ZZ1000 names + akshare spot fallback)
name_map = dict(zip(sel["stock_code"], sel["stock_name"]))

# Fill missing names via akshare stock_zh_a_spot (Sina, not blocked)
# Cache spot data to avoid repeated slow queries
SPOT_CACHE = OUT_DIR / "akshare_spot_cache.parquet"
missing_codes = set(summary["stock_code"]) - set(name_map.keys())
if missing_codes:
    spot = None
    if SPOT_CACHE.exists():
        spot = pd.read_parquet(SPOT_CACHE)
    else:
        try:
            import akshare as ak
            spot = ak.stock_zh_a_spot()
            spot.to_parquet(SPOT_CACHE)
            print(f"  Spot data cached: {len(spot)} stocks")
        except Exception as e:
            print(f"  akshare spot query failed: {e}")
    if spot is not None:
        spot["code_clean"] = spot["代码"].str.extract(r"(\d{6})$")[0]
        spot_map = {}
        for _, r in spot.iterrows():
            c = r.get("code_clean")
            if c and c in missing_codes:
                spot_map[c] = r.get("名称", "")
        name_map.update(spot_map)
        print(f"  Names filled: {len(spot_map)}/{len(missing_codes)}")

# Merge with top100 for avg_amount_yuan (daily turnover)
df = summary.merge(
    top100[["stock_code", "avg_amount_yuan"]],
    on="stock_code", how="left"
)
df["stock_name"] = df["stock_code"].map(name_map)

# ---- Task 1: Assign groups ----
def classify(row):
    if row["n_buy_ops"] >= 100 and row["avg_fwd_5d"] > 0 and row["win_5d"] > 0.52 \
            and row["avg_universe_excess_fwd_5d"] > 0 and row["stability"] > 0:
        return "A_valid_candidate"
    if row["avg_fwd_5d"] > 0 and row["avg_universe_excess_fwd_5d"] <= 0:
        return "C_beta_only"
    if row["avg_fwd_5d"] <= 0:
        return "D_invalid"
    if row["avg_fwd_5d"] > 0:
        return "B_positive_but_weak"
    return "D_invalid"

df["candidate_group"] = df.apply(classify, axis=1)
df["is_production_candidate"] = df["stock_code"].isin(candidates)

# Select and reorder columns for output profile
profile_cols = [
    "stock_code", "stock_name", "n_buy_ops", "n_days", "avg_daily_ops",
    "avg_amount_wan", "total_buy_wan", "avg_order_count",
    "avg_time_span_min", "avg_vwap_deviation_pct",
    "avg_fwd_5d", "win_5d", "avg_universe_excess_fwd_5d",
    "universe_excess_win_5d", "stability", "score",
    "avg_fwd_10d", "win_10d", "avg_fwd_20d", "win_20d",
    "avg_amount_yuan", "candidate_group", "is_production_candidate",
]
# Only keep cols that exist
profile_cols = [c for c in profile_cols if c in df.columns]
profile_out = df[profile_cols].copy()
profile_out["stock_code"] = profile_out["stock_code"].astype(str).str.zfill(6)
profile_out.to_csv(OUT_PROFILE, index=False)
print(f"Profile saved: {OUT_PROFILE} ({len(df)} stocks)")

# Group counts
for g in ["A_valid_candidate", "B_positive_but_weak", "C_beta_only", "D_invalid"]:
    n = (df["candidate_group"] == g).sum()
    print(f"  {g}: {n}")

# ---- Task 2 & 3: Comparative analysis ----

# Define comparison metrics
METRICS = {
    "n_buy_ops": ("BUY 信号数量", "count"),
    "n_days": ("有信号交易日数", "count"),
    "avg_amount_wan": ("单笔 BUY 均额(万)", "amount"),
    "total_buy_wan": ("总 BUY 金额(万)", "amount"),
    "avg_order_count": ("聚类内平均委托数", "cluster"),
    "avg_time_span_min": ("聚类平均时长(分钟)", "cluster"),
    "avg_vwap_deviation_pct": ("VWAP 偏离(%)", "price"),
    "stability": ("收益稳定性(stability)", "risk"),
    "avg_amount_yuan": ("日均成交额(元)", "liquidity"),
    "avg_fwd_5d": ("5日平均收益(%)", "return"),
    "win_5d": ("5日胜率", "return"),
    "avg_universe_excess_fwd_5d": ("5日超额收益(%)", "alpha"),
    "universe_excess_win_5d": ("5日超额胜率", "alpha"),
    "avg_fwd_10d": ("10日平均收益(%)", "return"),
    "win_10d": ("10日胜率", "return"),
    "avg_fwd_20d": ("20日平均收益(%)", "return"),
    "win_20d": ("20日胜率", "return"),
}

group_A = df[df["candidate_group"] == "A_valid_candidate"]
group_B = df[df["candidate_group"] == "B_positive_but_weak"]
group_C = df[df["candidate_group"] == "C_beta_only"]
group_D = df[df["candidate_group"] == "D_invalid"]
group_rest = df[df["candidate_group"] != "A_valid_candidate"]

def fmt_mean_std(series):
    return f"{series.mean():.2f} ± {series.std():.2f}"

def fmt_pct(series):
    return f"{series.mean():.1%}"

# Build report
rpt = []
rpt.append("# DBSCAN BUY 有效 vs 无效股票分析报告\n")
rpt.append(f"生成时间: {pd.Timestamp.now().strftime('%Y-%m-%d')}\n")
rpt.append("---\n")
rpt.append("## 分组概览\n")
rpt.append("| 分组 | 股票数 | 占比 | 含义 |")
rpt.append("|------|--------|------|------|")
for g, label in [("A_valid_candidate", "A: 有效候选"),
                  ("B_positive_but_weak", "B: 正收益但弱"),
                  ("C_beta_only", "C: 仅 Beta"),
                  ("D_invalid", "D: 无效")]:
    n = (df["candidate_group"] == g).sum()
    pct = n / len(df) * 100
    rpt.append(f"| {label} | {n} | {pct:.0f}% | |")
rpt.append(f"\n**Top100 总计: {len(df)} 只**\n")

# A group detail
rpt.append("---\n")
rpt.append("## A 组: 有效候选 (Production Candidates)\n")
rpt.append(f"共 {len(group_A)} 只。\n")
rpt.append("| 代码 | 名称 | BUY数 | Fwd5d | Win5d | Excess5d | Stability | Score |")
rpt.append("|------|------|-------|-------|-------|----------|-----------|-------|")
for _, row in group_A.sort_values("score", ascending=False).iterrows():
    name = row.get("stock_name", "") or ""
    rpt.append(f"| {row['stock_code']} | {name} | {int(row['n_buy_ops'])} | "
               f"{row['avg_fwd_5d']:+.2f}% | {row['win_5d']:.1%} | "
               f"{row['avg_universe_excess_fwd_5d']:+.2f}% | {row['stability']:.3f} | "
               f"{row['score']:.2f} |")

# ---- Comparison table: A vs rest ----
rpt.append("\n---\n")
rpt.append("## A 组 vs 其他组: 核心指标对比\n")
rpt.append("| 指标 | A_valid (n={}) | B_weak (n={}) | C_beta (n={}) | D_invalid (n={}) |".format(
    len(group_A), len(group_B), len(group_C), len(group_D)))
rpt.append("|------|------|------|------|------|")

for col, (label, _) in METRICS.items():
    if col not in df.columns:
        continue
    vals = []
    for gdf in [group_A, group_B, group_C, group_D]:
        if len(gdf) > 0:
            vals.append(fmt_mean_std(gdf[col]))
        else:
            vals.append("—")
    rpt.append(f"| {label} | {vals[0]} | {vals[1]} | {vals[2]} | {vals[3]} |")

# ---- Task 3 answers ----
rpt.append("\n---\n")
rpt.append("## 有效股票共同特征分析\n")

def compare_q(label, col, higher_is_better=True):
    """Compare A vs rest on a metric, return analysis text."""
    if col not in df.columns:
        return f"**{label}**: 数据不可用。\n"
    a_mean = group_A[col].mean()
    a_std = group_A[col].std()
    r_mean = group_rest[col].mean()
    r_std = group_rest[col].std()
    ratio = a_mean / r_mean if r_mean != 0 else float('inf')
    direction = "高于" if a_mean > r_mean else "低于"
    return (f"**{label}**: A组均值 {a_mean:.2f}, 其他组均值 {r_mean:.2f}, "
            f"A组{direction}其他组 {abs(ratio-1)*100:.0f}%。\n")

# Q1: More signals?
n1 = len(group_A)
n_rest = len(group_rest)
rpt.append(f"### 1. 有效股票是不是信号数量更多？\n")
rpt.append(compare_q("n_buy_ops", "n_buy_ops"))
rpt.append(f"A组信号量范围: {group_A['n_buy_ops'].min():.0f} - {group_A['n_buy_ops'].max():.0f}\n")
rpt.append(f"其他组信号量范围: {group_rest['n_buy_ops'].min():.0f} - {group_rest['n_buy_ops'].max():.0f}\n\n")

# Q2: Larger amounts?
rpt.append(f"### 2. 有效股票是不是大单金额更集中？\n")
rpt.append(compare_q("avg_amount_wan (单笔均额)", "avg_amount_wan"))
rpt.append(compare_q("total_buy_wan (总买入额)", "total_buy_wan"))
rpt.append("\n")

# Q3: Longer time span?
rpt.append(f"### 3. 有效股票是不是 time_span 更长（更像持续建仓）？\n")
rpt.append(compare_q("avg_time_span_min", "avg_time_span_min"))
rpt.append(compare_q("avg_order_count", "avg_order_count"))
rpt.append("\n")

# Q4: Lower VWAP deviation?
rpt.append(f"### 4. 有效股票是不是 VWAP 偏离更低（买入更温和）？\n")
rpt.append(compare_q("avg_vwap_deviation_pct", "avg_vwap_deviation_pct"))
# Check if D group has higher vwap deviation ("chasing")
if len(group_D) > 0 and "avg_vwap_deviation_pct" in df.columns:
    d_vwap = group_D["avg_vwap_deviation_pct"].mean()
    a_vwap = group_A["avg_vwap_deviation_pct"].mean()
    rpt.append(f"D组(无效) VWAP偏离均值: {d_vwap:.3f}% vs A组: {a_vwap:.3f}%\n")
rpt.append("\n")

# Q5: Industry/style bias — unavailable
rpt.append(f"### 5. 有效股票是否偏周期股/资源股/小盘股/高波动股？\n")
rpt.append("**行业数据暂不可用。** akshare 东方财富来源行业分类接口已被封，无法批量获取申万/中信行业分类。\n")
rpt.append("待接入替代数据源（如 tushare 或本地行业映射表）后可补充此分析。\n\n")
# Check if there's size/style signal from avg_amount_yuan
if "avg_amount_yuan" in df.columns:
    rpt.append(f"从**日均成交额**（流动性代理变量）来看：\n")
    rpt.append(compare_q("avg_amount_yuan", "avg_amount_yuan"))
    rpt.append(f"A组日均成交额范围: {group_A['avg_amount_yuan'].min()/1e8:.1f}亿 - {group_A['avg_amount_yuan'].max()/1e8:.1f}亿\n")
    rpt.append(f"其他组日均成交额范围: {group_rest['avg_amount_yuan'].min()/1e8:.1f}亿 - {group_rest['avg_amount_yuan'].max()/1e8:.1f}亿\n")
rpt.append("\n")

# Q6: Chasing feature in D group
rpt.append('### 6. 无效股票是否存在"买入即追高"的特征？\n')
if len(group_D) > 0:
    rpt.append(f"D组({len(group_D)}只) 特征:\n")
    for col, label in [("avg_vwap_deviation_pct", "VWAP偏离(%)"),
                        ("avg_fwd_5d", "Fwd5d(%)"),
                        ("avg_universe_excess_fwd_5d", "UniverseExcess5d(%)")]:
        if col in df.columns:
            rpt.append(f"- {label}: {group_D[col].mean():.3f} (A组: {group_A[col].mean():.3f})\n")
    rpt.append(f"\n")
    # List D group stocks with their key metrics
    rpt.append(f"D组股票明细:\n\n")
    rpt.append(f"| 代码 | Fwd5d | Win5d | Excess5d | VWAP偏离 | TimeSpan | OrderCnt |")
    rpt.append(f"|------|-------|-------|----------|----------|----------|----------|")
    for _, row in group_D.sort_values("avg_fwd_5d").iterrows():
        rpt.append(f"| {row['stock_code']} | {row['avg_fwd_5d']:+.2f}% | {row['win_5d']:.1%} | "
                   f"{row['avg_universe_excess_fwd_5d']:+.2f}% | {row['avg_vwap_deviation_pct']:.3f} | "
                   f"{row['avg_time_span_min']:.1f} | {int(row['avg_order_count'])} |")
rpt.append("\n")

# ---- Group distribution analysis ----
rpt.append("---\n")
rpt.append("## 分组分布特征\n")

# Correlation between key DBSCAN features and effectiveness
rpt.append("### DBSCAN 聚类特征与信号有效性的关系\n")
rpt.append("| 特征 | A组均值 | C+B组均值 | D组均值 | 趋势 |")
rpt.append("|------|---------|-----------|---------|------|")
for col, label in [
    ("avg_order_count", "聚类委托数"),
    ("avg_time_span_min", "聚类时长(min)"),
    ("avg_vwap_deviation_pct", "VWAP偏离(%)"),
    ("avg_amount_wan", "单笔均额(万)"),
]:
    if col not in df.columns:
        continue
    a_v = group_A[col].mean()
    bc_v = df[df["candidate_group"].isin(["B_positive_but_weak", "C_beta_only"])][col].mean()
    d_v = group_D[col].mean() if len(group_D) > 0 else float('nan')
    # determine trend
    trend = ""
    if a_v > bc_v > d_v:
        trend = "递减 ↓"
    elif a_v < bc_v < d_v:
        trend = "递增 ↑"
    elif a_v > bc_v and a_v > d_v:
        trend = "A组最高 ∧"
    elif a_v < bc_v and a_v < d_v:
        trend = "A组最低 ∨"
    else:
        trend = "非单调"
    rpt.append(f"| {label} | {a_v:.2f} | {bc_v:.2f} | {d_v:.2f} | {trend} |")

# ---- Stability analysis ----
rpt.append("\n---\n")
rpt.append("## Stability 分析\n")
rpt.append(f"Stability = avg_universe_excess_fwd_5d / std(universe_excess_fwd_5d)\n\n")
rpt.append(f"- A组 stability 均值: {group_A['stability'].mean():.4f}\n")
rpt.append(f"- 其他组 stability 均值: {group_rest['stability'].mean():.4f}\n")
rpt.append(f"- 全样本 stability 中位数: {df['stability'].median():.4f}\n")
rpt.append(f"- A组 stability 全部 > 0 (筛选条件)\n")
rpt.append(f"- 其他组中 stability > 0 的有 {(group_rest['stability'] > 0).sum()}/{len(group_rest)} 只\n")
rpt.append(f"- 其他组中 stability <= 0 的有 {(group_rest['stability'] <= 0).sum()}/{len(group_rest)} 只\n")

# ---- Signal dilution analysis ----
rpt.append("\n---\n")
rpt.append("## 信号稀释分析：为什么扩容失败？\n")
rpt.append(f"从 A_valid ({len(group_A)}只) → Top100 ({len(df)}只)：\n\n")
rpt.append(f"| 指标 | A组 | 全部Top100 | 稀释幅度 |")
rpt.append(f"|------|-----|-----------|----------|")
for col, label in [
    ("avg_fwd_5d", "Fwd5d(%)"),
    ("win_5d", "Win5d"),
    ("avg_universe_excess_fwd_5d", "Excess5d(%)"),
]:
    if col not in df.columns:
        continue
    a_v = group_A[col].mean()
    all_v = df[col].mean()
    dilution = (all_v - a_v) / abs(a_v) * 100 if a_v != 0 else 0
    rpt.append(f"| {label} | {a_v:+.3f} | {all_v:+.3f} | {dilution:+.0f}% |")

rpt.append(f"\n稀释来源:\n")
rpt.append(f"- C_beta_only ({len(group_C)}只): 有正收益但无超额，纯市场beta\n")
rpt.append(f"- D_invalid ({len(group_D)}只): DBSCAN信号在这些股票上完全无效\n")
rpt.append(f"- B_positive_but_weak ({len(group_B)}只): 有正收益但不够稳健\n\n")
rpt.append(f"**C+D 组合计 {len(group_C)+len(group_D)} 只（{((len(group_C)+len(group_D))/len(df)*100):.0f}%），是信号稀释的主因。**\n")

# ---- Production candidate check ----
rpt.append("\n---\n")
rpt.append("## Production Candidates 验证\n")
pc_in_summary = df[df["is_production_candidate"]]
rpt.append(f"Production candidates 在 Top100 summary 中的分组情况:\n")
for g in ["A_valid_candidate", "B_positive_but_weak", "C_beta_only", "D_invalid"]:
    n = (pc_in_summary["candidate_group"] == g).sum()
    if n > 0:
        rpt.append(f"- {g}: {n} 只\n")
rpt.append(f"\n所有 13 只 production candidates 都应落入 A 组。如有不一致，说明分组规则或 candidate 筛选有差异。\n")

with open(OUT_REPORT, "w") as f:
    f.write("\n".join(rpt))
print(f"Report saved: {OUT_REPORT}")

# ---- Task 4: Selection rules ----
rules = []
rules.append("# DBSCAN BUY 候选股票筛选规则 (v1)\n")
rules.append(f"生成时间: {pd.Timestamp.now().strftime('%Y-%m-%d')}\n")
rules.append("> 基于 Top100 DBSCAN BUY 信号有效性分析自动生成。\n")
rules.append("> 这些是候选规则，需 OOS 验证后才能用于实盘。\n")
rules.append("---\n")

# Derive rule thresholds from A vs rest comparison
rules.append("## 规则来源\n")
rules.append(f"A组({len(group_A)}只) vs 其他组({len(group_rest)}只) 在各维度上的差异，取 A 组 25-75 分位作为建议阈值。\n")

rules.append("\n## 一级规则（必须满足）\n")
rules.append("| 条件 | 阈值 | 依据 |")
rules.append("|------|------|------|")

# n_buy_ops
a_min = group_A["n_buy_ops"].min()
rules.append(f"| n_buy_ops | >= {a_min:.0f} | A组最小值，确保信号量充足 |")

# win_5d
rules.append(f"| win_5d | > 52% | 扩容标准，A组均值 {group_A['win_5d'].mean():.1%} |")

# universe_excess
rules.append(f"| avg_universe_excess_fwd_5d | > 0 | 必须有正向选股能力，A组均值 {group_A['avg_universe_excess_fwd_5d'].mean():.2f}% |")

# stability
rules.append(f"| stability | > 0 | 收益稳定性为正，A组均值 {group_A['stability'].mean():.4f} |")

# avg_fwd_5d
rules.append(f"| avg_fwd_5d | > 0 | 绝对收益为正，A组均值 {group_A['avg_fwd_5d'].mean():.2f}% |")

rules.append("\n## 二级规则（建议满足）\n")
rules.append("| 条件 | 建议范围 | 依据 |")
rules.append("|------|----------|------|")

# Derive ranges from A group quartiles
for col, label, fmt_val in [
    ("avg_order_count", "聚类平均委托数", lambda v: f"{v:.0f}"),
    ("avg_time_span_min", "聚类平均时长(分钟)", lambda v: f"{v:.1f}"),
    ("avg_vwap_deviation_pct", "VWAP偏离(%)", lambda v: f"{v:.3f}"),
    ("avg_amount_wan", "单笔BUY均额(万)", lambda v: f"{v:.1f}"),
]:
    if col not in group_A.columns:
        continue
    q25 = group_A[col].quantile(0.25)
    q75 = group_A[col].quantile(0.75)
    rules.append(f"| {label} | {fmt_val(q25)} - {fmt_val(q75)} | A组25-75分位范围 |")

rules.append("\n## 规则应用说明\n")
rules.append("1. **一级规则**全部满足 → 进入 production candidate 候选池\n")
rules.append("2. **二级规则**作为参考，不满足不排除，但需标记为\"需人工审核\"\n")
rules.append("3. 当前规则基于 2025 年全年样本内数据，需 2026 H1 OOS 验证\n")
rules.append("4. 规则仅适用于 DBSCAN BUY 信号（v6 流水线），不适用于其他信号类型\n")
rules.append(f"5. 在 Top100 池中，一级规则筛选出 {len(group_A)} 只（{len(group_A)/len(df)*100:.0f}%）\n")

# Add note about what didn't work
rules.append("\n## 已知无效的信号特征\n")
rules.append("以下特征的股票在 Top100 中 DBSCAN BUY 信号无效:\n")
if len(group_D) > 0:
    rules.append(f"- **D 组({len(group_D)}只)**: avg_fwd_5d <= 0，信号完全没有预测能力\n")
rules.append(f"- **C 组({len(group_C)}只)**: avg_fwd_5d > 0 但 universe_excess <= 0，收益来自市场beta而非alpha\n")
rules.append(f"- **B 组({len(group_B)}只)**: 正收益但不满足全部条件，信号不够稳健\n")
rules.append(f"\n**C+D 组共 {len(group_C)+len(group_D)} 只（{((len(group_C)+len(group_D))/len(df)*100):.0f}%），扩容的主要噪音来源。**\n")

rules.append("\n## 下一步\n")
rules.append("1. 用本规则在更大的股票池（如 ZZ1000 全部 736 只）中筛选候选\n")
rules.append("2. 对筛选出的候选做 2026 H1 OOS 验证\n")
rules.append("3. 验证通过后，分析候选股票的共同行业/市值特征\n")
rules.append("4. 接入行业分类数据后，补充行业维度的筛选规则\n")

with open(OUT_RULES, "w") as f:
    f.write("\n".join(rules))
print(f"Rules saved: {OUT_RULES}")
print("Sprint 0.7 analysis complete.")
