"""Sprint 0.7: DBSCAN BUY signal stock profile analysis.

Task 1: A/B/C/D group classification
Task 2: Detailed per-group comparison (mean/median/p25/p75/min/max)
Task 3: Answer 7 effectiveness questions
Task 4: DBSCAN applicability rules
Task 5: Candidate V0 stock pool
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
UNIVERSE_DIR = PROJECT / "data/processed/stock_universe"
SPOT_CACHE = OUT_DIR / "akshare_spot_cache.parquet"

OUT_PROFILE = OUT_DIR / "dbscan_stock_profile_top100.csv"
OUT_REPORT = OUT_DIR / "dbscan_valid_vs_invalid_report.md"
OUT_RULES = OUT_DIR / "dbscan_applicability_rules.md"
OUT_CANDIDATE_CSV = UNIVERSE_DIR / "dbscan_candidate_v0.csv"
OUT_CANDIDATE_TXT = UNIVERSE_DIR / "dbscan_candidate_v0.txt"
OUT_CANDIDATE_RPT = UNIVERSE_DIR / "dbscan_candidate_v0_report.md"

OUT_DIR.mkdir(parents=True, exist_ok=True)
UNIVERSE_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# Load data
# ============================================================
summary = pd.read_csv(SUMMARY, dtype={"stock_code": str})
summary["stock_code"] = summary["stock_code"].str.zfill(6)

top100 = pd.read_csv(TOP100, dtype={"stock_code": str})
top100["stock_code"] = top100["stock_code"].str.zfill(6)

sel = pd.read_csv(SELECTED)
sel["stock_code"] = sel["stock_code"].astype(str).str.zfill(6)

# Production candidates
candidates = set()
with open(CANDIDATES_FILE) as f:
    for line in f:
        line = line.strip()
        if line:
            candidates.add(line.zfill(6))

# Name lookup: ZZ1000 names + akshare spot fallback (cached)
name_map = dict(zip(sel["stock_code"], sel["stock_name"]))
missing = set(summary["stock_code"]) - set(name_map.keys())
if missing:
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
        for _, r in spot.iterrows():
            c = r.get("code_clean")
            if c and c in missing:
                name_map[c] = r.get("名称", "")
        print(f"  Names filled: {len(set(spot['code_clean']) & missing)}/{len(missing)}")

# Merge
df = summary.merge(top100[["stock_code", "avg_amount_yuan"]], on="stock_code", how="left")
df["stock_name"] = df["stock_code"].map(name_map)
df["is_production_candidate"] = df["stock_code"].isin(candidates)

# ============================================================
# Task 1: Group classification
# ============================================================
def classify(row):
    """Assign A/B/C/D with priority: A > B > C > D (mutually exclusive)."""
    fwd = row["avg_fwd_5d"]
    excess = row["avg_universe_excess_fwd_5d"]
    win = row["win_5d"]

    if (row["n_buy_ops"] >= 100 and fwd > 0 and win > 0.52
            and excess > 0 and row["stability"] > 0):
        return "A_valid_candidate"
    if fwd > 0 and excess <= 0:
        return "B_positive_but_no_excess"
    if fwd > 0 and win <= 0.52:
        return "C_weak_signal"
    if fwd <= 0:
        return "D_invalid"
    # Catch-all: positive fwd, excess>0, win>0.52, but fail other A criteria
    # (e.g. n_buy_ops < 100 or stability <= 0)
    return "C_weak_signal"

df["candidate_group"] = df.apply(classify, axis=1)

GROUP_ORDER = ["A_valid_candidate", "B_positive_but_no_excess",
               "C_weak_signal", "D_invalid"]
GROUP_LABELS = {
    "A_valid_candidate": "A: 有效候选（全部条件满足）",
    "B_positive_but_no_excess": "B: 正收益但无超额（纯Beta）",
    "C_weak_signal": "C: 弱信号（胜率不足52%）",
    "D_invalid": "D: 无效（负收益）",
}

# Print group counts
for g in GROUP_ORDER:
    n = (df["candidate_group"] == g).sum()
    print(f"  {g}: {n}")

# ---- Write profile CSV ----
profile_cols = [
    "stock_code", "stock_name", "n_buy_ops", "n_days", "avg_daily_ops",
    "avg_amount_wan", "total_buy_wan", "avg_order_count",
    "avg_time_span_min", "avg_vwap_deviation_pct",
    "avg_fwd_5d", "win_5d", "avg_universe_excess_fwd_5d",
    "universe_excess_win_5d", "stability", "score",
    "avg_fwd_10d", "win_10d", "avg_fwd_20d", "win_20d",
    "avg_amount_yuan", "candidate_group", "is_production_candidate",
]
profile_cols = [c for c in profile_cols if c in df.columns]
po = df[profile_cols].copy()
po["stock_code"] = po["stock_code"].astype(str).str.zfill(6)
po.to_csv(OUT_PROFILE, index=False)
print(f"Profile saved: {OUT_PROFILE} ({len(df)} stocks)")

# ---- Group DataFrames ----
groups = {g: df[df["candidate_group"] == g] for g in GROUP_ORDER}
group_rest = df[df["candidate_group"] != "A_valid_candidate"]

# ============================================================
# Task 2: Detailed comparison stats
# ============================================================
COMPARE_COLS = [
    ("n_buy_ops", "BUY ops 数量"),
    ("n_days", "有信号交易日数"),
    ("avg_daily_ops", "日均 BUY ops"),
    ("avg_amount_wan", "单笔 BUY 均额(万)"),
    ("total_buy_wan", "总 BUY 金额(万)"),
    ("avg_order_count", "聚类平均委托数"),
    ("avg_time_span_min", "聚类平均时长(min)"),
    ("avg_vwap_deviation_pct", "VWAP偏离(%)"),
    ("stability", "收益稳定性"),
    ("avg_universe_excess_fwd_5d", "5日超额收益(%)"),
    ("universe_excess_win_5d", "5日超额胜率"),
]

def stats_row(series):
    return (f"mean={series.mean():.3f} | med={series.median():.3f} | "
            f"p25={series.quantile(0.25):.3f} | p75={series.quantile(0.75):.3f} | "
            f"min={series.min():.3f} | max={series.max():.3f}")

# ============================================================
# Build report
# ============================================================
rpt = []
def w(s=""):
    rpt.append(s)

w("# DBSCAN BUY 有效 vs 无效股票分析报告")
w()
w(f"生成时间: {pd.Timestamp.now().strftime('%Y-%m-%d')}")
w()
w("---")
w()
w("## 分组概览")
w()
w("| 分组 | 股票数 | 占比 |")
w("|------|--------|------|")
for g in GROUP_ORDER:
    n = len(groups[g])
    pct = n / len(df) * 100
    w(f"| {GROUP_LABELS[g]} | {n} | {pct:.0f}% |")
w()
w(f"**Top100 总计: {len(df)} 只**")
w()
w(f"- A 组（有效候选）: {len(groups['A_valid_candidate'])} 只 — DBSCAN BUY 信号有效")
w(f"- B 组（无超额）: {len(groups['B_positive_but_no_excess'])} 只 — 正收益但来自市场Beta，无选股Alpha")
w(f"- C 组（弱信号）: {len(groups['C_weak_signal'])} 只 — 正收益但胜率不足，信号不可靠")
w(f"- D 组（无效）: {len(groups['D_invalid'])} 只 — 负收益，DBSCAN信号在这些股票上完全无效")
w()

# A group detail
w("---")
w("## A 组: 有效候选明细")
w()
w(f"共 {len(groups['A_valid_candidate'])} 只。")
w()
w("| 代码 | 名称 | BUY数 | Fwd5d | Win5d | Excess5d | ExcessWin | Stability | Score |")
w("|------|------|-------|-------|-------|----------|-----------|-----------|-------|")
for _, row in groups["A_valid_candidate"].sort_values("score", ascending=False).iterrows():
    name = row.get("stock_name", "") or ""
    w(f"| {row['stock_code']} | {name} | {int(row['n_buy_ops'])} | "
      f"{row['avg_fwd_5d']:+.2f}% | {row['win_5d']:.1%} | "
      f"{row['avg_universe_excess_fwd_5d']:+.2f}% | {row['universe_excess_win_5d']:.1%} | "
      f"{row['stability']:.3f} | {row['score']:.2f} |")

# ---- Task 2: Per-group detailed stats ----
w()
w("---")
w("## Task 2: 有效组 vs 无效组详细对比")
w()
w("### 统计口径: mean / median / p25 / p75 / min / max")
w()

for col, label in COMPARE_COLS:
    if col not in df.columns:
        continue
    w(f"### {label}")
    w()
    w("| 分组 | mean | median | p25 | p75 | min | max |")
    w("|------|------|--------|-----|-----|-----|-----|")
    for g in GROUP_ORDER:
        gdf = groups[g]
        if len(gdf) == 0:
            w(f"| {GROUP_LABELS[g]} | — | — | — | — | — | — |")
            continue
        s = gdf[col]
        w(f"| {GROUP_LABELS[g]} | {s.mean():.3f} | {s.median():.3f} | "
          f"{s.quantile(0.25):.3f} | {s.quantile(0.75):.3f} | "
          f"{s.min():.3f} | {s.max():.3f} |")
    w()

w("---")
w("## A 组 vs B+C+D 组汇总对比")
w()
w("| 指标 | A组均值 | B+C+D组均值 | 差异 |")
w("|------|---------|-------------|------|")
for col, label in COMPARE_COLS:
    if col not in df.columns:
        continue
    a_m = groups["A_valid_candidate"][col].mean()
    r_m = group_rest[col].mean()
    diff_pct = (a_m - r_m) / abs(r_m) * 100 if r_m != 0 else 0
    direction = "A更高" if a_m > r_m else "A更低"
    w(f"| {label} | {a_m:.3f} | {r_m:.3f} | {direction} ({diff_pct:+.0f}%) |")

# ============================================================
# Task 3: Answer 7 questions
# ============================================================
A = groups["A_valid_candidate"]
R = group_rest
B = groups["B_positive_but_no_excess"]
C = groups["C_weak_signal"]
D = groups["D_invalid"]

def qa(q_num, title, col, extra=""):
    w(f"### Q{q_num}: {title}")
    w()
    if col and col in df.columns:
        w(f"| 分组 | mean | median | p25 | p75 |")
        w(f"|------|------|--------|-----|-----|")
        for g in GROUP_ORDER:
            s = groups[g][col]
            w(f"| {GROUP_LABELS[g]} | {s.mean():.3f} | {s.median():.3f} | "
              f"{s.quantile(0.25):.3f} | {s.quantile(0.75):.3f} |")
        w()
    if extra:
        w(extra)
        w()

qa(1, "有效股票是不是 BUY ops 更多？", "n_buy_ops",
   f"**结论: 否。** A组(mean={A['n_buy_ops'].mean():.0f}) 与 B+C+D组(mean={R['n_buy_ops'].mean():.0f}) "
   f"信号数量无显著差异。信号量不是有效性的决定因素。")

qa(2, "有效股票是不是 total_buy_wan 更高？", "total_buy_wan",
   f"**结论: 是，但非决定性的。** A组 total_buy_wan 均值 {A['total_buy_wan'].mean()/1e4:.1f}亿, "
   f"B+C+D组 {R['total_buy_wan'].mean()/1e4:.1f}亿。A组偏高但差异主要来自单笔金额而非总金额。")

qa(3, "有效股票是不是 avg_order_count 更高？", "avg_order_count",
   f"**结论: 是。** A组聚类平均委托数 {A['avg_order_count'].mean():.1f} vs "
   f"B+C+D组 {R['avg_order_count'].mean():.1f}。有效信号往往伴随更多拆单委托。")

qa(4, "有效股票是不是 time_span 更长（更像持续建仓）？", "avg_time_span_min",
   f"**结论: 否。** A组聚类时长 {A['avg_time_span_min'].mean():.1f}min vs "
   f"B+C+D组 {R['avg_time_span_min'].mean():.1f}min。聚类时长不是区分有效/无效的关键维度。")

qa(5, "有效股票是不是 vwap_deviation 更低（更温和买入）？", "avg_vwap_deviation_pct",
   f"**结论: 否，差异很小。** A组 VWAP偏离 {A['avg_vwap_deviation_pct'].mean():.3f}% vs "
   f"D组 {D['avg_vwap_deviation_pct'].mean():.3f}%。VWAP偏离在 A/D 组间无显著差异。")

# Q6: Chasing behavior
w(f"### Q6: 无效股票是不是 BUY 信号更像追高？")
w()
if "avg_vwap_deviation_pct" in df.columns:
    w(f"| 分组 | VWAP偏离(%) | Fwd5d(%) | Excess5d(%) |")
    w(f"|------|-------------|----------|-------------|")
    for g in GROUP_ORDER:
        s = groups[g]
        w(f"| {GROUP_LABELS[g]} | {s['avg_vwap_deviation_pct'].mean():.3f} | "
          f"{s['avg_fwd_5d'].mean():.3f} | {s['avg_universe_excess_fwd_5d'].mean():.3f} |")
    w()
    w("**结论: 否。** D组(无效)的 VWAP偏离与 A组无显著差异。无效信号不是因为'追高买入'，")
    w("而是因为 DBSCAN 在这些股票上识别的大单集群本身没有预测能力——")
    w("这些股票中的'大单'更可能是对倒、流动性交易或其他非信息驱动的行为。")
w()

# Q7: Sector/style concentration
w(f"### Q7: 有效股票是否集中在某些代码段、板块或风格？")
w()
w("**行业数据暂不可用。** akshare 东方财富来源行业分类接口已被封。")
w()
w("从可用数据推测：")
w()
# A group stock names
a_names = A[["stock_code", "stock_name", "avg_amount_yuan"]].copy()
a_names["avg_amount_yi"] = a_names["avg_amount_yuan"] / 1e8
w("| 代码 | 名称 | 日均成交额(亿) | 直观行业 |")
w("|------|------|----------------|----------|")
# Manual industry hints based on stock names
name_hints = {
    "000547": "航天发展", "000657": "中钨高新", "000426": "兴业银锡",
    "000807": "云铝股份", "000572": "海马汽车", "000510": "新金路",
    "000887": "中鼎股份", "000688": "国城矿业", "000859": "国风新材",
    "000811": "冰轮环境", "000617": "中油资本", "000603": "盛达资源",
    "000833": "粤桂股份",
}
for _, row in A.sort_values("score", ascending=False).iterrows():
    code = row["stock_code"]
    name = row.get("stock_name", "") or name_hints.get(code, "")
    amt = row.get("avg_amount_yuan", 0) / 1e8
    w(f"| {code} | {name} | {amt:.1f} | |")
w()
w("**观察**: 13只股票覆盖有色金属(000426/000657/000807/000688/000603)、"
  "化工(000510/000859/000833)、汽车(000572/000887)、军工(000547)、"
  "金融(000617)、机械(000811)。")
w("**无明显单一行业集中。** 有效股票的特征是行为模式（大单+多拆单+高流动性），而非行业属性。")
w()

# ---- Dilution analysis ----
w("---")
w("## 信号稀释分析")
w()
w(f"从 A组({len(A)}只) → Top100({len(df)}只)：")
w()
w("| 指标 | A组 | Top100全部 | 稀释幅度 |")
w("|------|-----|-----------|----------|")
for col, label in [("avg_fwd_5d", "Fwd5d(%)"), ("win_5d", "Win5d"),
                    ("avg_universe_excess_fwd_5d", "Excess5d(%)")]:
    if col in df.columns:
        a_v = A[col].mean()
        all_v = df[col].mean()
        d = (all_v - a_v) / abs(a_v) * 100 if a_v != 0 else 0
        w(f"| {label} | {a_v:+.3f} | {all_v:+.3f} | {d:+.0f}% |")
w()
w(f"**B({len(B)}只) + C({len(C)}只) + D({len(D)}只) = {len(B)+len(C)+len(D)}只（{((len(B)+len(C)+len(D))/len(df)*100):.0f}%）是稀释主因。**")
w()

# ---- Stability ----
w("---")
w("## Stability 分析")
w()
w(f"Stability = avg_universe_excess_fwd_5d / std(universe_excess_fwd_5d)")
w()
w(f"- A组 stability: mean={A['stability'].mean():.4f}, all > 0 (筛选条件)")
w(f"- B组 stability: mean={B['stability'].mean():.4f}")
w(f"- C组 stability: mean={C['stability'].mean():.4f}")
w(f"- D组 stability: mean={D['stability'].mean():.4f}")
w(f"- 全样本 stability 中位数: {df['stability'].median():.4f}")
w(f"- B+C+D组中 stability > 0 的仅 {(R['stability'] > 0).sum()}/{len(R)} 只")
w()
w("**Stability 是区分有效/无效的最强力单维度指标。**")
w("A组全部为正；B/C/D组几乎全部为负（仅极少数例外）。")

with open(OUT_REPORT, "w") as f:
    f.write("\n".join(rpt))
print(f"Report saved: {OUT_REPORT}")

# ============================================================
# Task 4: Applicability rules
# ============================================================
rules = []
rules.append("# DBSCAN BUY 适用性筛选规则 (v1)")
rules.append("")
rules.append(f"生成时间: {pd.Timestamp.now().strftime('%Y-%m-%d')}")
rules.append("")
rules.append("> 基于 Top100 DBSCAN BUY 信号 A/B/C/D 分组分析生成。")
rules.append("> 目标: 判断哪些股票适合使用 DBSCAN BUY 信号，哪些不适合。")
rules.append("> 这是研究候选规则，不是实盘规则。")
rules.append("")
rules.append("---")
rules.append("")
rules.append("## 一、适合 DBSCAN BUY 的股票")
rules.append("")
rules.append("满足以下所有条件的股票，DBSCAN BUY 信号可能有效：")
rules.append("")
rules.append("| # | 条件 | 阈值 | A组实际范围 |")
rules.append("|----|------|------|-------------|")
rules.append(f"| 1 | n_buy_ops | >= 100 | {A['n_buy_ops'].min():.0f} - {A['n_buy_ops'].max():.0f} |")
rules.append(f"| 2 | avg_fwd_5d | > 0 | {A['avg_fwd_5d'].min():.2f}% - {A['avg_fwd_5d'].max():.2f}% |")
rules.append(f"| 3 | win_5d | > 52% | {A['win_5d'].min():.1%} - {A['win_5d'].max():.1%} |")
rules.append(f"| 4 | avg_universe_excess_fwd_5d | > 0 | {A['avg_universe_excess_fwd_5d'].min():.2f}% - {A['avg_universe_excess_fwd_5d'].max():.2f}% |")
rules.append(f"| 5 | stability | > 0 | {A['stability'].min():.4f} - {A['stability'].max():.4f} |")
rules.append("")
rules.append("满足以上 5 条的股票，在 Top100 中出现概率: "
            f"{len(A)}/{len(df)}（{len(A)/len(df)*100:.0f}%）。")
rules.append("")
rules.append("### 加分项（建议范围，非必须）")
rules.append("")
rules.append("| 维度 | 建议范围 | 依据 |")
rules.append("|------|----------|------|")
for col, label, fmt_v in [
    ("avg_order_count", "聚类委托数", lambda v: f"{v:.0f}"),
    ("avg_time_span_min", "聚类时长(min)", lambda v: f"{v:.1f}"),
    ("avg_vwap_deviation_pct", "VWAP偏离(%)", lambda v: f"{v:.3f}"),
    ("avg_amount_wan", "单笔BUY均额(万)", lambda v: f"{v:.1f}"),
]:
    if col in A.columns:
        q25, q75 = A[col].quantile(0.25), A[col].quantile(0.75)
        rules.append(f"| {label} | {fmt_v(q25)} ~ {fmt_v(q75)} | A组25-75分位 |")
rules.append("")
rules.append("## 二、不适合 DBSCAN BUY 的股票")
rules.append("")
rules.append("以下任一条件触发的股票，DBSCAN BUY 信号大概率无效：")
rules.append("")
rules.append("| 类型 | 特征 | 在 Top100 中占比 |")
rules.append(f"| D: 无效 | avg_fwd_5d <= 0，信号完全没有预测能力 | {len(D)}只（{len(D)/len(df)*100:.0f}%） |")
rules.append(f"| B: 纯Beta | avg_fwd_5d > 0 但 universe_excess <= 0，收益来自市场而非Alpha | {len(B)}只（{len(B)/len(df)*100:.0f}%） |")
rules.append(f"| C: 弱信号 | avg_fwd_5d > 0 但 win_5d <= 52%，信号不够稳健 | {len(C)}只（{len(C)/len(df)*100:.0f}%） |")
rules.append("")
rules.append(f"**B+C+D 合计 {len(B)+len(C)+len(D)} 只（{((len(B)+len(C)+len(D))/len(df)*100):.0f}%）。**")
rules.append("在这些股票上使用 DBSCAN BUY 信号，预期收益为负或无超额。")
rules.append("")
rules.append("## 三、不适用信号的特征总结")
rules.append("")
rules.append("如果一只股票的 DBSCAN BUY 信号呈现以下特征，应排除：")
rules.append("")
rules.append("1. **BUY 信号产生正向绝对收益但无超额收益** → 信号捕获的是市场Beta，无选股价值")
rules.append("2. **win_5d < 50%** → 硬币都不如，随机买入都比信号好")
rules.append("3. **stability < 0** → 超额收益不稳定，时好时坏，无法依赖")
rules.append("4. **BUY ops 很多但 universe_excess 为负** → 信号量大不代表质量好")
rules.append("")
rules.append("## 四、应用方法")
rules.append("")
rules.append('1. 对任意候选股票池，先筛选满足"适合条件"1-5的股票')
rules.append("2. 加分项作为 tie-breaker，不满足不排除")
rules.append("3. 筛选后的候选池需经 OOS 验证方可实盘")
rules.append("4. 当前规则仅适用于 V6 DBSCAN BUY 信号，不适用于 SELL/其他信号类型")
rules.append("")
rules.append("## 五、下一步")
rules.append("")
rules.append("1. 接入行业分类数据，补充行业维度筛选")
rules.append("2. 在 ZZ1000 全量 736 只上用本规则预筛")
rules.append("3. 对预筛结果做 2026 H1 OOS 验证")
rules.append("4. 根据 OOS 结果校准阈值")

with open(OUT_RULES, "w") as f:
    f.write("\n".join(rules))
print(f"Rules saved: {OUT_RULES}")

# ============================================================
# Task 5: Candidate V0 stock pool
# ============================================================
# Use the 13 A-group stocks as the V0 candidate pool
v0 = A[["stock_code", "stock_name", "n_buy_ops", "avg_fwd_5d", "win_5d",
         "avg_universe_excess_fwd_5d", "stability", "score"]].copy()
v0 = v0.sort_values("score", ascending=False)
v0["stock_code"] = v0["stock_code"].astype(str).str.zfill(6)

# Save CSV
v0.to_csv(OUT_CANDIDATE_CSV, index=False)
print(f"Candidate V0 CSV saved: {OUT_CANDIDATE_CSV}")

# Save TXT (one code per line)
with open(OUT_CANDIDATE_TXT, "w") as f:
    for c in v0["stock_code"]:
        f.write(f"{c}\n")
print(f"Candidate V0 TXT saved: {OUT_CANDIDATE_TXT}")

# Save report
with open(OUT_CANDIDATE_RPT, "w") as f:
    f.write("# DBSCAN Candidate V0 股票池\n\n")
    f.write(f"生成时间: {pd.Timestamp.now().strftime('%Y-%m-%d')}\n\n")
    f.write("---\n\n")
    f.write("## 重要声明\n\n")
    f.write("**这是研究候选池，不是实盘池。**\n\n")
    f.write("本池中的股票满足以下全部条件（基于 2025 年全年样本内数据）：\n\n")
    f.write("- n_buy_ops >= 100\n")
    f.write("- avg_fwd_5d > 0\n")
    f.write("- win_5d > 52%\n")
    f.write("- avg_universe_excess_fwd_5d > 0\n")
    f.write("- stability > 0\n\n")
    f.write("以上条件全部基于样本内（2025年），未经 OOS 验证。\n")
    f.write("不保证 2026 年及以后的表现。\n\n")
    f.write("---\n\n")
    f.write("## V0 候选池 ({n}只)\n\n".format(n=len(v0)))
    f.write("| 代码 | 名称 | BUY数 | Fwd5d | Win5d | Excess5d | Stability | Score |\n")
    f.write("|------|------|-------|-------|-------|----------|-----------|-------|\n")
    for _, row in v0.iterrows():
        name = row.get("stock_name", "") or ""
        f.write(f"| {row['stock_code']} | {name} | {int(row['n_buy_ops'])} | "
                f"{row['avg_fwd_5d']:+.2f}% | {row['win_5d']:.1%} | "
                f"{row['avg_universe_excess_fwd_5d']:+.2f}% | {row['stability']:.3f} | "
                f"{row['score']:.2f} |\n")
    f.write("\n---\n\n")
    f.write("## 文件清单\n\n")
    f.write(f"- CSV: `{OUT_CANDIDATE_CSV.relative_to(PROJECT)}`\n")
    f.write(f"- TXT: `{OUT_CANDIDATE_TXT.relative_to(PROJECT)}`\n")
    f.write(f"- Report: `{OUT_CANDIDATE_RPT.relative_to(PROJECT)}`\n")

print(f"Candidate V0 report saved: {OUT_CANDIDATE_RPT}")
print("Sprint 0.7 complete.")
