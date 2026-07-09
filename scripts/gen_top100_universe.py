"""Generate Top100 universe from DBSCAN 300 stocks sorted by liquidity."""
import pandas as pd
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
UNIVERSE_DIR = PROJECT / "data/processed/stock_universe"
DAILY_DIR = PROJECT / "data/daily"

# Load 300 DBSCAN stocks
sel300 = pd.read_csv(UNIVERSE_DIR / "selected_stocks.csv")
sel300["stock_code"] = sel300["stock"].astype(str).str.zfill(6)

# Compute avg_amount from daily cache
rows = []
for _, r in sel300.iterrows():
    code = r["stock_code"]
    cache = DAILY_DIR / f"{code}.parquet"
    if not cache.exists():
        continue
    df = pd.read_parquet(cache)
    df = df.sort_values("date").tail(60)
    if len(df) < 5:
        continue
    rows.append({
        "stock_code": code,
        "avg_amount_yuan": df["amount"].mean(),
        "avg_close_yuan": df["close"].mean(),
        "n_trade_days": len(df),
        "first_date": str(df["date"].iloc[0])[:10],
    })

df_all = pd.DataFrame(rows)

# ZZ1000 liquid for overlap check
zz1000 = pd.read_csv(UNIVERSE_DIR / "zz1000_liquid_selected.csv")
zz1000["code"] = zz1000["stock_code"].astype(str).str.zfill(6)
zz1000_set = set(zz1000["code"])

# Priority 25
p25_path = PROJECT / "config/v6_priority_stocks.txt"
p25_set = set()
if p25_path.exists():
    with open(p25_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                p25_set.add(line.zfill(6))

# Top 100 by amount
top100 = df_all.nlargest(100, "avg_amount_yuan").copy()
top100["rank"] = range(1, 101)
top100["in_zz1000"] = top100["stock_code"].isin(zz1000_set)
top100["in_priority25"] = top100["stock_code"].isin(p25_set)

overlap_p25 = top100[top100["in_priority25"]]
overlap_zz1000 = top100[top100["in_zz1000"]]

# Save
top100.to_csv(UNIVERSE_DIR / "zz1000_liquid_top100.csv", index=False)
with open(UNIVERSE_DIR / "zz1000_liquid_top100.txt", "w") as f:
    for c in top100["stock_code"]:
        f.write(f"{c}\n")

# Report
with open(UNIVERSE_DIR / "zz1000_liquid_top100_report.md", "w") as f:
    f.write("# DBSCAN Top100 Universe (按成交额)\n\n")
    f.write(f"生成时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}\n\n")
    f.write("## 重要说明\n\n")
    f.write("**此Top100来自已有DBSCAN输出的300只股票，按日均成交额降序取Top100。**\n\n")
    f.write("原计划从ZZ1000筛选池取Top100，但ZZ1000 Top100与level2_ops零重叠（仅45只ZZ1000股票有ops数据）。\n")
    f.write("需Windows重新跑DBSCAN才能覆盖ZZ1000股票池。当前先用DBSCAN 300只中的Top100按成交额跑验证。\n\n")
    f.write("## 概览\n\n")
    f.write(f"| 指标 | 数值 |\n")
    f.write(f"|------|------|\n")
    f.write(f"| Top100 股票数 | {len(top100)} |\n")
    f.write(f"| 与 Priority 25 重叠 | {len(overlap_p25)} |\n")
    f.write(f"| 与 ZZ1000 重叠 | {len(overlap_zz1000)} |\n")
    f.write(f"| 日均成交额范围 | {top100['avg_amount_yuan'].min()/1e8:,.1f}亿 - {top100['avg_amount_yuan'].max()/1e8:,.1f}亿 |\n")
    f.write(f"| 日均成交额中位数 | {top100['avg_amount_yuan'].median()/1e8:,.1f}亿 |\n\n")
    f.write(f"## 与 Priority 25 重叠\n\n")
    if len(overlap_p25) > 0:
        f.write(f"重叠 {len(overlap_p25)} 只: {', '.join(sorted(overlap_p25['stock_code']))}\n\n")
    else:
        f.write("无重叠。\n\n")
    f.write(f"## Top100 清单\n\n")
    f.write(f"| 排名 | 代码 | 日均成交额(亿) | 均价 | 天数 | ZZ1000 | Pri25 |\n")
    f.write(f"|------|------|---------------|------|------|--------|-------|\n")
    for _, r in top100.iterrows():
        f.write(f"| {r['rank']} | {r['stock_code']} | {r['avg_amount_yuan']/1e8:,.1f} | "
                f"{r['avg_close_yuan']:.1f} | {int(r['n_trade_days'])} | "
                f"{'Y' if r['in_zz1000'] else ''} | "
                f"{'Y' if r['in_priority25'] else ''} |\n")

print(f"Top100: {len(top100)} stocks")
print(f"  Priority 25 overlap: {len(overlap_p25)}")
print(f"  ZZ1000 overlap: {len(overlap_zz1000)}")
print(f"  Amount: {top100['avg_amount_yuan'].min()/1e8:,.1f} - {top100['avg_amount_yuan'].max()/1e8:,.1f}亿")
print(f"Done.")
