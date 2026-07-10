"""构建 Universe_A / Universe_B / Universe_C 并生成报告。

用法:
    python3 scripts/build_universe.py

产出:
    data/processed/universe/Universe_A.parquet  (fast_debug, 300 只)
    data/processed/universe/Universe_B.parquet  (小盘主研究池, ZZ1000 过滤后)
    data/processed/universe/Universe_C.parquet  (Level-2 可用池)
    reports/universe_report.md
    signal_zoo/registry/universe_registry.csv    (重写为 A/B/C 三行)
"""
import sys
from pathlib import Path
import yaml
import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))
from src.registry import universe_registry as reg

CONFIG = PROJECT / "configs" / "universe.yaml"
REPORT = PROJECT / "reports" / "universe_report.md"


def read_code_file(path: Path) -> list[str]:
    """读取股票池文件（每行 'code' 或 'code status'）。"""
    codes = []
    for line in open(path):
        parts = line.strip().split()
        if parts:
            codes.append(parts[0].zfill(6))
    return codes


def signal_covered_codes() -> set[str]:
    """已生成 Alpha191 信号覆盖的股票（用于标注 Universe_C 是否可直接验证）。"""
    sig_dir = PROJECT / "data" / "processed" / "signals" / "price_alpha191_full"
    sf = sorted(sig_dir.glob("signal*.parquet"))
    if not sf:
        return set()
    s = pd.read_parquet(sf[0])
    return set(s["stock_code"].astype(str).str.zfill(6).unique())


def monthly_coverage(members: list[str], start: str, end: str) -> pd.Series:
    """每月有交易的成员数量。"""
    counts = {}
    months = pd.period_range(start=start, end=end, freq="M")
    per_month = {str(m): 0 for m in months}
    for code in members:
        p = PROJECT / "data" / "daily" / f"{code}.parquet"
        if not p.exists():
            continue
        df = pd.read_parquet(p)
        df["date"] = pd.to_datetime(df["date"])
        df = df[(df["date"] >= start) & (df["date"] <= end) & (df["volume"] > 0)]
        for m in df["date"].dt.to_period("M").astype(str).unique():
            if m in per_month:
                per_month[m] += 1
    return pd.Series(per_month)


def dist_table(series: pd.Series, unit_div: float = 1.0, unit: str = "") -> str:
    """分位数分布字符串。"""
    s = series.dropna() / unit_div
    if s.empty:
        return "无数据"
    q = s.quantile([0, 0.25, 0.5, 0.75, 1.0])
    return (f"min={q[0]:.2f}{unit}, p25={q[0.25]:.2f}{unit}, "
            f"median={q[0.5]:.2f}{unit}, p75={q[0.75]:.2f}{unit}, max={q[1.0]:.2f}{unit}")


def main():
    cfg = yaml.safe_load(open(CONFIG))
    start, end = cfg["period"]["start_date"], cfg["period"]["end_date"]
    filters = cfg["filters"]
    sig_codes = signal_covered_codes()

    print(f"Building universes for {start} ~ {end}")
    print(f"Filters: {filters}")

    # ---- Universe_B: 小盘主研究池 (ZZ1000 流动性成分 → 过滤) ----
    b_cfg = cfg["universes"]["Universe_B"]
    b_base = read_code_file(PROJECT / b_cfg["base_file"])
    print(f"\nUniverse_B base (ZZ1000 liquid): {len(b_base)} candidates")
    b_inc, b_exc = reg.build_membership(b_base, start, end, filters)
    b_inc["in_signals"] = b_inc["symbol"].isin(sig_codes)
    print(f"  -> included {len(b_inc)}, excluded {len(b_exc)}")

    # ---- Universe_A: fast_debug (B 的确定性 300 子集) ----
    a_cfg = cfg["universes"]["Universe_A"]
    a_size = a_cfg["size"]
    a_syms = sorted(b_inc["symbol"].tolist())[:a_size]
    a_inc = b_inc[b_inc["symbol"].isin(a_syms)].reset_index(drop=True)
    a_exc = pd.DataFrame(columns=["symbol", "exclude_reason"])
    print(f"\nUniverse_A (subset of B): {len(a_inc)}")

    # ---- Universe_C: Level-2 可用池 ----
    c_cfg = cfg["universes"]["Universe_C"]
    base_dir = PROJECT / c_cfg["base_dir"]
    c_base = sorted(d.name.zfill(6) for d in base_dir.iterdir()
                    if d.is_dir() and d.name.isdigit())
    print(f"\nUniverse_C base (single_stock dirs): {len(c_base)} candidates")
    c_inc, c_exc = reg.build_membership(c_base, start, end, filters)
    c_inc["in_signals"] = c_inc["symbol"].isin(sig_codes)
    print(f"  -> included {len(c_inc)}, excluded {len(c_exc)}, "
          f"of which {c_inc['in_signals'].sum()} have Alpha191 signals")

    # ---- Save parquets ----
    reg.save_universe(a_inc, "Universe_A", a_cfg["name"], a_cfg["purpose"])
    reg.save_universe(b_inc, "Universe_B", b_cfg["name"], b_cfg["purpose"])
    reg.save_universe(c_inc, "Universe_C", c_cfg["name"], c_cfg["purpose"])

    # ---- Update registry CSV ----
    reg.update_registry([
        {"universe_id": "Universe_A", "universe_name": a_cfg["name"],
         "data_requirement": "Daily OHLCV", "source": "Universe_B 确定性 300 子集",
         "stock_count": len(a_inc), "start_date": start, "end_date": end,
         "status": "Active", "notes": "快速开发/调试池"},
        {"universe_id": "Universe_B", "universe_name": b_cfg["name"],
         "data_requirement": "Daily OHLCV", "source": "ZZ1000 流动性成分 + 通用过滤",
         "stock_count": len(b_inc), "start_date": start, "end_date": end,
         "status": "Active", "notes": "小盘主研究池；ST 剔除因缺数据源未执行"},
        {"universe_id": "Universe_C", "universe_name": c_cfg["name"],
         "data_requirement": "Level-2 + Daily", "source": "data/single_stock 有逐笔衍生数据且日线完整",
         "stock_count": len(c_inc), "start_date": start, "end_date": end,
         "status": "Active", "notes": f"Level-2 研究池；{int(c_inc['in_signals'].sum())} 只有 Alpha191 信号"},
    ])

    # ---- Report ----
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT, "w") as f:
        f.write("# Universe 报告（Universe_A / B / C）\n\n")
        f.write(f"生成时间: {pd.Timestamp.now():%Y-%m-%d}  |  窗口: {start} ~ {end}\n\n")
        f.write("> market cap 口径 = 真实价(amount/volume) × outstanding_share，单位亿元。\n")
        f.write("> ST 剔除因无名单数据源未执行；行业分布因无行业映射数据为 N/A。\n\n---\n\n")

        f.write("## 概览\n\n")
        f.write("| Universe | 名称 | 用途 | 成员数 | 剔除数 |\n|---|---|---|---|---|\n")
        for uid, inc, exc, ucfg in [
            ("Universe_A", a_inc, a_exc, a_cfg),
            ("Universe_B", b_inc, b_exc, b_cfg),
            ("Universe_C", c_inc, c_exc, c_cfg),
        ]:
            f.write(f"| {uid} | {ucfg['name']} | {ucfg['purpose']} | {len(inc)} | {len(exc)} |\n")
        f.write("\n")

        for uid, inc, exc in [
            ("Universe_A", a_inc, a_exc),
            ("Universe_B", b_inc, b_exc),
            ("Universe_C", c_inc, c_exc),
        ]:
            f.write(f"## {uid}\n\n")
            f.write(f"- 成员数: **{len(inc)}**\n")
            if "market_cap_est" in inc.columns and len(inc):
                f.write(f"- 市值分布(亿元): {dist_table(inc['market_cap_est'], 1e8, '')}\n")
                f.write(f"- 日均额分布(万元): {dist_table(inc['median_amount'], 1e4, '')}\n")
                f.write(f"- 换手率均值分布(%): {dist_table(inc['avg_turnover']*100, 1.0, '')}\n")
            if "in_signals" in inc.columns and len(inc):
                f.write(f"- 有 Alpha191 信号覆盖: {int(inc['in_signals'].sum())}/{len(inc)}\n")
            # 剔除原因
            if len(exc):
                rc = exc["exclude_reason"].value_counts()
                f.write(f"- 剔除原因: " + ", ".join(f"{k}={v}" for k, v in rc.items()) + "\n")
            # 月度覆盖
            mc = monthly_coverage(inc["symbol"].tolist(), start, end)
            f.write(f"- 月度覆盖数量: " + ", ".join(f"{m[-2:]}月:{v}" for m, v in mc.items()) + "\n\n")

        f.write("---\n\n## 已知缺口\n\n")
        f.write("- **ST/*ST 剔除**：无 ST 名单数据源，`exclude_st` 无法执行。\n")
        f.write("- **行业分布/集中度**：无行业分类映射，标注 N/A。\n")
        f.write("- **新股判定**：以「窗口内有效交易日 < 60」近似，非真实 IPO 日。\n")

    print(f"\nReport: {REPORT}")
    print(f"Registry updated: {reg.REGISTRY_CSV}")
    print(f"\nUniverse sizes: A={len(a_inc)}, B={len(b_inc)}, C={len(c_inc)}")


if __name__ == "__main__":
    main()
