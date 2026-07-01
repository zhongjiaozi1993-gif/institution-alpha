"""
SOFIA 机构订单簿生成器 — 为每个匿名机构建立按时间顺序的买卖记录

输出:
  data/single_stock/{stock}/sofia_v4/orderbook/
    ANON-001_orderbook.csv    — 单机构完整买卖日历
    ANON-002_orderbook.csv
    ...
    master_orderbook.csv      — 所有机构汇总
    orderbook_report.md       — 人读报告
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT = Path(__file__).parent.parent
STOCK_DIR = PROJECT / "data" / "single_stock"


def build_orderbook(stock: str = "002516"):
    """从SOFIA v4注册表构建机构订单簿。"""
    sofia_dir = STOCK_DIR / stock / "sofia_v4"
    registry_path = sofia_dir / "institution_registry.json"
    price_path = STOCK_DIR / stock / "price_daily.csv"

    if not registry_path.exists():
        print(f"SOFIA v4 注册表不存在: {registry_path}")
        print("请先运行: python3 scripts/sofia_v4_hunter.py --stock 002516 --year 2025")
        return

    registry = json.load(open(registry_path))
    prices = pd.read_csv(price_path)
    prices["日期"] = pd.to_datetime(prices["日期"])
    prices["date_str"] = prices["日期"].dt.strftime("%Y%m%d")

    out_dir = sofia_dir / "orderbook"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ═══════════════════════════════════════════════════
    # 1. 构建每机构的订单簿
    # ═══════════════════════════════════════════════════
    all_orders = []

    for inst in registry:
        aid = inst["anon_id"]
        rows = []

        for c in inst["all_clusters"]:
            date_str = c["date"]
            px_row = prices[prices["date_str"] == date_str]
            close_px = float(px_row["收盘"].values[0]) if len(px_row) > 0 else np.nan
            chg_pct = float(px_row["涨跌幅"].values[0]) if len(px_row) > 0 else np.nan

            rows.append({
                "date": date_str,
                "direction": c["direction"],
                "amount_wan": c["amount_wan"],
                "price_yuan": c["price_yuan"],
                "hfq_close": round(close_px, 2),
                "pct_chg": round(chg_pct, 2) if not np.isnan(chg_pct) else np.nan,
                "n_orders": c["n_orders"],
                "avg_id_gap": c["avg_id_gap"],
                "qty_cv": c["qty_cv"],
                "session": c["session"],
            })

        df = pd.DataFrame(rows)
        if df.empty:
            continue

        df = df.sort_values("date")

        # 累计统计
        df["cum_buy_wan"] = df["amount_wan"].where(df["direction"] == "BUY", 0).cumsum()
        df["cum_sell_wan"] = df["amount_wan"].where(df["direction"] == "SELL", 0).cumsum()
        df["cum_net_wan"] = df["cum_buy_wan"] - df["cum_sell_wan"]

        # 加权均价
        buy_mask = df["direction"] == "BUY"
        sell_mask = df["direction"] == "SELL"
        if buy_mask.any():
            df["vwap_buy"] = (df.loc[buy_mask, "amount_wan"] * df.loc[buy_mask, "price_yuan"]).cumsum() \
                             / df.loc[buy_mask, "amount_wan"].cumsum()
            df["vwap_buy"] = df["vwap_buy"].ffill()
        if sell_mask.any():
            df["vwap_sell"] = (df.loc[sell_mask, "amount_wan"] * df.loc[sell_mask, "price_yuan"]).cumsum() \
                              / df.loc[sell_mask, "amount_wan"].cumsum()
            df["vwap_sell"] = df["vwap_sell"].ffill()

        # 月度聚合
        df["month"] = df["date"].str[:6]
        monthly = df.groupby("month").agg(
            buy_times=("direction", lambda x: (x == "BUY").sum()),
            sell_times=("direction", lambda x: (x == "SELL").sum()),
            buy_wan=("amount_wan", lambda x: x[x.index.isin(df[df["direction"] == "BUY"].index)].sum()),
            sell_wan=("amount_wan", lambda x: x[x.index.isin(df[df["direction"] == "SELL"].index)].sum()),
            avg_price=("price_yuan", "mean"),
        ).reset_index()
        monthly["net_wan"] = monthly["buy_wan"] - monthly["sell_wan"]

        # 保存
        df.to_csv(out_dir / f"{aid}_orderbook.csv", index=False)
        monthly.to_csv(out_dir / f"{aid}_monthly.csv", index=False)

        # 汇总
        bh = inst["behavior"]
        for _, r in df.iterrows():
            all_orders.append({
                "anon_id": aid,
                "size_label": inst["size_label"],
                "date": r["date"],
                "direction": r["direction"],
                "amount_wan": r["amount_wan"],
                "price_yuan": r["price_yuan"],
                "hfq_close": r["hfq_close"],
                "pct_chg": r["pct_chg"],
                "n_orders": r["n_orders"],
                "avg_id_gap": r["avg_id_gap"],
                "qty_cv": r["qty_cv"],
                "session": r["session"],
                "buy_pct": inst["buy_pct"],
                "split_style": bh["split_style"],
                "typical_session": bh["typical_session"],
            })

    master = pd.DataFrame(all_orders)
    master.to_csv(out_dir / "master_orderbook.csv", index=False)

    # ═══════════════════════════════════════════════════
    # 2. 生成报告
    # ═══════════════════════════════════════════════════
    lines = [
        f"# {stock} SOFIA 机构订单簿",
        "",
        f"生成时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}",
        f"机构总数: {len(registry)}",
        f"总订单记录: {len(master)}",
        "",
        "## 十大交易事件日",
        "",
        "| 排名 | 日期 | 净流向(万) | 买入(万) | 卖出(万) | 最大单簇(万) | 笔数 | 关键机构 |",
        "|------|------|-----------|----------|----------|-------------|------|---------|",
    ]

    # Load daily summary
    daily = pd.read_csv(sofia_dir / "daily_algo_summary.csv")
    top10 = daily.nlargest(10, "net_wan") if "net_wan" in daily.columns else daily.head(10)

    for _, d in top10.iterrows():
        date_str = str(d["date"])
        day_orders = master[master["date"] == date_str]
        top_insts = day_orders.groupby("anon_id")["amount_wan"].sum().nlargest(3)
        inst_str = ", ".join(f"{k}({v:.0f}万)" for k, v in top_insts.items())

        net = d.get("net_wan", d.get("buy_wan", 0) - d.get("sell_wan", 0))
        buy_val = d.get("buy_wan", 0)
        sell_val = d.get("sell_wan", 0)
        arrow = "▲▲" if net > 5000 else "▲" if net > 1000 else "▼▼" if net < -5000 else "▼"

        lines.append(f"| {date_str} | {arrow}{net:+.0f} | {buy_val:.0f} | {sell_val:.0f} | "
                     f"{d.get('top_cluster_wan', '-')} | {d.get('top_cluster_orders', '-')} | {inst_str} |")

    # 十大机构详解
    lines.extend([
        "",
        "## 十大匿名机构订单簿",
        "",
        "### 机构总览",
        "",
        "| 机构 | 规模 | 覆盖天数 | 总买入(万) | 总卖出(万) | 净流向(万) | 时段偏好 | IDgap | 买入占比 | 操作风格 |",
        "|------|------|---------|-----------|-----------|-----------|---------|-------|---------|---------|",
    ])

    for inst in registry[:10]:
        bh = inst["behavior"]
        fp = inst["fingerprint_summary"]
        lines.append(
            f"| {inst['anon_id']} | {inst['size_label']} | {inst['n_days']} | "
            f"{inst['buy_wan']:.0f} | {inst['sell_wan']:.0f} | {inst['net_wan']:+.0f} | "
            f"{bh['typical_session']} | {fp['avg_id_gap']:.0f} | {inst['buy_pct']:.0f}% | "
            f"{bh['operation_style'][:15]} |"
        )

    # 逐机构详解
    for inst in registry[:10]:
        df = pd.read_csv(out_dir / f"{inst['anon_id']}_orderbook.csv")
        if df.empty:
            continue

        bh = inst["behavior"]
        fp = inst["fingerprint_summary"]
        rep = inst["representative"]

        lines.extend([
            "",
            f"### {inst['anon_id']} [{inst['size_label']}] — {inst['date_range']}",
            "",
            f"- **总足迹**: {inst['total_footprint_wan']:.0f}万 (买{inst['buy_wan']:.0f} / 卖{inst['sell_wan']:.0f} / 净{inst['net_wan']:+.0f})",
            f"- **代表操作**: {rep['date']} {rep['direction']} {rep['amount_wan']:.0f}万 @{rep['price_yuan']:.2f}元 ×{rep['n_orders']}笔",
            f"- **覆盖天数**: {inst['n_days']}天, 共{inst['n_clusters']}次操作",
            "",
            f"#### 拆单手法",
            f"- {bh['split_style']}",
            f"- 拆单CV均值: {fp['avg_qty_cv']:.3f} | ID间隔均值: {fp['avg_id_gap']:.0f} | 笔间间隔: {fp.get('avg_time_gap_sec', 0):.1f}s",
            f"- 时段分布: {fp['session_distribution']}",
            "",
            f"#### 操作风格",
            f"- {bh['operation_style']}",
            f"- {bh['size_stability']}",
            f"- {bh['time_preference']}",
            "",
        ])

        # 月度汇总表
        monthly = pd.read_csv(out_dir / f"{inst['anon_id']}_monthly.csv")
        if len(monthly) > 0:
            lines.append("#### 月度买卖汇总")
            lines.append("| 月份 | 买入次数 | 卖出次数 | 买入(万) | 卖出(万) | 净(万) | 均价 |")
            lines.append("|------|---------|---------|---------|---------|--------|------|")
            for _, m in monthly.iterrows():
                lines.append(
                    f"| {m['month']} | {int(m['buy_times'])} | {int(m['sell_times'])} | "
                    f"{m['buy_wan']:.0f} | {m['sell_wan']:.0f} | {m['net_wan']:+.0f} | "
                    f"{m['avg_price']:.2f} |"
                )

        # TOP 10 最大单日操作
        top_ops = df.nlargest(10, "amount_wan")
        lines.extend([
            "",
            "#### TOP 10 最大单笔操作",
            "| 日期 | 方向 | 金额(万) | 价格 | HFQ收盘 | 涨跌% | 笔数 | IDgap | CV | 时段 |",
            "|------|------|---------|------|---------|-------|------|-------|-----|------|",
        ])
        for _, r in top_ops.iterrows():
            dir_sym = "买入" if r['direction'] == 'BUY' else "卖出" if r['direction'] == 'SELL' else "混合"
            lines.append(
                f"| {r['date']} | {dir_sym} | {r['amount_wan']:.0f} | {r['price_yuan']:.2f} | "
                f"{r['hfq_close']:.2f} | {r['pct_chg']:+.2f}% | {int(r['n_orders'])} | "
                f"{r['avg_id_gap']:.0f} | {r['qty_cv']:.3f} | {r['session']} |"
            )

    # 跨机构规律总结
    lines.extend([
        "",
        "## 跨机构规律总结",
        "",
        "### 买方阵营 (净买入Top 5)",
        "",
        *[f"- **{inst['anon_id']}**: 净+{inst['net_wan']:.0f}万, "
          f"{inst['behavior']['typical_session']}偏好, "
          f"IDgap≈{inst['fingerprint_summary']['avg_id_gap']:.0f}, "
          f"{inst['behavior']['split_style'][:30]}"
          for inst in sorted(registry, key=lambda x: x['net_wan'], reverse=True)[:5]],
        "",
        "### 卖方阵营 (净卖出Top 5)",
        "",
        *[f"- **{inst['anon_id']}**: 净{inst['net_wan']:.0f}万, "
          f"{inst['behavior']['typical_session']}偏好, "
          f"IDgap≈{inst['fingerprint_summary']['avg_id_gap']:.0f}, "
          f"{inst['behavior']['split_style'][:30]}"
          for inst in sorted(registry, key=lambda x: x['net_wan'])[:5]],
        "",
        "### 时段与方向的关系",
        "",
        "| 时段 | 主要买方 | 主要卖方 | 特征 |",
        "|------|---------|---------|------|",
    ])

    sessions = ["AUCTION", "OPEN", "MORNING", "LATE_MORNING", "EARLY_AFTER", "AFTERNOON", "CLOSE"]
    for sess in sessions:
        buyers = [inst['anon_id'] for inst in registry[:10]
                  if inst['behavior']['typical_session'] == sess and inst['net_wan'] > 0]
        sellers = [inst['anon_id'] for inst in registry[:10]
                   if inst['behavior']['typical_session'] == sess and inst['net_wan'] < 0]
        lines.append(f"| {sess} | {', '.join(buyers[:3]) or '—'} | {', '.join(sellers[:3]) or '—'} | |")

    # 写入
    report_path = out_dir / "orderbook_report.md"
    report_path.write_text("\n".join(lines))

    print(f"\n订单簿已生成:")
    print(f"  主订单簿: {out_dir}/master_orderbook.csv ({len(master)}条)")
    for inst in registry[:10]:
        df = pd.read_csv(out_dir / f"{inst['anon_id']}_orderbook.csv")
        print(f"  {inst['anon_id']}: {len(df)}条记录, "
              f"净{inst['net_wan']:+.0f}万, "
              f"{inst['date_range']}")
    print(f"  报告: {report_path}")

    return registry, master


if __name__ == "__main__":
    stock = sys.argv[1] if len(sys.argv) > 1 else "002516"
    build_orderbook(stock)
