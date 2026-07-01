#!/usr/bin/env python3
"""SOFIA event-window replay.

Focuses on a short event window and joins:
- anonymous SOFIA v6 Level-2 institution behavior
- public evidence-chain events
- daily price action

Default window is the 002516 control-change window around 2025-09-08.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

STOCK = "002516"
STOCK_DIR = PROJECT / "data" / "single_stock" / STOCK
V6_DIR = STOCK_DIR / "sofia_v6"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stock", default=STOCK)
    parser.add_argument("--start", default="20250908")
    parser.add_argument("--end", default="20250911")
    parser.add_argument(
        "--event-days-before",
        type=int,
        default=3,
        help="Include public evidence this many calendar days before --start.",
    )
    parser.add_argument(
        "--event-days-after",
        type=int,
        default=0,
        help="Include public evidence this many calendar days after --end.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Default: data/single_stock/<stock>/sofia_v6/event_window_<start>_<end>.md",
    )
    return parser.parse_args()


def load_registry(stock_dir: Path) -> list[dict]:
    with open(stock_dir / "sofia_v6" / "institution_registry.json", encoding="utf-8") as f:
        return json.load(f)


def load_price(stock_dir: Path) -> pd.DataFrame:
    price = pd.read_csv(stock_dir / "price_daily.csv")
    price["date"] = pd.to_datetime(price["日期"]).dt.strftime("%Y%m%d")
    return price.sort_values("date").reset_index(drop=True)


def load_public_events(stock_dir: Path) -> pd.DataFrame:
    path = stock_dir / "evidence" / "public_evidence.csv"
    if not path.exists():
        return pd.DataFrame()
    events = pd.read_csv(path)
    if events.empty:
        return events
    events["date_str"] = events["date"].astype(str).str.replace(".0", "", regex=False).str.zfill(8)
    return events.sort_values(["date_str", "source", "title"]).reset_index(drop=True)


def load_merge_log(stock_dir: Path) -> pd.DataFrame:
    path = stock_dir / "sofia_v6" / "merge_log.csv"
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def extract_ops(registry: list[dict], start: str, end: str) -> pd.DataFrame:
    rows = []
    for inst in registry:
        for op in inst["operations"]:
            if start <= op["date"] <= end:
                rows.append(
                    {
                        "date": op["date"],
                        "anon_id": inst["anon_id"],
                        "confidence": inst["confidence"],
                        "behavior_type": inst["behavior_type"],
                        "direction": op["direction"],
                        "amount_wan": float(op["amount_wan"]),
                        "price_yuan": float(op["price_yuan"]),
                        "n_orders": int(op["n_orders"]),
                        "avg_id_gap": float(op["avg_id_gap"]),
                        "qty_cv": float(op["qty_cv"]),
                        "session": op["session"],
                        "inst_total_net_wan": inst.get("net_wan"),
                        "inst_buy_pct": inst.get("buy_pct"),
                    }
                )
    return pd.DataFrame(rows)


def build_daily_matrix(ops: pd.DataFrame) -> pd.DataFrame:
    if ops.empty:
        return pd.DataFrame()
    matrix = ops.groupby(["date", "anon_id", "direction"])["amount_wan"].sum().unstack(fill_value=0)
    for col in ["BUY", "SELL", "MIXED"]:
        if col not in matrix.columns:
            matrix[col] = 0.0
    matrix["net_wan"] = matrix["BUY"] - matrix["SELL"]
    return matrix.reset_index().sort_values(["date", "net_wan"], ascending=[True, False])


def build_report(
    stock: str,
    start: str,
    end: str,
    event_start: str,
    event_end: str,
    ops: pd.DataFrame,
    price: pd.DataFrame,
    events: pd.DataFrame,
    merge_log: pd.DataFrame,
) -> str:
    lines: list[str] = [
        f"# {stock} SOFIA事件窗口复盘 ({start}-{end})",
        "",
        "## 结论摘要",
        "",
    ]

    matrix = build_daily_matrix(ops)
    if matrix.empty:
        lines.append("- 窗口内无SOFIA v6机构操作。")
        return "\n".join(lines) + "\n"

    daily = matrix.groupby("date").agg(
        buy_wan=("BUY", "sum"),
        sell_wan=("SELL", "sum"),
        mixed_wan=("MIXED", "sum"),
        net_wan=("net_wan", "sum"),
        n_institutions=("anon_id", "nunique"),
    ).reset_index()

    max_buy = daily.sort_values("net_wan", ascending=False).iloc[0]
    max_sell = daily.sort_values("net_wan", ascending=True).iloc[0]
    lines.append(
        f"- 最大净买入日：{max_buy['date']}，净买 {max_buy['net_wan']:.0f} 万，"
        f"参与 {int(max_buy['n_institutions'])} 个匿名主体。"
    )
    lines.append(
        f"- 最大净卖出/兑现日：{max_sell['date']}，净流 {max_sell['net_wan']:.0f} 万。"
    )
    lines.append(
        f"- 公开事件窗口：{event_start}-{event_end}；交易行为窗口：{start}-{end}。"
    )
    lines.append("- 公开事件主线：控制权变更、股份转让协议、权益变动报告书。")
    lines.append("- 审计提示：匿名机构ID是行为指纹，不等于公开股东名称；不要把 ANON-* 直接归因给野村/HKSCC/启创一号。")

    if not merge_log.empty:
        merged_text = "; ".join(
            f"{r['merged']}→{r['into']}({r['conditions']}/7)" for _, r in merge_log.head(8).iterrows()
        )
        lines.append(f"- 合并视图提示：v6 merge_log 中存在 {merged_text}。事件窗口仍保留原始匿名ID时，需避免重复解读。")

    lines.extend(["", "## 公开事件", ""])
    ev = events[(events["date_str"] >= event_start) & (events["date_str"] <= event_end)]
    if ev.empty:
        lines.append("无。")
    else:
        view = ev[["date_str", "evidence_type", "source", "title"]].head(30).copy()
        lines.extend(markdown_table(view))

    lines.extend(["", "## 价格走势", ""])
    p = price[(price["date"] >= start) & (price["date"] <= end)]
    price_cols = ["date", "开盘", "收盘", "最高", "最低", "涨跌幅", "换手率", "成交额"]
    lines.extend(markdown_table(p[price_cols]))

    lines.extend(["", "## 日度资金矩阵", ""])
    lines.extend(markdown_table(daily.round(2)))

    lines.extend(["", "## 匿名机构净流向", ""])
    pivot = matrix.pivot_table(index="anon_id", values="net_wan", aggfunc="sum").reset_index()
    pivot = pivot.sort_values("net_wan", ascending=False)
    lines.extend(markdown_table(pivot.round(2)))

    lines.extend(["", "## 每日明细", ""])
    detail_cols = [
        "date",
        "anon_id",
        "confidence",
        "behavior_type",
        "direction",
        "amount_wan",
        "price_yuan",
        "n_orders",
        "avg_id_gap",
        "qty_cv",
        "session",
    ]
    detail = ops.sort_values(["date", "amount_wan"], ascending=[True, False])[detail_cols]
    lines.extend(markdown_table(detail.round(3)))

    lines.extend(
        [
            "",
            "## 解读",
            "",
            "- 2025-09-08 的主买是 ANON-002 和 ANON-004，ANON-001只小额参与；因此不能用 ANON-001 长期Alpha直接解释当日扫货。",
            "- 2025-09-09 出现明显分歧，ANON-005/006/003/007开始卖出，ANON-004仍在买。",
            "- 2025-09-10 权益变动报告书落地，ANON-004继续买，ANON-005/006继续卖，说明事件兑现窗口里多空分歧扩大。",
            "- 2025-09-11 高换手下卖方继续兑现，ANON-005是主要卖方之一。",
        ]
    )
    return "\n".join(lines) + "\n"


def markdown_table(frame: pd.DataFrame) -> list[str]:
    if frame.empty:
        return ["无。"]
    clean = frame.copy()
    clean = clean.where(pd.notna(clean), "")
    headers = list(clean.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in clean.iterrows():
        values = [str(row[h]).replace("|", "/") for h in headers]
        lines.append("| " + " | ".join(values) + " |")
    return lines


def main() -> None:
    args = parse_args()
    stock_dir = PROJECT / "data" / "single_stock" / args.stock
    out_path = (
        Path(args.output)
        if args.output
        else stock_dir / "sofia_v6" / f"event_window_{args.start}_{args.end}.md"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    registry = load_registry(stock_dir)
    ops = extract_ops(registry, args.start, args.end)
    start_dt = pd.to_datetime(args.start, format="%Y%m%d")
    end_dt = pd.to_datetime(args.end, format="%Y%m%d")
    event_start = (start_dt - pd.Timedelta(days=args.event_days_before)).strftime("%Y%m%d")
    event_end = (end_dt + pd.Timedelta(days=args.event_days_after)).strftime("%Y%m%d")
    report = build_report(
        stock=args.stock,
        start=args.start,
        end=args.end,
        event_start=event_start,
        event_end=event_end,
        ops=ops,
        price=load_price(stock_dir),
        events=load_public_events(stock_dir),
        merge_log=load_merge_log(stock_dir),
    )
    out_path.write_text(report, encoding="utf-8")

    print(f"event-window report: {out_path}")
    print(f"ops: {len(ops)}")
    if not ops.empty:
        matrix = build_daily_matrix(ops)
        print(matrix.groupby("date")["net_wan"].sum().round(0).to_string())


if __name__ == "__main__":
    main()
