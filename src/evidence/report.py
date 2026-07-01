"""Markdown reporting for institution behavior evidence chains."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def write_markdown_report(
    report_path: Path,
    stock_code: str,
    daily: pd.DataFrame,
    notable: pd.DataFrame,
    public_evidence: pd.DataFrame,
    holder_changes: pd.DataFrame,
    source_status: pd.DataFrame,
) -> None:
    """Write a compact report that Claude Code can continue from."""
    lines: list[str] = []
    lines.append(f"# {stock_code} 机构行为证据链")
    lines.append("")
    lines.append("## 结论摘要")
    lines.extend(_summary_lines(daily, notable, public_evidence))
    lines.append("")
    lines.append("## 重点行为日")
    lines.extend(_notable_table(notable))
    lines.append("")
    lines.append("## 公开证据源状态")
    lines.extend(_source_status_table(source_status))
    lines.append("")
    lines.append("## 滞后持仓变化线索")
    lines.extend(_holder_change_lines(holder_changes))
    lines.append("")
    lines.append("## 公开事件匹配")
    lines.extend(_public_event_lines(public_evidence))
    lines.append("")
    lines.append("## CC 接手提示")
    lines.append("")
    lines.append("- 先看 `notable_events.csv`，这是交易/投机最有价值的行为锚点。")
    lines.append("- `daily_evidence.csv` 保留全量日级行为，可继续接模型或回测。")
    lines.append("- `public_evidence.csv` 是外部公开源原始归一化结果，没匹配到不代表没有资金行为。")
    lines.append("- `holder_changes.csv` 是基金/股东持仓的报表期变化，适合做滞后确认。")
    lines.append("- 当前 Level-2 价格与日线价格可能不同口径，收益列只做事件参考，正式回测要修价格口径。")
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _summary_lines(
    daily: pd.DataFrame,
    notable: pd.DataFrame,
    public_evidence: pd.DataFrame,
) -> list[str]:
    lines: list[str] = []
    if daily.empty:
        return ["- 无本地行为数据。"]

    max_buy = daily.sort_values("net_wan", ascending=False).iloc[0]
    max_sell = daily.sort_values("net_wan", ascending=True).iloc[0]
    super_buy_days = int((daily.get("n_super_buy", 0) > 0).sum())
    super_sell_days = int((daily.get("n_super_sell", 0) > 0).sum())
    warning = ""
    if "price_unit_warning" in daily.columns:
        warnings = [w for w in daily["price_unit_warning"].dropna().unique().tolist() if w]
        warning = warnings[0] if warnings else ""

    lines.append(f"- 全部行为日：{len(daily)} 天；重点行为日：{len(notable)} 天。")
    lines.append(f"- 超级买入日：{super_buy_days} 天；超级卖出日：{super_sell_days} 天。")
    lines.append(
        f"- 最大净买入：{max_buy['date']}，净买 {max_buy['net_wan']:.0f} 万，"
        f"类型：{max_buy['behavior_type']}。"
    )
    lines.append(
        f"- 最大净卖出：{max_sell['date']}，净卖 {abs(max_sell['net_wan']):.0f} 万，"
        f"类型：{max_sell['behavior_type']}。"
    )
    if public_evidence.empty:
        lines.append("- 公开源暂未匹配到可直接解释重点行为日的龙虎榜/公告证据。")
    else:
        lines.append(f"- 已抓取公开证据 {len(public_evidence)} 条，可继续做人读复核。")
    if warning:
        lines.append(f"- 数据口径警告：{warning}。")
    return lines


def _notable_table(notable: pd.DataFrame) -> list[str]:
    if notable.empty:
        return ["无重点行为日。"]

    cols = [
        "date",
        "behavior_type",
        "behavior_confidence",
        "buy_wan",
        "sell_wan",
        "net_wan",
        "max_op_wan",
        "max_op_direction",
        "max_op_start",
        "pct_chg",
        "turnover",
        "fwd_5d_t1open_pct",
        "fwd_10d_t1open_pct",
        "public_event_count",
    ]
    available = [c for c in cols if c in notable.columns]
    view = notable.sort_values("gross_wan", ascending=False)[available].head(30).copy()
    for col in view.select_dtypes("number").columns:
        view[col] = view[col].round(2)
    return _markdown_table(view)


def _source_status_table(source_status: pd.DataFrame) -> list[str]:
    if source_status.empty:
        return ["未启用公开源抓取。"]
    return _markdown_table(source_status)


def _public_event_lines(public_evidence: pd.DataFrame) -> list[str]:
    if public_evidence.empty:
        return ["未抓到可归一化的公开事件。"]
    frame = public_evidence.copy()
    frame["date_str"] = frame["date"].astype(str).str.replace(".0", "", regex=False).str.zfill(8)
    recent = frame[frame["date_str"] >= "20250801"]
    if recent.empty:
        recent = frame.tail(50)
    view = recent[["date", "source", "evidence_type", "title", "buy_amt", "sell_amt", "net_amt"]].head(50)
    return _markdown_table(view)


def _holder_change_lines(holder_changes: pd.DataFrame) -> list[str]:
    if holder_changes.empty:
        return ["暂无可计算的持仓变化。"]

    frame = holder_changes.copy()
    frame["date_str"] = frame["date"].astype(str).str.replace(".0", "", regex=False).str.zfill(8)
    frame = frame[frame["date_str"] >= "20250930"]
    frame = frame.dropna(subset=["share_delta"])
    if frame.empty:
        return ["暂无 2025-09-30 之后的持仓变化。"]

    cols = [
        "date",
        "holder_type",
        "holder_name",
        "shares",
        "prev_date",
        "prev_shares",
        "share_delta",
        "ratio",
        "ratio_delta",
    ]
    increases = frame.sort_values("share_delta", ascending=False).head(12)[cols]
    decreases = frame.sort_values("share_delta", ascending=True).head(12)[cols]
    for table in [increases, decreases]:
        for col in ["shares", "prev_shares", "share_delta", "ratio", "ratio_delta"]:
            if col in table.columns:
                table[col] = pd.to_numeric(table[col], errors="coerce").round(4)

    lines = ["### 增持/新增靠前", ""]
    lines.extend(_markdown_table(increases))
    lines.append("")
    lines.append("### 减持/退出靠前")
    lines.append("")
    lines.extend(_markdown_table(decreases))
    return lines


def _markdown_table(frame: pd.DataFrame) -> list[str]:
    if frame.empty:
        return ["无。"]
    headers = list(frame.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in frame.iterrows():
        vals = []
        for col in headers:
            val = row[col]
            if pd.isna(val):
                vals.append("")
            else:
                vals.append(str(val).replace("|", "/"))
        lines.append("| " + " | ".join(vals) + " |")
    return lines
