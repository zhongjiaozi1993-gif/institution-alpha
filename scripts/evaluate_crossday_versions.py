"""
跨日匿名机构识别版本评估 (v4 vs v6) — OOT 外推验证

读取 v4/v6 的 institution_registry.json，统一字段、计算 L2 口径未来收益，
输出匿名机构级和版本级的 Alpha 评估报告。

用法:
  python3 scripts/evaluate_crossday_versions.py --stocks 002516
  python3 scripts/evaluate_crossday_versions.py --stocks 301529,300100
  python3 scripts/evaluate_crossday_versions.py --stocks all --versions v6

输出 (每只股票):
  - data/processed/oot/{stock}/l2_daily_ohlc.csv
  - data/processed/oot/{stock}/crossday_operations_unified.csv
  - data/processed/oot/{stock}/crossday_anon_eval.csv
  - data/processed/oot/{stock}/crossday_version_eval.csv

汇总:
  - data/processed/oot/v6_oot_summary.csv
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

DATA_DIR = PROJECT / "data" / "single_stock"
OOT_DIR = PROJECT / "data" / "processed" / "oot"
PRICE_SCALE = 10000
FWD_HORIZONS = [1, 3, 5, 10]


def wind_code(stock: str) -> str:
    """补全交易所后缀: 002516→002516.SZ, 600519→600519.SH"""
    suffix = "SH" if stock.startswith(("6", "9")) else "SZ"
    return f"{stock}.{suffix}"


# ─── L2 OHLC ───────────────────────────────────────────────────────


def _l2_daily_ohlc(stock: str, date_str: str) -> dict | None:
    cj_path = DATA_DIR / stock / "raw" / date_str / wind_code(stock) / "逐笔成交.csv"
    if not cj_path.exists():
        return None
    for enc in ["gb18030", "gbk", "utf-8"]:
        try:
            df = pd.read_csv(cj_path, encoding=enc, low_memory=False)
            break
        except (UnicodeDecodeError, UnicodeError):
            continue
    else:
        return None
    if "成交代码" in df.columns:
        df = df[df["成交代码"] != "C"]
    if df.empty:
        return None
    df = df.sort_values("时间")
    px = df["成交价格"].astype(float) / PRICE_SCALE
    return {"open": float(px.iloc[0]), "high": float(px.max()),
            "low": float(px.min()), "close": float(px.iloc[-1])}


def build_l2_ohlc_table(stock: str) -> pd.DataFrame:
    raw_dir = DATA_DIR / stock / "raw"
    if not raw_dir.exists():
        return pd.DataFrame()
    dates = sorted([d.name for d in raw_dir.iterdir()
                    if d.is_dir() and len(d.name) == 8 and (d / wind_code(stock)).exists()])
    rows = []
    for d in dates:
        ohlc = _l2_daily_ohlc(stock, d)
        if ohlc:
            rows.append({"date": d, **ohlc})
    return pd.DataFrame(rows)


def attach_forward_returns(ops: pd.DataFrame,
                           l2_ohlc: pd.DataFrame,
                           stock: str = "") -> pd.DataFrame:
    """为每条 BUY 操作附加未来 L2 收益"""
    l2 = l2_ohlc.set_index("date")
    for h in FWD_HORIZONS:
        ops[f"fwd_{h}d"] = np.nan
    ops["win_5d"] = np.nan

    buy_mask = ops["direction"] == "BUY"
    stock_mask = ops["stock_code"] == stock if stock else pd.Series(True, index=ops.index)
    mask = buy_mask & stock_mask
    buy_dates = ops.loc[mask, "date"].values
    all_dates = sorted(l2.index)

    for i, d in enumerate(buy_dates):
        if d not in all_dates:
            continue
        base_close = l2.loc[d, "close"]
        idx = all_dates.index(d)
        for h in FWD_HORIZONS:
            tidx = idx + h
            if tidx < len(all_dates):
                fwd_close = l2.loc[all_dates[tidx], "close"]
                ret = (fwd_close / base_close - 1) * 100
                ops.loc[(ops["date"] == d) & mask, f"fwd_{h}d"] = ret
        ret5 = ops.loc[(ops["date"] == d) & mask, "fwd_5d"].values
        ops.loc[(ops["date"] == d) & mask, "win_5d"] = (ret5 > 0).astype(float)

    return ops


# ─── Adapters ──────────────────────────────────────────────────────


def _v4_confidence(inst: dict) -> str:
    size = inst.get("size_label", "")
    buy_pct = inst.get("buy_pct", 0)
    if size in ("巨鲸", "大型") and buy_pct >= 90:
        return "HIGH"
    if size in ("大型", "中型") and buy_pct >= 70:
        return "MEDIUM"
    return "LOW"


def _v4_behavior_type(inst: dict) -> str:
    b = inst.get("behavior", {})
    return b.get("operation_style", "")


def _v6_confidence(inst: dict) -> str:
    return inst.get("confidence", "LOW")


def _v6_behavior_type(inst: dict) -> str:
    return inst.get("behavior_type", "")


def load_v4(stock: str) -> pd.DataFrame:
    path = DATA_DIR / stock / "sofia_v4" / "institution_registry.json"
    if not path.exists():
        print(f"  ⚠ v4 registry not found: {path}")
        return pd.DataFrame()

    with open(path, "rb") as f:
        raw = f.read()
    for enc in ["utf-8", "gb18030", "gbk"]:
        try:
            data = json.loads(raw.decode(enc))
            break
        except (UnicodeDecodeError, json.JSONDecodeError):
            continue
    else:
        raise ValueError(f"Cannot decode {path}")

    rows = []
    for inst in data:
        anon = inst["anon_id"]
        conf = _v4_confidence(inst)
        btype = _v4_behavior_type(inst)
        for c in inst.get("all_clusters", []):
            rows.append({
                "version": "v4",
                "anon_id": f"{stock}:{anon}",
                "stock_code": stock,
                "date": str(c["date"]),
                "direction": c["direction"],
                "amount_wan": c["amount_wan"],
                "price_yuan": c.get("price_yuan"),
                "n_orders": c.get("n_orders"),
                "session": c.get("session", ""),
                "confidence": conf,
                "behavior_type": btype,
                "source_file": str(path),
            })
    print(f"  v4: {len(data)} 个机构, {len(rows)} 条操作")
    return pd.DataFrame(rows)


def load_v6(stock: str) -> pd.DataFrame:
    path = DATA_DIR / stock / "sofia_v6" / "institution_registry.json"
    if not path.exists():
        print(f"  ⚠ v6 registry not found: {path}")
        return pd.DataFrame()

    with open(path, "rb") as f:
        raw = f.read()
    for enc in ["utf-8", "gb18030", "gbk"]:
        try:
            data = json.loads(raw.decode(enc))
            break
        except (UnicodeDecodeError, json.JSONDecodeError):
            continue
    else:
        raise ValueError(f"Cannot decode {path}")

    rows = []
    for inst in data:
        anon = inst["anon_id"]
        conf = _v6_confidence(inst)
        btype = _v6_behavior_type(inst)
        for op in inst.get("operations", []):
            rows.append({
                "version": "v6",
                "anon_id": f"{stock}:{anon}",
                "stock_code": stock,
                "date": str(op["date"]),
                "direction": op["direction"],
                "amount_wan": op["amount_wan"],
                "price_yuan": op.get("price_yuan"),
                "n_orders": op.get("n_orders"),
                "session": op.get("session", ""),
                "confidence": conf,
                "behavior_type": btype,
                "source_file": str(path),
            })
    print(f"  v6: {len(data)} 个机构, {len(rows)} 条操作")
    return pd.DataFrame(rows)


# ─── Evaluation ─────────────────────────────────────────────────────


def eval_anon_level(ops: pd.DataFrame) -> pd.DataFrame:
    """匿名机构级评估"""
    grouped = ops.groupby(["version", "anon_id"])
    rows = []
    for (ver, anon), g in grouped:
        buys = g[g["direction"] == "BUY"]
        sells = g[g["direction"] == "SELL"]
        n_buy = len(buys)
        n_sell = len(sells)
        n_total = n_buy + n_sell
        total_buy = buys["amount_wan"].sum()
        total_sell = sells["amount_wan"].sum()
        net = total_buy - total_sell
        buy_ratio = n_buy / max(1, n_total)

        conf = g["confidence"].iloc[0]

        fwd_cols = [f"fwd_{h}d" for h in FWD_HORIZONS]
        fwd_stats = {}
        for col in fwd_cols:
            vals = buys[col].dropna()
            fwd_stats[f"avg_{col}"] = round(float(vals.mean()), 2) if len(vals) > 0 else None

        win5_vals = buys["win_5d"].dropna()
        win5 = round(float(win5_vals.mean()), 3) if len(win5_vals) > 0 else None

        rows.append({
            "version": ver,
            "anon_id": anon,
            "confidence": conf,
            "behavior_type": g["behavior_type"].iloc[0],
            "active_days": g["date"].nunique(),
            "buy_days": buys["date"].nunique(),
            "sell_days": sells["date"].nunique(),
            "buy_ops": n_buy,
            "sell_ops": n_sell,
            "buy_ratio": round(buy_ratio, 3),
            "total_buy_wan": round(total_buy, 1),
            "total_sell_wan": round(total_sell, 1),
            "net_buy_wan": round(net, 1),
            "signal_count": n_buy,
            **fwd_stats,
            "win_5d": win5,
        })
    return pd.DataFrame(rows)


def eval_version_level(anon_eval: pd.DataFrame) -> pd.DataFrame:
    """版本级评估"""
    rows = []
    for ver, g in anon_eval.groupby("version"):
        high = g[g["confidence"] == "HIGH"]
        med = g[g["confidence"] == "MEDIUM"]
        low = g[g["confidence"] == "LOW"]

        def _safe_mean(s):
            return round(float(s.mean()), 2) if len(s) > 0 else None

        top10 = g.nlargest(10, "net_buy_wan")

        row = {
            "version": ver,
            "n_anon": len(g),
            "n_high": len(high),
            "n_medium": len(med),
            "n_low": len(low),
            "low_ratio": round(len(low) / max(1, len(g)), 3),
            "avg_active_days": _safe_mean(g["active_days"]),
            "median_active_days": round(float(g["active_days"].median()), 1),
            "avg_buy_ratio": _safe_mean(g["buy_ratio"]),
            "avg_net_buy_wan": _safe_mean(g["net_buy_wan"]),
            "win_5d": _safe_mean(g["win_5d"].dropna()),
            "fragmentation_score": round(
                low["signal_count"].sum() / max(1, g["signal_count"].sum()), 3),
        }
        for h in FWD_HORIZONS:
            col = f"avg_fwd_{h}d"
            row[col] = _safe_mean(g[col].dropna()) if col in g.columns else None
            row[f"top10_{col}"] = _safe_mean(top10[col].dropna()) if col in top10.columns else None
        rows.append(row)
    return pd.DataFrame(rows)


# ─── Per-stock pipeline ────────────────────────────────────────────


def evaluate_stock(stock: str, versions: list[str], out_dir: Path):
    """对单只股票跑完整评估管线，输出到 out_dir"""
    out_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: L2 OHLC
    print(f"  Step 1: 构建 L2 OHLC 基准价格表...")
    l2_ohlc = build_l2_ohlc_table(stock)
    if l2_ohlc.empty:
        print(f"  ✗ 无 L2 数据，跳过 {stock}")
        return None
    l2_path = out_dir / "l2_daily_ohlc.csv"
    l2_ohlc.to_csv(l2_path, index=False)
    print(f"    → {l2_path} ({len(l2_ohlc)} 天)")

    # Step 2: 加载版本
    print(f"  Step 2: 加载机构注册表...")
    loaders = {"v4": load_v4, "v6": load_v6}
    dfs = []
    for ver in versions:
        if ver in loaders:
            df = loaders[ver](stock)
            if not df.empty:
                dfs.append(df)

    if not dfs:
        print(f"  ✗ 没有找到任何版本的机构数据")
        return None

    ops = pd.concat(dfs, ignore_index=True)
    ops["date"] = ops["date"].astype(str)

    # Step 3: 附加前向收益
    print(f"  Step 3: 附加 L2 口径未来收益...")
    ops = attach_forward_returns(ops, l2_ohlc, stock)

    # Step 4: 输出统一操作表
    print(f"  Step 4: 输出统一操作表...")
    unified_path = out_dir / "crossday_operations_unified.csv"
    ops.to_csv(unified_path, index=False)
    print(f"    → {unified_path} ({len(ops)} 条)")

    for ver in ops["version"].unique():
        vops = ops[ops["version"] == ver]
        buys = vops[vops["direction"] == "BUY"]
        print(f"    {ver}: {len(vops)} ops, BUY={len(buys)}, "
              f"fwd_5d均值={buys['fwd_5d'].dropna().mean():.2f}%, "
              f"win_5d={buys['win_5d'].dropna().mean():.3f}")

    # Step 5: 机构级评估
    print(f"  Step 5: 匿名机构级评估...")
    anon_eval = eval_anon_level(ops)
    anon_eval = anon_eval.sort_values(["version", "net_buy_wan"], ascending=[True, False])
    anon_path = out_dir / "crossday_anon_eval.csv"
    anon_eval.to_csv(anon_path, index=False)
    print(f"    → {anon_path} ({len(anon_eval)} 个机构)")

    # Step 6: 版本级评估
    print(f"  Step 6: 版本级评估...")
    ver_eval = eval_version_level(anon_eval)
    ver_path = out_dir / "crossday_version_eval.csv"
    ver_eval.to_csv(ver_path, index=False)
    print(f"    → {ver_path}")

    # 打印版本对比
    print(f"\n  {'─' * 60}")
    print_cols = ["version", "n_anon", "n_high", "n_medium", "n_low",
                  "avg_buy_ratio", "avg_net_buy_wan", "avg_fwd_5d", "win_5d",
                  "top10_avg_fwd_5d", "top10_win_5d", "fragmentation_score"]
    available_print = [c for c in print_cols if c in ver_eval.columns]
    print(f"  {ver_eval[available_print].to_string(index=False)}")

    # Top 机构
    top_cols = ["version", "anon_id", "confidence", "behavior_type",
                "active_days", "buy_ratio", "net_buy_wan", "avg_fwd_5d", "win_5d"]
    print(f"\n  Top 5 机构 (按净买入额):")
    print(f"  {anon_eval.head(5)[top_cols].to_string(index=False)}")

    return {"anon_eval": anon_eval, "ver_eval": ver_eval}


# ─── OOT Summary ───────────────────────────────────────────────────


def build_oot_summary(ver_eval_list: list[pd.DataFrame]) -> pd.DataFrame:
    """从各股票版本级评估拼接 OOT 汇总表"""
    if not ver_eval_list:
        return pd.DataFrame()

    df = pd.concat(ver_eval_list, ignore_index=True)

    summary_cols = [
        "stock_code", "version",
        "n_anon", "n_high", "n_medium", "n_low", "low_ratio",
        "avg_active_days", "avg_buy_ratio", "avg_net_buy_wan",
        "avg_fwd_1d", "avg_fwd_3d", "avg_fwd_5d", "avg_fwd_10d",
        "win_5d", "top10_avg_fwd_5d", "top10_win_5d",
        "fragmentation_score",
    ]

    available = [c for c in summary_cols if c in df.columns]
    return df[available]


# ─── Main ───────────────────────────────────────────────────────────


def main():
    ap = argparse.ArgumentParser(description="跨日匿名机构版本评估 (OOT)")
    ap.add_argument("--stocks", default="",
                    help="股票代码逗号分隔, 如 '301529,300100'。默认自动发现全部")
    ap.add_argument("--versions", default="v4,v6",
                    help="评估版本, 逗号分隔 (默认 v4,v6)")
    args = ap.parse_args()

    # 确定股票列表
    if args.stocks and args.stocks.lower() != "all":
        stocks = [s.strip() for s in args.stocks.split(",") if s.strip()]
    else:
        stocks = sorted([d.name for d in DATA_DIR.iterdir()
                        if d.is_dir() and (d / "raw").exists()])

    if not stocks:
        print("未找到股票数据。用 --stocks 指定或确保 data/single_stock/<code>/raw/ 存在")
        sys.exit(1)

    versions = [v.strip() for v in args.versions.split(",")]

    print(f"评估股票: {stocks}")
    print(f"版本: {versions}")
    print(f"输出: {OOT_DIR}\n")

    OOT_DIR.mkdir(parents=True, exist_ok=True)

    all_ver_evals = []

    for stock in stocks:
        print(f"{'=' * 60}")
        print(f"股票: {stock}")
        print(f"{'=' * 60}")

        stock_out = OOT_DIR / stock
        result = evaluate_stock(stock, versions, stock_out)

        if result and "ver_eval" in result:
            ve = result["ver_eval"].copy()
            ve["stock_code"] = stock
            all_ver_evals.append(ve)
        print()

    # ─── 汇总 ───
    if all_ver_evals:
        summary = build_oot_summary(all_ver_evals)

        # v6-only summary (the OOT validation)
        v6_summary = summary[summary["version"] == "v6"].drop(columns=["version"], errors="ignore")
        summary_path = OOT_DIR / "v6_oot_summary.csv"
        v6_summary.to_csv(summary_path, index=False)
        print(f"\nOOT 汇总: {summary_path}")
        print(f"{'=' * 80}")
        print(v6_summary.to_string(index=False))

        # Also save full summary (v4+v6)
        full_path = OOT_DIR / "oot_summary_full.csv"
        summary.to_csv(full_path, index=False)
        print(f"\n完整汇总: {full_path}")
    else:
        print("没有成功评估任何股票")


if __name__ == "__main__":
    main()
