"""
跨日匿名机构识别版本评估 (v4 vs v6)

读取 v4/v6 的 institution_registry.json，统一字段、计算 L2 口径未来收益，
输出匿名机构级和版本级的 Alpha 评估报告。

v5 预留接口，当前跳过。

输出:
  - data/processed/l2_daily_ohlc.csv        L2逐笔OHLC基准价
  - data/processed/crossday_operations_unified.csv  统一操作明细
  - data/processed/crossday_anon_eval.csv            匿名机构级评估
  - data/processed/crossday_version_eval.csv         版本级评估
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

DATA_DIR = PROJECT / "data" / "single_stock"
OUT_DIR = PROJECT / "data" / "processed"
PRICE_SCALE = 10000
FWD_HORIZONS = [1, 3, 5, 10]

# ─── L2 OHLC ───────────────────────────────────────────────────────


def _l2_daily_ohlc(stock: str, date_str: str) -> dict | None:
    cj_path = DATA_DIR / stock / "raw" / date_str / f"{stock}.SZ" / "逐笔成交.csv"
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
    dates = sorted([d.name for d in raw_dir.iterdir()
                    if d.is_dir() and len(d.name) == 8 and (d / f"{stock}.SZ").exists()])
    rows = []
    for d in dates:
        ohlc = _l2_daily_ohlc(stock, d)
        if ohlc:
            rows.append({"date": d, **ohlc})
    return pd.DataFrame(rows)


def attach_forward_returns(ops: pd.DataFrame,
                           l2_ohlc: pd.DataFrame) -> pd.DataFrame:
    """为每条 BUY 操作附加未来 L2 收益"""
    l2 = l2_ohlc.set_index("date")
    for h in FWD_HORIZONS:
        ops[f"fwd_{h}d"] = np.nan
    ops["win_5d"] = np.nan

    buy_mask = ops["direction"] == "BUY"
    buy_dates = ops.loc[buy_mask, "date"].values
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
                ops.loc[(ops["date"] == d) & buy_mask, f"fwd_{h}d"] = ret
        ret5 = ops.loc[(ops["date"] == d) & buy_mask, "fwd_5d"].values
        ops.loc[(ops["date"] == d) & buy_mask, "win_5d"] = (ret5 > 0).astype(float)

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

    with open(path) as f:
        data = json.load(f)

    rows = []
    for inst in data:
        anon = inst["anon_id"]
        conf = _v4_confidence(inst)
        btype = _v4_behavior_type(inst)
        for c in inst.get("all_clusters", []):
            rows.append({
                "version": "v4",
                "anon_id": anon,
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

    with open(path) as f:
        data = json.load(f)

    rows = []
    for inst in data:
        anon = inst["anon_id"]
        conf = _v6_confidence(inst)
        btype = _v6_behavior_type(inst)
        for op in inst.get("operations", []):
            rows.append({
                "version": "v6",
                "anon_id": anon,
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


def load_v5(stock: str) -> pd.DataFrame:
    """预留: v5 暂不评估"""
    path = DATA_DIR / stock / "sofia_v5" / "institution_registry.json"
    if not path.exists():
        print(f"  v5: registry not found, skipping")
        return pd.DataFrame()
    print(f"  v5: found but not yet implemented, skipping")
    return pd.DataFrame()


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
            "buy_days": n_buy,
            "sell_days": n_sell,
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

        # Top 10 by net_buy_wan
        top10 = g.nlargest(10, "net_buy_wan")

        rows.append({
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
            "avg_fwd_5d": _safe_mean(g["avg_fwd_5d"].dropna()),
            "win_5d": _safe_mean(g["win_5d"].dropna()),
            "top10_avg_fwd_5d": _safe_mean(top10["avg_fwd_5d"].dropna()),
            "top10_win_5d": _safe_mean(top10["win_5d"].dropna()),
            # Fragmentation: share of ops from LOW-confidence institutions
            "fragmentation_score": round(
                low["signal_count"].sum() / max(1, g["signal_count"].sum()), 3),
        })
    return pd.DataFrame(rows)


# ─── Main ───────────────────────────────────────────────────────────


def main():
    stocks = [d.name for d in DATA_DIR.iterdir()
              if d.is_dir() and (d / "raw").exists()]

    if not stocks:
        print("未找到股票数据")
        sys.exit(1)

    # 只处理有 L2 数据的股票
    stock = stocks[0]  # 当前只有 002516
    print(f"评估股票: {stock}\n")

    # ─── Step 1: L2 OHLC ───
    print("Step 1: 构建 L2 OHLC 基准价格表...")
    l2_ohlc = build_l2_ohlc_table(stock)
    l2_path = OUT_DIR / "l2_daily_ohlc.csv"
    l2_ohlc.to_csv(l2_path, index=False)
    print(f"  → {l2_path} ({len(l2_ohlc)} 天)")

    # ─── Step 2: 加载各版本 ───
    print("\nStep 2: 加载各版本机构注册表...")
    dfs = []
    for loader, label in [(load_v4, "v4"), (load_v6, "v6"), (load_v5, "v5")]:
        df = loader(stock)
        if not df.empty:
            dfs.append(df)

    if not dfs:
        print("没有找到任何版本的机构数据")
        sys.exit(1)

    ops = pd.concat(dfs, ignore_index=True)
    ops["date"] = ops["date"].astype(str)

    # ─── Step 3: 附加前向收益 ───
    print("\nStep 3: 附加 L2 口径未来收益...")
    ops = attach_forward_returns(ops, l2_ohlc)

    # ─── Step 4: 输出统一操作表 ───
    print("\nStep 4: 输出统一操作表...")
    unified_path = OUT_DIR / "crossday_operations_unified.csv"
    ops.to_csv(unified_path, index=False)
    print(f"  → {unified_path} ({len(ops)} 条)")

    # 快速检查
    for ver in ops["version"].unique():
        vops = ops[ops["version"] == ver]
        buys = vops[vops["direction"] == "BUY"]
        print(f"  {ver}: {len(vops)} ops, BUY={len(buys)}, "
              f"fwd_5d均值={buys['fwd_5d'].dropna().mean():.2f}%, "
              f"win_5d={buys['win_5d'].dropna().mean():.3f}")

    # ─── Step 5: 机构级评估 ───
    print("\nStep 5: 匿名机构级评估...")
    anon_eval = eval_anon_level(ops)
    anon_eval = anon_eval.sort_values(["version", "net_buy_wan"], ascending=[True, False])
    anon_path = OUT_DIR / "crossday_anon_eval.csv"
    anon_eval.to_csv(anon_path, index=False)
    print(f"  → {anon_path} ({len(anon_eval)} 个机构)")

    # ─── Step 6: 版本级评估 ───
    print("\nStep 6: 版本级评估...")
    ver_eval = eval_version_level(anon_eval)
    ver_path = OUT_DIR / "crossday_version_eval.csv"
    ver_eval.to_csv(ver_path, index=False)
    print(f"  → {ver_path}")

    # ─── 打印版本对比 ───
    print("\n" + "=" * 80)
    print("版本对比摘要")
    print("=" * 80)
    cols = ["version", "n_anon", "n_high", "n_medium", "n_low",
            "avg_buy_ratio", "avg_net_buy_wan", "avg_fwd_5d", "win_5d",
            "top10_avg_fwd_5d", "top10_win_5d", "fragmentation_score"]
    print(ver_eval[cols].to_string(index=False))

    # 打印 Top 机构
    print("\n" + "=" * 80)
    print("Top 10 机构 (按净买入额)")
    print("=" * 80)
    top_cols = ["version", "anon_id", "confidence", "behavior_type",
                "active_days", "buy_ratio", "net_buy_wan", "avg_fwd_5d", "win_5d"]
    print(anon_eval.head(10)[top_cols].to_string(index=False))


if __name__ == "__main__":
    main()
