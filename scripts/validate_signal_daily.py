"""Sprint 2+: Single-signal validation with vectorized fwd returns.

Computes IC, RankIC, quintile returns, 3-class win rates, coverage,
discrete-signal detection, and direction classification for one Signal.

Usage:
    python3 scripts/validate_signal_daily.py \
        --signal-file data/processed/signals/price_alpha191/signal017.parquet \
        --output-prefix signal017
"""
import argparse
import pandas as pd
import numpy as np
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
DAILY_DIR = PROJECT / "data" / "daily"
VAL_DIR = PROJECT / "data" / "processed" / "validation"
VAL_DIR.mkdir(parents=True, exist_ok=True)

FWD_HORIZONS = [1, 3, 5, 10, 20]
N_QUINTILES = 5


# ---------------------------------------------------------------------------
# Forward-return precomputation (vectorised)
# ---------------------------------------------------------------------------

def precompute_forward_returns(
    stock_codes: list[str],
    start_date: str, end_date: str,
) -> pd.DataFrame:
    """Build a (trade_date, stock_code) -> fwd returns lookup table.

    Vectorised per stock: close.shift(-h) instead of row-by-row loop.
    Returns DataFrame with columns [trade_date, stock_code, fwd_1d, ... fwd_20d, win_1d, ... win_20d].
    """
    frames = []
    for code in stock_codes:
        p = DAILY_DIR / f"{code}.parquet"
        if not p.exists():
            continue
        pdf = pd.read_parquet(p)
        pdf = pdf.sort_values("date").reset_index(drop=True)
        mask = (pdf["date"] >= start_date) & (pdf["date"] <= end_date)
        pdf = pdf[mask].copy()
        if pdf.empty:
            continue

        close = pdf["close"].values
        n = len(close)
        row = {"trade_date": pdf["date"].values, "stock_code": code}
        for h in FWD_HORIZONS:
            fwd = np.full(n, np.nan)
            valid = np.arange(n - h)
            fwd[valid] = (close[valid + h] / close[valid] - 1) * 100
            row[f"fwd_{h}d"] = fwd
            row[f"win_{h}d"] = (fwd > 0).astype(float)
        frames.append(pd.DataFrame(row))

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# IC / RankIC
# ---------------------------------------------------------------------------

def compute_ic(df: pd.DataFrame) -> pd.DataFrame:
    """Compute daily IC and RankIC (Spearman) per horizon."""
    rows = []
    for date, g in df.groupby("trade_date"):
        if len(g) < 5:
            continue
        for h in FWD_HORIZONS:
            col = f"fwd_{h}d"
            valid = g[["signal_value", col]].dropna()
            if len(valid) < 5:
                continue
            pearson = valid["signal_value"].corr(valid[col], method="pearson")
            spearman = valid["signal_value"].corr(valid[col], method="spearman")
            rows.append({
                "trade_date": date, "horizon": h,
                "IC": pearson, "RankIC": spearman, "n_stocks": len(valid),
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Quintile returns (with discrete-signal detection)
# ---------------------------------------------------------------------------

def compute_quintile_returns(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Compute forward return by signal quintile per date.

    Returns (quintile_df, discrete_info) where discrete_info is
    {unique_value_count, top_value_ratio, warning: bool}.
    """
    n_unique = df["signal_value"].dropna().nunique()
    vc = df["signal_value"].value_counts(normalize=True)
    top_ratio = vc.iloc[0] if len(vc) > 0 else 1.0
    discrete_info = {
        "unique_value_count": n_unique,
        "top_value_ratio": round(top_ratio, 4),
        "discrete_warning": n_unique < N_QUINTILES * 2,
    }

    rows = []
    for date, g in df.groupby("trade_date"):
        g_clean = g.dropna(subset=["signal_value"])
        if len(g_clean) < N_QUINTILES * 2:
            continue
        try:
            g_clean["quintile"] = pd.qcut(
                g_clean["signal_value"], N_QUINTILES,
                labels=False, duplicates="drop",
            )
        except ValueError:
            continue
        for h in FWD_HORIZONS:
            col = f"fwd_{h}d"
            q_means = g_clean.groupby("quintile")[col].mean()
            for q, v in q_means.items():
                rows.append({
                    "trade_date": date, "horizon": h,
                    "quintile": int(q), "avg_fwd": v,
                    "n_stocks": (g_clean["quintile"] == q).sum(),
                })
    return pd.DataFrame(rows), discrete_info


# ---------------------------------------------------------------------------
# Direction detection
# ---------------------------------------------------------------------------

def classify_direction(rankic_5d: float, spread_5d: float) -> dict:
    """Classify signal direction based on RankIC and spread at 5d horizon."""
    if rankic_5d > 0.01 and spread_5d > 0:
        raw = "positive"
        recommended = "original"
    elif rankic_5d < -0.01 and spread_5d < 0:
        raw = "negative"
        recommended = "inverse"
    elif rankic_5d > 0.01 and (np.isnan(spread_5d) or spread_5d <= 0):
        raw = "positive"
        recommended = "unclear"
    elif rankic_5d < -0.01 and (np.isnan(spread_5d) or spread_5d >= 0):
        raw = "negative"
        recommended = "unclear"
    else:
        raw = "unclear"
        recommended = "unclear"
    return {"raw_direction": raw, "recommended_direction": recommended}


# ---------------------------------------------------------------------------
# Main validation entry point
# ---------------------------------------------------------------------------

def validate_signal(
    signal_file: str, output_prefix: str,
    start_date: str = "2025-01-01", end_date: str = "2025-12-31",
    fwd_df: pd.DataFrame | None = None,
):
    """Run full validation for one Signal.

    If fwd_df is provided, forward returns are merged from it
    (avoiding repeated price loads in batch mode).
    """
    sig = pd.read_parquet(signal_file)
    sig["trade_date"] = pd.to_datetime(sig["trade_date"])
    sig = sig[(sig["trade_date"] >= start_date) & (sig["trade_date"] <= end_date)]

    signal_id = sig["signal_id"].iloc[0] if "signal_id" in sig.columns else "Unknown"
    signal_name = sig["signal_name"].iloc[0] if "signal_name" in sig.columns else "Unknown"

    stock_codes = sorted(sig["stock_code"].unique())
    n_stocks = len(stock_codes)
    n_dates = sig["trade_date"].nunique()
    total_rows = len(sig)
    missing_pct = sig["signal_value"].isna().mean() * 100

    print(f"  Signal: {signal_id} ({signal_name})")
    print(f"  Universe: {n_stocks} stocks, {n_dates} dates, {total_rows} rows")
    print(f"  Signal missing: {missing_pct:.1f}%")

    # Attach forward returns
    if fwd_df is not None:
        df = sig.merge(fwd_df, on=["trade_date", "stock_code"], how="left")
    else:
        prices = _load_prices(stock_codes, start_date, end_date)
        df = _attach_forward_returns_legacy(sig, prices)

    # Coverage
    coverage = {}
    for h in FWD_HORIZONS:
        col = f"fwd_{h}d"
        coverage[h] = df[col].notna().mean() * 100

    # IC analysis
    ic_df = compute_ic(df)
    ic_summary = {}
    for h in FWD_HORIZONS:
        sub = ic_df[ic_df["horizon"] == h]
        if len(sub) == 0:
            ic_summary[h] = dict.fromkeys(
                ["IC_mean", "IC_std", "ICIR", "RankIC_mean", "RankIC_std",
                 "RankICIR", "IC_pos_pct", "RankIC_pos_pct"], np.nan)
            continue
        ic_summary[h] = {
            "IC_mean": sub["IC"].mean(),
            "IC_std": sub["IC"].std(),
            "ICIR": sub["IC"].mean() / sub["IC"].std() if sub["IC"].std() > 0 else 0,
            "RankIC_mean": sub["RankIC"].mean(),
            "RankIC_std": sub["RankIC"].std(),
            "RankICIR": sub["RankIC"].mean() / sub["RankIC"].std() if sub["RankIC"].std() > 0 else 0,
            "IC_pos_pct": (sub["IC"] > 0).mean() * 100,
            "RankIC_pos_pct": (sub["RankIC"] > 0).mean() * 100,
        }

    # Quintile analysis + discrete detection
    q_df, discrete_info = compute_quintile_returns(df)
    quintile_summary = {}
    for h in FWD_HORIZONS:
        sub = q_df[q_df["horizon"] == h]
        q_means = sub.groupby("quintile")["avg_fwd"].mean()
        if 0 in q_means.index and (N_QUINTILES - 1) in q_means.index:
            top = q_means[N_QUINTILES - 1]
            bottom = q_means[0]
            spread = top - bottom
        else:
            top = bottom = spread = np.nan
        quintile_summary[h] = {
            "top_quintile_avg": top,
            "bottom_quintile_avg": bottom,
            "spread": spread,
        }

    # --- Win rates (3 types) ---
    win_rate = {}
    for h in FWD_HORIZONS:
        col = f"win_{h}d"
        win_rate[h] = {
            "baseline_win_rate": df[col].mean() * 100,
        }
        sub = q_df[q_df["horizon"] == h]
        if len(sub) > 0:
            top_dates = sub[sub["quintile"] == (N_QUINTILES - 1)]
            bottom_dates = sub[sub["quintile"] == 0]
            win_rate[h]["top_quantile_win_rate"] = (top_dates["avg_fwd"] > 0).mean() * 100
            win_rate[h]["bottom_quantile_win_rate"] = (bottom_dates["avg_fwd"] > 0).mean() * 100
            # long_short: fraction of dates where top > bottom
            merged = top_dates[["trade_date", "avg_fwd"]].merge(
                bottom_dates[["trade_date", "avg_fwd"]],
                on="trade_date", suffixes=("_top", "_bottom"),
            )
            if len(merged) > 0:
                win_rate[h]["long_short_win_rate"] = (
                    merged["avg_fwd_top"] > merged["avg_fwd_bottom"]
                ).mean() * 100
            else:
                win_rate[h]["long_short_win_rate"] = np.nan
        else:
            win_rate[h]["top_quantile_win_rate"] = np.nan
            win_rate[h]["bottom_quantile_win_rate"] = np.nan
            win_rate[h]["long_short_win_rate"] = np.nan

    # Direction
    ic5 = ic_summary[5]
    qs5 = quintile_summary[5]
    direction = classify_direction(
        ic5.get("RankIC_mean", np.nan),
        qs5.get("spread", np.nan),
    )

    # ---- Generate report ----
    rpt_path = VAL_DIR / f"{output_prefix}_validation_report.md"
    sum_path = VAL_DIR / f"{output_prefix}_validation_summary.csv"

    with open(rpt_path, "w") as f:
        f.write(f"# {signal_id}: {signal_name} — Validation Report\n\n")
        f.write(f"生成时间: {pd.Timestamp.now().strftime('%Y-%m-%d')}\n\n")
        f.write("---\n\n")
        f.write("## 基本信息\n\n")
        f.write(f"| 指标 | 数值 |\n")
        f.write(f"|------|------|\n")
        f.write(f"| Signal ID | {signal_id} |\n")
        f.write(f"| Signal Name | {signal_name} |\n")
        f.write(f"| Universe | {n_stocks} stocks |\n")
        f.write(f"| Period | {start_date} ~ {end_date} |\n")
        f.write(f"| Trading days | {n_dates} |\n")
        f.write(f"| Signal missing rate | {missing_pct:.1f}% |\n")
        f.write(f"| Direction | raw={direction['raw_direction']}, recommended={direction['recommended_direction']} |\n")
        f.write(f"| Unique values | {discrete_info['unique_value_count']} |\n")
        if discrete_info["discrete_warning"]:
            f.write(f"| ⚠️ DISCRETE_SIGNAL | 有效值过少({discrete_info['unique_value_count']}<10)，qcut 分组不稳定 |\n")
        f.write("\n")

        f.write("## Coverage\n\n")
        f.write("| Horizon | Coverage |\n")
        f.write("|---------|----------|\n")
        for h in FWD_HORIZONS:
            f.write(f"| {h}d | {coverage[h]:.1f}% |\n")
        f.write("\n")

        f.write("## IC / RankIC\n\n")
        f.write("| Horizon | IC Mean | ICIR | RankIC Mean | RankICIR | IC>0% | RankIC>0% |\n")
        f.write("|---------|---------|------|-------------|----------|-------|------------|\n")
        for h in FWD_HORIZONS:
            s = ic_summary[h]
            f.write(f"| {h}d | {s['IC_mean']:.4f} | {s['ICIR']:.2f} | "
                    f"{s['RankIC_mean']:.4f} | {s['RankICIR']:.2f} | "
                    f"{s['IC_pos_pct']:.0f}% | {s['RankIC_pos_pct']:.0f}% |\n")
        f.write("\n")

        f.write("## Quintile Returns\n\n")
        f.write("| Horizon | Top Q | Bottom Q | Spread |\n")
        f.write("|---------|-------|----------|--------|\n")
        for h in FWD_HORIZONS:
            s = quintile_summary[h]
            f.write(f"| {h}d | {s['top_quintile_avg']:+.3f}% | "
                    f"{s['bottom_quintile_avg']:+.3f}% | {s['spread']:+.3f}% |\n")
        f.write("\n")

        f.write("## Win Rate (3 types)\n\n")
        f.write("| Horizon | Baseline | TopQ Win | BottomQ Win | Long-Short Win |\n")
        f.write("|---------|----------|----------|-------------|----------------|\n")
        for h in FWD_HORIZONS:
            w = win_rate[h]
            f.write(f"| {h}d | {w['baseline_win_rate']:.1f}% | "
                    f"{w['top_quantile_win_rate']:.1f}% | "
                    f"{w['bottom_quantile_win_rate']:.1f}% | "
                    f"{w['long_short_win_rate']:.1f}% |\n")
        f.write("\n")
        f.write("> baseline = 全样本胜率（非因子选股胜率）；TopQ/BottomQ = 分组胜率；Long-Short = Top>Bottom 日期占比\n\n")

        # Judgment
        f.write("## 判断\n\n")
        checks = []
        if not np.isnan(ic5.get("RankIC_mean", np.nan)):
            if ic5["RankIC_mean"] > 0:
                checks.append(f"RankIC_5d = {ic5['RankIC_mean']:.4f} > 0 → 信号有正向选股能力")
            else:
                checks.append(f"RankIC_5d = {ic5['RankIC_mean']:.4f} ≤ 0 → 信号无正向选股能力")
        if not np.isnan(qs5.get("spread", np.nan)) and qs5["spread"] > 0:
            checks.append(f"Top-Bottom Spread = {qs5['spread']:+.3f}% > 0 → 多空分组有效")
        elif not np.isnan(qs5.get("spread", np.nan)):
            checks.append(f"Top-Bottom Spread = {qs5['spread']:+.3f}% ≤ 0 → 分组无效或反向")
        if discrete_info["discrete_warning"]:
            checks.append(f"离散信号警告: 仅 {discrete_info['unique_value_count']} 个唯一值，qcut 分组不可靠")
        for c in checks:
            f.write(f"- {c}\n")
        f.write(f"- Direction: {direction['recommended_direction']}\n")
        f.write("\n")

        f.write("## 结论\n\n")
        rankicir5 = ic5.get("RankICIR", np.nan)
        wr5 = win_rate[5]["long_short_win_rate"]
        if not np.isnan(rankicir5) and rankicir5 > 0.3 and not np.isnan(wr5) and wr5 > 52:
            f.write("**推荐进入 Candidate。**\n")
        elif not np.isnan(ic5.get("RankIC_mean", np.nan)) and ic5["RankIC_mean"] > 0:
            f.write("**有研究价值但暂不满足候选标准。** 需观察更长时间或更大股票池。\n")
        elif direction["recommended_direction"] == "inverse":
            f.write("**负向有效，建议反用。** 对 signal_value 取反后可作候选。\n")
        else:
            f.write("**暂不推荐。** 信号在当前股票池上无明显预测能力。\n")

    print(f"  Report: {rpt_path}")

    # Summary CSV
    summary = {
        "signal_id": signal_id,
        "signal_name": signal_name,
        "n_stocks": n_stocks,
        "n_dates": n_dates,
        "missing_pct": missing_pct,
        "unique_values": discrete_info["unique_value_count"],
        "discrete_warning": discrete_info["discrete_warning"],
        "raw_direction": direction["raw_direction"],
        "recommended_direction": direction["recommended_direction"],
    }
    for h in FWD_HORIZONS:
        s = ic_summary[h]
        summary[f"IC_mean_{h}d"] = s.get("IC_mean", np.nan)
        summary[f"RankIC_mean_{h}d"] = s.get("RankIC_mean", np.nan)
        summary[f"RankICIR_{h}d"] = s.get("RankICIR", np.nan)
        summary[f"IC_pos_pct_{h}d"] = s.get("IC_pos_pct", np.nan)
        qs = quintile_summary[h]
        summary[f"spread_{h}d"] = qs["spread"]
        w = win_rate[h]
        summary[f"baseline_win_{h}d"] = w["baseline_win_rate"]
        summary[f"topq_win_{h}d"] = w["top_quantile_win_rate"]
        summary[f"longshort_win_{h}d"] = w["long_short_win_rate"]

    pd.DataFrame([summary]).to_csv(sum_path, index=False)
    print(f"  Summary: {sum_path}")

    return summary


# ---------------------------------------------------------------------------
# Legacy fallback (used when fwd_df is not provided)
# ---------------------------------------------------------------------------

def _load_prices(stock_codes: list[str], start_date: str, end_date: str) -> dict[str, pd.DataFrame]:
    prices = {}
    for code in stock_codes:
        p = DAILY_DIR / f"{code}.parquet"
        if not p.exists():
            continue
        df = pd.read_parquet(p)
        df = df.sort_values("date").reset_index(drop=True)
        mask = (df["date"] >= start_date) & (df["date"] <= end_date)
        prices[code] = df[mask].copy()
    return prices


def _attach_forward_returns_legacy(signal_df: pd.DataFrame, prices: dict) -> pd.DataFrame:
    df = signal_df.copy()
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    for h in FWD_HORIZONS:
        df[f"fwd_{h}d"] = np.nan
        df[f"win_{h}d"] = np.nan
    for code in df["stock_code"].unique():
        if code not in prices:
            continue
        pdf = prices[code]
        close_series = pdf.set_index("date")["close"].sort_index()
        dates_arr = close_series.index
        mask = df["stock_code"] == code
        idxs = df.loc[mask].index
        for i in idxs:
            t = df.loc[i, "trade_date"]
            if t not in dates_arr:
                continue
            pos = dates_arr.get_loc(t)
            for h in FWD_HORIZONS:
                tgt_pos = pos + h
                if tgt_pos < len(dates_arr):
                    c_t = close_series.iloc[pos]
                    c_fwd = close_series.iloc[tgt_pos]
                    fwd = (c_fwd / c_t - 1) * 100
                    df.loc[i, f"fwd_{h}d"] = fwd
                    df.loc[i, f"win_{h}d"] = 1.0 if fwd > 0 else 0.0
    return df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Validate a single Signal")
    parser.add_argument("--signal-file", required=True)
    parser.add_argument("--output-prefix", required=True)
    parser.add_argument("--start-date", default="2025-01-01")
    parser.add_argument("--end-date", default="2025-12-31")
    args = parser.parse_args()
    validate_signal(args.signal_file, args.output_prefix, args.start_date, args.end_date)


if __name__ == "__main__":
    main()
