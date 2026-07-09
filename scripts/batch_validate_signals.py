"""Sprint 3: Batch validation of multiple Signals on 300-stock universe.

Precomputes forward returns once, then validates all signals in one pass.

Usage:
    python3 scripts/batch_validate_signals.py \
        --signal-dir data/processed/signals/price_alpha191_300 \
        --universe-file data/processed/stock_universe/validation_300.txt \
        --start-date 2025-01-01 \
        --end-date 2025-12-31 \
        --output-dir data/processed/validation/alpha191_sprint3
"""
import argparse
import sys
import pandas as pd
import numpy as np
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT / "scripts"))
from validate_signal_daily import (
    precompute_forward_returns, compute_ic, compute_quintile_returns,
    classify_direction, FWD_HORIZONS, N_QUINTILES,
)


def load_universe(universe_file: str) -> list[str]:
    """Load stock codes from validation_300.txt (code status format)."""
    codes = []
    with open(universe_file) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2 and parts[1] == "ok":
                codes.append(parts[0].zfill(6))
    return codes


def validate_one_signal(
    signal_path: Path, fwd_df: pd.DataFrame,
    start_date: str, end_date: str,
) -> dict:
    """Validate a single signal parquet against precomputed forward returns."""
    sig = pd.read_parquet(signal_path)
    sig["trade_date"] = pd.to_datetime(sig["trade_date"])
    sig = sig[(sig["trade_date"] >= start_date) & (sig["trade_date"] <= end_date)]

    sid = sig["signal_id"].iloc[0] if "signal_id" in sig.columns else signal_path.stem
    sname = sig["signal_name"].iloc[0] if "signal_name" in sig.columns else ""
    sfid = sig["source_formula_id"].iloc[0] if "source_formula_id" in sig.columns else ""

    n_stocks = sig["stock_code"].nunique()
    n_dates = sig["trade_date"].nunique()
    missing_pct = sig["signal_value"].isna().mean() * 100

    df = sig.merge(fwd_df, on=["trade_date", "stock_code"], how="left")

    # Coverage
    coverage = {}
    for h in FWD_HORIZONS:
        col = f"fwd_{h}d"
        coverage[h] = df[col].notna().mean() * 100

    # IC
    ic_df = compute_ic(df)
    ic_summary = {}
    for h in FWD_HORIZONS:
        sub = ic_df[ic_df["horizon"] == h]
        if len(sub) == 0:
            ic_summary[h] = {"IC_mean": np.nan, "RankIC_mean": np.nan, "RankICIR": np.nan,
                             "IC_pos_pct": np.nan, "RankIC_pos_pct": np.nan}
            continue
        ic_summary[h] = {
            "IC_mean": sub["IC"].mean(),
            "RankIC_mean": sub["RankIC"].mean(),
            "RankICIR": sub["RankIC"].mean() / sub["RankIC"].std() if sub["RankIC"].std() > 0 else 0,
            "IC_pos_pct": (sub["IC"] > 0).mean() * 100,
            "RankIC_pos_pct": (sub["RankIC"] > 0).mean() * 100,
        }

    # Quintile + discrete
    q_df, discrete_info = compute_quintile_returns(df)
    q_summary = {}
    for h in FWD_HORIZONS:
        sub = q_df[q_df["horizon"] == h]
        q_means = sub.groupby("quintile")["avg_fwd"].mean()
        if 0 in q_means.index and (N_QUINTILES - 1) in q_means.index:
            spread = q_means[N_QUINTILES - 1] - q_means[0]
        else:
            spread = np.nan
        # long_short win rate
        top_dates = sub[sub["quintile"] == (N_QUINTILES - 1)]
        bottom_dates = sub[sub["quintile"] == 0]
        long_short_win = np.nan
        top_q_win = np.nan
        if len(top_dates) > 0 and len(bottom_dates) > 0:
            top_q_win = (top_dates["avg_fwd"] > 0).mean() * 100
            merged = top_dates[["trade_date", "avg_fwd"]].merge(
                bottom_dates[["trade_date", "avg_fwd"]],
                on="trade_date", suffixes=("_top", "_bottom"),
            )
            if len(merged) > 0:
                long_short_win = (merged["avg_fwd_top"] > merged["avg_fwd_bottom"]).mean() * 100
        q_summary[h] = {"spread": spread, "topq_win": top_q_win, "longshort_win": long_short_win}

    # Direction
    ic5 = ic_summary[5]
    qs5 = q_summary[5]
    direction = classify_direction(
        ic5.get("RankIC_mean", np.nan),
        qs5.get("spread", np.nan),
    )

    # Baseline win rate
    win_baseline = {}
    for h in FWD_HORIZONS:
        col = f"win_{h}d"
        win_baseline[h] = df[col].mean() * 100

    return {
        "signal_id": sid,
        "signal_name": sname,
        "source_formula_id": sfid,
        "n_stocks": n_stocks,
        "n_dates": n_dates,
        "missing_pct": missing_pct,
        "unique_values": discrete_info["unique_value_count"],
        "discrete_warning": discrete_info["discrete_warning"],
        "raw_direction": direction["raw_direction"],
        "recommended_direction": direction["recommended_direction"],
        **{f"coverage_{h}d": coverage[h] for h in FWD_HORIZONS},
        **{f"IC_mean_{h}d": ic_summary[h]["IC_mean"] for h in FWD_HORIZONS},
        **{f"RankIC_mean_{h}d": ic_summary[h]["RankIC_mean"] for h in FWD_HORIZONS},
        **{f"RankICIR_{h}d": ic_summary[h]["RankICIR"] for h in FWD_HORIZONS},
        **{f"RankIC_pos_{h}d": ic_summary[h]["RankIC_pos_pct"] for h in FWD_HORIZONS},
        **{f"spread_{h}d": q_summary[h]["spread"] for h in FWD_HORIZONS},
        **{f"topq_win_{h}d": q_summary[h]["topq_win"] for h in FWD_HORIZONS},
        **{f"longshort_win_{h}d": q_summary[h]["longshort_win"] for h in FWD_HORIZONS},
        **{f"baseline_win_{h}d": win_baseline[h] for h in FWD_HORIZONS},
    }


# ---------------------------------------------------------------------------
# Candidate screening
# ---------------------------------------------------------------------------

def screen_candidates(summary_df: pd.DataFrame) -> pd.DataFrame:
    """Apply Sprint 3 candidate criteria to batch summary."""
    candidates = []
    for _, row in summary_df.iterrows():
        reasons = []
        rk5 = row.get("RankIC_mean_5d", np.nan)
        rkir5 = row.get("RankICIR_5d", np.nan)
        rk10 = row.get("RankIC_mean_10d", np.nan)
        rkir10 = row.get("RankICIR_10d", np.nan)
        spread5 = row.get("spread_5d", np.nan)

        if (not np.isnan(rk5) and not np.isnan(rkir5) and
                rk5 > 0.02 and rkir5 > 0.2):
            reasons.append("A: RankIC_5d>0.02 & RankICIR_5d>0.2")
        if (not np.isnan(rk10) and not np.isnan(rkir10) and
                rk10 > 0.025 and rkir10 > 0.2):
            reasons.append("B: RankIC_10d>0.025 & RankICIR_10d>0.2")
        if not np.isnan(spread5) and spread5 > 0.3:
            reasons.append(f"C: spread_5d={spread5:+.3f}%>0.3%")
        if (not np.isnan(rk5) and not np.isnan(rkir5) and
                rk5 < -0.02 and abs(rkir5) > 0.2):
            reasons.append("D: inverse_candidate (RankIC_5d<-0.02)")

        if reasons:
            candidates.append({
                "signal_id": row["signal_id"],
                "signal_name": row["signal_name"],
                "source_formula_id": row.get("source_formula_id", ""),
                "category": row.get("sub_category", ""),
                "recommended_direction": row.get("recommended_direction", ""),
                "rankic_5d": rk5,
                "rankicir_5d": rkir5,
                "spread_5d": spread5,
                "coverage": row.get("coverage_5d", np.nan),
                "candidate_reason": "; ".join(reasons),
            })
    return pd.DataFrame(candidates)


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def generate_report(summary_df: pd.DataFrame, candidates_df: pd.DataFrame, output_dir: Path):
    """Generate Sprint 3 summary report in markdown."""
    total = len(summary_df)
    n_positive = (summary_df["recommended_direction"] == "original").sum()
    n_inverse = (summary_df["recommended_direction"] == "inverse").sum()
    n_unclear = (summary_df["recommended_direction"] == "unclear").sum()
    n_discrete = summary_df["discrete_warning"].sum()
    n_high_cov = (summary_df["coverage_5d"] > 95).sum()

    rpt = output_dir / "alpha191_sprint3_report.md"
    with open(rpt, "w") as f:
        f.write("# Sprint 3: Alpha191 × 300-stock Validation Report\n\n")
        f.write(f"生成时间: {pd.Timestamp.now().strftime('%Y-%m-%d')}\n\n")
        f.write("---\n\n")

        f.write("## 1. 概览\n\n")
        f.write(f"| 指标 | 数值 |\n")
        f.write(f"|------|------|\n")
        f.write(f"| 接入因子总数 | {total} |\n")
        f.write(f"| 覆盖率 > 95% | {n_high_cov} |\n")
        f.write(f"| 离散信号 | {n_discrete} |\n")
        f.write(f"| 正向有效 (original) | {n_positive} |\n")
        f.write(f"| 负向有效 (inverse) | {n_inverse} |\n")
        f.write(f"| 方向不明 (unclear) | {n_unclear} |\n")
        f.write(f"| Candidate 数量 | {len(candidates_df)} |\n\n")

        # RankIC_5d Top 10
        top_ric = summary_df.dropna(subset=["RankIC_mean_5d"]).nlargest(10, "RankIC_mean_5d")
        f.write("## 2. RankIC_5d Top 10\n\n")
        f.write("| Signal | Name | RankIC_5d | RankICIR_5d | Spread_5d | Direction |\n")
        f.write("|--------|------|-----------|-------------|-----------|-----------|\n")
        for _, r in top_ric.iterrows():
            f.write(f"| {r['signal_id']} | {r['signal_name']} | {r['RankIC_mean_5d']:.4f} | "
                    f"{r['RankICIR_5d']:.2f} | {r['spread_5d']:+.3f}% | {r['recommended_direction']} |\n")
        f.write("\n")

        # RankICIR_5d Top 10
        top_icir = summary_df.dropna(subset=["RankICIR_5d"]).nlargest(10, "RankICIR_5d")
        f.write("## 3. RankICIR_5d Top 10\n\n")
        f.write("| Signal | Name | RankIC_5d | RankICIR_5d | Spread_5d | Direction |\n")
        f.write("|--------|------|-----------|-------------|-----------|-----------|\n")
        for _, r in top_icir.iterrows():
            f.write(f"| {r['signal_id']} | {r['signal_name']} | {r['RankIC_mean_5d']:.4f} | "
                    f"{r['RankICIR_5d']:.2f} | {r['spread_5d']:+.3f}% | {r['recommended_direction']} |\n")
        f.write("\n")

        # Spread_5d Top 10
        top_spread = summary_df.dropna(subset=["spread_5d"]).nlargest(10, "spread_5d")
        f.write("## 4. Top-Bottom Spread_5d Top 10\n\n")
        f.write("| Signal | Name | RankIC_5d | RankICIR_5d | Spread_5d | Direction |\n")
        f.write("|--------|------|-----------|-------------|-----------|-----------|\n")
        for _, r in top_spread.iterrows():
            f.write(f"| {r['signal_id']} | {r['signal_name']} | {r['RankIC_mean_5d']:.4f} | "
                    f"{r['RankICIR_5d']:.2f} | {r['spread_5d']:+.3f}% | {r['recommended_direction']} |\n")
        f.write("\n")

        # Discrete signals
        if n_discrete > 0:
            f.write("## 5. 离散信号警告\n\n")
            discrete = summary_df[summary_df["discrete_warning"] == True]
            f.write("| Signal | Unique Values | Missing Pct |\n")
            f.write("|--------|---------------|-------------|\n")
            for _, r in discrete.iterrows():
                f.write(f"| {r['signal_id']} | {r['unique_values']} | {r['missing_pct']:.1f}% |\n")
            f.write("\n")

        # Negative-effective
        inverse = summary_df[summary_df["recommended_direction"] == "inverse"]
        if len(inverse) > 0:
            f.write("## 6. 负向有效因子 (建议反用)\n\n")
            f.write("| Signal | Name | RankIC_5d | RankICIR_5d | Spread_5d |\n")
            f.write("|--------|------|-----------|-------------|-----------|\n")
            for _, r in inverse.iterrows():
                f.write(f"| {r['signal_id']} | {r['signal_name']} | "
                        f"{r['RankIC_mean_5d']:.4f} | {r['RankICIR_5d']:.2f} | {r['spread_5d']:+.3f}% |\n")
            f.write("\n")

        # Candidates
        f.write("## 7. Candidate Signals\n\n")
        if len(candidates_df) > 0:
            f.write("| Signal | Name | RankIC_5d | RankICIR_5d | Spread_5d | Reason |\n")
            f.write("|--------|------|-----------|-------------|-----------|--------|\n")
            for _, r in candidates_df.iterrows():
                f.write(f"| {r['signal_id']} | {r['signal_name']} | "
                        f"{r['rankic_5d']:.4f} | {r['rankicir_5d']:.2f} | "
                        f"{r['spread_5d']:+.3f}% | {r['candidate_reason']} |\n")
        else:
            f.write("无 Signal 通过 Candidate 标准。\n")
        f.write("\n")

        # Full table
        f.write("## 8. 全量汇总\n\n")
        cols = ["signal_id", "signal_name", "RankIC_mean_5d", "RankICIR_5d",
                "spread_5d", "longshort_win_5d", "recommended_direction", "discrete_warning"]
        available = [c for c in cols if c in summary_df.columns]
        summary_sorted = summary_df.dropna(subset=["RankICIR_5d"]).sort_values("RankICIR_5d", ascending=False)
        f.write(summary_sorted[available].to_string(index=False))
        f.write("\n")

    print(f"  Report: {rpt}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Batch validate multiple Signals on 300-stock universe")
    parser.add_argument("--signal-dir", required=True)
    parser.add_argument("--universe-file", required=True)
    parser.add_argument("--start-date", default="2025-01-01")
    parser.add_argument("--end-date", default="2025-12-31")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    signal_dir = Path(args.signal_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load universe
    stocks = load_universe(args.universe_file)
    print(f"Universe: {len(stocks)} stocks")

    # Precompute forward returns (once for all signals)
    fwd_cache = PROJECT / "data" / "processed" / "prices_300_fwd.parquet"
    if fwd_cache.exists():
        print(f"Loading cached fwd returns: {fwd_cache}")
        fwd_df = pd.read_parquet(fwd_cache)
    else:
        print("Precomputing forward returns for 292 stocks...")
        fwd_df = precompute_forward_returns(stocks, args.start_date, args.end_date)
        fwd_df.to_parquet(fwd_cache, index=False)
        print(f"  Cached: {fwd_cache} ({len(fwd_df)} rows)")

    # Find all signal parquets
    parquets = sorted(signal_dir.glob("signal*.parquet"))
    print(f"Signals to validate: {len(parquets)}")

    # Validate each signal
    summaries = []
    for sp in parquets:
        print(f"  {sp.stem}...", end=" ", flush=True)
        s = validate_one_signal(sp, fwd_df, args.start_date, args.end_date)
        summaries.append(s)
        print(f"RankIC_5d={s['RankIC_mean_5d']:.4f} dir={s['recommended_direction']}")

    summary_df = pd.DataFrame(summaries)

    # Save batch summary CSV
    sum_csv = output_dir / "alpha191_sprint3_summary.csv"
    summary_df.to_csv(sum_csv, index=False)
    print(f"\nBatch summary: {sum_csv} ({len(summary_df)} signals)")

    # Candidate screening
    candidates_df = screen_candidates(summary_df)
    cand_csv = output_dir / "candidate_signals.csv"
    candidates_df.to_csv(cand_csv, index=False)
    print(f"Candidates: {cand_csv} ({len(candidates_df)} signals)")

    # Generate report
    generate_report(summary_df, candidates_df, output_dir)

    # Print top findings
    print("\n=== Top 5 by RankICIR_5d ===")
    top5 = summary_df.dropna(subset=["RankICIR_5d"]).nlargest(5, "RankICIR_5d")
    for _, r in top5.iterrows():
        print(f"  {r['signal_id']} {r['signal_name']}: "
              f"RankICIR_5d={r['RankICIR_5d']:.2f} "
              f"IC={r['RankIC_mean_5d']:.4f} "
              f"spread={r['spread_5d']:+.3f}% "
              f"dir={r['recommended_direction']}")

    print(f"\nDone. Output: {output_dir}")


if __name__ == "__main__":
    main()
