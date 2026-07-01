#!/usr/bin/env python3
"""Build a single-stock institution behavior evidence chain.

Default target is 002516 because the project currently has full local Level-2
features for this stock only.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from src.evidence import build_evidence_chain  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stock", default="002516", help="A-share stock code, default: 002516")
    parser.add_argument("--secucode", default=None, help="Eastmoney secucode, default inferred")
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Skip public source fetching; only use local Level-2 and price data.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory. Default: data/single_stock/<stock>/evidence",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir) if args.output_dir else None
    result = build_evidence_chain(
        project_dir=PROJECT,
        stock_code=args.stock,
        secucode=args.secucode,
        fetch_public=not args.offline,
        output_dir=output_dir,
    )

    notable = result["notable"]
    print("=" * 80)
    print(f"{args.stock} institution behavior evidence chain")
    print("=" * 80)
    print(f"notable events: {len(notable)}")
    print(f"daily csv:      {result['daily_path']}")
    print(f"notable csv:    {result['notable_path']}")
    print(f"public csv:     {result['public_path']}")
    print(f"holder changes: {result['holder_changes_path']}")
    print(f"source status:  {result['status_path']}")
    print(f"report:         {result['report_path']}")

    if len(notable):
        cols = ["date", "behavior_type", "net_wan", "max_op_wan", "max_op_direction", "public_event_count"]
        cols = [c for c in cols if c in notable.columns]
        print("\nTop notable events:")
        print(notable.sort_values("gross_wan", ascending=False)[cols].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
