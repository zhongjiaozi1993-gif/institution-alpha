"""Evidence-chain orchestration."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .behavior import load_ops, summarize_behavior
from .market import attach_market_context, load_price
from .public_sources import compute_holder_changes, fetch_public_evidence, match_public_events
from .report import write_markdown_report


def build_evidence_chain(
    project_dir: Path,
    stock_code: str = "002516",
    secucode: str | None = None,
    fetch_public: bool = True,
    output_dir: Path | None = None,
) -> dict[str, pd.DataFrame | Path]:
    """Build local and public evidence chain for one stock."""
    secucode = secucode or _default_secucode(stock_code)
    single_stock_dir = project_dir / "data" / "single_stock" / stock_code
    output_dir = output_dir or single_stock_dir / "evidence"
    output_dir.mkdir(parents=True, exist_ok=True)

    ops = load_ops(single_stock_dir)
    daily, notable = summarize_behavior(ops)
    price = load_price(single_stock_dir)
    daily = attach_market_context(daily, price)
    notable = attach_market_context(notable, price)

    public_evidence = pd.DataFrame()
    holder_changes = pd.DataFrame()
    source_status = pd.DataFrame()
    if fetch_public:
        event_dates = notable["date"].astype(str).tolist()
        public_evidence, source_status = fetch_public_evidence(stock_code, secucode, event_dates)
        holder_changes = compute_holder_changes(public_evidence)
        daily = match_public_events(daily, public_evidence)
        notable = match_public_events(notable, public_evidence)
    else:
        daily = match_public_events(daily, public_evidence)
        notable = match_public_events(notable, public_evidence)

    daily_path = output_dir / "daily_evidence.csv"
    notable_path = output_dir / "notable_events.csv"
    public_path = output_dir / "public_evidence.csv"
    holder_changes_path = output_dir / "holder_changes.csv"
    status_path = output_dir / "source_status.csv"
    report_path = output_dir / "evidence_report.md"

    daily.to_csv(daily_path, index=False)
    notable.to_csv(notable_path, index=False)
    public_evidence.to_csv(public_path, index=False)
    holder_changes.to_csv(holder_changes_path, index=False)
    source_status.to_csv(status_path, index=False)
    write_markdown_report(
        report_path=report_path,
        stock_code=stock_code,
        daily=daily,
        notable=notable,
        public_evidence=public_evidence,
        holder_changes=holder_changes,
        source_status=source_status,
    )

    return {
        "daily": daily,
        "notable": notable,
        "public_evidence": public_evidence,
        "holder_changes": holder_changes,
        "source_status": source_status,
        "daily_path": daily_path,
        "notable_path": notable_path,
        "public_path": public_path,
        "holder_changes_path": holder_changes_path,
        "status_path": status_path,
        "report_path": report_path,
    }


def _default_secucode(stock_code: str) -> str:
    suffix = "SZ" if stock_code.startswith(("0", "3")) else "SH"
    return f"{stock_code}.{suffix}"
