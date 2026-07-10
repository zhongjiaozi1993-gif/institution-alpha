"""
Run v4+v6 for the 4 heavy stocks that timed out, using --quick mode for v4.
000547, 000887, 000021, 000630 — each 1.9-2.3 GB, 161 days.
"""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
SCRIPTS = PROJECT / "scripts"

STOCKS = ["000547", "000887", "000021", "000630"]


def run_stock(stock: str) -> dict:
    t0 = time.time()
    result = {"stock": stock, "v4_ok": False, "v6_ok": False}

    # Step 1: v4 with --quick (60 days only)
    t1 = time.time()
    print(f"  v4 --quick ...", end=" ", flush=True)
    r = subprocess.run(
        [sys.executable, str(SCRIPTS / "sofia_v4_hunter.py"),
         "--stock", stock, "--year", "2025", "--quick"],
        capture_output=True, text=True, timeout=1200,
        cwd=str(PROJECT),
    )
    result["v4_time"] = round(time.time() - t1, 1)
    result["v4_ok"] = r.returncode == 0
    if result["v4_ok"]:
        print(f"OK ({result['v4_time']}s)", end=" ")
    else:
        err = r.stderr[-150:] if r.stderr else f"exit={r.returncode}"
        result["v4_error"] = err
        print(f"FAIL: {err[:80]}")
        return result

    # Step 2: v6
    t2 = time.time()
    print(f"| v6 ...", end=" ", flush=True)
    r = subprocess.run(
        [sys.executable, str(SCRIPTS / "sofia_v6_enhanced.py"),
         "--stock", stock],
        capture_output=True, text=True, timeout=300,
        cwd=str(PROJECT),
    )
    result["v6_time"] = round(time.time() - t2, 1)
    result["v6_ok"] = r.returncode == 0
    if result["v6_ok"]:
        print(f"OK ({result['v6_time']}s)")
    else:
        err = r.stderr[-150:] if r.stderr else f"exit={r.returncode}"
        result["v6_error"] = err
        print(f"FAIL: {err[:80]}")

    result["total_time"] = round(time.time() - t0, 1)
    return result


def main():
    print(f"Running v4 --quick + v6 for {len(STOCKS)} heavy stocks\n")

    ok, fail = 0, 0
    failures = []
    t_start = time.time()

    for i, stock in enumerate(STOCKS):
        print(f"[{i+1}/{len(STOCKS)}] {stock}")
        r = run_stock(stock)
        if r["v4_ok"] and r["v6_ok"]:
            ok += 1
        else:
            fail += 1
            failures.append(r)

    elapsed = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"Done: OK={ok} FAIL={fail} / {len(STOCKS)} | {elapsed/60:.0f} min")
    if failures:
        print("Failures:")
        for f in failures:
            print(f"  {f['stock']}: v4={f.get('v4_error','ok')} v6={f.get('v6_error','ok')}")


if __name__ == "__main__":
    main()
