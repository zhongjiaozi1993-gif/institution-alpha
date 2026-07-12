"""日线补数脚本：对给定股票清单下载/合并日线，单票失败不中断、失败落盘。

底层 load_stock_daily 已带 socket 超时（见 price_loader.READ_TIMEOUT）+ 指数退避重试，
本脚本再套一层：逐只 try/except，任何单票异常或空数据都记录并继续，全部失败清单落盘，
避免再次出现补数进程永久挂起 / 单票失败整批终止。

用法:
    python3 scripts/backfill_daily.py                                  # 默认补 L2 缺日线清单
    python3 scripts/backfill_daily.py --symbols-file X.txt --start 2024-12-01 --end 2026-07-11
"""
import argparse
import sys
import time
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))
from src.data.price_loader import load_stock_daily

DEFAULT_SYMBOLS = PROJECT / "data" / "processed" / "level2" / "level2_missing_daily_symbols.txt"
DEFAULT_FAILED_OUT = PROJECT / "data" / "processed" / "level2" / "level2_backfill_failed.txt"


def main():
    ap = argparse.ArgumentParser(description="日线补数（超时/重试/单票失败不中断/失败落盘）")
    ap.add_argument("--symbols-file", default=str(DEFAULT_SYMBOLS))
    ap.add_argument("--start", default="2024-12-01",
                    help="前置到窗口起点之前，供年初特征/流动性回看")
    ap.add_argument("--end", default=time.strftime("%Y-%m-%d"),
                    help="后延覆盖年末远期标签（10/20日）")
    ap.add_argument("--adjust", default="hfq")
    ap.add_argument("--failed-out", default=str(DEFAULT_FAILED_OUT))
    args = ap.parse_args()

    codes = [ln.strip().zfill(6) for ln in open(args.symbols_file, encoding="utf-8") if ln.strip()]
    print(f"[backfill] {len(codes)} symbols {args.start}..{args.end} adjust={args.adjust}", flush=True)

    ok = empty = fail = 0
    failed: list[str] = []
    t0 = time.time()
    for i, code in enumerate(codes, 1):
        try:
            df = load_stock_daily(code, args.start, args.end, adjust=args.adjust)
            if df is None or len(df) == 0:
                empty += 1
                failed.append(f"{code}\tempty")
            else:
                ok += 1
        except Exception as e:  # noqa: BLE001 — 单票失败不影响其它
            fail += 1
            failed.append(f"{code}\t{type(e).__name__}: {e}"[:160])
        if i % 25 == 0 or i == len(codes):
            print(f"  [{i}/{len(codes)}] ok={ok} empty={empty} fail={fail} "
                  f"elapsed={time.time()-t0:.0f}s", flush=True)

    out = Path(args.failed_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(failed) + ("\n" if failed else ""), encoding="utf-8")
    print(f"[backfill] DONE ok={ok} empty={empty} fail={fail} | failed_list -> {out}", flush=True)


if __name__ == "__main__":
    main()
