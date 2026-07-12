"""Level-2 × 日线 股票交集审计（Phase 5.2B 验证前置）。

Windows 产出的 Level-2 特征回传 Mac 后，先做交集审计：哪些 L2 股票有/无日线。
**关键**：日线缺失 ≠ Level-2 缺失。缺日线的股票要进补数清单，不能在验证报告里
被误判成 Level-2 覆盖问题。

产出:
    data/processed/level2/level2_daily_intersection_audit.csv   # 每股一行 + 分类
    data/processed/level2/level2_missing_daily_symbols.txt      # 补数清单（有L2无日线）
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

L2_FEATURES = PROJECT / "data" / "processed" / "level2" / "level2_daily_features.parquet"
DAILY_DIR = PROJECT / "data" / "daily"
OUT_DIR = PROJECT / "data" / "processed" / "level2"
AUDIT_OUT = OUT_DIR / "level2_daily_intersection_audit.csv"
MISSING_OUT = OUT_DIR / "level2_missing_daily_symbols.txt"

# 已知数据源限制的特殊证券：有 L2 逐笔但当前 akshare-Sina 日线源不支持，非 L2 缺失、非读取失败。
# 后续统一做「特殊证券数据源适配」（CDR / 北交所 / 其他特殊代码）时再补。
SOURCE_LIMIT = {
    "689009": "689009 / 九号公司-WD / CDR / current AkShare-Sina source unsupported",
}


def daily_symbols() -> set[str]:
    if not DAILY_DIR.exists():
        return set()
    return {p.stem.zfill(6) for p in DAILY_DIR.glob("*.parquet")}


def main():
    ap = argparse.ArgumentParser(description="Level-2 × 日线 股票交集审计")
    ap.add_argument("--l2-features", default=str(L2_FEATURES))
    args = ap.parse_args()

    l2 = pd.read_parquet(args.l2_features, columns=["trade_date", "symbol"])
    l2["symbol"] = l2["symbol"].astype(str).str.zfill(6)
    l2_days = l2.groupby("symbol")["trade_date"].nunique()
    l2_syms = set(l2_days.index)

    daily_syms = daily_symbols()

    common = sorted(l2_syms & daily_syms)
    missing_daily_all = sorted(l2_syms - daily_syms)   # 有 L2 无日线
    source_limit = [s for s in missing_daily_all if s in SOURCE_LIMIT]      # 源限制，不阻塞验证
    missing_backfill = [s for s in missing_daily_all if s not in SOURCE_LIMIT]  # 可补数清单
    missing_l2 = sorted(daily_syms - l2_syms)      # 有日线无 L2（信息用）

    rows = []
    for s in sorted(l2_syms | daily_syms):
        in_l2, in_daily = s in l2_syms, s in daily_syms
        note = ""
        if in_l2 and in_daily:
            cat = "common"
        elif in_l2 and s in SOURCE_LIMIT:
            cat = "missing_daily_source_limit"
            note = SOURCE_LIMIT[s]
        elif in_l2:
            cat = "l2_only_missing_daily"
        else:
            cat = "daily_only_missing_l2"
        rows.append({"symbol": s, "in_l2": in_l2, "in_daily": in_daily,
                     "n_l2_days": int(l2_days.get(s, 0)), "category": cat, "note": note})
    audit = pd.DataFrame(rows).sort_values(["category", "symbol"]).reset_index(drop=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    audit.to_csv(AUDIT_OUT, index=False)
    # 补数清单只含「可补数」的（排除源限制特殊证券），避免每次补数都对无法补的证券重试
    Path(MISSING_OUT).write_text("\n".join(missing_backfill) + ("\n" if missing_backfill else ""),
                                 encoding="utf-8")

    print(f"l2_symbols            = {len(l2_syms)}")
    print(f"daily_symbols         = {len(daily_syms)}")
    print(f"common_symbols        = {len(common)}")
    print(f"missing_daily_backfill= {len(missing_backfill)}  (有L2无日线, 可补 → {MISSING_OUT.name})")
    print(f"missing_daily_srclimit= {len(source_limit)}  (源限制, 不阻塞验证: {source_limit})")
    print(f"missing_l2_symbols    = {len(missing_l2)}  (有日线无L2, 信息用)")
    print(f"→ {AUDIT_OUT}")


if __name__ == "__main__":
    main()
