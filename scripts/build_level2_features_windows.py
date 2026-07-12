"""Level-2 全可用窗口特征生产入口（可部署到 Windows 训练机）。

权威 Level-2 原始数据在 Windows：data/single_stock/{code}/raw/{YYYYMMDD}/逐笔成交.csv
（1102 只，flat 结构，2025-03-03 ~ 2025-12-31）。本入口在 Windows 本地跑 DBSCAN 特征，
产出 parquet 后回传 Mac 做验证（Windows 无日线，标签/验证在 Mac）。

用法（Windows PowerShell 示例）:
    python scripts\\build_level2_features_windows.py ^
        --data-root  C:\\Users\\1\\Desktop\\institution-alpha\\data\\single_stock ^
        --output-root C:\\Users\\1\\Desktop\\l2_features_out ^
        --start-date 2025-03-03 --end-date 2025-12-31 ^
        --workers 6 --resume

断点续跑: 再次以 --resume 运行会自动跳过已完成（feature_version 一致）的股票。
--force 强制全部重算; --limit N 只跑前 N 只（冒烟）; --symbols-file 指定股票清单。

不硬编码 Mac 绝对路径; 所有输入输出由参数给定。
"""
import argparse
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from src.features import level2_batch as batch


def _read_symbols_file(path: str) -> list[str]:
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    return [ln.strip() for ln in lines if ln.strip() and not ln.startswith("#")]


def main():
    ap = argparse.ArgumentParser(description="Level-2 全可用窗口特征生产（可断点续跑）")
    ap.add_argument("--data-root", required=True,
                    help="single_stock 目录（含 {code}/raw/{YYYYMMDD}/...）")
    ap.add_argument("--output-root", required=True, help="输出目录（中间产物 + 合并结果）")
    ap.add_argument("--symbols-file", default=None,
                    help="股票清单文件（每行一个 code）；缺省则扫描 data-root")
    ap.add_argument("--start-date", default=None, help="YYYY-MM-DD 或 YYYYMMDD（含）")
    ap.add_argument("--end-date", default=None, help="YYYY-MM-DD 或 YYYYMMDD（含）")
    ap.add_argument("--workers", type=int, default=1, help="并行工作数")
    ap.add_argument("--executor", choices=["thread", "process"], default="thread",
                    help="thread=线程池（Windows 分离会话/计划任务下可用，默认）; "
                         "process=进程池（仅交互式会话可靠）")
    ap.add_argument("--resume", action="store_true", help="跳过已完成且版本一致的股票")
    ap.add_argument("--force", action="store_true", help="强制全部重算")
    ap.add_argument("--limit", type=int, default=None, help="只处理前 N 只（冒烟）")
    ap.add_argument("--shard-index", type=int, default=0, help="分片编号 (0..shard-count-1)")
    ap.add_argument("--shard-count", type=int, default=1, help="分片总数（轮转分片）")
    ap.add_argument("--no-merge", action="store_true",
                    help="只产 by_symbol 中间产物，不合并（分片 worker 用）")
    ap.add_argument("--merge-only", action="store_true",
                    help="不处理股票，只把已有 by_symbol 合并为最终产物 + manifest")
    args = ap.parse_args()

    symbols = _read_symbols_file(args.symbols_file) if args.symbols_file else None

    if args.merge_only:
        params = {"data_root": args.data_root, "output_root": args.output_root,
                  "start_date": args.start_date, "end_date": args.end_date,
                  "workers": args.workers, "resume": args.resume, "force": args.force,
                  "limit": args.limit, "executor": args.executor,
                  "shard_count": args.shard_count, "merge_only": True}
        m = batch.merge_and_write(args.output_root, params, project_root=PROJECT)
        print(f"[merge-only] feature_rows={m['feature_rows']} "
              f"symbols_with_ok={m['symbols_with_ok_days']} "
              f"stock_days_ok={m['stock_days_ok']}")
        return

    batch.run(
        data_root=args.data_root,
        output_root=args.output_root,
        symbols=symbols,
        start_date=args.start_date,
        end_date=args.end_date,
        workers=args.workers,
        executor=args.executor,
        resume=args.resume,
        force=args.force,
        limit=args.limit,
        shard_index=args.shard_index,
        shard_count=args.shard_count,
        do_merge=not args.no_merge,
        project_root=PROJECT,
    )


if __name__ == "__main__":
    main()
