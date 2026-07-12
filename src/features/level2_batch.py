"""Level-2 特征批量生产编排（可断点续跑，跨平台可部署）。

设计目标（Phase 5.2A）
----------------------
1102 只 × 约 200 交易日的逐笔 DBSCAN 特征生产**不能设计成一次失败全部重来**。
采用**分股票中间产物**：

    {output_root}/by_symbol/{symbol}.parquet         # 该股票全部 ok 特征行
    {output_root}/audit_by_symbol/{symbol}.csv       # 该股票全部 symbol-day 审计

- 单只股票成功后立即落盘；单只失败不影响其他股票。
- --resume 跳过「已落盘且 feature_version 一致」的股票。
- 全部完成后合并为 level2_daily_features.parquet / level2_coverage_audit.csv /
  level2_skipped_stock_days.csv / level2_feature_metadata.parquet / run_manifest.json。

本模块只做**特征生产 + 覆盖审计**，不下有效性结论、不做验证。所有路径均由参数传入，
不硬编码 Mac 绝对路径，Windows 可直接部署运行。
"""
from __future__ import annotations

import json
import os
import subprocess
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

from src.features import level2_feature_builder as fb

BY_SYMBOL = "by_symbol"
AUDIT_BY_SYMBOL = "audit_by_symbol"


# ─────────────────────────────────────────────────────────────
# 路径 / 完成判定
# ─────────────────────────────────────────────────────────────
def symbol_output_paths(output_root: str | Path, code: str) -> tuple[Path, Path]:
    root = Path(output_root)
    return (root / BY_SYMBOL / f"{code}.parquet",
            root / AUDIT_BY_SYMBOL / f"{code}.csv")


def is_symbol_complete(output_root: str | Path, code: str,
                       feature_version: str = fb.FEATURE_VERSION) -> bool:
    """已落盘（parquet + audit 均在）且 audit 里的 feature_version 与当前一致 → 视为已完成。

    空审计（该股票无任何日目录）也算完成，避免反复重扫空股票。
    """
    pq, audit = symbol_output_paths(output_root, code)
    if not (pq.exists() and audit.exists()):
        return False
    try:
        adf = pd.read_csv(audit)
    except Exception:
        return False
    if adf.empty:
        return True
    if "feature_version" not in adf.columns:
        return False
    return bool((adf["feature_version"].astype(str) == str(feature_version)).all())


def discover_symbols(data_root: str | Path) -> list[str]:
    """扫描 data_root 下形如 {code}/raw 的股票目录。"""
    root = Path(data_root)
    if not root.exists():
        return []
    out = []
    for d in sorted(os.listdir(root)):
        if (root / d / "raw").is_dir():
            out.append(d)
    return out


def select_symbols(all_symbols: list[str], output_root: str | Path,
                   feature_version: str, resume: bool, force: bool,
                   limit: int | None) -> list[str]:
    codes = list(all_symbols)
    # resume 先剔除已完成，再取 limit —— 支持"连着跑的分块"：每块 --resume --limit N 处理下一批
    if resume and not force:
        codes = [c for c in codes if not is_symbol_complete(output_root, c, feature_version)]
    if limit is not None:
        codes = codes[:limit]
    return codes


# ─────────────────────────────────────────────────────────────
# 单股票 worker（进程池可调用，必须 top-level）
# ─────────────────────────────────────────────────────────────
def process_and_write_symbol(code: str, data_root: str, output_root: str,
                             start_day: str | None, end_day: str | None) -> dict:
    """构建单股票特征 + 审计并立即落盘。返回结果摘要（供合并/manifest 统计）。"""
    result = {"symbol": code, "status": "ok", "feat_rows": 0,
              "audit_days": 0, "ok_days": 0, "error": ""}
    try:
        df, audit_rows = fb.build_stock_features(
            code, single_stock_root=data_root, start_day=start_day, end_day=end_day)
        audit_df = fb.audit_frame(audit_rows)
        pq, audit_path = symbol_output_paths(output_root, code)
        pq.parent.mkdir(parents=True, exist_ok=True)
        audit_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(pq, index=False)              # 可能为空表（schema 保留）
        audit_df.to_csv(audit_path, index=False)
        result["feat_rows"] = int(len(df))
        result["audit_days"] = int(len(audit_df))
        result["ok_days"] = int((audit_df["status"] == "ok").sum()) if not audit_df.empty else 0
    except Exception as e:  # noqa: BLE001 — 单股票失败不影响其它股票
        result["status"] = "failed"
        result["error"] = f"{type(e).__name__}: {e}"[:300]
    return result


# ─────────────────────────────────────────────────────────────
# 合并 / manifest
# ─────────────────────────────────────────────────────────────
def merge_outputs(output_root: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    root = Path(output_root)
    feat_cols = ["trade_date", "symbol", "layout_type"] + fb.FEATURE_NAMES

    frames = []
    for p in sorted((root / BY_SYMBOL).glob("*.parquet")):
        d = pd.read_parquet(p)
        if not d.empty:
            frames.append(d)
    features = (pd.concat(frames, ignore_index=True) if frames
                else pd.DataFrame(columns=feat_cols))
    if not features.empty:
        features["trade_date"] = pd.to_datetime(features["trade_date"])
        features["symbol"] = features["symbol"].astype(str).str.zfill(6)
        features = features.sort_values(["trade_date", "symbol"]).reset_index(drop=True)

    audits = []
    for p in sorted((root / AUDIT_BY_SYMBOL).glob("*.csv")):
        a = pd.read_csv(p, dtype={"symbol": str, "day": str})
        if not a.empty:
            audits.append(a)
    audit = (pd.concat(audits, ignore_index=True) if audits
             else fb.audit_frame([]))
    if not audit.empty:
        audit["symbol"] = audit["symbol"].astype(str).str.zfill(6)
        audit["trade_date"] = pd.to_datetime(audit["trade_date"])
        audit = audit.sort_values(["symbol", "day"]).reset_index(drop=True)
    return features, audit


def _git_commit(project_root: Path) -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                             cwd=str(project_root), capture_output=True, text=True, timeout=10)
        return out.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def build_manifest(features: pd.DataFrame, audit: pd.DataFrame, results: list[dict],
                   params: dict, project_root: Path) -> dict:
    layout_dist = ({} if audit.empty
                   else audit["selected_layout"].value_counts().to_dict())
    status_dist = ({} if audit.empty
                   else audit["status"].value_counts().to_dict())
    ok_audit = audit[audit["status"] == "ok"] if not audit.empty else audit
    run_failed = [r["symbol"] for r in results if r["status"] == "failed"]
    return {
        "feature_version": fb.FEATURE_VERSION,
        "dbscan": {"eps": fb.DBSCAN_EPS, "min_samples": fb.DBSCAN_MIN_SAMPLES,
                   "min_total_amount_wan": fb.DBSCAN_MIN_TOTAL_WAN},
        "git_commit": _git_commit(project_root),
        "data_root": params.get("data_root"),
        "output_root": params.get("output_root"),
        "start_date": params.get("start_date"),
        "end_date": params.get("end_date"),
        # ---- merged/global（跨所有 run，报告以此为准）----
        "symbols_total": int(audit["symbol"].nunique()) if not audit.empty else 0,
        "symbols_with_features": int(features["symbol"].nunique()) if not features.empty else 0,
        "symbols_with_ok_days": int(ok_audit["symbol"].nunique()) if not ok_audit.empty else 0,
        "stock_days_scanned": int(len(audit)),
        "stock_days_ok": int(len(ok_audit)),
        "feature_rows": int(len(features)),
        "coverage_start": (audit["trade_date"].min().strftime("%Y-%m-%d")
                           if not audit.empty else None),
        "coverage_end": (audit["trade_date"].max().strftime("%Y-%m-%d")
                         if not audit.empty else None),
        "layout_distribution": {str(k): int(v) for k, v in layout_dist.items()},
        "status_distribution": {str(k): int(v) for k, v in status_dist.items()},
        # ---- 本次 run（增量视角）----
        "run_symbols_processed": len(results),
        "run_symbols_success": sum(1 for r in results if r["status"] == "ok"),
        "run_symbols_failed": len(run_failed),
        "run_failed_symbols": run_failed[:100],
        "run_params": params,
        "generated_at": pd.Timestamp.now().isoformat(timespec="seconds"),
    }


# ─────────────────────────────────────────────────────────────
# 顶层编排
# ─────────────────────────────────────────────────────────────
def _norm_day(s: str | None) -> str | None:
    if not s:
        return None
    return s.replace("-", "").strip()


def merge_and_write(output_root: str | Path, params: dict, results: list[dict] | None = None,
                    project_root: Path | None = None) -> dict:
    """合并所有 by_symbol 产物 → 最终宽表/审计/skipped/metadata/manifest。"""
    output_root = str(Path(output_root))
    project_root = project_root or fb.PROJECT
    features, audit = merge_outputs(output_root)
    manifest = build_manifest(features, audit, results or [], params, Path(project_root))

    out = Path(output_root)
    features.to_parquet(out / "level2_daily_features.parquet", index=False)
    audit.to_csv(out / "level2_coverage_audit.csv", index=False)
    skipped = audit[audit["status"] != "ok"] if not audit.empty else audit
    skipped.to_csv(out / "level2_skipped_stock_days.csv", index=False)
    fb.feature_metadata().to_parquet(out / "level2_feature_metadata.parquet", index=False)
    with open(out / "run_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    return manifest


def run(data_root: str | Path, output_root: str | Path, *,
        symbols: list[str] | None = None, start_date: str | None = None,
        end_date: str | None = None, workers: int = 1, resume: bool = False,
        force: bool = False, limit: int | None = None, executor: str = "thread",
        shard_index: int = 0, shard_count: int = 1, do_merge: bool = True,
        project_root: Path | None = None) -> dict:
    data_root = str(Path(data_root))
    output_root = str(Path(output_root))
    project_root = project_root or fb.PROJECT
    start_day, end_day = _norm_day(start_date), _norm_day(end_date)

    all_syms = symbols if symbols is not None else discover_symbols(data_root)
    all_syms = [str(s).strip() for s in all_syms if str(s).strip()]
    if shard_count > 1:  # 轮转分片：大小盘均摊到各分片
        all_syms = all_syms[shard_index::shard_count]
    todo = select_symbols(all_syms, output_root, fb.FEATURE_VERSION, resume, force, limit)

    Path(output_root, BY_SYMBOL).mkdir(parents=True, exist_ok=True)
    Path(output_root, AUDIT_BY_SYMBOL).mkdir(parents=True, exist_ok=True)

    print(f"[level2_batch] data_root={data_root}", flush=True)
    print(f"[level2_batch] symbols_total={len(all_syms)} to_process={len(todo)} "
          f"shard={shard_index}/{shard_count} "
          f"(resume={resume} force={force} limit={limit}) workers={workers} "
          f"executor={executor} window={start_day or 'ALL'}..{end_day or 'ALL'}", flush=True)

    results: list[dict] = []
    done = 0

    def _log(r):
        nonlocal done
        done += 1
        if done % 25 == 0 or done == len(todo):
            ok = sum(1 for x in results if x["status"] == "ok")
            print(f"  [{done}/{len(todo)}] {r['symbol']}: {r['status']} "
                  f"feat_rows={r['feat_rows']} ok_days={r['ok_days']} success={ok}", flush=True)

    if workers <= 1:
        for code in todo:
            r = process_and_write_symbol(code, data_root, output_root, start_day, end_day)
            results.append(r)
            _log(r)
    else:
        # 线程池默认：本工作负载在 GB18030 解析 + sklearn DBSCAN 的 C 段释放 GIL，
        # 线程可并行，且无 spawn/控制台依赖 → 可在 Windows 分离会话（计划任务）下运行。
        # 进程池只在交互式会话可靠，分离会话下 worker 无法 bootstrap（全部 failed）。
        Pool = ThreadPoolExecutor if executor == "thread" else ProcessPoolExecutor
        with Pool(max_workers=workers) as ex:
            futs = {ex.submit(process_and_write_symbol, code, data_root, output_root,
                              start_day, end_day): code for code in todo}
            for fut in as_completed(futs):
                code = futs[fut]
                try:
                    r = fut.result()
                except Exception as e:  # noqa: BLE001 — worker 崩溃也记为该股票失败
                    r = {"symbol": code, "status": "failed", "feat_rows": 0,
                         "audit_days": 0, "ok_days": 0, "error": f"{type(e).__name__}: {e}"[:300]}
                results.append(r)
                _log(r)

    # 若 resume 且部分股票被跳过，合并阶段仍会纳入其已落盘产物。
    params = {"data_root": data_root, "output_root": output_root,
              "start_date": start_date, "end_date": end_date,
              "workers": workers, "resume": resume, "force": force, "limit": limit,
              "executor": executor, "shard_index": shard_index, "shard_count": shard_count}

    if not do_merge:
        ok = sum(1 for r in results if r["status"] == "ok")
        print(f"[level2_batch] SHARD {shard_index}/{shard_count} DONE (no merge) "
              f"processed={len(results)} success={ok}", flush=True)
        return {"shard_index": shard_index, "shard_count": shard_count,
                "run_symbols_processed": len(results),
                "run_symbols_success": ok, "do_merge": False}

    manifest = merge_and_write(output_root, params, results, project_root=Path(project_root))
    print(f"[level2_batch] DONE feature_rows={manifest['feature_rows']} "
          f"stock_days_ok={manifest['stock_days_ok']} "
          f"symbols_with_ok={manifest['symbols_with_ok_days']} "
          f"run_success={manifest['run_symbols_success']} "
          f"run_failed={manifest['run_symbols_failed']}", flush=True)
    print(f"[level2_batch] layout={manifest['layout_distribution']} "
          f"status={manifest['status_distribution']}", flush=True)
    return manifest
