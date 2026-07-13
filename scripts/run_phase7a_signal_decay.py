"""Phase 7A 驱动脚本 1：长期信号衰减验证。

核心问题：Alpha191 的稳定排序信息能否延伸到 20d/40d/60d？
长周期若无效，Phase 7A 后半段低换手引擎暂停，不继续堆规则。

对 5 个 horizon (5/10/20/40/60d) 做滚动 OOS（4 折双月度）：
  - 每折独立 purge（逐 horizon，不沿用固定截断日期）；
  - 每折在 train 上重新筛选因子/方向/去相关；
  - test 冻结评估 RankIC + 分位 spread；
  - equal_weight 主方案，best_single 对照；
  - 预固定合格门槛（不看完结果再定）。

产出 reports/phase7a_signal_decay_report.md。
"""
from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))
from src.fusion import alpha_rule_fusion as arf
from src.validation import factor_validator as fv

REPORT = PROJECT / "reports" / "phase7a_signal_decay_report.md"

DATA_START = "2025-01-02"
PRICE_END = "2026-03-31"
HORIZONS = (5, 10, 20, 40, 60)
EMBARGO = arf.EMBARGO
MAIN = "equal_weight"
CONTROL = "best_single"

# 双月度 test 区块（同 Phase 6B）
BLOCKS = [
    ("2025-04-30", "2025-05-01", "2025-06-30"),
    ("2025-06-30", "2025-07-01", "2025-08-31"),
    ("2025-08-31", "2025-09-01", "2025-10-31"),
    ("2025-10-31", "2025-11-01", "2025-12-31"),
]
ROLL_TRAIN_START = {
    "2025-04-30": "2025-01-02", "2025-06-30": "2025-03-01",
    "2025-08-31": "2025-05-01", "2025-10-31": "2025-07-01",
}

# ---- 预固定合格门槛（写完即定，不从 test 结果反推）----
GATE_RANKIC_POS_FOLDS = 3 / 4          # 至少 3/4 fold RankIC > 0
GATE_POOLED_RANKIC_POS = 0.0           # 合并 OOS RankIC > 0
GATE_POOLED_RANKICIR_MIN = 0.25        # 合并 RankICIR > 0.25
GATE_WORST_FOLD_RANKIC_MIN = -0.01     # 最差 fold RankIC >= -0.01
GATE_QUANTILE_CONSISTENT = 3 / 4       # 至少 3/4 fold 分位 spread 与 RankIC 同向


def _daily_score_ic(scores: pd.DataFrame, fwd: pd.DataFrame, h: int) -> pd.DataFrame:
    fc = f"fwd_{h}d"
    m = (scores.rename(columns={"final_score": "signal_value"})
         .merge(fwd[["trade_date", "symbol", fc]], on=["trade_date", "symbol"], how="inner"))
    return fv._daily_corr(m, fc)


def _score_quintile_spread(scores: pd.DataFrame, fwd: pd.DataFrame, h: int) -> dict:
    fc = f"fwd_{h}d"
    m = (scores.rename(columns={"final_score": "signal_value"})
         .merge(fwd[["trade_date", "symbol", fc]], on=["trade_date", "symbol"], how="inner"))
    spread, top_mean, bot_mean, _ = fv._quintile_spread_turnover(m, fc)
    return {"spread": spread, "top_mean": top_mean, "bot_mean": bot_mean}


def run_fold(mode: str, panel, fwd, block: tuple[str, str, str], h: int) -> dict:
    train_end, test_start, test_end = block
    train_start = DATA_START if mode == "expanding" else ROLL_TRAIN_START[train_end]
    sub_panel = panel[panel["trade_date"] >= pd.Timestamp(train_start)]

    # 逐 horizon 独立 purge
    fit = arf.fit_fusion(sub_panel, fwd, h, train_end=train_end,
                         test_start=test_start, embargo=EMBARGO)
    rec = {"mode": mode, "train_start": train_start, "train_end": train_end,
           "test_start": test_start, "test_end": test_end, "horizon": h}

    if "error" in fit:
        rec["error"] = fit["error"]; rec["kept"] = []
        return rec

    kept = fit["kept"]
    signs = {c: float(fit["schemes"][MAIN]["signs"][c]) for c in kept}
    rec.update({"kept": kept, "signs": signs, "n_kept": len(kept),
                "best_single_factor": fit["schemes"][CONTROL]["factors"][0],
                "purged_days": fit["purge"].get("purged_rows", np.nan)})

    # test 冻结评估
    test_block = panel[(panel["trade_date"] >= pd.Timestamp(test_start))
                       & (panel["trade_date"] <= pd.Timestamp(test_end))]
    for tag, scheme_name in (("main", MAIN), ("control", CONTROL)):
        scores = arf.build_scheme_scores(test_block, fit["schemes"][scheme_name])
        ic_df = _daily_score_ic(scores, fwd, h)
        if len(ic_df):
            rankic = float(ic_df["RankIC"].mean())
            rankicir = float(rankic / ic_df["RankIC"].std()) if ic_df["RankIC"].std() > 0 else np.nan
            pos_ratio = float((ic_df["RankIC"] > 0).mean())
        else:
            rankic, rankicir, pos_ratio = np.nan, np.nan, np.nan
        qs = _score_quintile_spread(scores, fwd, h)
        rec[tag] = {"rankic": rankic, "rankicir": rankicir, "ic_pos_ratio": pos_ratio,
                    "spread": qs["spread"], "top_mean": qs["top_mean"],
                    "bot_mean": qs["bot_mean"], "n_ic_days": int(len(ic_df)),
                    "daily_ics": list(ic_df["RankIC"].values) if len(ic_df) else []}
    return rec


def aggregate(folds: list[dict]) -> dict:
    ok = [f for f in folds if "error" not in f]
    n = len(ok)
    if n == 0:
        return {"n_folds": 0}

    ric = [f["main"]["rankic"] for f in ok]
    ricir = [f["main"]["rankicir"] for f in ok]
    spreads = [f["main"]["spread"] for f in ok if not np.isnan(f["main"]["spread"])]

    # 因子入选频率 & 方向翻转
    freq: dict[str, int] = {}
    dir_by_factor: dict[str, set] = {}
    for f in ok:
        for c in f["kept"]:
            freq[c] = freq.get(c, 0) + 1
            dir_by_factor.setdefault(c, set()).add(f["signs"][c])
    flipped = [c for c, s in dir_by_factor.items() if len(s) > 1]

    # kept 集合稳定性：相邻折 Jaccard
    jac = []
    for a, b in zip(ok[:-1], ok[1:]):
        sa, sb = set(a["kept"]), set(b["kept"])
        if sa or sb:
            jac.append(len(sa & sb) / len(sa | sb))

    # 合并 OOS（pooled）RankIC/ICIR：所有 test 日 IC 合并计算
    all_daily_ics = []
    for f in ok:
        all_daily_ics.extend(f["main"].get("daily_ics", []))
    if all_daily_ics:
        arr = np.array(all_daily_ics)
        pooled_ric = float(np.mean(arr))
        pooled_ricir = float(pooled_ric / np.std(arr)) if np.std(arr) > 0 else np.nan
    else:
        pooled_ric, pooled_ricir = np.nan, np.nan

    best_ric = [f["control"]["rankic"] for f in ok]

    return {
        "n_folds": n, "total_folds": len(folds),
        "rankic_pos_ratio": float(np.mean([r > 0 for r in ric])),
        "rankicir_min": float(np.nanmin(ricir)), "rankicir_max": float(np.nanmax(ricir)),
        "rankicir_mean": float(np.nanmean(ricir)),
        "worst_fold_rankic": float(np.nanmin(ric)),
        "pooled_rankic": pooled_ric,
        "pooled_rankicir": pooled_ricir,
        "spread_mean": float(np.nanmean(spreads)) if spreads else np.nan,
        "spread_pos_ratio": float(np.mean([s > 0 for s in spreads])) if spreads else np.nan,
        "n_kept_min": min(f["n_kept"] for f in ok), "n_kept_max": max(f["n_kept"] for f in ok),
        "factor_freq": dict(sorted(freq.items(), key=lambda kv: -kv[1])),
        "direction_flipped": flipped,
        "flip_rate": len(flipped) / len(dir_by_factor) if dir_by_factor else np.nan,
        "kept_jaccard_mean": float(np.mean(jac)) if jac else np.nan,
        "best_rankic_mean": float(np.nanmean(best_ric)) if best_ric else np.nan,
        "best_rankicir_mean": (float(np.nanmean(best_ric) / np.nanstd(best_ric))
                                if len(best_ric) > 1 and np.nanstd(best_ric) > 0 else np.nan),
    }


def _p(x, d=3, pct=False):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "n/a"
    return f"{x*100:+.{d}f}%" if pct else f"{x:+.{d}f}"


def qualify(agg: dict) -> dict:
    """应用预固定门槛，返回每个 horizon（+ mode）的合格判定。"""
    n = agg.get("n_folds", 0)
    if n == 0:
        return {"pass": False, "reason": "无有效折"}

    checks = {}
    checks["g1_pos_folds"] = agg["rankic_pos_ratio"] >= GATE_RANKIC_POS_FOLDS
    checks["g2_pooled_ric_pos"] = (agg["pooled_rankic"] is not None
                                    and agg["pooled_rankic"] > GATE_POOLED_RANKIC_POS)
    checks["g3_pooled_ricir"] = (agg["pooled_rankicir"] is not None
                                  and agg["pooled_rankicir"] > GATE_POOLED_RANKICIR_MIN)
    checks["g4_worst_fold"] = (agg["worst_fold_rankic"] is not None
                                and agg["worst_fold_rankic"] >= GATE_WORST_FOLD_RANKIC_MIN)
    checks["g5_quantile_consistent"] = (agg["spread_pos_ratio"] is not None
                                         and agg["spread_pos_ratio"] >= GATE_QUANTILE_CONSISTENT)
    # 不依赖单一fold：ICIR 极差 < 2×|均值|（即范围不过度宽于中心）
    icir_span = agg["rankicir_max"] - agg["rankicir_min"]
    checks["g6_not_single_fold"] = icir_span < abs(agg["rankicir_mean"]) * 2 + 1e-9
    all_pass = all(checks.values())
    failed = [k for k, v in checks.items() if not v]
    return {"pass": all_pass, "checks": checks, "failed_gates": failed}


def build_report(results: dict, agg: dict, qual: dict) -> str:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    L = ["# Phase 7A 长期信号衰减验证报告\n\n",
         f"生成时间: {ts}  |  主方案={MAIN}, 对照={CONTROL}  |  "
         f"horizon={list(HORIZONS)}, embargo={EMBARGO}\n\n",
         "> **目的**：验证 Alpha191 的稳定排序信息是否能延伸到 20d/40d/60d。\n",
         "> 长周期若无效，Phase 7A 后半段低换手引擎暂停，不继续堆规则。\n",
         "> 每折在其**自己的 train** 上独立确定因子/方向/权重；逐 horizon 独立 purge；test 冻结。\n",
         "> 合格门槛**预固定**（不看完结果再定），见文末。\n\n---\n"]

    # ---- 衰减曲线概览 ----
    L.append("\n## 衰减曲线：跨 horizon RankIC / RankICIR\n\n")
    L.append("| horizon | expanding pooled RankIC | expanding pooled RankICIR | "
             "rolling pooled RankIC | rolling pooled RankICIR | 合格? |\n")
    L.append("|---|---|---|---|---|---|\n")
    for h in HORIZONS:
        ae = agg.get((h, "expanding"), {})
        ar = agg.get((h, "rolling"), {})
        qe = qual.get((h, "expanding"), {})
        qr = qual.get((h, "rolling"), {})
        L.append(f"| {h}d | {_p(ae.get('pooled_rankic'))} | {_p(ae.get('pooled_rankicir'))} | "
                 f"{_p(ar.get('pooled_rankic'))} | {_p(ar.get('pooled_rankicir'))} | "
                 f"exp={'✓' if qe.get('pass') else '✗'} roll={'✓' if qr.get('pass') else '✗'} |\n")
    L.append("\n")

    # ---- Per-horizon detail ----
    for h in HORIZONS:
        L.append(f"\n## 持有 {h}d\n")
        for mode in ("expanding", "rolling"):
            folds = results[(h, mode)]
            a = agg[(h, mode)]
            q = qual[(h, mode)]
            L.append(f"\n### {mode} window\n")
            if a.get("n_folds", 0) == 0:
                L.append("无有效折（所有折均无因子通过筛选或全部报错）。\n")
                continue

            L.append("\n| 折 | train | test | 入选数 | 主RankIC | 主RankICIR | "
                     "IC正日比 | 分位spread | top均值 | bot均值 | "
                     "对照RankIC | 对照RankICIR |\n")
            L.append("|---|---|---|---|---|---|---|---|---|---|---|---|\n")
            for f in folds:
                if "error" in f:
                    L.append(f"| {f['test_start'][:7]} | {f['train_start'][:7]}→{f['train_end'][:7]} | "
                             f"{f['test_start'][:7]}→{f['test_end'][:7]} | 0 | "
                             f"无因子通过 | | | | | | | |\n")
                    continue
                mm, cc = f["main"], f["control"]
                L.append(f"| {f['test_start'][:7]} | {f['train_start'][:7]}→{f['train_end'][:7]} | "
                         f"{f['test_start'][:7]}→{f['test_end'][:7]} | {f['n_kept']} | "
                         f"{_p(mm['rankic'])} | {_p(mm['rankicir'])} | "
                         f"{_p(mm['ic_pos_ratio'], 0, True)} | "
                         f"{_p(mm['spread'], 2, True)} | {_p(mm['top_mean'], 2, True)} | "
                         f"{_p(mm['bot_mean'], 2, True)} | "
                         f"{_p(cc['rankic'])} | {_p(cc['rankicir'])} |\n")

            L.append(f"\n**{mode} 聚合（{a['n_folds']}/{a['total_folds']} 折）**：\n")
            L.append(f"- RankIC 正折比: {_p(a['rankic_pos_ratio'], 0, True)}  |  "
                     f"pooled RankIC: {_p(a['pooled_rankic'])}  |  pooled RankICIR: {_p(a['pooled_rankicir'])}\n")
            L.append(f"- RankICIR 跨折: [{_p(a['rankicir_min'], 2)}, {_p(a['rankicir_max'], 2)}] "
                     f"均值 {_p(a['rankicir_mean'], 2)}  |  最差折 RankIC: {_p(a['worst_fold_rankic'])}\n")
            L.append(f"- 分位 spread 均值: {_p(a['spread_mean'], 2, True)}  |  "
                     f"spread 正比率: {_p(a['spread_pos_ratio'], 0, True)}\n")
            L.append(f"- best_single RankIC 均值: {_p(a['best_rankic_mean'])}  |  "
                     f"best_single RankICIR 均值: {_p(a['best_rankicir_mean'])}\n")
            L.append(f"- 入选因子数: {a['n_kept_min']}–{a['n_kept_max']}  |  "
                     f"kept 相邻 Jaccard: {_p(a['kept_jaccard_mean'], 2)}  |  "
                     f"方向翻转: {a['direction_flipped'] or '无'}"
                     f"{' (' + _p(a['flip_rate'], 0, True) + ')' if not np.isnan(a.get('flip_rate', np.nan)) else ''}\n")
            L.append(f"- 因子入选频率: " + ", ".join(f"{k}×{v}" for k, v in a.get("factor_freq", {}).items()) + "\n")

            # 门槛判定
            if q:
                L.append(f"\n**预固定门槛判定**: {'✓ 通过' if q['pass'] else '✗ 未通过'}")
                if not q['pass']:
                    L.append(f"（未过: {', '.join(q['failed_gates'])}）")
                L.append("\n")
                for gk, gv in q["checks"].items():
                    L.append(f"  - {gk}: {'✓' if gv else '✗'}\n")

    # ---- 跨 horizon 因子稳定性 ----
    L.append("\n---\n\n## 跨 horizon 因子稳定性\n\n")
    L.append("| horizon | expanding 入选数范围 | expanding Jaccard | "
             "rolling 入选数范围 | rolling Jaccard | 翻转因子数 |\n")
    L.append("|---|---|---|---|---|---|\n")
    for h in HORIZONS:
        ae = agg.get((h, "expanding"), {})
        ar = agg.get((h, "rolling"), {})
        L.append(f"| {h}d | {ae.get('n_kept_min','-')}–{ae.get('n_kept_max','-')} | "
                 f"{_p(ae.get('kept_jaccard_mean'), 2)} | "
                 f"{ar.get('n_kept_min','-')}–{ar.get('n_kept_max','-')} | "
                 f"{_p(ar.get('kept_jaccard_mean'), 2)} | "
                 f"{len(ae.get('direction_flipped', [])) + len(ar.get('direction_flipped', []))} |\n")

    # ---- 合格 horizon 汇总 ----
    L.append("\n---\n\n## 合格 horizon 汇总\n\n")
    L.append("| horizon | expanding | rolling | 综合判定 |\n")
    L.append("|---|---|---|---|\n")
    for h in HORIZONS:
        qe_pass = qual.get((h, "expanding"), {}).get("pass", False)
        qr_pass = qual.get((h, "rolling"), {}).get("pass", False)
        both = qe_pass and qr_pass
        either = qe_pass or qr_pass
        verdict = "✓✓ 双窗通过" if both else ("△ 单窗通过" if either else "✗ 未通过")
        L.append(f"| {h}d | {'✓' if qe_pass else '✗'} | {'✓' if qr_pass else '✗'} | {verdict} |\n")

    L.append("\n### 预固定合格门槛定义\n\n")
    L.append(f"1. 至少 {int(GATE_RANKIC_POS_FOLDS*4)}/4 fold RankIC > 0\n")
    L.append(f"2. 合并 OOS RankIC > {GATE_POOLED_RANKIC_POS}\n")
    L.append(f"3. 合并 RankICIR > {GATE_POOLED_RANKICIR_MIN}\n")
    L.append(f"4. 最差 fold RankIC ≥ {GATE_WORST_FOLD_RANKIC_MIN}\n")
    L.append("5. 高分位相对低分位方向与 RankIC 一致（≥3/4 折 spread > 0）\n")
    L.append("6. 不依赖单一 fold（ICIR 跨折极差 < 2×|均值|）\n")
    L.append("7. 因子方向翻转率不显著高于 5d/10d 基准（在衰减曲线表格中对比）\n\n")

    L.append("### 下一步决策\n\n")
    L.append("- **有合格长周期（20d+）**：继续 Phase 7A Commit 2 低换手状态机。\n")
    L.append("- **仅 20d 勉强有效**：缩小 Phase 7A，只做 20d 低换手。\n")
    L.append("- **20/40/60d 全部失效**：停止死磕 Alpha191，转入 Phase 8 多源因子工厂。\n")

    return "".join(L)


def main():
    print("加载面板/标签/行情 ...")
    panel = arf.load_alpha_panel(DATA_START, "2025-12-31")
    fwd = arf.load_fwd(HORIZONS)
    codes = sorted(panel["symbol"].unique())
    print(f"  panel {panel['trade_date'].nunique()}日 {len(codes)}股  |  "
          f"horizon={list(HORIZONS)}")

    results, agg, qual = {}, {}, {}
    for h in HORIZONS:
        for mode in ("expanding", "rolling"):
            folds = [run_fold(mode, panel, fwd, blk, h) for blk in BLOCKS]
            results[(h, mode)] = folds
            a = aggregate(folds)
            agg[(h, mode)] = a
            q = qualify(a)
            qual[(h, mode)] = q
            print(f"  {h}d {mode}: folds={a.get('n_folds')}/{a.get('total_folds')} "
                  f"RankICpos={_p(a.get('rankic_pos_ratio'),0,True)} "
                  f"pooledRIC={_p(a.get('pooled_rankic'))} "
                  f"pooledRICIR={_p(a.get('pooled_rankicir'))} "
                  f"qualify={'PASS' if q.get('pass') else 'FAIL'} "
                  f"({', '.join(q.get('failed_gates', []))})")

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(build_report(results, agg, qual), encoding="utf-8")
    print(f"\n报告已写入: {REPORT}")


if __name__ == "__main__":
    main()
