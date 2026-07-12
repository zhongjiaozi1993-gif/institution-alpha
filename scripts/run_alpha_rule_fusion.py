"""Phase 6A 驱动脚本 1：Alpha191 规则融合筛选 + IC 验证 + 暴露审计 + L2 shadow。

产出 reports/alpha_rule_fusion_report.md，并把 test 期各方案打分落盘（数据产物，.gitignore 保护）。

严格 OOS：因子筛选/去相关/方向/权重只在 train 上定；test 冻结评估，不用 test 标签。
分别对 5d/10d 独立执行，不混用两个 horizon 的方向和权重。
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

HORIZONS = (5, 10)
TOP_Q = 5
OUT_SCORES = PROJECT / "data" / "processed" / "fusion"
REPORT = PROJECT / "reports" / "alpha_rule_fusion_report.md"


# ======================================================================
# 打分评估
# ======================================================================
def _quintiles(merged: pd.DataFrame, fc: str) -> dict:
    """每日按 final_score 分 5 组，返回各组均值(%)、spread、扣费 spread、top 换手。"""
    qret = {q: [] for q in range(TOP_Q)}
    top_sets = []
    for date, g in merged.groupby("trade_date"):
        gc = g.dropna(subset=["signal_value", fc])
        if len(gc) < TOP_Q * 2:
            continue
        try:
            q = pd.qcut(gc["signal_value"], TOP_Q, labels=False, duplicates="drop")
        except ValueError:
            continue
        gc = gc.assign(q=q)
        qm = gc.groupby("q")[fc].mean()
        if not all(k in qm.index for k in (0, TOP_Q - 1)):
            continue
        for k in range(TOP_Q):
            if k in qm.index:
                qret[k].append(qm[k])
        top_sets.append((date, set(gc[gc["q"] == TOP_Q - 1]["symbol"])))
    means = {k: (float(np.mean(v)) if v else np.nan) for k, v in qret.items()}
    spread = means[TOP_Q - 1] - means[0] if not (np.isnan(means[TOP_Q - 1]) or np.isnan(means[0])) else np.nan
    return {
        "q_means": means, "spread": spread,
        "cost_adj_spread": (spread - arf.COST_ROUNDTRIP_PCT if not np.isnan(spread) else np.nan),
        "turnover": fv._turnover(top_sets),
        "spread_n_days": len(top_sets),
    }


def eval_score(score_df: pd.DataFrame, fwd: pd.DataFrame, horizon: int) -> dict:
    """给定打分面板评估：RankIC/RankICIR/IC正日比/月度RankIC/五分位/扣费spread/turnover/n。"""
    fc = f"fwd_{horizon}d"
    m = (score_df.rename(columns={"final_score": "signal_value"})
         .merge(fwd[["trade_date", "symbol", fc]], on=["trade_date", "symbol"], how="left"))
    ic = fv._daily_corr(m, fc)
    res = {"n_stocks": int(score_df["symbol"].nunique()),
           "n_dates": int(score_df["trade_date"].nunique())}
    if len(ic):
        std = float(ic["RankIC"].std())
        res["rankic"] = float(ic["RankIC"].mean())
        res["rankicir"] = float(res["rankic"] / std) if std > 0 else 0.0
        res["ic_pos_ratio"] = float((ic["RankIC"] > 0).mean())
        mic = ic.assign(m=ic["trade_date"].dt.to_period("M")).groupby("m")["RankIC"].mean()
        res["monthly_rankic"] = {str(k): round(float(v), 4) for k, v in mic.items()}
        res["ic_n_days"] = int(len(ic))
    else:
        res.update(rankic=np.nan, rankicir=np.nan, ic_pos_ratio=np.nan,
                   monthly_rankic={}, ic_n_days=0)
    res.update(_quintiles(m, fc))
    return res


def exposure_corr(score_df: pd.DataFrame, controls: pd.DataFrame, cols: list[str]) -> dict:
    """每日截面 Spearman(score, control) 的均值 → 各控制变量的暴露。"""
    m = score_df.merge(controls, on=["trade_date", "symbol"], how="inner")
    out = {}
    for c in cols:
        daily = []
        for _, g in m.groupby("trade_date"):
            v = g[["final_score", c]].dropna()
            if len(v) < 10:
                continue
            corr = v["final_score"].corr(v[c], method="spearman")
            if pd.notna(corr):
                daily.append(corr)
        out[c] = round(float(np.mean(daily)), 4) if daily else np.nan
    return out


# ======================================================================
# L2 shadow：alpha 组合分 vs L2 信息（不纳入主结论）
# ======================================================================
def l2_composite_shadow(feat_df: pd.DataFrame) -> pd.DataFrame:
    """35 个 L2 特征等权 z(rank) 综合分（仅作 shadow 对照，不定方向、不参与正式融合）。"""
    from src.features.level2_feature_builder import FEATURE_NAMES
    cols = [c for c in FEATURE_NAMES if c in feat_df.columns]
    parts = []
    for date, g in feat_df.groupby("trade_date"):
        if len(g) < 10:
            continue
        r = g[cols].rank()
        z = (r - r.mean()) / r.std(ddof=0)
        comp = z.mean(axis=1)
        gg = g[["trade_date", "symbol"]].copy()
        gg["l2_composite"] = comp.to_numpy()
        parts.append(gg)
    return pd.concat(parts, ignore_index=True)


def daily_corr_two(a: pd.DataFrame, acol: str, b: pd.DataFrame, bcol: str) -> tuple[float, int]:
    """两个信号面板每日截面 Spearman 均值。"""
    m = a.merge(b, on=["trade_date", "symbol"], how="inner")
    daily = []
    for _, g in m.groupby("trade_date"):
        v = g[[acol, bcol]].dropna()
        if len(v) < 10:
            continue
        c = v[acol].corr(v[bcol], method="spearman")
        if pd.notna(c):
            daily.append(c)
    return (round(float(np.mean(daily)), 4) if daily else np.nan), len(daily)


# ======================================================================
# 报告
# ======================================================================
def _fmt(x, p=4):
    return "n/a" if (x is None or (isinstance(x, float) and np.isnan(x))) else f"{x:+.{p}f}"


def build_report(results: dict, shadow: dict) -> str:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    L = []
    L.append("# Alpha191 规则融合 baseline 报告（Phase 6A）\n")
    L.append(f"生成时间: {ts}  |  train {arf.TRAIN_END} 前 / test {arf.TEST_START} 起  |  "
             f"正式 final_score 仅用 Alpha191，Level-2 仅 shadow\n")
    L.append("> 目标：证明多因子规则融合是否比最优单因子更**稳定**（而非样本内最高收益）。"
             "因子筛选/去相关/方向/权重全部只在 train 定，test 冻结评估。5d/10d 独立，不混用。\n")
    L.append("\n---\n")

    for h in HORIZONS:
        r = results[h]
        pg = r["purge"]
        L.append(f"\n## Horizon {h}d\n")
        L.append(f"purge: horizon={h} embargo={arf.EMBARGO} | 末train信号日 {pg['last_train_trade_date']} "
                 f"(label结束 {pg['last_train_label_end_date']}) < 首test {pg['first_test_trade_date']} | "
                 f"purged_rows={pg['purged_rows']} (horizon {pg['purged_horizon_rows']} + embargo {pg['embargo_rows']})\n")

        # --- 因子筛选 + 去相关 ---
        L.append(f"\n### 1. 因子筛选（train）与去相关\n")
        sd = r["screen"]
        passed = sd[sd["pass"]]
        L.append(f"- 通过入池门槛(coverage≥{arf.MIN_COVERAGE}, |RankICIR|≥{arf.MIN_ABS_RANKICIR}, "
                 f"月度稳定≥{arf.MIN_MONTHLY_CONSISTENCY}, pre-cost分位方向一致)：**{len(passed)}** 个\n")
        L.append(f"- 去相关后代表因子（|日均截面相关|<{arf.CORR_CLUSTER_THRESH}）：**{r['kept']}**\n")
        L.append("\n| 因子 | 方向 | coverage | RankIC | RankICIR | 月度稳定 | 季度稳定 | 扣费spread% | 入池 | 保留 |\n")
        L.append("|---|---|---|---|---|---|---|---|---|---|\n")
        for _, row in sd.iterrows():
            keep = "✓" if row["factor"] in r["kept"] else ""
            pas = "✓" if row["pass"] else ""
            dirn = "正" if row["sign"] > 0 else "反"
            L.append(f"| {row['factor']} | {dirn} | {row['coverage']:.2f} | {_fmt(row['rankic'])} | "
                     f"{_fmt(row['rankicir'],2)} | {_fmt(row['monthly_consistency'],2)} | "
                     f"{_fmt(row['quarterly_consistency'],2)} | {_fmt(row['cost_adj_spread'],2)} | {pas} | {keep} |\n")

        # --- 四方案权重 ---
        L.append(f"\n### 2. 四方案权重（train 冻结）\n")
        L.append("\n| 方案 | 权重 |\n|---|---|\n")
        for name, sc in r["schemes"].items():
            w = ", ".join(f"{k}:{v:.3f}" for k, v in sc["weights"].items())
            L.append(f"| {name} | {w} |\n")

        # --- IC 验证 train/test ---
        L.append(f"\n### 3. IC 验证（train / test）\n")
        L.append("\n| 方案 | 期 | RankIC | RankICIR | IC正日% | 分位spread% | 扣费spread% | top换手 | 有效日 | 有效股 |\n")
        L.append("|---|---|---|---|---|---|---|---|---|---|\n")
        for name in r["schemes"]:
            for split in ("train", "test"):
                e = r["eval"][name][split]
                L.append(f"| {name} | {split} | {_fmt(e['rankic'])} | {_fmt(e['rankicir'],2)} | "
                         f"{_fmt(e['ic_pos_ratio'],2)} | {_fmt(e['spread'],3)} | {_fmt(e['cost_adj_spread'],3)} | "
                         f"{_fmt(e['turnover'],2)} | {e['ic_n_days']} | {e['n_stocks']} |\n")

        # --- 相对 best_single 增量（test）---
        L.append(f"\n### 4. 相对 best_single 增量（test）\n")
        bs = r["eval"]["best_single"]["test"]
        L.append("\n| 方案 | ΔRankIC | ΔRankICIR | RankICIR(方案/基线) |\n|---|---|---|---|\n")
        for name in r["schemes"]:
            e = r["eval"][name]["test"]
            d_ic = e["rankic"] - bs["rankic"] if not (np.isnan(e["rankic"]) or np.isnan(bs["rankic"])) else np.nan
            d_ir = e["rankicir"] - bs["rankicir"] if not (np.isnan(e["rankicir"]) or np.isnan(bs["rankicir"])) else np.nan
            L.append(f"| {name} | {_fmt(d_ic)} | {_fmt(d_ir,2)} | {_fmt(e['rankicir'],2)} / {_fmt(bs['rankicir'],2)} |\n")

        # --- 五分位收益 (test) ---
        L.append(f"\n### 5. 五分位收益（test，各组 {h}d 均值%）\n")
        L.append("\n| 方案 | Q1(低) | Q2 | Q3 | Q4 | Q5(高) |\n|---|---|---|---|---|---|\n")
        for name in r["schemes"]:
            qm = r["eval"][name]["test"]["q_means"]
            L.append(f"| {name} | " + " | ".join(_fmt(qm.get(k), 3) for k in range(TOP_Q)) + " |\n")

        # --- 暴露审计 ---
        L.append(f"\n### 6. 暴露审计（test，每日截面 Spearman；行业无数据源 N/A）\n")
        L.append("\n| 方案 | score | log市值 | log成交额 | 换手 | 波动率 |\n|---|---|---|---|---|---|\n")
        for name in r["schemes"]:
            ex_raw = r["exposure"][name]["raw"]
            ex_neu = r["exposure"][name]["neutralized"]
            L.append(f"| {name} | raw | {_fmt(ex_raw['log_mktcap'])} | {_fmt(ex_raw['log_amount'])} | "
                     f"{_fmt(ex_raw['turnover'])} | {_fmt(ex_raw['volatility'])} |\n")
            L.append(f"| {name} | 中性化 | {_fmt(ex_neu['log_mktcap'])} | {_fmt(ex_neu['log_amount'])} | "
                     f"{_fmt(ex_neu['turnover'])} | {_fmt(ex_neu['volatility'])} |\n")
        L.append("\n> 中性化 score = 逐日对 [log市值, log成交额, 换手, 波动率] 截面 OLS 残差；"
                 "残差对控制变量暴露≈0（构造使然）。\n")
        L.append("\n| 方案 | test RankIC(raw) | test RankIC(中性化残差) | retained |\n|---|---|---|---|\n")
        for name in r["schemes"]:
            raw_ic = r["eval"][name]["test"]["rankic"]
            neu_ic = r["exposure"][name]["resid_rankic"]
            ret = (abs(neu_ic) / abs(raw_ic) * 100) if (not np.isnan(raw_ic) and abs(raw_ic) > 1e-9) else np.nan
            L.append(f"| {name} | {_fmt(raw_ic)} | {_fmt(neu_ic)} | "
                     f"{('%.0f%%' % ret) if not np.isnan(ret) else 'n/a'} |\n")
        L.append("\n> retained = |中性化后 RankIC| / |raw RankIC|。偏低 → 该方案 alpha 主要来自规模/流动性暴露。\n")

    # --- L2 shadow ---
    L.append("\n---\n\n## L2 shadow 对照（不纳入主结论）\n")
    L.append("\n| horizon | 方案 | corr(score, l2_amount_yi) | corr(score, L2_35composite) | 有效日 |\n")
    L.append("|---|---|---|---|---|\n")
    for h in HORIZONS:
        for name in ("stability_weight", "icir_weight"):
            s = shadow[h][name]
            L.append(f"| {h}d | {name} | {_fmt(s['corr_amount'])} | {_fmt(s['corr_l2comp'])} | {s['n_days']} |\n")
    L.append("\n> Alpha191 组合分与 l2_amount_yi / L2 综合分的每日截面相关。若组合分已高度暴露于成交额/规模，"
             "说明其信息与 L2（本质规模/流动性代理，见 5.2C）大幅重叠，L2 无独立增量空间。本阶段不做 L2 权重优化。\n")

    # --- 验收判定 ---
    L.append("\n---\n\n## 验收判定（规则融合是否优于单因子）\n")
    L.append(verdict_block(results))
    L.append("\n## 已知限制\n")
    L.append("1. 单一 train/test 切分（2025），无跨年滚动；test 仅 4 个月。\n")
    L.append("2. 暴露审计缺**行业**（无数据源），规模/流动性已控制但行业暴露未剔除。\n")
    L.append("3. 中性化为逐日线性 OLS，未做稳健/非线性控制。\n")
    L.append(f"4. Alpha191 覆盖 {results[5]['eval']['equal_weight']['train']['n_stocks']} 只（有日线因子的股票），非全市场。\n")
    return "".join(L)


def verdict_block(results: dict) -> str:
    """按验收标准逐 horizon 判定：test RankICIR 优于或相近但更稳、增量是否成立。"""
    lines = []
    any_pass = False
    for h in HORIZONS:
        r = results[h]
        bs = r["eval"]["best_single"]["test"]
        best_scheme, best_ir = None, -np.inf
        for name in ("equal_weight", "icir_weight", "stability_weight"):
            ir = abs(r["eval"][name]["test"]["rankicir"])
            if not np.isnan(ir) and ir > best_ir:
                best_scheme, best_ir = name, ir
        bs_ir = abs(bs["rankicir"]) if not np.isnan(bs["rankicir"]) else np.nan
        cond = (not np.isnan(best_ir) and not np.isnan(bs_ir) and best_ir >= bs_ir - 1e-9)
        any_pass = any_pass or cond
        lines.append(f"- **{h}d**：最优融合方案 = {best_scheme}，test |RankICIR|={best_ir:.2f} "
                     f"vs best_single {bs_ir:.2f} → {'融合 RankICIR ≥ 单因子 ✓' if cond else '未超过单因子 ✗'}\n")
    lines.append(f"\n> IC 层面结论：{'至少一个 horizon 融合 RankICIR ≥ best_single（稳定性不劣）' if any_pass else '融合未在 IC 层面超过 best_single'}。"
                 " 完整验收（Sharpe/回撤/收益集中度/暴露可解释）见回测报告 alpha_fusion_backtest_report.md。\n")
    return "".join(lines)


# ======================================================================
def main():
    print("加载面板/标签/控制变量 ...")
    panel = arf.load_alpha_panel("2025-01-01", "2025-12-31")
    fwd = arf.load_fwd(HORIZONS)
    controls = arf.load_exposure_controls()
    ctrl_cols = ["log_mktcap", "log_amount", "turnover", "volatility"]
    feat = arf.lv.load_l2_features() if hasattr(arf.lv, "load_l2_features") else None
    l2_amount = None
    l2_comp = None
    if feat is not None:
        l2_amount = feat[["trade_date", "symbol", "l2_amount_yi"]].copy()
        l2_comp = l2_composite_shadow(feat)

    results, shadow = {}, {}
    for h in HORIZONS:
        print(f"\n=== horizon {h}d ===")
        fit = arf.fit_fusion(panel, fwd, h)
        train_panel, test_panel, _ = arf.purge_train_test(panel, h)
        r = {"purge": fit["purge"], "screen": fit["screen"], "kept": fit["kept"],
             "schemes": fit["schemes"], "eval": {}, "exposure": {}}
        OUT_SCORES.mkdir(parents=True, exist_ok=True)
        test_scores_all = []
        for name, sc in fit["schemes"].items():
            tr = arf.build_scheme_scores(train_panel, sc)
            te = arf.build_scheme_scores(test_panel, sc)
            r["eval"][name] = {"train": eval_score(tr, fwd, h), "test": eval_score(te, fwd, h)}
            # 暴露 raw + 中性化
            ex_raw = exposure_corr(te, controls, ctrl_cols)
            resid = arf.neutralize_scores(te, controls, ctrl_cols)
            resid_eval = eval_score(resid.rename(columns={"resid": "final_score"}), fwd, h)
            ex_neu = exposure_corr(resid.rename(columns={"resid": "final_score"}), controls, ctrl_cols)
            r["exposure"][name] = {"raw": ex_raw, "neutralized": ex_neu,
                                   "resid_rankic": resid_eval["rankic"]}
            te2 = te.copy(); te2["scheme"] = name; te2["horizon"] = h
            test_scores_all.append(te2)
            print(f"  {name:16s} test RankIC={_fmt(r['eval'][name]['test']['rankic'])} "
                  f"RankICIR={_fmt(r['eval'][name]['test']['rankicir'],2)}")
        pd.concat(test_scores_all, ignore_index=True).to_parquet(
            OUT_SCORES / f"test_scores_{h}d.parquet", index=False)
        results[h] = r

        # shadow
        shadow[h] = {}
        stab_te = arf.build_scheme_scores(test_panel, fit["schemes"]["stability_weight"])
        icir_te = arf.build_scheme_scores(test_panel, fit["schemes"]["icir_weight"])
        for name, sc_te in (("stability_weight", stab_te), ("icir_weight", icir_te)):
            ca, na = daily_corr_two(sc_te, "final_score", l2_amount, "l2_amount_yi") if l2_amount is not None else (np.nan, 0)
            cc, nc = daily_corr_two(sc_te, "final_score", l2_comp, "l2_composite") if l2_comp is not None else (np.nan, 0)
            shadow[h][name] = {"corr_amount": ca, "corr_l2comp": cc, "n_days": max(na, nc)}

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(build_report(results, shadow), encoding="utf-8")
    print(f"\n报告已写入: {REPORT}")
    print(f"test 打分已落盘: {OUT_SCORES}/test_scores_[5|10]d.parquet")


if __name__ == "__main__":
    main()
