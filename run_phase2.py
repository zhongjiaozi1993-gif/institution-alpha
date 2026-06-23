"""
Phase 2 — 行为模式检测与Alpha预测（B方案）

不关心"谁"在买，只关心"怎么买"：
  Step 1: 检测拆单聚类 → 提取行为指纹
  Step 2: 计算买入后N日收益 → 构建 行为→收益 数据库
  Step 3: 交叉验证：相似行为是否预测相似收益？
  Step 4: 分层分析 + 跟买信号逻辑

用法:
  python run_phase2.py
  python run_phase2.py --data D:\\level2_data --dates 2026-01-05,2026-01-06
  LEVEL2_DATA=D:\\level2_data python run_phase2.py
"""
from __future__ import annotations
import argparse
import os
from pathlib import Path
import pandas as pd
import numpy as np

from src.data.level2_reader import read_level2_stock_dir, match_orders_to_trades
from src.cluster.split_detector import detect_institution_operations
from src.cluster.behavior_db import BehaviorDB, extract_behavior_fp
from src.data.price_loader import load_stock_daily

# ═══════════ 默认配置（可通过 CLI 或环境变量覆盖） ═══════════
DEFAULT_DATA_ROOT = os.environ.get(
    "LEVEL2_DATA",
    str(Path(__file__).parent.parent / "level2_data"),
)
DEFAULT_DATES = "2026-01-05,2026-01-06,2026-01-07,2026-01-08,2026-01-09,2026-01-12"
FORWARD_HORIZONS = [5, 10, 20]
EPS = 0.15
MIN_SAMPLES = 5
MIN_AMOUNT_WAN = 100


def parse_args():
    p = argparse.ArgumentParser(description="Phase 2 — 行为模式检测与Alpha预测")
    p.add_argument("--data", "-d", default=DEFAULT_DATA_ROOT,
                   help=f"Level-2数据根目录 (默认: {DEFAULT_DATA_ROOT})")
    p.add_argument("--dates", default=DEFAULT_DATES,
                   help=f"逗号分隔日期列表 (默认: {DEFAULT_DATES})")
    p.add_argument("--force", action="store_true",
                   help="强制重新检测，忽略缓存")
    p.add_argument("--min-amount", type=float, default=MIN_AMOUNT_WAN,
                   help=f"最少聚类总金额/万元 (默认: {MIN_AMOUNT_WAN})")
    return p.parse_args()


def main():
    args = parse_args()
    data_root = Path(os.path.expanduser(args.data))
    dates = [d.strip() for d in args.dates.split(",") if d.strip()]
    force = args.force

    out_dir = Path(__file__).parent / "data" / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)

    ops_cache = out_dir / "phase2_operations.parquet"
    rets_cache = out_dir / "phase2_returns.parquet"

    # ---- 检查缓存，决定是否需要重新检测 ----
    if not force and ops_cache.exists() and rets_cache.exists():
        print("检测到缓存，跳过检测步骤，直接加载...")
        print(f"  (用 --force 强制重新检测)")
        ops_df = pd.read_parquet(ops_cache)
        returns_df = pd.read_parquet(rets_cache)
    else:
        if force:
            print("--force: 强制重新检测")
        print(f"数据根目录: {data_root}")
        print(f"日期: {dates}")

        # ---- Step 1: 多日拆单检测 ----
        print("\n" + "=" * 60)
        print("Step 1: 多日拆单检测")
        all_ops = []

        for date in dates:
            day_dir = data_root / date.replace("-", "")
            if not day_dir.exists():
                print(f"  {date}: 目录不存在, 跳过")
                continue

            day_ops = 0
            for stock_dir in sorted(day_dir.iterdir()):
                if not stock_dir.is_dir():
                    continue
                csvs = list(stock_dir.glob("*.csv"))
                if len(csvs) < 3:
                    continue

                try:
                    data = read_level2_stock_dir(stock_dir)
                    if "逐笔委托" not in data or "逐笔成交" not in data:
                        continue
                    wtcj = match_orders_to_trades(data["逐笔委托"], data["逐笔成交"])
                    if wtcj.empty:
                        continue
                    ops = detect_institution_operations(
                        wtcj, eps=EPS, min_samples=MIN_SAMPLES,
                        min_total_amount_wan=args.min_amount,
                    )
                    for op in ops:
                        op["stock_code"] = stock_dir.name.replace(".SZ", "").replace(".SH", "")
                        op["date"] = date
                    all_ops.extend(ops)
                    day_ops += len(ops)
                except Exception:
                    pass

            print(f"  {date}: {day_ops} 个机构操作")

        if not all_ops:
            print("未检测到任何机构操作!")
            return

        ops_df = pd.DataFrame(all_ops)
        ops_df.to_parquet(ops_cache, index=False)

        print(f"\n总计: {len(ops_df)} 个操作, "
              f"{ops_df['stock_code'].nunique()} 只股票")

        # ---- Step 2: 价格下载 + 收益计算 ----
        print("\n" + "=" * 60)
        print("Step 2: 下载价格数据 + 计算N日收益")

        unique_stocks = ops_df["stock_code"].unique()
        print(f"  需下载 {len(unique_stocks)} 只股票的价格...")

        price_data = {}
        failed = 0
        for i, code in enumerate(unique_stocks):
            try:
                df = load_stock_daily(
                    code, start_date="2026-01-01", end_date="2026-03-31",
                    adjust="hfq",
                )
                if not df.empty:
                    df["date"] = pd.to_datetime(df["date"])
                    price_data[code] = df
            except Exception:
                failed += 1
            if (i + 1) % 50 == 0:
                print(f"    {i+1}/{len(unique_stocks)} ...")

        print(f"  下载完成: {len(price_data)} 只有数据, {failed} 失败")

        # ---- 计算每条操作的N日收益 ----
        all_returns = []
        for _, op in ops_df.iterrows():
            stock = op["stock_code"]
            date = pd.to_datetime(op["date"])
            avg_price = op["avg_price"]

            if stock not in price_data or avg_price <= 0:
                continue

            prices = price_data[stock]
            future = prices[prices["date"] >= date].reset_index(drop=True)
            if len(future) < 2:
                continue

            entry_open = future.iloc[0]["open"]
            if entry_open <= 0:
                continue

            row = {
                "stock_code": stock,
                "entry_date": str(date.date()),
                "avg_price": avg_price,
                "entry_open": entry_open,
                "total_amount_wan": op["total_amount_wan"],
                "direction": op["direction"],
                "order_count": op["order_count"],
                "cluster_id": op["cluster_id"],
            }
            for h in FORWARD_HORIZONS:
                if h < len(future):
                    row[f"ret_{h}d"] = round(
                        (future.iloc[h]["close"] - entry_open) / entry_open, 4
                    )
                else:
                    row[f"ret_{h}d"] = np.nan

            all_returns.append(row)

        returns_df = pd.DataFrame(all_returns)
        returns_df.to_parquet(rets_cache, index=False)
        ops_df.to_parquet(ops_cache, index=False)

    # ---- Step 3: 构建行为模式数据库 ----
    print("\n" + "=" * 60)
    print("Step 3: 构建行为模式数据库")

    db = BehaviorDB()

    # 用 ops_df 的索引关联 returns_df
    # returns_df 和 ops_df 行对应（未过滤掉的价格缺失行除外）
    # 重建关联：用 stock_code + entry_date + cluster_id
    ret_lookup = returns_df.set_index(
        [returns_df["stock_code"], returns_df["entry_date"], returns_df["cluster_id"]]
    )

    matched = 0
    for _, op in ops_df.iterrows():
        key = (op["stock_code"], str(pd.to_datetime(op["date"]).date()), op["cluster_id"])
        if key not in ret_lookup.index:
            continue

        ret_row = ret_lookup.loc[key]
        # ret_lookup.loc[key] 可能返回多行(duplicate keys)，取第一行
        if isinstance(ret_row, pd.DataFrame):
            ret_row = ret_row.iloc[0]

        fp = extract_behavior_fp(op.to_dict())
        returns = {
            "ret_5d": ret_row.get("ret_5d", np.nan),
            "ret_10d": ret_row.get("ret_10d", np.nan),
            "ret_20d": ret_row.get("ret_20d", np.nan),
        }
        meta = {
            "stock_code": op["stock_code"],
            "date": op["date"],
            "direction": op["direction"],
            "order_count": int(op["order_count"]),
            "total_amount_wan": float(op["total_amount_wan"]),
            "avg_price": float(op["avg_price"]),
            "time_span_min": float(op.get("time_span_min", 0)),
            "vwap_deviation_pct": float(op.get("vwap_deviation_pct", 0) or 0),
        }
        db.add(fp, returns, meta)
        matched += 1

    print(f"  入库: {len(db)} 条行为模式 (匹配率 {matched}/{len(ops_df)})")

    stats = db.stats()
    print(f"\n  行为模式库统计:")
    for h, s in stats['returns'].items():
        print(f"    {h}: 均值{s['mean']:.2%}, "
              f"中位数{s['median']:.2%}, 胜率{s['win_rate']:.1%}, 样本{s['n']}")

    # ---- Step 4: 交叉验证（时间切分） ----
    print("\n" + "=" * 60)
    print("Step 4: 行为模式交叉验证（时间切分）")

    # 按日期分组
    date_order = sorted(dates)
    split_idx = max(4, len(date_order) * 2 // 3)  # 前2/3训练，后1/3测试
    train_dates = set(date_order[:split_idx])
    test_dates = set(date_order[split_idx:])

    train_db = BehaviorDB()
    test_db = BehaviorDB()

    for i in range(len(db)):
        meta = db.metadata[i]
        if meta["date"] in train_dates:
            train_db.add(db.fingerprints[i], db.returns[i], meta)
        elif meta["date"] in test_dates:
            test_db.add(db.fingerprints[i], db.returns[i], meta)

    print(f"  训练集(Jan5-8): {len(train_db)} 条")
    print(f"  测试集(Jan9,12): {len(test_db)} 条")

    if len(train_db) >= 5 and len(test_db) >= 5:
        print(f"\n  对测试集每条行为，在训练集中找相似模式预测收益...")
        predictions = []
        for i in range(len(test_db)):
            pred = train_db.predict(
                test_db.fingerprints[i],
                horizon="ret_20d",
                top_k=20,
                min_similarity=0.7,
                min_samples=5,
            )
            if pred is None:
                continue

            actual = test_db.returns[i].get("ret_20d", np.nan)
            if np.isnan(actual):
                continue

            predictions.append({
                **test_db.metadata[i],
                "predicted": pred["predicted_return"],
                "actual": actual,
                "n_similar": pred["n_samples"],
                "mean_similarity": pred["mean_similarity"],
                "pred_strength": pred["prediction_strength"],
            })

        pred_df = pd.DataFrame(predictions)
        print(f"  有效预测: {len(pred_df)} 条 (覆盖 {len(pred_df)/len(test_db)*100:.0f}% 测试集)")

        if len(pred_df) >= 10:
            # 按预测强度分组
            print(f"\n  {'预测强度':<16} {'样本':>6} {'预测收':>8} {'实际收':>8} {'偏差':>8} {'胜率':>7}")
            print(f"  {'-'*55}")
            for lo, hi, label in [(-99, -0.01, "负向"), (-0.01, 0.01, "中性"),
                                    (0.01, 0.05, "弱正向"), (0.05, 0.10, "中正向"),
                                    (0.10, 99, "强正向")]:
                seg = pred_df[(pred_df["pred_strength"] >= lo) & (pred_df["pred_strength"] < hi)]
                if len(seg) >= 3:
                    print(f"  {label:<16} {len(seg):>6} "
                          f"{seg['predicted'].mean():>7.1%} {seg['actual'].mean():>7.1%} "
                          f"{(seg['actual'].mean()-seg['predicted'].mean()):>7.1%} "
                          f"{(seg['actual']>0).mean():>7.1%}")

            # 预测vs实际的秩相关
            from scipy.stats import spearmanr
            corr, pval = spearmanr(pred_df["predicted"], pred_df["actual"])
            print(f"\n  Spearman相关: {corr:.3f} (p={pval:.3f})")
            print(f"  若>0.1且p<0.1 → 行为相似度对收益有预测力")
            print(f"  若≈0 → 当前数据量不足以建立行为-收益关联(需更多天)")

    else:
        print("  训练或测试集不足，跳过交叉验证")

    # ---- Step 5: 分层分析（按行为特征） ----
    print("\n" + "=" * 60)
    print("Step 5: 按行为特征分层分析")

    # 构建带收益的完整DataFrame
    all_records = []
    for i in range(len(db)):
        all_records.append({
            **db.metadata[i],
            **db.returns[i],
        })
    df = pd.DataFrame(all_records)
    df = df.dropna(subset=["ret_20d"])

    buy = df[df["direction"] == "BUY"]
    print(f"\n  买入操作: {len(buy)} 笔, 卖出: {len(df)-len(buy)} 笔")

    # 按拆单数分层
    print(f"\n  --- 按拆单数分层 ---")
    print(f"  {'拆单数':<12} {'样本':>6} {'5日收':>8} {'10日收':>8} {'20日收':>8} {'胜率(20日)':>10}")
    print(f"  {'-'*55}")
    for lo, hi, label in [(5, 10, "5-10笔"), (10, 30, "10-30笔"),
                            (30, 100, "30-100笔"), (100, 9999, "100+笔")]:
        seg = buy[(buy["order_count"] >= lo) & (buy["order_count"] < hi)]
        if len(seg) > 0:
            print(f"  {label:<12} {len(seg):>6} "
                  f"{seg['ret_5d'].mean():>7.1%} {seg['ret_10d'].mean():>7.1%} "
                  f"{seg['ret_20d'].mean():>7.1%} {(seg['ret_20d']>0).mean():>9.1%}")

    # 按金额分层
    print(f"\n  --- 按金额分层 ---")
    print(f"  {'金额':<12} {'样本':>6} {'5日收':>8} {'10日收':>8} {'20日收':>8} {'胜率(20日)':>10}")
    print(f"  {'-'*55}")
    for lo, hi, label in [(100, 300, "100-300万"), (300, 1000, "300-1000万"),
                            (1000, 3000, "1000-3000万"), (3000, 999999, "3000万+")]:
        seg = buy[(buy["total_amount_wan"] >= lo) & (buy["total_amount_wan"] < hi)]
        if len(seg) > 0:
            print(f"  {label:<12} {len(seg):>6} "
                  f"{seg['ret_5d'].mean():>7.1%} {seg['ret_10d'].mean():>7.1%} "
                  f"{seg['ret_20d'].mean():>7.1%} {(seg['ret_20d']>0).mean():>9.1%}")

    # 买卖方向对比
    print(f"\n  --- 买卖方向对比 ---")
    print(f"  {'方向':<12} {'样本':>6} {'5日收':>8} {'10日收':>8} {'20日收':>8} {'胜率(20日)':>10}")
    print(f"  {'-'*55}")
    for d in ["BUY", "SELL"]:
        seg = df[df["direction"] == d]
        if len(seg) > 0:
            print(f"  {d:<12} {len(seg):>6} "
                  f"{seg['ret_5d'].mean():>7.1%} {seg['ret_10d'].mean():>7.1%} "
                  f"{seg['ret_20d'].mean():>7.1%} {(seg['ret_20d']>0).mean():>9.1%}")

    # 时间跨度分层
    print(f"\n  --- 按时间跨度分层 ---")
    print(f"  {'时间跨度':<12} {'样本':>6} {'5日收':>8} {'10日收':>8} {'20日收':>8} {'胜率(20日)':>10}")
    print(f"  {'-'*55}")
    for lo, hi, label in [(0, 10, "<10分钟"), (10, 30, "10-30分"),
                            (30, 120, "30-120分"), (120, 9999, ">120分钟")]:
        seg = buy[(buy["time_span_min"] >= lo) & (buy["time_span_min"] < hi)]
        if len(seg) > 0:
            print(f"  {label:<12} {len(seg):>6} "
                  f"{seg['ret_5d'].mean():>7.1%} {seg['ret_10d'].mean():>7.1%} "
                  f"{seg['ret_20d'].mean():>7.1%} {(seg['ret_20d']>0).mean():>9.1%}")

    # Top买入股票
    print(f"\n  --- 机构买入最集中的股票Top10 ---")
    top_stocks = buy.groupby("stock_code").agg(
        买入次数=("total_amount_wan", "count"),
        总买入金额_万=("total_amount_wan", "sum"),
        平均20日收益=("ret_20d", "mean"),
        胜率=("ret_20d", lambda x: (x > 0).mean()),
    ).sort_values("买入次数", ascending=False).head(10)

    for code, row in top_stocks.iterrows():
        print(f"  {code}: {int(row['买入次数'])}次买入, "
              f"总{row['总买入金额_万']:.0f}万元, "
              f"20日{row['平均20日收益']:.1%}, "
              f"胜率{row['胜率']:.0%}")

    # ---- 保存 ----
    db.save(out_dir / "behavior_db")
    print(f"\n结果已保存到 {out_dir}/")

    # ---- 跟买信号逻辑演示 ----
    print("\n" + "=" * 60)
    print("跟买信号逻辑演示（基于行为模式数据库）")
    print("""
    每日收盘后:
      1. 检测当日新拆单聚类
      2. 提取行为指纹
      3. 在DB中找K个最相似的历史行为
      4. 若相似行为 historially 20日收益>0且胜率>55% → 生成跟买信号
      5. 信号强度 = 预测收益 × 胜率 × min(相似样本数/20, 1)

    当前数据积累不足时的 fallback（分层统计）:
      - 大拆单(100+笔)买入 → 历史胜率XX% → 跟
      - 小拆单(5-10笔)卖出 → 历史胜率XX% → 忽略
      - 高金额(3000万+)买入 → 历史收益率XX% → 跟
    """)

    # 演示：取10条买入操作，展示信号生成
    print("  演示（10条买入操作）:")
    buy_indices = [i for i in range(len(db)) if db.metadata[i]["direction"] == "BUY"]
    demo_indices = buy_indices[:10]

    print(f"  {'股票':<8} {'日期':<12} {'笔数':>5} {'金额(万)':>10} "
          f"{'预测20日':>9} {'相似样本':>8} {'信号强度':>8} {'建议':<8}")
    print(f"  {'-'*70}")

    for idx in demo_indices:
        pred = db.predict(
            db.fingerprints[idx], horizon="ret_20d",
            top_k=20, min_similarity=0.7, min_samples=5,
            exclude_indices={idx},
        )
        meta = db.metadata[idx]

        if pred is None:
            print(f"  {meta['stock_code']:<8} {meta['date']:<12} "
                  f"{meta['order_count']:>5} {meta['total_amount_wan']:>10.0f} "
                  f"{'N/A':>9} {'N/A':>8} {'N/A':>8} {'数据不足':<8}")
        else:
            strength = pred["prediction_strength"]
            if strength > 0.05 and pred["win_rate"] > 0.55:
                advice = "跟买✓"
            elif strength > 0 and pred["win_rate"] > 0.5:
                advice = "观察"
            elif strength < -0.02:
                advice = "回避"
            else:
                advice = "中性"

            print(f"  {meta['stock_code']:<8} {meta['date']:<12} "
                  f"{meta['order_count']:>5} {meta['total_amount_wan']:>10.0f} "
                  f"{pred['predicted_return']:>8.1%} {pred['n_samples']:>8} "
                  f"{strength:>8.3f} {advice:<8}")


if __name__ == "__main__":
    main()
