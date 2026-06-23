"""
每日运行管线
盘后一键执行：下载数据 → Alpha归因 → 信号生成 → 风险过滤 → 输出交易计划

使用 akshare 新浪来源API：
  - stock_lhb_jgmx_sina() → 机构席位买卖明细（股票代码、日期、买入额、卖出额）
  - stock_lhb_detail_daily_sina(date) → 每日上榜股票列表（备用）
  - stock_lhb_yytj_sina() → 营业部排名统计（备用）
"""
from __future__ import annotations
import sys
from pathlib import Path
import pandas as pd
import yaml
try:
    from loguru import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.lhb_collector import (
    download_lhb_seat_detail_range, download_lhb_seat_detail,
    download_lhb_jgmx, build_trade_records,
    download_lhb_stocks_daily, download_lhb_seat_stats,
)
from src.data.price_loader import load_stock_daily, load_index_daily
from src.alpha.return_calculator import calculate_future_returns
from src.alpha.alpha_profiler import build_alpha_registry, rank_seats
from src.alpha.dynamic_scorer import score_all_seats
from src.signal.generator import generate_signals, generate_composite_signals
from src.signal.composite import filter_signals
from src.backtest.engine import BacktestEngine
from src.backtest.metrics import compute_full_metrics, print_metrics
from src.risk.regime import detect_regime, adjust_for_regime
from src.risk.decay_monitor import AlphaDecayMonitor
from src.risk.crowding import filter_crowded_stocks


def load_config(config_path: str | None = None) -> dict:
    """加载配置文件"""
    cfg_path = Path(config_path) if config_path else PROJECT_ROOT / "config" / "settings.yaml"
    if cfg_path.exists():
        with open(cfg_path, "r") as f:
            return yaml.safe_load(f)
    return {}


def run_pipeline(
    start_date: str,
    end_date: str,
    config_path: str | None = None,
) -> dict:
    """
    运行完整管线：数据采集 → Alpha归因 → 信号生成 → 回测
    """
    cfg = load_config(config_path)
    alpha_cfg = cfg.get("alpha", {})
    signal_cfg = cfg.get("signal", {})
    trading_cfg = cfg.get("trading", {})
    risk_cfg = cfg.get("risk", {})

    horizons = alpha_cfg.get("horizons", [1, 5, 10, 20, 60])
    max_horizon = max(horizons)
    half_life = alpha_cfg.get("dynamic_half_life", 30)
    lookback = alpha_cfg.get("dynamic_lookback", 90)

    # ============================================================
    # Step 1: 下载机构席位买卖明细（Sina汇总数据）
    # ============================================================
    logger.info(f"Step 1: 下载机构席位买卖明细")
    jgmx = download_lhb_jgmx(force_refresh=True)

    if jgmx.empty:
        logger.error("未获取到机构席位买卖明细")
        return {}

    logger.info(f"机构席位明细列: {list(jgmx.columns)}")

    trade_records = build_trade_records(jgmx, start_date, end_date)

    if trade_records.empty:
        logger.error(f"区间 {start_date}~{end_date} 无交易记录")
        return {}

    logger.info(f"交易记录: {len(trade_records)} 条, "
                f"日期范围 {trade_records['lhb_date'].min()} ~ {trade_records['lhb_date'].max()}")

    # ============================================================
    # Step 2: 下载涉及的股票日线
    # ============================================================
    stock_codes = trade_records["stock_code"].unique().tolist()
    logger.info(f"Step 2: 下载 {len(stock_codes)} 只股票日线")

    # 价格数据需覆盖到 LHB 日期之后 max_horizon 个交易日，才能计算未来收益
    from datetime import datetime, timedelta
    price_end = (datetime.strptime(end_date, "%Y-%m-%d") + timedelta(days=max_horizon * 2)).strftime("%Y-%m-%d")

    price_data = {}
    for code in stock_codes:
        df = load_stock_daily(str(code), start_date, price_end)
        if not df.empty:
            price_data[str(code)] = df

    logger.info(f"成功加载 {len(price_data)} 只股票日线")

    # ============================================================
    # Step 3: 下载基准指数
    # ============================================================
    benchmark_code = alpha_cfg.get("benchmark", "000852")
    logger.info(f"Step 3: 下载基准指数 {benchmark_code}")
    bench_data = load_index_daily(benchmark_code, start_date, price_end)

    # ============================================================
    # Step 4: Alpha归因 — 计算机构买入后N日收益
    # ============================================================
    logger.info("Step 4: Alpha归因")
    seat_histories = calculate_future_returns(
        trade_records, price_data, horizons, bench_data
    )

    if seat_histories.empty:
        logger.error("无有效的交易历史（可能价格数据不足）")
        return {}

    logger.info(f"计算出 {len(seat_histories)} 条带收益记录")

    # ============================================================
    # Step 5: Alpha画像 + 动态评分
    # ============================================================
    logger.info("Step 5: 构建Alpha注册表 + 动态评分")
    registry = build_alpha_registry(
        seat_histories, horizon=20,
        min_trades=alpha_cfg.get("min_trades", 3)
    )

    last_date = str(seat_histories["lhb_date"].max().date()) \
        if hasattr(seat_histories["lhb_date"].max(), "date") \
        else str(seat_histories["lhb_date"].max())
    seat_scores = score_all_seats(
        registry, seat_histories, last_date,
        half_life=half_life, lookback_days=lookback
    )

    if seat_scores.empty:
        logger.error("无合格的机构评分（可能交易次数不足）")
        return {}

    logger.info(f"机构评分 Top 10:")
    top10 = seat_scores.head(10)[["seat_name", "dynamic_score", "tier", "recent_trades"]]
    for _, r in top10.iterrows():
        logger.info(f"  {r['seat_name']:30s} score={r['dynamic_score']:5.1f} tier={r['tier']} trades={r['recent_trades']}")

    # ============================================================
    # Step 6: 生成交易信号
    # ============================================================
    logger.info("Step 6: 生成交易信号")

    signals = generate_signals(
        trade_records, seat_scores,
        min_signal_strength=signal_cfg.get("min_signal_strength", 0.3),
        tier_weights=signal_cfg.get("tier_weights", None),
    )

    if not signals.empty:
        signals = generate_composite_signals(
            signals,
            resonance_bonus_max=signal_cfg.get("resonance_bonus_max", 0.3),
        )

    logger.info(f"生成 {len(signals)} 个交易信号")

    # ============================================================
    # Step 7: 风险过滤
    # ============================================================
    logger.info("Step 7: 风险过滤")

    if not bench_data.empty:
        regime = detect_regime(bench_data)
        signals = adjust_for_regime(signals, regime)
    else:
        regime = "OSCILLATE"

    decay_monitor = AlphaDecayMonitor(
        window=risk_cfg.get("alpha_decay_window", 10),
        threshold=risk_cfg.get("alpha_decay_threshold", -0.05),
    )

    signals = filter_crowded_stocks(
        signals, signals,
        lookback=risk_cfg.get("crowding_lookback", 20),
        threshold=risk_cfg.get("crowding_threshold", 0.8),
    )

    signals = filter_signals(signals, max_signals_per_day=20)
    logger.info(f"过滤后剩余 {len(signals)} 个信号")

    # ============================================================
    # Step 8: 回测
    # ============================================================
    logger.info("Step 8: 执行回测")

    engine = BacktestEngine(
        initial_capital=trading_cfg.get("initial_capital", 1_000_000),
        commission=trading_cfg.get("commission", 0.0003),
        slippage=trading_cfg.get("slippage", 0.001),
        stop_loss=trading_cfg.get("stop_loss", 0.08),
        take_profit=trading_cfg.get("take_profit", 0.15),
        max_holding_days=trading_cfg.get("max_holding_days", 20),
        max_positions=10,
    )

    nav_df, trades_df = engine.run(signals, price_data, start_date, end_date)

    if nav_df.empty:
        logger.error("回测未产生净值数据")
        return {}

    # ============================================================
    # Step 9: 绩效报告
    # ============================================================
    logger.info("Step 9: 计算绩效指标")

    metrics = compute_full_metrics(nav_df, trades_df)
    metrics["market_regime"] = regime
    metrics["total_signals"] = len(signals)
    metrics["total_seats_rated"] = len(registry)

    print_metrics(metrics)

    # 保存结果
    processed_dir = Path(cfg.get("data", {}).get("processed_dir", "data/processed"))
    processed_path = PROJECT_ROOT / processed_dir
    processed_path.mkdir(parents=True, exist_ok=True)

    nav_df.to_csv(processed_path / "nav_history.csv", index=False)
    if not trades_df.empty:
        trades_df.to_csv(processed_path / "trades.csv", index=False)
    seat_scores.to_csv(processed_path / "seat_scores.csv", index=False)
    logger.info(f"结果已保存到 {processed_path}")

    return metrics


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="机构Alpha追踪 — 每日运行管线")
    parser.add_argument("--start", default="2025-01-01", help="起始日期")
    parser.add_argument("--end", default="2025-12-31", help="结束日期")
    parser.add_argument("--config", default=None, help="配置文件路径")
    args = parser.parse_args()

    logger.info(f"开始运行管线: {args.start} ~ {args.end}")
    metrics = run_pipeline(args.start, args.end, args.config)

    if metrics:
        ann_ret = metrics.get('annualized_return', 0)
        logger.info(f"管线完成。年化收益: {ann_ret*100:.2f}%")
    else:
        logger.error("管线运行失败")
