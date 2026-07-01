import pandas as pd

from src.data.level2_reader import (
    BIG_VOLUME,
    classify_orders_by_size,
    compute_big_order_summary,
    compute_period_flow,
    match_orders_to_trades,
)


def test_match_orders_uses_order_id_not_exchange_order_id():
    orders = pd.DataFrame([
        {
            "委托编号": "1001",
            "交易所委托号": "0",
            "委托类型": "A",
            "委托代码": "B",
            "委托价格": 100000,
            "委托数量": 1000,
        },
        {
            "委托编号": "1002",
            "交易所委托号": "0",
            "委托类型": "A",
            "委托代码": "S",
            "委托价格": 101000,
            "委托数量": 2000,
        },
    ])
    trades = pd.DataFrame([
        {
            "叫买序号": "1001",
            "叫卖序号": "9001",
            "成交数量": 1000,
            "成交金额": 100000 * 1000,
        },
        {
            "叫买序号": "9002",
            "叫卖序号": "1002",
            "成交数量": 2000,
            "成交金额": 101000 * 2000,
        },
    ])

    matched = match_orders_to_trades(orders, trades)

    assert set(matched["委托编号"]) == {"1001", "1002"}
    assert matched["成交数量"].sum() == 3000


def test_big_volume_threshold_is_100k_shares():
    assert BIG_VOLUME == 100000

    wtcj = pd.DataFrame([
        {
            "委托代码": "B",
            "委托价格": 10000,
            "委托数量": 100000,
            "成交数量": 100000,
            "成交金额": 10000 * 100000,
        }
    ])

    classified = classify_orders_by_size(wtcj)
    assert len(classified["big"]) == 1


def test_amount_units_are_yi_for_summaries():
    cjdf = pd.DataFrame([
        {
            "时间": "093000000",
            "成交价格": 330000,
            "成交数量": 30000,
            "成交金额": 330000 * 30000,
            "叫买序号": "1001",
            "叫卖序号": "2001",
            "委托代码": "B",
        }
    ])

    flow = compute_period_flow(cjdf)
    assert flow["amount_yi"] == 0.01
    assert flow["big_buy"] == 0.01

    wtcj = pd.DataFrame([
        {
            "委托代码": "B",
            "委托价格": 330000,
            "委托数量": 30000,
            "成交数量": 30000,
            "成交金额": 330000 * 30000,
        }
    ])
    summary = compute_big_order_summary(wtcj)
    assert summary["big_buy"] == 0.01
    assert summary["big_buy_avg_price"] == 33.0



def test_match_orders_falls_back_to_exchange_order_id_when_order_id_is_zero():
    orders = pd.DataFrame([
        {
            "委托编号": "0",
            "交易所委托号": "1001",
            "委托类型": "A",
            "委托代码": "B",
            "委托价格": 100000,
            "委托数量": 1000,
        },
        {
            "委托编号": "0",
            "交易所委托号": "1002",
            "委托类型": "A",
            "委托代码": "S",
            "委托价格": 101000,
            "委托数量": 2000,
        },
    ])
    trades = pd.DataFrame([
        {
            "叫买序号": "1001",
            "叫卖序号": "9001",
            "成交数量": 1000,
            "成交金额": 100000 * 1000,
        },
        {
            "叫买序号": "9002",
            "叫卖序号": "1002",
            "成交数量": 2000,
            "成交金额": 101000 * 2000,
        },
    ])

    matched = match_orders_to_trades(orders, trades)

    assert set(matched["交易所委托号"]) == {"1001", "1002"}
    assert set(matched["match_key"]) == {"交易所委托号"}
    assert matched["成交数量"].sum() == 3000
