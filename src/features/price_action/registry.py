"""Price Action 因子注册表。"""
from __future__ import annotations

from src.features.price_action.breakout import breakout_close_quality

PRICE_ACTION_FACTORS: dict[str, dict] = {
    "breakout_close_quality": {
        "function": breakout_close_quality,
        "params": {
            "L": [20, 40, 60],
            "ATR_N": [14, 20],
            "VOL_N": [20, 40],
        },
        "default_params": {"L": 20, "ATR_N": 20, "VOL_N": 20},
        "available_time": "T_close",
        "earliest_use": "T+1_open",
        "output_type": "continuous",
        "direction": "higher_is_better",
    },
}
