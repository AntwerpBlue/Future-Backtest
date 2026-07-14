"""
feat_basic.py — 基础特征组
=====================================
包含从原始 feature_engineering.py 迁移的 15 个特征组函数，
覆盖期货日内交易的核心因子体系。

函数协议
--------
每个函数签名统一为：

    func(df: pl.DataFrame, bars: dict) -> Tuple[pl.DataFrame, List[str], List[str]]

参数：
    df   : 当前的 working_df（5m K线 + 已计算的前序特征列）
           _add_basic_features 已执行，含 shifted_* / cum_price_gap 等基础列。
    bars : 多周期K线字典 {"1m", "5m", "30m", "1d"}
           当前15个函数仅用 df，bars 为跨周期特征预留接口，保持签名统一。

返回：
    (enriched_df, feature_names_list, categorical_feature_names_list)

执行顺序
--------
由本文件末尾的模块级 PIPELINE 列表定义。后续特征若依赖本文件的输出列，
应放在字母序更靠后的文件中（如 feat_derived.py）。
"""

import numpy as np
import polars as pl
import talib
from typing import Tuple, List


# ==============================================================
# 1. 技术指标
# ==============================================================

def feat_technical_indicators(
    df: pl.DataFrame,
    bars: dict,
) -> Tuple[pl.DataFrame, List[str], List[str]]:
    """
    布林带、ATR、RSI、ADX、MACD。
    全部基于 shifted_* 去跳空价格序列，避免隔夜缺口污染指标。
    """
    BBAND_period    = 30
    BBAND_STD_ratio = 1.0

    # 布林带（去跳空典型价计算，还原到原始坐标系）
    df = df.with_columns([
        ((pl.col('shifted_high') + pl.col('shifted_low') + pl.col('shifted_close')) / 3.0)
        .rolling_mean(BBAND_period).alias('BBAND_mid'),
        ((pl.col('shifted_high') + pl.col('shifted_low') + pl.col('shifted_close')) / 3.0)
        .rolling_std(BBAND_period).alias('BBAND_std'),
    ]).with_columns([
        pl.when(pl.col('BBAND_std') == 0).then(pl.lit(None))
          .otherwise(pl.col('BBAND_std')).alias('BBAND_std'),
    ]).with_columns([
        (pl.col('BBAND_mid') + pl.col('cum_price_gap')).alias('BBAND_mid'),
        (pl.col('BBAND_mid') + BBAND_STD_ratio * pl.col('BBAND_std')
         + pl.col('cum_price_gap')).alias('BBAND_up'),
        (pl.col('BBAND_mid') - BBAND_STD_ratio * pl.col('BBAND_std')
         + pl.col('cum_price_gap')).alias('BBAND_down'),
    ])

    # ATR、RSI、ADX
    df = df.with_columns([
        pl.map_batches(['shifted_high', 'shifted_low', 'shifted_close'],
                       lambda hlc: talib.ATR(hlc[0], hlc[1], hlc[2]))
          .fill_nan(None).alias('ATR'),
        pl.map_batches(['shifted_close'],
                       lambda close: talib.RSI(close[0]))
          .fill_nan(None).alias('RSI'),
        pl.map_batches(['shifted_high', 'shifted_low', 'shifted_close'],
                       lambda hlc: talib.ADX(hlc[0], hlc[1], hlc[2]))
          .fill_nan(None).alias('ADX'),
    ])

    # RSI 衍生
    df = df.with_columns([
        (pl.col('RSI') - 50).abs().alias('RSI_mid'),
    ])

    # MACD
    df = df.with_columns([
        pl.col('shifted_close').ewm_mean(span=12, adjust=False).alias('ema12'),
        pl.col('shifted_close').ewm_mean(span=26, adjust=False).alias('ema26'),
    ]).with_columns([
        (pl.col('ema12') - pl.col('ema26')).alias('macd_dif'),
    ]).with_columns([
        pl.col('macd_dif').ewm_mean(span=9, adjust=False).alias('macd_dea'),
    ]).with_columns([
        (pl.col('macd_dif') - pl.col('macd_dea')).alias('macd_hist'),
        (pl.col('macd_dif') / (pl.col('ATR') + 1e-9)).alias('macd_norm'),
    ])

    features    = ['ATR', 'RSI', 'RSI_mid', 'ADX', 'macd_hist', 'macd_norm']
    categorical = []
    return df, features, categorical


# ==============================================================
# 2. 价格位置
# ==============================================================

def feat_price_position(
    df: pl.DataFrame,
    bars: dict,
) -> Tuple[pl.DataFrame, List[str], List[str]]:
    """
    价格在 1h / 1d / 5d 时间窗口内的相对位置。
    使用 shifted_high/low 消除跳空影响。
    """
    df = df.with_columns([
        pl.col('shifted_high').rolling_max(12).alias('high_1h'),
        pl.col('shifted_high').rolling_max(69).alias('high_1d'),
        pl.col('shifted_high').rolling_max(345).alias('high_5d'),
        pl.col('shifted_low').rolling_min(12).alias('low_1h'),
        pl.col('shifted_low').rolling_min(69).alias('low_1d'),
        pl.col('shifted_low').rolling_min(345).alias('low_5d'),
    ]).with_columns([
        (pl.col('high_1h') - pl.col('low_1h')).alias('width_1h'),
        (pl.col('high_1d') - pl.col('low_1d')).alias('width_1d'),
        (pl.col('high_5d') - pl.col('low_5d')).alias('width_5d'),
    ]).with_columns([
        ((pl.col('shifted_close') - pl.col('low_1h'))
         / (1e-9 + pl.col('width_1h'))).alias('pos_1h'),
        ((pl.col('shifted_close') - pl.col('low_1d'))
         / (1e-9 + pl.col('width_1d'))).alias('pos_1d'),
        ((pl.col('shifted_close') - pl.col('low_5d'))
         / (1e-9 + pl.col('width_5d'))).alias('pos_5d'),
    ])

    features    = ['pos_1h', 'pos_1d', 'pos_5d']
    categorical = []
    return df, features, categorical


# ==============================================================
# 3. 成交量
# ==============================================================

def feat_volume_basic(
    df: pl.DataFrame,
    bars: dict,
) -> Tuple[pl.DataFrame, List[str], List[str]]:
    """成交量相对强度（当前量 / EMA5 量）。"""
    df = df.with_columns([
        pl.col('volume')
          .rolling_mean(5, weights=np.exp(np.linspace(-1, 0, 5)).tolist())
          .alias('volume_ema5'),
    ]).with_columns([
        (pl.col('volume') / pl.col('volume_ema5')).alias('vol_ratio'),
    ])

    features    = ['vol_ratio']
    categorical = []
    return df, features, categorical


# ==============================================================
# 4. 持仓量
# ==============================================================

def feat_open_interest(
    df: pl.DataFrame,
    bars: dict,
) -> Tuple[pl.DataFrame, List[str], List[str]]:
    """持仓量变化、趋势、波动率等期货特有指标。"""
    df = df.with_columns([
        (pl.col('open_interest') / pl.col('open_interest').shift(1) - 1).alias('oi_change'),
        pl.col('open_interest').rolling_mean(12).alias('oi_ma_1h'),
        pl.col('open_interest').rolling_mean(69).alias('oi_ma_1d'),
    ]).with_columns([
        (pl.col('open_interest') / (pl.col('oi_ma_1d') + 1e-9)).alias('oi_ratio'),
        (pl.col('volume') / (pl.col('open_interest').diff().abs() + 1)).alias('vol_oi_ratio'),
        pl.col('oi_change').rolling_sum(12).alias('oi_change_cum_1h'),
        pl.col('oi_change').rolling_sum(69).alias('oi_change_cum_1d'),
        (pl.col('open_interest').rolling_rank(method='average', window_size=69)
         / 69).alias('oi_rank_1d'),
        pl.col('oi_change').rolling_std(12).alias('oi_volatility'),
    ]).with_columns([
        (pl.col('oi_change_cum_1h')
         / (pl.col('oi_volatility') + 1e-9)).alias('oi_trend_strength'),
    ])

    features = [
        'oi_change', 'oi_ratio', 'vol_oi_ratio',
        'oi_change_cum_1h', 'oi_change_cum_1d', 'oi_rank_1d',
        'oi_volatility', 'oi_trend_strength',
    ]
    categorical = []
    return df, features, categorical


# ==============================================================
# 5. 价格动量
# ==============================================================

def feat_price_momentum(
    df: pl.DataFrame,
    bars: dict,
) -> Tuple[pl.DataFrame, List[str], List[str]]:
    """
    收益率、均线偏离、均线排列、ROC、价格加速度。
    全部使用 shifted_close 消除隔夜跳空对动量的污染。
    """
    df = df.with_columns([
        (pl.col('shifted_close') / pl.col('shifted_close').shift(6) - 1).alias('ret_30min'),
        (pl.col('shifted_close') / pl.col('shifted_close').shift(12) - 1).alias('ret_1h'),
        (pl.col('shifted_close') / pl.col('shifted_close').shift(69) - 1).alias('ret_1d'),
        ((pl.col('shifted_close') - pl.col('shifted_close').shift(6)) -
         (pl.col('shifted_close').shift(6) - pl.col('shifted_close').shift(12)))
        .alias('momentum_accel'),
        pl.col('shifted_close').rolling_mean(12).alias('ma_1h'),
        pl.col('shifted_close').rolling_mean(69).alias('ma_1d'),
    ]).with_columns([
        (pl.col('shifted_close') / (pl.col('ma_1h') + 1e-9) - 1).alias('ma_dev_1h'),
        (pl.col('shifted_close') / (pl.col('ma_1d') + 1e-9) - 1).alias('ma_dev_1d'),
    ])

    # 均线排列（三状态：0=多头，1=空头，2=混合）
    df = df.with_columns([
        pl.when(
            (pl.col('ma_1h') > pl.col('ma_1d')) &
            (pl.col('shifted_close') > pl.col('ma_1h'))
        ).then(0)
        .when(
            (pl.col('ma_1h') < pl.col('ma_1d')) &
            (pl.col('shifted_close') < pl.col('ma_1h'))
        ).then(1)
        .otherwise(2).alias('ma_alignment'),

        ((pl.col('ma_1h') - pl.col('ma_1h').shift(6))
         / (pl.col('ATR') + 1e-9)).alias('ma_slope_1h'),
        ((pl.col('ma_1d') - pl.col('ma_1d').shift(69))
         / (pl.col('ATR') + 1e-9)).alias('ma_slope_1d'),
        ((pl.col('shifted_close') - pl.col('shifted_close').shift(24))
         / (pl.col('shifted_close').shift(24) + 1e-9)).alias('roc_2h'),
        (pl.col('ret_30min') - pl.col('ret_30min').shift(6)).alias('price_accel_raw'),
    ]).with_columns([
        (pl.col('price_accel_raw')
         / (pl.col('ATR') / pl.col('shifted_close') + 1e-9)).alias('price_acceleration'),
    ])

    features = [
        'ret_30min', 'ret_1h', 'ret_1d', 'momentum_accel',
        'ma_dev_1h', 'ma_dev_1d',
        'ma_alignment', 'ma_slope_1h', 'ma_slope_1d',
        'roc_2h', 'price_acceleration',
    ]
    categorical = ['ma_alignment']  # 0, 1, 2
    return df, features, categorical


# ==============================================================
# 6. 突破特征
# ==============================================================

def feat_breakout(
    df: pl.DataFrame,
    bars: dict,
) -> Tuple[pl.DataFrame, List[str], List[str]]:
    """1h / 1d 高低点突破标志 + ATR 归一化突破强度。"""
    df = df.with_columns([
        (pl.col('close') > pl.col('high_1h').shift(1)).cast(pl.Int8).alias('breakout_high_1h'),
        (pl.col('close') > pl.col('high_1d').shift(1)).cast(pl.Int8).alias('breakout_high_1d'),
        (pl.col('close') < pl.col('low_1h').shift(1)).cast(pl.Int8).alias('breakout_low_1h'),
        (pl.col('close') < pl.col('low_1d').shift(1)).cast(pl.Int8).alias('breakout_low_1d'),
        pl.max_horizontal([
            (pl.col('high') - pl.col('high_1h').shift(1)) / (pl.col('ATR') + 1e-9),
            (pl.col('low_1h').shift(1) - pl.col('low'))  / (pl.col('ATR') + 1e-9),
        ]).alias('range_break_strength'),
    ])

    features = [
        'breakout_high_1h', 'breakout_high_1d',
        'breakout_low_1h',  'breakout_low_1d',
        'range_break_strength',
    ]
    categorical = [
        'breakout_high_1h', 'breakout_high_1d',
        'breakout_low_1h',  'breakout_low_1d',
    ]
    return df, features, categorical


# ==============================================================
# 7. 波动率
# ==============================================================

def feat_volatility(
    df: pl.DataFrame,
    bars: dict,
) -> Tuple[pl.DataFrame, List[str], List[str]]:
    """
    已实现波动率、ATR 趋势、Parkinson 波动率、波动率偏度。
    收益率序列使用 shifted_close 消除跳空影响。
    """
    df = df.with_columns([
        (pl.col('shifted_close') / pl.col('shifted_close').shift(1) - 1).alias('returns'),
    ]).with_columns([
        ((pl.col('high') - pl.col('low'))
         / (pl.col('shifted_close') + 1e-9)).alias('range_pct'),
        pl.col('returns').rolling_std(12).alias('realized_vol_1h'),
        pl.col('returns').rolling_std(69).alias('realized_vol_1d'),
        pl.when(pl.col('close') > pl.col('open'))
          .then(pl.col('high') - pl.col('close'))
          .otherwise(0).rolling_mean(12).alias('vol_up'),
        pl.when(pl.col('close') < pl.col('open'))
          .then(pl.col('close') - pl.col('low'))
          .otherwise(0).rolling_mean(12).alias('vol_down'),
    ]).with_columns([
        (pl.col('realized_vol_1h')
         / (pl.col('realized_vol_1d') + 1e-9)).alias('vol_ratio_st_lt'),
        (pl.col('ATR') / pl.col('ATR').rolling_mean(69)).alias('atr_trend'),
        (((pl.col('high') / (pl.col('low') + 1e-9)).log() ** 2) / (4 * np.log(2)))
        .sqrt().rolling_mean(12).alias('parkinson_vol'),
        (pl.col('vol_up') / (pl.col('vol_down') + 1e-9)).alias('vol_skew'),
    ])

    features = [
        'realized_vol_1h', 'realized_vol_1d',
        'vol_ratio_st_lt', 'atr_trend', 'parkinson_vol',
        'range_pct', 'vol_skew',
    ]
    categorical = []
    return df, features, categorical


# ==============================================================
# 8. K 线形态
# ==============================================================

def feat_candlestick(
    df: pl.DataFrame,
    bars: dict,
) -> Tuple[pl.DataFrame, List[str], List[str]]:
    """实体比、上下影线、连续趋势计数、收盘强度、盘整计数。"""
    df = df.with_columns([
        ((pl.col('close') - pl.col('open')).abs()
         / (pl.col('ATR') + 1e-9)).alias('body_ratio'),
        ((pl.col('high') - pl.max_horizontal(['open', 'close']))
         / (pl.col('high') - pl.col('low') + 1e-9)).alias('upper_shadow'),
        ((pl.min_horizontal(['open', 'close']) - pl.col('low'))
         / (pl.col('high') - pl.col('low') + 1e-9)).alias('lower_shadow'),
        ((pl.col('close') - pl.col('low'))
         / (pl.col('high') - pl.col('low') + 1e-9)).alias('close_position'),
        pl.when(pl.col('close') > pl.col('open')).then(1)
          .when(pl.col('close') < pl.col('open')).then(-1)
          .otherwise(0).rolling_sum(6).alias('consecutive_trend'),
        ((pl.col('close') - pl.col('open'))
         / (pl.col('high') - pl.col('low') + 1e-9)).alias('buy_pressure_proxy'),
        ((2 * pl.col('close') - pl.col('high') - pl.col('low'))
         / (pl.col('high') - pl.col('low') + 1e-9)).alias('close_strength'),
        ((pl.col('high') <= pl.col('high_1h').shift(1)) &
         (pl.col('low')  >= pl.col('low_1h').shift(1)))
        .cast(pl.Int8).rolling_sum(12).alias('consolidation_count'),
    ])

    features = [
        'body_ratio', 'upper_shadow', 'lower_shadow',
        'close_position', 'consecutive_trend',
        'buy_pressure_proxy', 'close_strength',
        'consolidation_count',
    ]
    categorical = []
    return df, features, categorical


# ==============================================================
# 9. 量价配合
# ==============================================================

def feat_volume_price(
    df: pl.DataFrame,
    bars: dict,
) -> Tuple[pl.DataFrame, List[str], List[str]]:
    """量价背离、OBV、VWAP 偏离、量价确认信号、成交量集中度。"""
    # 量价背离
    df = df.with_columns([
        pl.when(pl.col('close') > pl.col('open'))
          .then(pl.col('volume')).otherwise(0)
          .rolling_sum(12).alias('volume_up'),
        pl.when(pl.col('close') < pl.col('open'))
          .then(pl.col('volume')).otherwise(0)
          .rolling_sum(12).alias('volume_down'),
    ]).with_columns([
        (pl.col('volume_up') / (pl.col('volume_down') + 1)).alias('volume_divergence'),
        (pl.col('volume') * pl.col('ATR')
         / (pl.col('shifted_close') + 1e-9)).alias('volume_volatility_adj'),
    ])

    # OBV（使用 shifted_close 判断方向，避免跨 session 虚假涨跌）
    df = df.with_columns([
        pl.when(pl.col('shifted_close') > pl.col('shifted_close').shift(1))
          .then(pl.col('volume'))
          .when(pl.col('shifted_close') < pl.col('shifted_close').shift(1))
          .then(-pl.col('volume'))
          .otherwise(0)
          .cum_sum()
          .alias('obv_raw'),
    ]).with_columns([
        (pl.col('obv_raw')
         / (pl.col('obv_raw').rolling_std(69) + 1e-9)).alias('obv_norm'),
        ((pl.col('obv_raw') - pl.col('obv_raw').shift(12))
         / (pl.col('volume').rolling_sum(12) + 1)).alias('obv_trend'),
    ])

    # VWAP（shifted 典型价）
    df = df.with_columns([
        (((pl.col('shifted_high') + pl.col('shifted_low') + pl.col('shifted_close')) / 3.0
          * pl.col('volume')).rolling_sum(69)
         / (pl.col('volume').rolling_sum(69) + 1e-9)).alias('vwap_1d'),
    ]).with_columns([
        ((pl.col('shifted_close') - pl.col('vwap_1d'))
         / (pl.col('ATR') + 1e-9)).alias('vwap_deviation'),
        ((pl.col('vwap_1d') - pl.col('vwap_1d').shift(12))
         / (pl.col('ATR') + 1e-9)).alias('vwap_slope'),
    ])

    # 量价确认信号
    df = df.with_columns([
        ((pl.col('shifted_close') > pl.col('shifted_close').shift(1)) &
         (pl.col('volume') > pl.col('volume').shift(1)))
        .cast(pl.Int8).rolling_sum(6).alias('price_up_vol_up'),
        ((pl.col('shifted_close') < pl.col('shifted_close').shift(1)) &
         (pl.col('volume') > pl.col('volume').shift(1)))
        .cast(pl.Int8).rolling_sum(6).alias('price_down_vol_up'),
        (pl.col('volume').rolling_sum(12)
         / (pl.col('volume').rolling_sum(69) + 1e-9)).alias('volume_concentration'),
    ])

    features = [
        'volume_divergence', 'volume_volatility_adj',
        'obv_norm', 'obv_trend',
        'vwap_deviation', 'vwap_slope',
        'price_up_vol_up', 'price_down_vol_up',
        'volume_concentration',
    ]
    categorical = []
    return df, features, categorical


# ==============================================================
# 10. 仓量价关系（期货特有）
# ==============================================================

def feat_oi_volume_price(
    df: pl.DataFrame,
    bars: dict,
) -> Tuple[pl.DataFrame, List[str], List[str]]:
    """价涨量增持仓增（bullish_trio）、价跌量增持仓减（背离）等期货三重确认信号。"""
    df = df.with_columns([
        ((pl.col('shifted_close') > pl.col('shifted_close').shift(1)) &
         (pl.col('volume') > pl.col('volume').shift(1)) &
         (pl.col('oi_change') > 0))
        .cast(pl.Int8).rolling_sum(6).alias('bullish_trio'),

        ((pl.col('shifted_close') < pl.col('shifted_close').shift(1)) &
         (pl.col('volume') > pl.col('volume').shift(1)) &
         (pl.col('oi_change') > 0))
        .cast(pl.Int8).rolling_sum(6).alias('bearish_trio'),

        ((pl.col('shifted_close') > pl.col('shifted_close').shift(6)) &
         (pl.col('open_interest') < pl.col('open_interest').shift(6)))
        .cast(pl.Int8).alias('oi_price_div_bull'),

        ((pl.col('shifted_close') < pl.col('shifted_close').shift(6)) &
         (pl.col('open_interest') < pl.col('open_interest').shift(6)))
        .cast(pl.Int8).alias('oi_price_div_bear'),
    ])

    features    = ['bullish_trio', 'bearish_trio', 'oi_price_div_bull', 'oi_price_div_bear']
    categorical = []
    return df, features, categorical


# ==============================================================
# 11. 开盘特征
# ==============================================================

def feat_opening(
    df: pl.DataFrame,
    bars: dict,
) -> Tuple[pl.DataFrame, List[str], List[str]]:
    """
    开盘动量（ATR 归一化）和开盘区间宽度。
    在 session 第2根 bar 时计算，之后在 session 内前向填充。
    """
    df = df.with_columns([
        pl.col('high').rolling_max(2).alias('open_high_temp'),
        pl.col('low').rolling_min(2).alias('open_low_temp'),
    ]).with_columns([
        pl.when(pl.col('bar_idx_in_session') == 2)
          .then(pl.col('open_high_temp')).otherwise(None)
          .fill_null(strategy='forward')
          .over('continuous_session_id').alias('open_high'),
        pl.when(pl.col('bar_idx_in_session') == 2)
          .then(pl.col('open_low_temp')).otherwise(None)
          .fill_null(strategy='forward')
          .over('continuous_session_id').alias('open_low'),
        pl.when(pl.col('bar_idx_in_session') == 2)
          .then((pl.col('close') - pl.col('close').shift(1))
                / (pl.col('ATR') + 1e-9))
          .otherwise(None)
          .fill_null(strategy='forward')
          .over('continuous_session_id').alias('open_momentum'),
    ]).with_columns([
        (pl.col('open_high') - pl.col('open_low')).alias('open_width'),
    ])

    features    = ['open_momentum', 'open_width']
    categorical = []
    return df, features, categorical


# ==============================================================
# 12. 时间特征
# ==============================================================

def feat_time(
    df: pl.DataFrame,
    bars: dict,
) -> Tuple[pl.DataFrame, List[str], List[str]]:
    """session 类型、星期、session 内进度、bar 序号。"""
    df = df.with_columns([
        pl.when(pl.col('datetime').dt.time().is_between(pl.time(9,  0), pl.time(9,  30))).then(1)
          .when(pl.col('datetime').dt.time().is_between(pl.time(14, 30), pl.time(15,  0))).then(2)
          .when(pl.col('datetime').dt.time().is_between(pl.time(21,  0), pl.time(21, 30))).then(3)
          .when(pl.col('datetime').dt.time().is_between(pl.time(22, 30), pl.time(23,  0))).then(4)
          .when(pl.col('datetime').dt.time().is_between(pl.time(9,  30), pl.time(14, 30))).then(5)
          .when(pl.col('datetime').dt.time().is_between(pl.time(21, 30), pl.time(22, 30))).then(6)
          .otherwise(0).alias('session_type'),
        pl.col('datetime').dt.weekday().alias('weekday'),
        (pl.col('bar_idx_in_session').cast(pl.Float64) /
         pl.col('bar_idx_in_session').max().over('continuous_session_id'))
        .alias('session_progress'),
    ])

    features    = ['session_type', 'weekday', 'session_progress', 'bar_idx_in_session']
    categorical = ['session_type', 'weekday']
    return df, features, categorical


# ==============================================================
# 13. 相对强度
# ==============================================================

def feat_relative_strength(
    df: pl.DataFrame,
    bars: dict,
) -> Tuple[pl.DataFrame, List[str], List[str]]:
    """
    相对开盘收益率、突破强度、价格百分位排名、收益率偏度。
    依赖 feat_price_position（high_1d/low_1d）和 feat_volatility（returns）。
    """
    df = df.with_columns([
        ((pl.col('close') - pl.col('open').first().over('continuous_session_id'))
         / (pl.col('ATR') + 1e-9)).alias('return_from_open'),
        pl.max_horizontal([
            (pl.col('shifted_close') - pl.col('high_1d')) / (pl.col('ATR') + 1e-9),
            (pl.col('low_1d') - pl.col('shifted_close')) / (pl.col('ATR') + 1e-9),
        ]).alias('breakout_strength'),
        (pl.col('shifted_close').rolling_rank(method='average', window_size=69)
         / 69).alias('price_rank_1d'),
        pl.col('returns').rolling_skew(69).alias('returns_skew_1d'),
    ])

    features    = ['return_from_open', 'breakout_strength', 'price_rank_1d', 'returns_skew_1d']
    categorical = []
    return df, features, categorical


# ==============================================================
# 14. 多周期一致性
# ==============================================================

def feat_multi_timeframe(
    df: pl.DataFrame,
    bars: dict,
) -> Tuple[pl.DataFrame, List[str], List[str]]:
    """
    跨 30m / 1h / 1d 的收益方向一致性计数，以及动量加速度与近期动量的方向一致性。
    依赖 feat_price_momentum（ret_30min / ret_1h / ret_1d / momentum_accel）。
    """
    df = df.with_columns([
        (
            (pl.col('ret_30min').sign() == pl.col('ret_1h').sign()).cast(pl.Int8) +
            (pl.col('ret_1h').sign()    == pl.col('ret_1d').sign()).cast(pl.Int8) +
            (pl.col('ret_30min').sign() == pl.col('ret_1d').sign()).cast(pl.Int8)
        ).alias('multi_tf_consistency'),
        (pl.col('momentum_accel').sign() == pl.col('ret_30min').sign())
        .cast(pl.Int8).alias('momentum_consistency'),
    ])

    features    = ['multi_tf_consistency', 'momentum_consistency']
    categorical = []
    return df, features, categorical


# ==============================================================
# 15. 标准化特征
# ==============================================================

def feat_normalized(
    df: pl.DataFrame,
    bars: dict,
) -> Tuple[pl.DataFrame, List[str], List[str]]:
    """
    滚动 IQR Z-score 标准化 + 异常成交量检测。
    依赖几乎所有前序特征列（ATR / ADX / RSI_mid / BBAND_std / width_* / vol_ratio 等）。
    必须放在 PIPELINE 末尾。
    """
    def rolling_z(colname):
        return (
            (pl.col(colname) - pl.col(colname).rolling_median(100)) /
            (pl.col(colname).rolling_quantile(0.9, window_size=100) -
             pl.col(colname).rolling_quantile(0.1, window_size=100) + 1e-9)
        )

    df = df.with_columns([
        rolling_z('ATR').alias('ATR_z'),
        rolling_z('ADX').alias('ADX_z'),
        rolling_z('RSI_mid').alias('RSI_mid_z'),
        rolling_z('vol_ratio').alias('vol_ratio_z'),
        rolling_z('volume').alias('vol_z'),
        rolling_z('BBAND_std').alias('BBAND_std_z'),
        (pl.col('width_1h') / (pl.col('BBAND_std') + 1e-9)).alias('width_1h_z'),
        (pl.col('width_1d') / (pl.col('BBAND_std') + 1e-9)).alias('width_1d_z'),
        (pl.col('width_5d') / (pl.col('BBAND_std') + 1e-9)).alias('width_5d_z'),
    ])

    # 异常成交量
    df = df.with_columns([
        (pl.col('vol_z') > 2).cast(pl.Int8).alias('volume_spike'),
        (pl.col('volume') > pl.col('volume').rolling_mean(12))
        .cast(pl.Int8).rolling_sum(6).alias('sustained_high_vol'),
        (pl.col('volume') < pl.col('volume').rolling_mean(12) * 0.7)
        .cast(pl.Int8).rolling_sum(6).alias('low_volume_count'),
    ])

    features = [
        'ATR_z', 'ADX_z', 'RSI_mid_z',
        'vol_ratio_z', 'vol_z', 'BBAND_std_z',
        'width_1h_z', 'width_1d_z', 'width_5d_z',
        'volume_spike', 'sustained_high_vol', 'low_volume_count',
    ]
    categorical = []
    return df, features, categorical


# ==============================================================
# 本文件的执行顺序声明
# ==============================================================
#
# FeatureEngineer._compute_all_features 会自动发现 factors/ 目录下所有 .py 文件，
# 按字母序逐个 import，读取其 PIPELINE 并串联执行。
#
# 新增特征到本文件：在上方实现函数，然后在此列表末尾追加一行即可。
# 新增特征文件：新建 factors/feat_xxx.py 并提供 PIPELINE，无需修改本文件。
#
# 注意：后续特征文件若依赖本文件的输出列（如 ATR / ATR_z 等），
# 需确保其文件名在字母序上晚于 feat_basic.py（如 feat_derived.py）。

PIPELINE = [
    ('技术指标',     feat_technical_indicators),
    ('价格位置',     feat_price_position),
    ('成交量',       feat_volume_basic),
    ('持仓量',       feat_open_interest),
    ('价格动量',     feat_price_momentum),
    ('突破特征',     feat_breakout),
    ('波动率',       feat_volatility),
    ('K线形态',      feat_candlestick),
    ('量价配合',     feat_volume_price),
    ('仓量价关系',   feat_oi_volume_price),
    ('开盘特征',     feat_opening),
    ('时间特征',     feat_time),
    ('相对强度',     feat_relative_strength),
    ('多周期一致性', feat_multi_timeframe),
    ('标准化特征',   feat_normalized),
]
