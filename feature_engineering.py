"""
特征工程模块
采用模块化设计，每个函数负责一类特征的计算。

特征函数统一存放在 factors/ 子包中，FeatureEngineer 通过自动发现机制加载：
  - 扫描 factors/*.py（按字母序），读取每个文件的模块级 PIPELINE 列表
  - PIPELINE = [(组名, func), ...]，func 签名：(df, bars) -> (df, features, categoricals)
  - 无需维护集中注册表：新增文件或新增函数只需改对应的 factors/*.py
"""

import importlib
import numpy as np
import polars as pl
import talib
from pathlib import Path
from typing import Tuple, List, Dict


class FeatureEngineer:
    """特征工程类"""
    
    def __init__(self, label_bars: int = 6, label_threshold: float = 0.5,
                 atr_long_window: int = 400,
                 horizon_bars: List[int] = None):
        """
        初始化特征工程器
        
        Parameters:
        -----------
        label_bars : int
            预测未来几个bar（默认6个bar = 30分钟）
        label_threshold : float
            标签阈值，以ATR的倍数表示
        atr_long_window : int
            长基线ATR的回望窗口（bar数），用于归一化收益率。
            默认400个5分钟bar ≈ 2000分钟 ≈ 14个交易日。
        horizon_bars : List[int]
            生成多周期收益率的 horizon 列表（bar数）。
            默认 [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]（5~60分钟全序列），
            用于 FactorLab.horizon_only 分析。
            训练管线不使用这些列，修改不影响模型特征。
        """
        self.label_bars = label_bars
        self.label_threshold = label_threshold
        self.atr_long_window = atr_long_window
        self.horizon_bars = horizon_bars if horizon_bars is not None \
            else list(range(1, 13))

        # 多周期K线字典（原始数据，只读）
        self.bars: dict[str, pl.DataFrame] = {}

        # 特征标签扁平表（用于模型训练）
        self.features_df: pl.DataFrame = None

        # 特征元信息（在计算后填充）
        self.feature_cols = []
        self.categorical_features = []
        self.feature_groups_info = None
        
    # ========================================
    # 主函数
    # ========================================
    
    def _bar_preprocess(self, bar1min: pl.DataFrame) -> pl.DataFrame:
        """
        对原始数据进行预处理，包括
        1. 将原始的1分钟K的datetime时间戳转换为iso标准格式
        2. 还原出交易日字段
        3. 添加bar结束时间戳 bar_end_ts
        """
        # 整理datetime的类型
        datetime_dtype = bar1min['datetime'].dtype
        if datetime_dtype == pl.Utf8:   # 来自历史行情
            bar1min = bar1min.with_columns([
                pl.col('datetime').str.strptime(
                    dtype = pl.Datetime('ns'),
                    format = '%Y-%m-%d %H:%M:%S.%9f'
                )
            ])
        # 来自实时行情
        elif (datetime_dtype == pl.Int64) or (datetime_dtype == pl.UInt64) or (datetime_dtype == pl.Float64): 
            bar1min = bar1min.with_columns([
                pl.col('datetime').alias('datetime_nano') # 还原 unix 时间戳
            ]).with_columns([
                (pl.col('datetime') + 8 * 60 * 60 * 1000_000_000) # 原始时间戳是 naive 时间戳，要加上时区 Asia/Shanghai
                .cast(pl.Datetime('ns')) # 得到 iso 时间戳
            ])
        elif datetime_dtype == pl.Datetime('ns'): # 不用转换
            pass
        else:
            raise ValueError(f"数据时间戳原始类型不正确！类型：{datetime_dtype}")

        # 得到1分钟K的结束时间
        minute = 60 * 1000_000_000
        bar1min = bar1min.with_columns([
            (pl.col('datetime_nano') + 1 * minute).alias('bar_end_ts')
        ])

        # 还原交易日字段
        hour = 60 * minute
        bar1min = bar1min.with_columns([
            # 1. 计算与上一根K线的时间差
            (pl.col('datetime_nano') - pl.col('datetime_nano').shift(1))
            .fill_null(0)
            .alias('time_gap_ns'),
            
            # 2. 获取上一根K线所在的小时数 (用于判断跳跃前是日盘还是夜盘)
            pl.col('datetime').dt.hour().shift(1).alias('prev_hour')
        ]).with_columns([
            # 核心判断逻辑：
            # 条件1：时间差距大于4小时
            # 条件2：跳跃前的K线必须是白天 (8点~16点之间，通常是 14:59 或 15:00)
            pl.when(
                (pl.col('time_gap_ns') > 4 * hour) & 
                (pl.col('prev_hour') >= 8) & 
                (pl.col('prev_hour') <= 16)
            )
            .then(1)
            .otherwise(0)
            .cum_sum()
            .alias('session_id')
        ])

        temp_df = bar1min.group_by('session_id').agg([
            pl.col('datetime').last().dt.date().alias('trading_date') # 通过最后一根bar的date还原出交易日字段
        ]).select(['session_id', 'trading_date'])
        bar1min = bar1min.join(temp_df, on = 'session_id').drop(['time_gap_ns', 'prev_hour', 'session_id'])
        # group_by 聚合后 join 不保证行序，显式排序确保 group_by_dynamic 的 index 列有序
        bar1min = bar1min.sort('datetime')
        return bar1min
    
    def bar_downsample_to(self, period: str, bar1min: pl.DataFrame) -> pl.DataFrame:
        """
        bar1min为调用_bar_preprocess后的df
        降采样为n分钟K，这里的n不宜过大，否则会产生很多不完整K。因为这种聚合不允许跨越不连续的交易时段。
        """
        
        barNmin = bar1min.group_by_dynamic(
            'datetime', every = period,
            group_by = ['trading_date'] # 保证在同一个交易日内，不跨交易时段
        ).agg(
            pl.col('datetime_nano').first().alias('datetime_nano'),
            pl.col('open').first().alias('open'),
            pl.col('high').max().alias('high'),
            pl.col('low').min().alias('low'),
            pl.col('close').last().alias('close'),
            pl.col('volume').sum().alias('volume'),
            pl.col('open_oi').first().alias('open_oi'),
            pl.col('close_oi').last().alias('close_oi'),
            pl.col('bar_end_ts').last().alias('bar_end_ts'),
        ).sort(['trading_date', 'datetime_nano'])

        return barNmin
    
    def bar_downsample_to_day(self, bar1min: pl.DataFrame) -> pl.DataFrame:
        """
        通过1分钟K还原出日K
        """
        day_bar = bar1min.group_by('trading_date').agg([
            pl.col('datetime').first().alias('datetime'),
            pl.col('datetime_nano').first().alias('datetime_nano'),
            pl.col('open').first().alias('open'),
            pl.col('high').max().alias('high'),
            pl.col('low').min().alias('low'),
            pl.col('close').last().alias('close'),
            pl.col('volume').sum().alias('volume'),
            pl.col('open_oi').first().alias('open_oi'),
            pl.col('close_oi').last().alias('close_oi'),
            
            # 日K的 bar_end_ts 同样取最后一根 1分钟 K 的时间戳
            pl.col('bar_end_ts').last().alias('bar_end_ts'), 
        ]).sort(by=['trading_date', 'datetime_nano'])
        
        return day_bar

    def create_features_and_labels(self, 
                                   bar1min: pl.DataFrame,
                                   verbose: bool = True) -> pl.DataFrame:
        """
        主函数：从1分钟K线创建特征和标签
        
        数据流程：
        1. 预处理1分钟K → 聚合多周期K → 存储到 self.bars
        2. 基于5分钟K计算特征 → working_df（临时变量）
        3. 生成标签 → working_df
        4. 提取特征+标签列 → self.features_df
        
        Returns:
        --------
        features_df : pl.DataFrame
            特征标签扁平表，包含：
            - datetime: ISO格式时间戳（便于查看）
            - datetime_nano: Unix纳秒时间戳（用于对齐）
            - 所有特征列（86个）
            - label, ror_future, price_change: 标签列
        """
        
        if verbose:
            print("\n" + "="*80)
            print("特征工程开始")
            print("="*80)
        
        # 0. 1分钟K线（原子数据）的预处理
        bar1min = self._bar_preprocess(bar1min)

        # 1. 聚合到各个时间周期（5分钟K是目前模型的执行频率，主要K线）
        bar5min  = self.bar_downsample_to('5m',  bar1min)
        bar30min = self.bar_downsample_to('30m', bar1min)
        bar1d    = self.bar_downsample_to_day(bar1min)

        # 2. 基于5分钟K计算特征（working_df是临时变量）
        if verbose:
            print("步骤 1/2: 基础预处理...")
        working_df = self._add_basic_features(bar5min)

        # 将结构性元数据（session_id / cum_price_gap / bar_idx）附加到裸 bar5min，
        # 存入 self.bars["5m"]。这样 FactorLab 和训练管线都能通过 bars["5m"]
        # 直接访问这三列，无需重新推导 session 边界。
        bar5min_with_meta = bar5min.join(
            working_df.select(['datetime',
                               'continuous_session_id',
                               'cum_price_gap',
                               'bar_idx_in_session']),
            on='datetime', how='left'
        ).sort(['datetime'])
        self.bars = {"1m": bar1min, "5m": bar5min_with_meta,
                     "30m": bar30min, "1d": bar1d}
        
        # 3. 计算所有特征（总控函数）
        if verbose:
            print("步骤 2/2: 计算所有特征...")
        working_df, all_features, all_categorical = self._compute_all_features(working_df, verbose=verbose)
        
        # 4. 生成标签
        if verbose:
            print("\n生成标签...")
        working_df = self._generate_labels(working_df)
        
        # 5. 保存特征元信息
        self.feature_cols = all_features
        self.categorical_features = all_categorical
        
        # 6. 提取最终的特征标签表
        available_features = [f for f in all_features if f in working_df.columns]
        
        if len(available_features) != len(all_features):
            missing = set(all_features) - set(available_features)
            if verbose:
                print(f"  ⚠️  警告：以下特征在数据中不存在: {missing}")
        
        # 分析专用列（不参与模型训练）
        # - ATR_long, ror_future_Nb, ror_norm_Nb: IC分析与回测用
        # - continuous_session_id: 回测中判断session边界（强制平仓）
        # - open: 回测中用于下一根bar的开盘成交价
        # - bar_end_ts: 预留，用于未来精确时间对齐
        # - trading_date: IC分析按交易日分组
        analysis_cols = (
            ['ATR_long'] +
            [f'ror_future_{b}b' for b in self.horizon_bars] +
            [f'ror_norm_{b}b'   for b in self.horizon_bars] +
            ['continuous_session_id', 'open', 'trading_date']
        )
        available_analysis = [c for c in analysis_cols if c in working_df.columns]

        self.features_df = working_df.select([
            'datetime',
            'datetime_nano',
            *available_features,
            *available_analysis,    # 分析专用列，不在 self.feature_cols 中
            'label',
            'ror_future',
            'price_change',
        ])
        
        # 7. 显式删除 working_df 释放内存
        del working_df
        
        if verbose:
            print(f"\n{'='*80}")
            print(f"特征工程完成!")
            print(f"{'='*80}")
            print(f"总特征数: {len(available_features)}")
            print(f"类别特征数: {len(self.categorical_features)}")
            print(f"数据行数: {len(self.features_df):,}")
            
            # 按组统计
            print(f"\n特征分组统计:")
            for group_name, info in self.feature_groups_info.items():
                n_feat = len(info['features'])
                n_cat = len(info['categorical'])
                print(f"  {group_name}: {n_feat} 个特征", end="")
                if n_cat > 0:
                    print(f" (含 {n_cat} 个类别特征)")
                else:
                    print()
            
            print(f"\n标签分布:")
            label_dist = self.features_df.group_by('label').agg(pl.len()).sort('label')
            print(label_dist)
            print(f"{'='*80}\n")
        
        return self.features_df
    
    # ========================================
    # 总控函数：自动发现并调用所有特征计算函数
    # ========================================
    
    def _compute_all_features(self, df: pl.DataFrame,
                              verbose: bool = True) -> Tuple[pl.DataFrame, List[str], List[str]]:
        """
        自动发现 factors/ 目录下所有 .py 文件（排除 __init__.py），
        按字母序依次 import，读取每个文件的模块级 PIPELINE 列表并串联执行。

        PIPELINE 格式（每个 factors/*.py 文件必须提供）：
            PIPELINE = [
                ('组名', func),   # func(df, bars) -> (df, features, categoricals)
                ...
            ]

        新增特征组：在 factors/ 下新建 .py 文件，提供符合协议的函数和 PIPELINE，
        无需修改本文件。

        Returns
        -------
        df            : 包含所有特征列的 DataFrame
        all_features  : 所有特征名列表
        all_categorical: 所有类别特征名列表
        """
        all_features    = []
        all_categorical = []
        self.feature_groups_info = {}

        # factors/ 与本文件同级
        factors_dir = Path(__file__).parent / 'factors'
        factor_files = sorted(
            p for p in factors_dir.glob('*.py')
            if p.name != '__init__.py'
        )

        if not factor_files:
            if verbose:
                print("  ⚠️  factors/ 目录下未找到任何特征文件，跳过特征计算。")
            return df, all_features, all_categorical

        # 确保 factors 包可以被 import（将 feature_mining/ 加入搜索路径）
        import sys
        feature_mining_dir = str(Path(__file__).parent)
        if feature_mining_dir not in sys.path:
            sys.path.insert(0, feature_mining_dir)

        for py_file in factor_files:
            module_name = f'factors.{py_file.stem}'   # e.g. 'factors.feat_basic'
            mod = importlib.import_module(module_name)

            pipeline = getattr(mod, 'PIPELINE', [])
            if not pipeline:
                if verbose:
                    print(f"  ⚠️  {py_file.name} 未提供 PIPELINE，已跳过。")
                continue

            for group_name, func in pipeline:
                if verbose:
                    print(f"    - {group_name}...", end=" ")

                df, features, categorical = func(df, self.bars)

                all_features.extend(features)
                all_categorical.extend(categorical)
                self.feature_groups_info[group_name] = {
                    'features':    features,
                    'categorical': categorical,
                }

                if verbose:
                    cat_info = f" (含{len(categorical)}个类别)" if categorical else ""
                    print(f"{len(features)}个特征{cat_info}")

        return df, all_features, all_categorical
    
    # ========================================
    # 辅助方法
    # ========================================
    def _add_basic_features(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        基础预处理：
        1. 寻找非连续交易段的边界 (>= 2.5小时)
        2. 基于边界计算连续的跳空缺口 (cum_gap)，生成平滑价格
        3. 计算典型价格 (typical_price)（基于原始价格）
        4. 生成段内K线索引 (bar_idx_in_session)
        """
        
        # 1. 基础变量与时间跳跃检测
        minute = 60 * 1000_000_000
        df = df.with_columns([
            pl.col('close').shift(1).alias('prev_close'),
            pl.col('close_oi').alias('open_interest'), # 改名
            
            # 计算相邻两根K线的物理时间差
            (pl.col('datetime_nano') - pl.col('datetime_nano').shift(1))
            .fill_null(0) # 填充第一根K线
            .alias('time_diff_nano')
        ])
        
        # 2. 识别连续交易时段 (session) 和 跳空缺口 (gap)
        df = df.with_columns([
            # 如果时间跳跃 >= 150分钟 (2.5小时)，则判定为新时段的开端
            pl.when(pl.col('time_diff_nano') >= 150 * minute)
            .then(1).otherwise(0)
            .alias('is_new_session')
        ]).with_columns([
            # 基于新时段标志，累加生成唯一时段ID
            pl.col('is_new_session').cum_sum().alias('continuous_session_id'),
            
            # 核心优化：只在新时段开端计算一次跳空缺口，其余全为0，再做累加
            pl.when(pl.col('is_new_session') == 1)
            .then(pl.col('open') - pl.col('prev_close'))
            .otherwise(0)
            .fill_null(0) # 应对极端的首行 null 情况
            .cum_sum()
            .alias('cum_price_gap')
        ])
        # 3. 计算平移价格与典型价格
        #    shifted_* : 去掉累计跳空的连续价格序列，用于跨bar技术指标（ATR/RSI/MA等）
        #    typical_price : 基于原始OHLC的典型价格，用于收益率/标签分母（语义正确）
        df = df.with_columns([
            (pl.col('open') - pl.col('cum_price_gap')).alias('shifted_open'),
            (pl.col('high') - pl.col('cum_price_gap')).alias('shifted_high'),
            (pl.col('low')  - pl.col('cum_price_gap')).alias('shifted_low'),
            (pl.col('close') - pl.col('cum_price_gap')).alias('shifted_close'),
        ]).with_columns([
            # 原始典型价格：用于 ror/label 分母，不受跳空平移影响
            ((pl.col('high') + pl.col('low') + pl.col('close')) / 3.0)
            .alias('typical_price')
        ])
        # 4. 生成时段内 bar 序号
        df = df.with_columns([
            # 在每个 continuous_session_id 分组内，从 1 开始生成递增序号
            pl.int_range(1, pl.len() + 1).over('continuous_session_id').alias('bar_idx_in_session')
        ])
        # 清理辅助列 (保持 DataFrame 干净)
        df = df.drop(['time_diff_nano', 'prev_close', 'is_new_session'])
        
        return df
    
    def _generate_labels(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        生成标签及因子分析专用的多周期收益率列。

        收益率设计原则
        --------------
        - typical_price = (high + low + close) / 3，基于原始 OHLC，不受跳空平移影响
        - price_change = future_typical - current_typical
        - ror_future = price_change / typical_price
        - label 阈值比较：|price_change| vs threshold × ATR
          price_change 为原始价差（gap 相消），ATR 基于 shifted 序列（消除跳空跳变）
          两侧绝对量均不受平移影响

        分析专用列（不参与模型训练）
        ----------------------------
        - ATR_long       : 长基线ATR（atr_long_window个5分钟bar），用于归一化
        - ror_future_Nb  : 未来N棒的原始典型价格收益率
        - ror_norm_Nb    : ror_future_Nb / (ATR_long / typical_price)，波动率归一化收益
          其中 N ∈ {1, 3, 6, 12}
        """

        # ── 原始标签 ──────────────────────────────────────────────
        df = df.with_columns([
            (pl.col('typical_price').shift(-self.label_bars).over('continuous_session_id') -
             pl.col('typical_price')).alias('price_change'),
        ]).with_columns([
            (pl.col('price_change') / pl.col('typical_price')).alias('ror_future'),
        ]).with_columns([
            pl.when(pl.col('price_change').abs() >= (self.label_threshold * pl.col('ATR')))
            .then(pl.col('price_change').sign())
            .when(pl.col('price_change').abs() < (self.label_threshold * pl.col('ATR')))
            .then(0)
            .otherwise(None)
            .alias('label')
        ])

        # ── 分析专用：长基线ATR（用于 ror_norm 归一化）────────────
        # 使用 shifted_high/low/close 消除跳空跳变，与 ATR14 保持一致
        df = df.with_columns([
            pl.map_batches(
                ['shifted_high', 'shifted_low', 'shifted_close'],
                lambda hlc: talib.ATR(hlc[0], hlc[1], hlc[2],
                                      timeperiod=self.atr_long_window)
            ).fill_nan(None).alias('ATR_long')
        ])

        # ── 分析专用：多周期收益率
        # over('continuous_session_id') 保证不跨 session 计算
        for bars in self.horizon_bars:
            col_raw  = f'ror_future_{bars}b'
            col_norm = f'ror_norm_{bars}b'
            df = df.with_columns([
                pl.col('open').shift(-1).over('continuous_session_id').alias('start_price'), # 下1根bar的开盘
                pl.col('open').shift(-(1 + bars)).over('continuous_session_id').alias('end_price') #从下1根bar开始，持有bars以后的开盘价格
            ]).with_columns([
                (pl.col('end_price') / pl.col('start_price') - 1).alias(col_raw),
            ]).with_columns([
                (
                    pl.col(col_raw) / 
                    (   
                        1e-9 + pl.col('ATR_long') / (1e-9 + pl.col('start_price'))
                    )
                ).alias(col_norm)
            ]).drop(['start_price', 'end_price'])

        return df

    # ========================================
    # FactorLab 专用轻量初始化
    # ========================================

    def _build_lab_infra(self, bar1min: pl.DataFrame,
                         verbose: bool = True) -> None:
        """
        FactorLab 专用轻量初始化，只走最小依赖链。

        对比 create_features_and_labels()，跳过了全部 _compute_all_features()
        （86个特征，15个特征组），只计算 FactorLab 实际需要的内容：

        最小依赖链
        ----------
        _bar_preprocess
          → bar_downsample_to × 3
            → _add_basic_features（session_id, cum_price_gap, shifted_*, typical_price）
              → ATR14（标签阈值所需）
                → _generate_labels（label, ror_future_Nb, ror_norm_Nb, ATR_long）

        完成后填充
        ----------
        self.bars     : 多周期K线（bars["5m"] 含三列结构性元数据）
        self.infra_df : 因子分析基础设施列（框架内部用，不暴露给用户）
                        包含：datetime, datetime_nano, trading_date, open,
                              continuous_session_id, label, ror_future, ATR_long,
                              ror_future_1/3/6/12b, ror_norm_1/3/6/12b
        """
        if verbose:
            print("FactorLab 基础设施初始化（轻量模式，跳过特征计算）...")

        # Step 1: 1m 预处理 + 多周期聚合
        bar1min  = self._bar_preprocess(bar1min)
        bar5min  = self.bar_downsample_to('5m',  bar1min)
        bar30min = self.bar_downsample_to('30m', bar1min)
        bar1d    = self.bar_downsample_to_day(bar1min)

        # Step 2: 基础特征（session_id / cum_price_gap / shifted_* / typical_price）
        working_df = self._add_basic_features(bar5min)

        # Step 3: 单独计算 ATR14（_generate_labels 的标签阈值依赖 ATR）
        # 使用与训练管线完全一致的计算方式（shifted_high/low/close）
        working_df = working_df.with_columns([
            pl.map_batches(
                ['shifted_high', 'shifted_low', 'shifted_close'],
                lambda hlc: talib.ATR(hlc[0], hlc[1], hlc[2])
            ).fill_nan(None).alias('ATR')
        ])

        # Step 4: 生成标签和多周期收益率
        working_df = self._generate_labels(working_df)

        # Step 5: 将结构性元数据 join 到裸 bar5min，存入 self.bars
        bar5min_with_meta = bar5min.join(
            working_df.select(['datetime',
                               'continuous_session_id',
                               'cum_price_gap',
                               'bar_idx_in_session']),
            on='datetime', how='left'
        ).sort(['datetime'])
        self.bars = {
            "1m":  bar1min,
            "5m":  bar5min_with_meta,
            "30m": bar30min,
            "1d":  bar1d,
        }

        # Step 6: 构建基础设施 DataFrame（框架内部用，不对外暴露）
        # open 用于回测的开盘成交价（t+1 bar 开盘入场）
        infra_cols = (
            ['datetime', 'datetime_nano', 'trading_date', 'open',
             'continuous_session_id', 'label', 'ror_future', 'ATR_long'] +
            [f'ror_future_{b}b' for b in self.horizon_bars] +
            [f'ror_norm_{b}b'   for b in self.horizon_bars]
        )
        available = [c for c in infra_cols if c in working_df.columns]
        self.infra_df = working_df.select(available)

        if verbose:
            n_rows = len(self.infra_df)
            dt_min = bar5min['datetime'].min()
            dt_max = bar5min['datetime'].max()
            print(f"  完成：{n_rows:,} 行，时间范围 {dt_min} ~ {dt_max}")
            missing = [c for c in infra_cols if c not in working_df.columns]
            if missing:
                print(f"  ⚠️  以下基础设施列未生成: {missing}")

    # ========================================
    # 工具方法
    # ========================================


    def get_filtered_categorical_features(self, selected_features: List[str]) -> List[str]:
        """
        根据筛选后的特征列表，返回对应的类别特征
        
        Parameters:
        -----------
        selected_features : List[str]
            筛选后保留的特征列表
        
        Returns:
        --------
        filtered_categorical : List[str]
            筛选后的类别特征列表
        """
        selected_set = set(selected_features)
        filtered_categorical = [f for f in self.categorical_features if f in selected_set]
        return filtered_categorical
    
    def save_features(self, 
                     output_dir: str,
                     file_format: str = 'parquet'):
        """
        保存特征和标签到文件
        
        Parameters:
        -----------
        output_dir : str
            输出目录路径
        file_format : str
            文件格式 ('parquet' 或 'csv')
        
        Note:
        -----
        使用 self.features_df 作为数据源
        """
        
        if self.features_df is None:
            raise ValueError("features_df 未初始化，请先调用 create_features_and_labels()")
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 提取特征列和标签列
        feature_cols = ['datetime', 'datetime_nano'] + self.feature_cols
        label_cols = ['datetime', 'datetime_nano', 'label', 'ror_future', 'price_change']
        
        features_df = self.features_df.select(feature_cols)
        labels_df = self.features_df.select(label_cols)
        
        # 保存文件
        if file_format == 'parquet':
            features_df.write_parquet(output_dir / 'features.parquet')
            labels_df.write_parquet(output_dir / 'labels.parquet')
        elif file_format == 'csv':
            features_df.write_csv(output_dir / 'features.csv')
            labels_df.write_csv(output_dir / 'labels.csv')
        else:
            raise ValueError(f"不支持的文件格式: {file_format}")
        
        # 保存特征列表
        feature_list_data = []
        for i, feat in enumerate(self.feature_cols):
            is_categorical = feat in self.categorical_features
            
            # 找到特征所属组
            group_name = "未分组"
            for grp_name, info in self.feature_groups_info.items():
                if feat in info['features']:
                    group_name = grp_name
                    break
            
            feature_list_data.append({
                'index': i,
                'feature_name': feat,
                'is_categorical': is_categorical,
                'group': group_name,
            })
        
        feature_list_df = pl.DataFrame(feature_list_data)
        feature_list_df.write_csv(output_dir / 'feature_list.csv')
        
        # 保存类别特征列表
        if self.categorical_features:
            categorical_df = pl.DataFrame({
                'categorical_feature': self.categorical_features
            })
            categorical_df.write_csv(output_dir / 'categorical_features.csv')
        
        print(f"  ✓ 特征已保存: {output_dir}")
        print(f"  ✓ 类别特征: {len(self.categorical_features)} 个")
