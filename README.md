# feature_mining

中国商品期货市场的**因子挖掘与模型训练框架**，覆盖从原始1分钟K线到三分类信号模型的完整流程，同时提供独立的单因子分析管线。

---

## 目录结构

```
feature_mining/
├── config.json             # 全局配置（品种、特征、训练、因子分析）
├── main.py                 # 主入口，支持 train / factor_analysis 两种模式
├── data_loader.py          # 数据加载，从本地 CSV 读取并缓存为 parquet
├── feature_engineering.py  # 特征工程：多周期聚合、自动发现特征模块、标签生成
├── factors/                # 特征函数子包（按文件字母序自动加载）
│   ├── __init__.py         # 包声明
│   └── feat_basic.py       # 基础特征组（15组86个特征）
├── model_train.py          # LightGBM 训练：两阶段训练、SHAP筛选、评估
├── factor_analysis.py      # 单因子分析：IC分析、OLS Beta、多阈值回测、相关性矩阵
├── factor_lab.ipynb        # 交互式因子实验室（Jupyter Notebook）
├── temp_data/              # 数据缓存（1m K线 parquet 文件）
├── train_result/           # 模型训练结果
└── factor_result/          # 因子分析结果
```

---

## 快速开始

```bash
# 模型训练（全部品种）
uv run python main.py --mode train

# 单因子批量分析
uv run python main.py --mode factor_analysis

# 指定因子和品种
uv run python main.py --mode factor_analysis --factors RSI_mid,ret_1h --symbols SHFE_au

# 交互式因子实验（Notebook）
uv run jupyter notebook factor_lab.ipynb
```

---

## 两大工作流

### 工作流 A：模型训练

```
1m K线（CSV）
  → data_loader       读取、清洗、缓存 parquet
  → FeatureEngineer   多周期聚合 + 86个特征 + 标签
  → ModelTrainer      两阶段 LightGBM 训练
  → train_result/
```

**时序切分**（严格无未来信息）：

| 集合 | 范围 | 用途 |
|---|---|---|
| 训练集 | `train_start_year` ~ `train_end_year` 前70% | 模型拟合 |
| 验证集 | `train_start_year` ~ `train_end_year` 后30% | 早停、调参 |
| 测试集 | `train_end_year` 之后 | 样本外评估 |

两组之间有 `embargo` 根 bar 的间隔，防止标签泄露。

**两阶段训练**：先用全量特征预训练 → 用 SHAP 删去重要性末20%的特征（含 DUMMY 基准特征） → 在筛选后的特征集上重新训练最终模型。

---

### 工作流 B：单因子分析

```
1m K线
  → FeatureEngineer   同上（含 ror_norm 等分析专用列）
  → SingleFactorAnalyzer（逐品种 × 逐因子）
      IC 分析：非重叠窗口 IC + ICIR + t统计量 + p值
      OLS Beta：窗口级 OLS + 逐窗口 t统计量
      回测：持仓状态机 × 多阈值，含/不含成本净值曲线双轨对比
      分层分析：滚动分位数分层 × 时段热力图
  → CrossFactorAnalyzer  因子相关性矩阵 + 聚类树
  → factor_result/
```

#### IC 分析方法

采用**非重叠窗口 IC**：将样本期按交易日序号切分为长度 = `window_days`（默认20个交易日）的非重叠窗口，在每个窗口内用全量 bar 计算 Spearman IC。

- 每窗口约 `63 × window_days ≈ 1260` 根 bar，消除日内因子值单调结构对 IC 的虚高干扰
- 尾部不足 `window_days` 个交易日的窗口直接丢弃
- ICIR = `mean(IC) / std(IC)`（跨窗口序列，不年化）
- 统计量 `t = IC均值 × sqrt(N窗口) / IC标准差`，自由度 = N窗口 - 1

#### Rolling OLS Beta 分析方法

与 IC 并列的第二个统计验证维度：

- **X（因子值）**：先做滚动 z-score 标准化（`norm_window` 根 bar，仅用历史数据，无未来信息），适应不同 market regime
- **回归**：在每个非重叠窗口内对 `ror_norm_Nb ~ z_factor` 做 OLS，记录窗口 beta 和 R²
- **beta 含义**：因子值每增加1个近期标准差，预期未来收益变化 beta
- **汇总统计**：beta均值、t统计量（跨窗口）、beta>0占比、R²均值
- 输出多预测周期（1/3/6/12b）的 beta 衰减曲线

#### 回测方法

基于**持仓状态机（Position-State Model）**，使用滚动分位数动态阈值（无未来信息）产生方向信号，模拟真实 CTA 回测逻辑：

- **信号规则**：因子值 > 滚动上分位数（`hi_q`）→ 做多；因子值 < 滚动下分位数（`lo_q`）→ 做空；其余 → 空仓等待
- **成交规则**：信号产生后，下一根 bar 开盘价成交（与 `ror_future_Nb` 的 `start_price` 完全对齐）
- **持仓到期**：固定持有 `label_bars` 根 bar 后平仓；到期时若有新信号，同价平旧开新（续仓或反手），扣双边成本
- **反手规则**：持仓中出现反向信号，下一根 bar 开盘平旧仓开新仓，扣双边成本
- **Session 边界**：靠近 session 末的 bar（`ror_future_Nb` 为 null）禁止新开仓，自然平仓，不跨夜
- **成本**：`单边 = slippage_ticks × tick_size / 入场价`，开平仓各一次（往返共 2 次单边）
- **双轨输出**：每个阈值对同时输出 cost=0（毛收益，虚线）和 with cost（净收益，实线）净值曲线，可直观评估成本侵蚀
- **多阈值并行**：支持同时回测多组 `(lo_q, hi_q)` 阈值，净值曲线叠加对比

#### 因子分层分析方法

在 IC / OLS / 回测之外新增的第四个分析维度，回答**"因子值越极端，收益是否越高？不同交易时段的因子效果是否有差异？"**

**维度一：全样本分层收益分布（单调性验证）**

- 用 Polars `rolling_quantile`（向量化，无未来信息）计算因子值的滚动分位数边界（回望 `rolling_window` 根 bar）
- 将每根 bar 分配到 Q1 ~ Q5 层（Q1 最低，Q5 最高）
- 统计每层的均值收益、标准差、胜率、偏度等，绘制柱状图（含误差棒）和箱线图
- 输出单调性系数（Spearman）和多空价差（Q5 - Q1）

**维度二：时段 × 分层 交叉热力图（时段效应验证）**

- 根据品种 `session_template` 配置，将每根 bar 映射到对应时段
- 对每个时段内的 bar 独立计算各层均值收益，绘制时段 × 分层 热力图
- 样本量 < 30 的格子自动打灰色遮罩

---

## config.json 说明

### `symbols`

```json
{
  "code": "KQ.m@SHFE.au",
  "tick_size": 0.02,
  "session_template": "night_0230"
}
```

| 字段 | 说明 |
|---|---|
| `code` | 品种代码（TqSDK 主连格式） |
| `tick_size` | 最小变动价位（价格单位），用于将滑点从跳数换算为价格 |
| `session_template` | 交易时段模板名，用于因子分层分析的时段维度。**必填**，缺失时降级为 `night_2300` 并打印警告。支持内置模板名（字符串）或自定义模板（`list[dict]`，见下文） |

**内置交易时段模板：**

| 模板名 | 适用品种 | 夜盘 | 开盘 | 收盘 |
|---|---|---|---|---|
| `day_only` | 无夜盘商品（部分农产品等） | 无 | 09:00 | 15:00 |
| `night_2300` | 黑色系（焦炭j、铁矿i、纯碱SA、玻璃FG等） | 21:00 ~ 23:00 | 09:00 | 15:00 |
| `night_0100` | 有色金属（铜cu、铝al等） | 21:00 ~ 01:00 | 09:00 | 15:00 |
| `night_0230` | 贵金属（黄金au、白银ag） | 21:00 ~ 02:30 | 09:00 | 15:00 |
| `stock_index` | 股指期货（IF/IC/IH/IM，CFFEX） | 无 | **09:30** | **15:00** |
| `bond` | 国债期货（T/TF/TS/TL，CFFEX） | 无 | **09:15** | **15:15** |

**自定义模板**：直接在 `session_template` 字段传入 `list[dict]`，数据结构与内置模板一致：

```json
{
  "code": "KQ.m@XXX.yy",
  "tick_size": 1.0,
  "session_template": [
    {"name": "特殊段", "start": "20:00", "end": "21:00"},
    {"name": "日盘",   "start": "09:00", "end": "15:00"}
  ]
}
```

每个时段条目格式为 `{"name": "<时段名>", "start": "HH:MM", "end": "HH:MM"}`，支持跨日时段（`end < start` 时自动识别为跨日，如 `"start": "23:00", "end": "01:00"`）。

### `data_split`

| 参数 | 说明 |
|---|---|
| `train_start_year` | 训练集起始年份（之前的数据用于技术指标预热，不参与统计） |
| `train_end_year` | 样本内外的严格边界，因子分析和训练均不使用此后数据 |
| `valid_ratio` | 验证集占训练+验证集的比例 |
| `embargo` | 训练集与验证集之间的间隔 bar 数 |

### `feature_engineering`

| 参数 | 说明 |
|---|---|
| `label_bars` | 预测窗口（根数），6 = 未来30分钟。同时决定回测持仓时长 |
| `label_threshold` | 标签阈值（ATR 倍数），`\|价格变动\| >= threshold × ATR` 才产生方向标签 |
| `atr_long_window` | 长基线 ATR 窗口（bar数），用于 `ror_norm` 的波动率归一化 |

### `factor_analysis`

| 参数 | 默认值 | 说明 |
|---|---|---|
| `ic_forward_periods` | `[1, 3, 6, 12]` | IC / OLS 的多预测周期（bar数） |
| `use_ror_norm` | `true` | 使用 ATR 归一化收益率（`ror_norm_Nb`）还是原始收益率 |
| `window_days` | `20` | IC / OLS 的非重叠窗口交易日数 |
| `ols_norm_window` | `200` | OLS X（因子值）的滚动 z-score 回望窗口（bar数） |

### `factor_analysis.backtest`

| 参数 | 默认值 | 说明 |
|---|---|---|
| `enabled` | `true` | 是否在批量因子分析中运行回测（`false` 时跳过该步骤） |
| `slippage_ticks` | `1.5` | 单边滑点（跳数），开仓和平仓各扣一次（往返共 3 跳） |
| `rolling_window` | `500` | 滚动分位数回望窗口（bar数），与 `quantile_analysis.rolling_window` 含义一致 |
| `min_samples` | `20` | 滚动分位数最小样本数，不足时该 bar 信号置为 0 |
| `thresholds` | `[[0.2,0.8],[0.25,0.75],[0.3,0.7]]` | 多阈值对列表，每对 `[lo_q, hi_q]` 独立运行状态机，结果并排对比 |

### `factor_analysis.quantile_analysis`

因子分层收益分析的参数配置。

| 参数 | 默认值 | 说明 |
|---|---|---|
| `n_quantiles` | `5` | 分层数（Q1 ~ Q5），因子值从低到高等频分层 |
| `rolling_window` | `500` | 滚动分位数边界的回望窗口（bar数），前 `rolling_window` 根 bar 为预热期，不参与分析 |
| `min_samples` | `20` | 滚动分位数计算的最小样本数，不足时该 bar 分层标签置 null |

> **时段配置**已移至品种级 `session_template` 字段，不再在此处配置全局 `session_slots`。

---

## 特征工程

### 标签设计

分析专用收益率标签与实盘信号逻辑完全对齐：

```
start_price = open[t+1]         （下一根bar开盘入场）
end_price   = open[t+1+N]       （持有N根bar后下一根开盘平仓）
ror_future_Nb = end_price / start_price - 1

ror_norm_Nb = ror_future_Nb / (ATR_long / start_price)   （ATR 归一化）
```

模型训练的三分类标签 `label` 仍使用典型价格（`typical_price`）计算价格变动，由 `label_threshold × ATR14` 决定中性区间宽度。

### 跳空消除

`_add_basic_features` 在 session 边界处累计跳空金额 `cum_price_gap`，生成连续价格序列：

```
shifted_close = close - cum_price_gap
```

**计算规则**：

- `shifted_*`：去跳空价格，用于所有跨 bar 的技术指标（RSI、MA、ATR 等）
- `typical_price = (high + low + close) / 3`：原始价格，用于模型训练的标签计算

### 特征模块化架构

所有特征函数统一存放在 `factors/` 子包，`FeatureEngineer` 通过自动发现机制加载：

1. 扫描 `factors/*.py`（按文件名字母序）
2. 读取每个文件的模块级 `PIPELINE = [(组名, func), ...]`
3. 按顺序串联执行所有函数

**函数协议**（所有特征函数统一签名）：

```python
def feat_xxx(
    df:   pl.DataFrame,   # 当前 working_df（5m K线 + 已计算的前序特征列）
    bars: dict,           # 多周期K线字典 {"1m", "5m", "30m", "1d"}
) -> Tuple[pl.DataFrame, List[str], List[str]]:
    # 返回 (enriched_df, feature_names, categorical_feature_names)
```

**新增特征的两种方式**：

```python
# 方式A：向现有文件加特征（只改 feat_basic.py）
def feat_new_alpha(df, bars):
    df = df.with_columns([...])
    return df, ['new_col'], []

PIPELINE = [..., ('新特征组', feat_new_alpha)]  # 追加一行


# 方式B：新建特征文件（feature_engineering.py 零改动）
# 新建 factors/feat_derived.py，提供符合协议的函数和 PIPELINE 即可
# 文件名字母序靠后的文件，可依赖靠前文件的输出列
```

### 15 大特征组（86 个特征）

| 特征组 | 代表特征 |
|---|---|
| 技术指标 | ATR、RSI、ADX、MACD |
| 价格位置 | pos_1h/1d/5d（去跳空高低区间内的相对位置）|
| 成交量 | vol_ratio、volume_spike |
| 持仓量 | oi_change、oi_trend_strength |
| 价格动量 | ret_30min/1h/1d（去跳空）、MA 偏离度 |
| 突破 | breakout_high/low_1h/1d、range_break_strength |
| 波动率 | realized_vol、parkinson_vol、atr_trend |
| K线形态 | body_ratio、upper/lower_shadow、close_strength |
| 量价配合 | OBV、vwap_deviation、price_up_vol_up |
| 仓量价 | bullish/bearish_trio、oi_price_div |
| 开盘特征 | open_momentum、open_width |
| 时间特征 | session_type、weekday、bar_idx_in_session |
| 相对强度 | return_from_open、price_rank_1d |
| 多周期一致性 | multi_tf_consistency、ma_alignment |
| 标准化因子 | ATR_z、RSI_mid_z 等滚动 IQR z-score |

---

## 交互式因子实验室

`factor_lab.ipynb` 提供 Notebook 内的单因子快速验证，初始化一次后可反复测试因子，无需重新加载数据。

```python
from factor_analysis import FactorLab

lab = FactorLab("SHFE_au")   # 初始化（约10-30秒，只需一次）

def my_factor(bars: dict) -> pl.Series:
    bar5m = bars['5m']
    sc = bar5m['close'] - bar5m['cum_price_gap']
    ma20 = bar5m.with_columns(
        (pl.col('close') - pl.col('cum_price_gap')).alias('_sc')
    ).with_columns(
        pl.col('_sc').rolling_mean(20).alias('_ma20')
    )['_ma20']
    return -(sc - ma20) / (ma20.abs() + 1e-9)

# IC 分析（非重叠20交易日窗口）
stats = lab.ic_only(my_factor, "ma20_dev")
stats = lab.ic_only(my_factor, "ma20_dev", window_days=60)   # 自定义窗口

# Rolling OLS Beta 分析
stats = lab.ols_only(my_factor, "ma20_dev")
stats = lab.ols_only(my_factor, "ma20_dev", window_days=20, norm_window=200)

# 单因子回测（持仓状态机，含/不含成本双轨）
stats = lab.backtest_only(
    my_factor, "ma20_dev",
    thresholds=[(0.2, 0.8), (0.25, 0.75)],   # None → 读 config 默认值
    rolling_window=500,
    slippage_ticks=1.5,
)

# 因子分层收益分析（使用内置时段模板）
stats = lab.quantile_only(my_factor, "ma20_dev", session_template="night_0230")
stats = lab.quantile_only(my_factor, "ma20_dev", session_template="stock_index")
stats = lab.quantile_only(my_factor, "ma20_dev", session_template="bond")

# 因子分层收益分析（自定义时段）
stats = lab.quantile_only(my_factor, "ma20_dev", session_template=[
    {"name": "尾盘15分钟", "start": "14:45", "end": "15:00"},
    {"name": "开盘30分钟", "start": "09:00", "end": "09:30"},
])

# 完整分析（IC + 回测）
stats = lab.analyze(my_factor, "ma20_dev")
```

### `bars` 字典

| key | 内容 |
|---|---|
| `bars["1m"]` | 1分钟 OHLCV + `trading_date` |
| `bars["5m"]` | 5分钟 OHLCV + `trading_date` + `continuous_session_id` + `cum_price_gap` + `bar_idx_in_session` |
| `bars["30m"]` | 30分钟 OHLCV + `trading_date` |
| `bars["1d"]` | 日线 OHLCV + `trading_date` |

**数据隔离**：`bars` 中的数据严格不包含 `train_end_year` 之后的行情。IC 分析和回测仅在 `[train_start_year, train_end_year]` 区间内运行。

> **⚠️ `.over()` 使用规则**  
> `.over()` 是 `pl.Expr` 的方法，不能对 `pl.Series` 直接调用。  
> 凡是需要 session 内 rolling/shift 的计算，必须在 `with_columns()` 内使用 `pl.col(...)` 表达式。

### 数据泄露检测

`FactorLab.check_leakage()` 基于注入测试（Perturbation Test），可以严格验证因子函数是否存在前向偏差（look-ahead bias）：

**原理**：选定注入点 `inject_idx`，将该点及之后所有1m bar 的 OHLC（+`inject_delta`）、volume 和 OI（×2）全部修改，然后对比注入前后两次完整 Pipeline 输出。注入点之前的输出（安全区）若有任何变化，即判定为数据泄露。

```python
# 默认：注入数据集中间位置，同时检查多周期K线和因子值
result = lab.check_leakage(my_factor, name='ma20_dev')

# 只检查因子值（更快）
result = lab.check_leakage(my_factor, name='ma20_dev', check_bars=False)

# 指定注入位置（第1000根1m bar所在的5m窗口）
result = lab.check_leakage(my_factor, name='ma20_dev', inject_idx=1000)

# 返回值
result['leaked']              # bool：汇总结论
result['factor']['leaked']    # bool：因子值是否泄露
result['5m_bars']['leaked']   # bool：5m K线是否泄露
result['5m_bars']['leaked_columns']  # list：泄露的列及首次泄露时间
```

> **注意**：`check_leakage` 不会修改 `self._bar1min`（Polars DataFrame 不可变），调用后继续使用 `ic_only` / `ols_only` 等方法，结果与检测前完全一致。

---

## 输出目录

### `train_result/`

```
train_result/
├── report_{timestamp}.txt          # 全品种汇总报告
└── {EXCHANGE_SYMBOL}/
    ├── features/features.parquet   # 特征标签表
    ├── models/                     # 训练好的模型文件
    ├── figures/                    # SHAP图、混淆矩阵等
    └── results/                    # 预测结果、参数扫描结果
```

### `factor_result/`

```
factor_result/
├── factor_summary_rank.csv         # 所有因子 × 品种的 IC/OLS/回测/分层综合排名
├── ic_summary_rank.csv             # IC 排名
├── ols_summary_rank.csv            # OLS Beta 排名
├── backtest_summary_rank.csv       # 回测排名
├── quantile_summary_rank.csv       # 分层分析排名（单调性系数、多空价差）
├── {EXCHANGE_SYMBOL}/
│   ├── ic/
│   │   ├── ic_stats_all_factors.csv
│   │   ├── ic_decay_{factor}.csv       # 各预测周期的 IC 衰减统计（含 window_days 字段）
│   │   └── figures/
│   │       ├── ic_heatmap.png          # 因子 × 预测周期 ICIR 热图
│   │       └── ic_timeseries_{factor}.png  # 窗口级 IC 时序柱状图
│   ├── ols/
│   │   ├── ols_stats_all_factors.csv
│   │   ├── ols_decay_{factor}.csv      # 各预测周期的 OLS Beta 衰减统计
│   │   └── figures/
│   │       └── ols_timeseries_{factor}.png  # 窗口级 Beta 时序图 + R² 散点
│   ├── backtest/
│   │   ├── figures/
│   │   │   ├── backtest_{factor}.png           # 净值曲线 + 回撤（多阈值双轨，cost=0虚线/有成本实线）
│   │   │   └── backtest_metrics_{factor}.png   # 绩效指标汇总表（各阈值 × 含/不含成本）
│   │   ├── backtest_{factor}.csv               # 逐阈值 × 含/不含成本的绩效指标
│   │   └── backtest_all_factors.csv            # 品种内所有因子的最优阈值汇总
│   └── quantile/                       # 因子分层分析输出
│       ├── quantile_stats_all_factors.csv   # 各因子的分层摘要（单调性系数、多空价差）
│       ├── quantile_stats_{factor}.csv      # 单因子各层统计（均值、std、胜率、偏度等）
│       ├── quantile_timeslot_{factor}.csv   # 单因子各层 × 各时段均值收益
│       └── figures/
│           ├── quantile_bar_{factor}.png             # 分层均值收益柱状图（含误差棒、胜率折线）
│           ├── quantile_boxplot_{factor}.png          # 分层收益箱线图
│           └── quantile_timeslot_heatmap_{factor}.png # 时段 × 分层 均值收益热力图
└── cross_factor/
    ├── correlation_matrix.csv
    └── figures/
        ├── heatmap.png                 # 因子相关性热度图（按特征组排序）
        └── dendrogram.png              # 层次聚类树
```

---

## 注意事项

**中文字体**：图表使用 `Noto Sans CJK`，如显示方框请执行：

```bash
sudo apt-get install fonts-noto-cjk
rm -rf ~/.cache/matplotlib/
```

**依赖**：

```bash
uv sync
```
