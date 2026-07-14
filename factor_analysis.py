"""
单因子分析模块

分析框架说明
------------
本模块面向 CTA 时序因子研究，不照搬股票截面 IC 分析，具体体现在：

1. IC 计算采用「非重叠窗口 IC」：
   - 将样本期按交易日序号切分为长度 = window_days（默认20个交易日）的非重叠窗口
   - 在每个窗口内用全量 bar 计算 Pearson IC（因子值与未来收益率的线性相关，约1260根bar/窗口）
   - 消除日内因子值单调结构对 IC 的虚高干扰，统计量更可靠
   - ICIR = mean(IC_window) / std(IC_window)（不年化，以窗口为单位）
   - 同时输出 t 统计量和 p 值；尾部不完整窗口直接丢弃

2. Rolling OLS Beta 分析：
   - X（因子值）先做滚动 z-score（norm_window 根 bar，仅用历史，适应 regime 变化）
   - 在每个非重叠窗口内对 ror_norm_Nb ~ z_factor 做 OLS，记录窗口 beta 和 R²
   - 跨窗口汇总：beta 均值、t 统计量（= beta_mean × sqrt(N_windows) / beta_std）、beta>0 占比
   - 逐窗口 OLS 已消除跨窗口序列相关，无需 Newey-West 修正
   - beta 含义：因子值每增加1个近期标准差，预期未来收益变化 beta

3. 资金曲线回测：
   - 滚动分位数信号（动态阈值，无未来信息）
    - 下一根 bar 开盘成交，扣除单边摩擦成本（slippage_ticks 跳，含滑点与手续费）
   - session 末强制平仓（不跨夜）
   - 支持多阈值并行回测，输出净值曲线、回撤曲线对比图

4. 因子分层收益分析（Quantile Analysis）：
   - 用 Polars rolling_quantile 向量化计算滚动分位数边界（无未来信息）
   - 将每根 bar 分配到 Q1~Qn 层（Q1 最低，Qn 最高），预热期自动丢弃
   - 维度一（单调性）：全样本各层均值收益柱状图 + 箱线图，输出
     Spearman 单调性系数和多空价差（Qn - Q1）
   - 维度二（时段效应）：按品种交易时段模板将 bar 映射到对应时段，
     输出「时段 × 分层」均值收益热力图（样本量 < 30 的格子打灰色遮罩）

5. 因子间相关性分析：
   - 使用非重叠样本计算 Spearman 相关矩阵
   - 按特征组排序的热度图，标注高相关因子对
   - 层次聚类树

交易时段模板（SESSION_TEMPLATES）
----------------------------------
因子分层分析的时段维度通过品种级 session_template 配置，内置六套模板：

  day_only     无夜盘商品（部分农产品等）      09:00 开盘，15:00 收盘
  night_2300   黑色系（j/i/SA/FG 等）         夜盘 21:00~23:00
  night_0100   有色金属（cu/al 等）            夜盘 21:00~01:00
  night_0230   贵金属（au/ag）                夜盘 21:00~02:30
  stock_index  股指期货（IF/IC/IH/IM，CFFEX）  09:30 开盘，15:00 收盘，无夜盘
  bond         国债期货（T/TF/TS/TL，CFFEX）   09:15 开盘，15:15 收盘，无夜盘

也支持在 config.json 的 symbols[i].session_template 中直接传入
list[dict] 格式的自定义模板，结构与内置模板一致：
  [{"name": "时段名", "start": "HH:MM", "end": "HH:MM"}, ...]
跨日时段（end < start）自动识别（如 "start": "23:00", "end": "01:00"）。
品种未配置 session_template 时，打印警告并降级为 night_2300 默认模板。

运行方式
--------
python main.py --mode factor_analysis [--factors RSI_mid,ATR_z] [--symbols SHFE_au]
"""

import json
import datetime
import warnings
import numpy as np
import polars as pl
import matplotlib

def _is_notebook() -> bool:
    """检测当前是否在 Jupyter Notebook / IPython 环境中运行"""
    try:
        from IPython import get_ipython
        return get_ipython() is not None
    except ImportError:
        return False

# 批量任务（非 notebook）使用 Agg 后端，避免无头环境报错
# notebook 环境保留默认 inline 后端，支持 plt.show() 内嵌显示
if not _is_notebook():
    matplotlib.use('Agg')

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from scipy import stats
from scipy.cluster.hierarchy import linkage, dendrogram
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import statsmodels.api as sm

warnings.filterwarnings('ignore')

import sys
sys.path.append(str(Path(__file__).parent))

from data_loader import batch_download
from feature_engineering import FeatureEngineer
from main import parse_symbols, get_symbol_name

# ── 绘图全局设置 ──────────────────────────────────────────────────
from matplotlib import font_manager as _fm
_available_fonts = {f.name for f in _fm.fontManager.ttflist}
# Noto Sans CJK 在不同系统上可能以 SC/TC/JP 等不同 locale 注册，逐一尝试
_NOTO_CJK_CANDIDATES = [
    'Noto Sans CJK SC', 'Noto Sans CJK TC', 'Noto Sans CJK JP',
]
_chosen_font = next((f for f in _NOTO_CJK_CANDIDATES if f in _available_fonts), None)
if _chosen_font:
    plt.rcParams['font.family'] = _chosen_font
else:
    warnings.warn("未找到 Noto Sans CJK 字体，中文可能显示为方框。"
                  "请安装：sudo apt-get install fonts-noto-cjk"
                  "，并清除缓存：rm -rf ~/.cache/matplotlib/")
plt.rcParams['figure.dpi']       = 120
sns.set_style('whitegrid')
# 字体设置必须在 sns.set_style() 之后，否则会被 seaborn 的 rcParams 覆盖
if _chosen_font:
    plt.rcParams['font.family'] = _chosen_font
plt.rcParams['axes.unicode_minus'] = False


# ==============================================================
# 0. 内置交易时段模板
# ==============================================================

SESSION_TEMPLATES: Dict[str, List[Dict]] = {
    # ──────────────────────────────────────────────────────────────────
    # 时段划分说明
    # ──────────────────────────────────────────────────────────────────
    # 分层收益分析（run_quantile_analysis）在计算收益率时使用
    # `.over('continuous_session_id')` 约束，不允许跨 session 的 shift。
    # 因此每个 session 最后 label_bars 根 bar 的未来收益率为 null，
    # 会在预处理阶段被过滤掉，不参与任何时段的统计。
    #
    # 实际影响（以 5 分钟 bar、label_bars=6 为例）：
    #   每个 session 末尾 6 根 bar = 30 分钟 的数据将被排除。
    #   若某个时段恰好与这段末尾完全重合（如收盘段 14:30~15:00），
    #   该时段在热力图中将呈现为零样本（灰色遮罩）。
    #   这是正常现象，反映了"临近收盘时无法完整计算持仓收益"的事实。
    #   若需要该时段出现有效样本，可适当缩短 label_bars（如改为3），
    #   或将收盘段时间提前（如 14:00~14:30）。
    # ──────────────────────────────────────────────────────────────────

    # 纯白盘品种（无夜盘）：部分农产品等
    "day_only": [
        {"name": "开盘段",   "start": "09:00", "end": "10:00"},
        {"name": "上午中段", "start": "10:00", "end": "11:30"},
        {"name": "下午开盘", "start": "13:00", "end": "14:00"},
        {"name": "收盘段",   "start": "14:00", "end": "15:00"},
        # 注：14:00~15:00 末尾 label_bars 根 bar 无有效收益率，
        #     label_bars=6 时收盘段后30分钟样本为零。
    ],
    # 夜盘至23:00：黑色系（焦炭j、铁矿i、纯碱SA、玻璃FG等）
    "night_2300": [
        {"name": "夜盘",     "start": "21:00", "end": "23:00"},
        # 注：22:30~23:00（最后30分钟）样本为零（session 末尾 null 区）。
        {"name": "开盘段",   "start": "09:00", "end": "10:00"},
        {"name": "日盘中段", "start": "10:00", "end": "14:30"},
        {"name": "收盘段",   "start": "14:30", "end": "15:00"},
        # 注：label_bars=6 时收盘段（14:30~15:00）= 末尾 null 区，样本为零。
    ],
    # 夜盘至次日01:00：有色金属（铜cu、铝al等）
    "night_0100": [
        {"name": "夜盘前段", "start": "21:00", "end": "23:00"},
        {"name": "夜盘后段", "start": "23:00", "end": "01:00"},
        # 注：00:30~01:00（最后30分钟）样本为零。
        {"name": "开盘段",   "start": "09:00", "end": "10:00"},
        {"name": "日盘中段", "start": "10:00", "end": "14:30"},
        {"name": "收盘段",   "start": "14:30", "end": "15:00"},
        # 注：label_bars=6 时收盘段（14:30~15:00）样本为零。
    ],
    # 夜盘至次日02:30：贵金属（黄金au、白银ag）
    "night_0230": [
        {"name": "夜盘前段", "start": "21:00", "end": "23:00"},
        {"name": "夜盘后段", "start": "23:00", "end": "02:30"},
        # 注：02:00~02:30（最后30分钟）样本为零。
        {"name": "开盘段",   "start": "09:00", "end": "10:00"},
        {"name": "日盘中段", "start": "10:00", "end": "14:30"},
        {"name": "收盘段",   "start": "14:30", "end": "15:00"},
        # 注：label_bars=6 时收盘段（14:30~15:00）样本为零。
    ],
    # 股指期货（IF/IC/IH/IM，CFFEX）：09:30开盘，15:00收盘，无夜盘
    "stock_index": [
        {"name": "开盘段",   "start": "09:30", "end": "10:30"},
        {"name": "上午中段", "start": "10:30", "end": "11:30"},
        {"name": "下午开盘", "start": "13:00", "end": "14:00"},
        {"name": "收盘段",   "start": "14:00", "end": "15:00"},
        # 注：label_bars=6 时收盘段（14:30~15:00）样本为零。
    ],
    # 国债期货（T/TF/TS/TL，CFFEX）：09:15开盘，15:15收盘，无夜盘
    "bond": [
        {"name": "开盘段",   "start": "09:15", "end": "10:15"},
        {"name": "上午中段", "start": "10:15", "end": "11:30"},
        {"name": "下午开盘", "start": "13:00", "end": "14:15"},
        {"name": "收盘段",   "start": "14:15", "end": "15:15"},
        # 注：label_bars=6 时收盘段（14:45~15:15）样本为零。
    ],
}

# ==============================================================
# 1. SingleFactorAnalyzer
# ==============================================================

class SingleFactorAnalyzer:
    """
    单因子分析器：IC分析 + 资金曲线回测
    每次实例对应一个品种的完整数据集

    两种运行模式
    ------------
    批量模式（默认）：output_dir 不为 None，图表写入磁盘，不调用 plt.show()
    inline 模式：output_dir=None 或 inline=True，图表调用 plt.show() 内嵌显示，
                 不写任何磁盘文件，适合 Jupyter Notebook 交互探索
    """

    def __init__(self,
                 symbol_name: str,
                 label_bars: int,
                 ic_forward_periods: List[int],
                 use_ror_norm: bool,
                 tick_size: float,
                 output_dir: Optional[Path],
                 inline: bool = False,
                 norm_window: int = 200,
                 window_days: int = 20,
                 winsorize_ror: bool = True,
                 winsorize_quantiles: tuple = (0.01, 0.99)):
        self.symbol_name         = symbol_name
        self.label_bars          = label_bars
        self.ic_forward_periods  = ic_forward_periods
        self.use_ror_norm        = use_ror_norm
        self.tick_size           = tick_size
        self.output_dir          = output_dir
        self.norm_window         = norm_window
        self.window_days         = window_days
        self.winsorize_ror       = winsorize_ror
        self.winsorize_quantiles = winsorize_quantiles
        # output_dir 为 None 时自动切换到 inline 模式
        self.inline              = inline or (output_dir is None)

        if output_dir is not None:
            # 批量模式：创建磁盘目录
            self.ic_dir  = output_dir / 'ic'
            self.ols_dir = output_dir / 'ols'
            self.bt_dir  = output_dir / 'backtest'
            for d in [self.ic_dir  / 'figures',
                      self.ols_dir / 'figures',
                      self.bt_dir  / 'figures']:
                d.mkdir(parents=True, exist_ok=True)
        else:
            # inline 模式：无磁盘目录
            self.ic_dir  = None
            self.ols_dir = None
            self.bt_dir  = None

    # ──────────────────────────────────────────────────────────
    # 公共辅助方法
    # ──────────────────────────────────────────────────────────

    def _winsorize_array(self, arr: np.ndarray) -> np.ndarray:
        """
        对 numpy 数组做窗口内截尾（winsorize）。

        设计原则
        --------
        - 仅在 run_ic / run_ols 的窗口循环内部调用，每次传入单个窗口的
          收益率数组，基于该窗口自身的分位数做截断。
        - 窗口内截尾（而非全样本截尾）可自适应不同 regime 的收益分布，
          避免用牛市极端值截断熊市尾部（反之亦然）。
        - winsorize_ror=False 时直接返回原数组，调用方无需加 if 判断。
        - NaN / Inf 不参与分位数计算，clip 也不改变其值，
          不影响后续 mask = np.isfinite(x) & np.isfinite(y) 的有效性。

        截断位置由 self.winsorize_quantiles 控制，默认 (0.01, 0.99)。
        """
        if not self.winsorize_ror:
            return arr
        lo_q, hi_q = self.winsorize_quantiles
        finite_vals = arr[np.isfinite(arr)]
        if len(finite_vals) < 10:
            return arr
        lo = np.percentile(finite_vals, lo_q * 100)
        hi = np.percentile(finite_vals, hi_q * 100)
        return np.clip(arr, lo, hi)

    def _assign_window_ids(self, df: pl.DataFrame) -> Tuple[pl.DataFrame, Dict]:
        """
        按交易日排序序号将每根 bar 分配到非重叠的 window_days 大小窗口。

        原则
        ----
        - 以 trading_date 的排序序号为基准（不用日历天数），确保每个窗口
          恰好包含 window_days 个完整交易日。
        - 不足 window_days 个交易日的尾部窗口直接丢弃（不完整窗口）。
        - window_days=1 时退化为逐日行为（向后兼容）。

        Returns
        -------
        df_with_wid : pl.DataFrame
            增加了 '__window_id__' 列
        window_meta : Dict[int, date]
            window_id → 该窗口最后一个交易日（用作时间坐标）
        """
        # 1. 取有序唯一交易日列表
        sorted_dates = sorted(df['trading_date'].unique().to_list())
        n_dates      = len(sorted_dates)

        # 2. 按 index // window_days 分配窗口ID，尾部不完整窗口标记为 -1
        date_to_wid = {}
        wid_to_last_date = {}
        for idx, d in enumerate(sorted_dates):
            wid = idx // self.window_days
            # 最后一个完整窗口的 wid = (n_dates - 1) // window_days
            # 如果该 wid 对应的交易日不足 window_days 个则丢弃
            date_to_wid[d] = wid

        # 找出完整窗口（窗口内交易日数 == window_days）
        from collections import Counter
        wid_counts = Counter(date_to_wid.values())
        complete_wids = {wid for wid, cnt in wid_counts.items()
                         if cnt == self.window_days}

        # 不完整窗口打标记 -1
        date_to_wid_clean = {
            d: (wid if wid in complete_wids else -1)
            for d, wid in date_to_wid.items()
        }

        # 3. 每个完整窗口取最后一个交易日作为时间坐标
        wid_to_last_date = {}
        for d, wid in date_to_wid_clean.items():
            if wid < 0:
                continue
            if wid not in wid_to_last_date or d > wid_to_last_date[wid]:
                wid_to_last_date[wid] = d

        # 4. 将窗口 ID 注入 df
        wid_series = df['trading_date'].map_elements(
            lambda d: date_to_wid_clean.get(d, -1),
            return_dtype=pl.Int32
        )
        df_with_wid = df.with_columns(wid_series.alias('__window_id__'))

        return df_with_wid, wid_to_last_date

    # ──────────────────────────────────────────────────────────
    # IC 分析
    # ──────────────────────────────────────────────────────────

    def run_ic(self, df: pl.DataFrame, factor: str) -> Dict:
        """
        对单个因子运行完整 IC 分析。

        方法：非重叠 window_days 窗口 IC
        ----------------------------------
        将样本期按交易日序号切分为长度 = window_days 的非重叠窗口，
        在每个窗口内用全量 bar 计算 Pearson IC（因子值与未来收益率的线性相关）。

        - window_days=20（默认）：每窗口约 1260 根 bar，避免日内单调结构
          导致的 IC 虚高，统计量更可靠
        - window_days=1：退化为逐日行为（向后兼容）
        - 不足 window_days 个交易日的尾部窗口直接丢弃

        统计指标（跨窗口序列）
        ----------------------
        - ic_mean      : 窗口级 IC 均值
        - ic_std       : 窗口级 IC 标准差
        - icir          : ic_mean / ic_std（不再年化，因窗口长度可变）
        - t_stat        : ic_mean × sqrt(N_windows) / ic_std
        - p_value       : 双侧 p 值（t 分布，df = N_windows - 1）
        - ic_win_rate   : IC > 0 的窗口占比

        Returns: dict with per-period stats + main-period summary
        """
        if factor not in df.columns:
            return {}

        # 分配窗口 ID（非重叠，基于交易日序号）
        df_w, wid_to_last_date = self._assign_window_ids(df)

        results = {}

        for period in self.ic_forward_periods:
            ror_col = f'ror_norm_{period}b' if self.use_ror_norm else f'ror_future_{period}b'
            if ror_col not in df_w.columns:
                continue

            # 去除 null，过滤掉不完整尾部窗口（window_id == -1）
            period_df = (
                df_w
                .select(['__window_id__', factor, ror_col])
                .filter(pl.col('__window_id__') >= 0)
                .drop_nulls()
            )
            if len(period_df) < 30:
                continue

            # 每窗口最少 bar 数 = window_days × 30
            min_obs = max(10, self.window_days * 30)

            window_records = []
            for (wid,), grp in period_df.group_by(['__window_id__']):
                x    = grp[factor].to_numpy().astype(float)
                y    = grp[ror_col].to_numpy().astype(float)
                mask = np.isfinite(x) & np.isfinite(y)
                if mask.sum() < min_obs:
                    continue
                # 窗口内截尾：仅处理收益率 y，不改变 mask 索引
                y_w  = self._winsorize_array(y)
                rho, _ = stats.pearsonr(x[mask], y_w[mask])
                window_records.append({
                    'wid':   wid,
                    'date':  wid_to_last_date.get(wid),
                    'ic':    rho,
                    'n_obs': int(mask.sum()),
                })

            if len(window_records) < 3:
                continue

            # 按窗口序号排序
            window_records.sort(key=lambda r: r['wid'])

            ic_series  = np.array([r['ic']   for r in window_records])
            dates      = [r['date'] for r in window_records]
            n_windows  = len(ic_series)

            ic_mean  = float(np.nanmean(ic_series))
            ic_std   = float(np.nanstd(ic_series, ddof=1))
            icir     = ic_mean / ic_std if ic_std > 1e-9 else 0.0
            t_stat   = ic_mean * np.sqrt(n_windows) / ic_std if ic_std > 1e-9 else 0.0
            p_value  = float(2 * (1 - stats.t.cdf(abs(t_stat), df=n_windows - 1)))
            win_rate = float(np.mean(ic_series > 0))

            results[period] = {
                'ic_mean':   ic_mean,
                'ic_std':    ic_std,
                'icir':      icir,
                't_stat':    t_stat,
                'p_value':   p_value,
                'win_rate':  win_rate,
                'n_windows': n_windows,
                'ic_series': ic_series,
                'dates':     dates,
            }

        if not results:
            return {}

        # 保存 IC 衰减 CSV（批量模式）
        decay_rows = []
        for period, stat in results.items():
            decay_rows.append({
                'forward_bars':    period,
                'forward_minutes': period * 5,
                'ic_mean':         round(stat['ic_mean'],  6),
                'ic_std':          round(stat['ic_std'],   6),
                'icir':            round(stat['icir'],     4),
                't_stat':          round(stat['t_stat'],   4),
                'p_value':         round(stat['p_value'],  6),
                'win_rate':        round(stat['win_rate'], 4),
                'n_windows':       stat['n_windows'],
                'window_days':     self.window_days,
                'ror_type':        'ror_norm' if self.use_ror_norm else 'ror_raw',
            })
        if self.output_dir is not None:
            pl.DataFrame(decay_rows).write_csv(
                self.ic_dir / f'ic_decay_{factor}.csv')

        # 绘图
        self._plot_ic(factor, results)

        # 返回主周期（label_bars）统计，若无则用第一个
        key = self.label_bars if self.label_bars in results else list(results.keys())[0]
        return {
            'factor': factor,
            **{f'period_{p}_icir':   v['icir']    for p, v in results.items()},
            **{f'period_{p}_ic':     v['ic_mean'] for p, v in results.items()},
            **{f'period_{p}_tstat':  v['t_stat']  for p, v in results.items()},
            **{f'period_{p}_pvalue': v['p_value'] for p, v in results.items()},
            'main_ic_mean':   results[key]['ic_mean'],
            'main_icir':      results[key]['icir'],
            'main_t_stat':    results[key]['t_stat'],
            'main_p_value':   results[key]['p_value'],
            'main_win_rate':  results[key]['win_rate'],
            'main_n_windows': results[key]['n_windows'],
        }

    def _plot_ic(self, factor: str, results: Dict):
        """绘制窗口级 IC 时序图（所有预测周期） + IC 衰减柱状图"""

        n_periods = len(results)
        if n_periods == 0:
            return

        fig, axes = plt.subplots(n_periods + 1, 1,
                                 figsize=(14, 3.5 * (n_periods + 1)))
        if n_periods == 1:
            plt.close(fig)
            fig, axes = plt.subplots(n_periods + 1, 1,
                                     figsize=(14, 3.5 * (n_periods + 1)))

        ror_label = 'ror_norm (波动率(ATR)校正)' if self.use_ror_norm else 'ror_raw (原始收益)'

        # ── 逐预测周期：窗口 IC 时序图 ────────────────────────────
        for i, (period, stat) in enumerate(sorted(results.items())):
            ax = axes[i] if hasattr(axes, '__len__') else axes

            dates_dt = []
            for d in stat['dates']:
                if isinstance(d, datetime.date):
                    dates_dt.append(d)
                else:
                    try:
                        dates_dt.append(datetime.date.fromisoformat(str(d)))
                    except Exception:
                        dates_dt.append(d)

            ic_arr = stat['ic_series']

            # 裸柱状图（颜色区分正负），不画滚动均值
            bar_width = max(1.0, self.window_days * 0.6)
            colors = ['#d62728' if v < 0 else '#1f77b4' for v in ic_arr]
            ax.bar(dates_dt, ic_arr, color=colors, alpha=0.55, width=bar_width)

            ax.axhline(0, color='black', lw=0.8, ls='--')

            p = stat['p_value']
            sig = '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else ''))

            ax.set_title(
                f'{self.symbol_name} | {factor} | 预测{period}棒({period * 5}分钟)\n'
                f'IC均值={stat["ic_mean"]:.4f}  '
                f'ICIR={stat["icir"]:.3f}  '
                f't={stat["t_stat"]:.2f}  '
                f'p={stat["p_value"]:.4f}{sig}  '
                f'方向正确率={stat["win_rate"]:.1%}  '
                f'N={stat["n_windows"]}个窗口 × {self.window_days}交易日\n'
                f'[{ror_label}，非重叠窗口，每窗口≥{self.window_days * 30}根bar]',
                fontsize=8)
            ax.set_ylabel('窗口 IC')
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

        # ── 最后一行：IC 衰减图 ──────────────────────────────────
        ax_decay = axes[-1] if hasattr(axes, '__len__') else axes
        periods_sorted = sorted(results.keys())
        icir_vals = [results[p]['icir']   for p in periods_sorted]
        t_vals    = [results[p]['t_stat'] for p in periods_sorted]
        x = np.arange(len(periods_sorted))

        bar_colors = ['#2ca02c' if v > 0 else '#d62728' for v in icir_vals]
        ax_decay.bar(x - 0.2, icir_vals, width=0.35,
                     color=bar_colors, alpha=0.7, label='ICIR（左轴）')

        ax_t = ax_decay.twinx()
        ax_t.bar(x + 0.2, t_vals, width=0.35,
                 color='#9467bd', alpha=0.6, label='t统计量（右轴）')
        ax_t.axhline( 2.0, color='#9467bd', lw=0.8, ls='--', alpha=0.6)
        ax_t.axhline(-2.0, color='#9467bd', lw=0.8, ls='--', alpha=0.6)

        ax_decay.set_xticks(x)
        ax_decay.set_xticklabels([f'{p}棒\n{p * 5}min' for p in periods_sorted])
        ax_decay.axhline(0, color='black', lw=0.8, ls='--')
        ax_decay.set_title(
            f'{self.symbol_name} | {factor} | IC 衰减分析（窗口={self.window_days}交易日）\n'
            f'紫色虚线 = t = ±2（5% 显著性参考）',
            fontsize=9)
        ax_decay.set_ylabel('ICIR')
        ax_t.set_ylabel('t 统计量')

        lines1, labs1 = ax_decay.get_legend_handles_labels()
        lines2, labs2 = ax_t.get_legend_handles_labels()
        ax_decay.legend(lines1 + lines2, labs1 + labs2, fontsize=7, loc='upper right')

        plt.tight_layout()
        if self.output_dir is not None:
            fig.savefig(self.ic_dir / 'figures' / f'ic_timeseries_{factor}.png',
                        bbox_inches='tight')
        if self.inline:
            plt.show()
        else:
            plt.close(fig)

    # ──────────────────────────────────────────────────────────
    # Rolling OLS Beta 分析
    # ──────────────────────────────────────────────────────────

    def run_ols(self, df: pl.DataFrame, factor: str) -> Dict:
        """
        对单个因子运行非重叠窗口 OLS Beta 分析。

        方法：非重叠 window_days 窗口 OLS，滚动 z-score 标准化 X
        ----------------------------------------------------------
        Step 1：对因子值做滚动 z-score（norm_window 根 bar，仅用历史数据）
                z[t] = (f[t] - rolling_mean(f, norm_window)[t])
                       / rolling_std(f, norm_window)[t]
                预热期（前 norm_window 根 bar）置为 null，后续 dropna 跳过。

        Step 2：将样本期切分为长度 = window_days 的非重叠交易日窗口，
                在每个窗口内对 ror_norm_Nb ~ z_factor 做 OLS，记录：
                - beta  : OLS 斜率（每1σ因子值变化对应的预期收益）
                - r2    : 拟合优度 R²
                - n_obs : 窗口内有效 bar 数

        Step 3：跨窗口汇总统计
                - beta_mean     : 窗口级 beta 均值
                - beta_std      : 窗口级 beta 标准差
                - t_stat        : beta_mean × sqrt(N_windows) / beta_std
                - p_value       : 双侧 p 值（t 分布，df = N_windows - 1）
                - beta_pos_rate : beta > 0 的窗口占比

        注意
        ----
        - window_days=20（默认）：每窗口约 1260 根 bar，R² 回落到合理量级
        - norm_window 建议 ≥ window_days × bars_per_day（默认200根bar）
        - 逐窗口 OLS 消除了跨窗口序列相关，无需 Newey-West 修正
        """
        if factor not in df.columns:
            return {}

        # ── Step 1：全量因子序列的滚动 z-score ───────────────────
        factor_series = df[factor]
        roll_mean = factor_series.rolling_mean(self.norm_window, min_samples=self.norm_window)
        roll_std  = factor_series.rolling_std (self.norm_window, min_samples=self.norm_window)
        z_series  = (factor_series - roll_mean) / roll_std.map_elements(
            lambda v: v if (v is not None and v > 1e-10) else None,
            return_dtype=pl.Float64
        )
        z_col   = f'__z_{factor}__'
        work_df = df.with_columns(z_series.alias(z_col))

        # ── 分配窗口 ID（与 run_ic 共用辅助函数）──────────────────
        work_df, wid_to_last_date = self._assign_window_ids(work_df)

        results = {}

        for period in self.ic_forward_periods:
            ror_col = (f'ror_norm_{period}b' if self.use_ror_norm
                       else f'ror_future_{period}b')
            if ror_col not in work_df.columns:
                continue

            period_df = (
                work_df
                .select(['__window_id__', z_col, ror_col])
                .filter(pl.col('__window_id__') >= 0)
                .drop_nulls()
            )
            if len(period_df) < 30:
                continue

            min_obs = max(10, self.window_days * 30)

            window_records = []
            for (wid,), grp in period_df.group_by(['__window_id__']):
                x_arr = grp[z_col].to_numpy().astype(float)
                y_arr = grp[ror_col].to_numpy().astype(float)
                mask  = np.isfinite(x_arr) & np.isfinite(y_arr)
                if mask.sum() < min_obs:
                    continue

                # 窗口内截尾：仅处理收益率 y_arr，不改变 mask 索引
                y_w  = self._winsorize_array(y_arr)
                X = sm.add_constant(x_arr[mask], prepend=True)
                try:
                    res  = sm.OLS(y_w[mask], X).fit()
                    beta = float(res.params[1])
                    r2   = float(res.rsquared)
                except Exception:
                    continue

                window_records.append({
                    'wid':   wid,
                    'date':  wid_to_last_date.get(wid),
                    'beta':  beta,
                    'r2':    r2,
                    'n_obs': int(mask.sum()),
                })

            if len(window_records) < 3:
                continue

            window_records.sort(key=lambda r: r['wid'])

            beta_arr  = np.array([r['beta'] for r in window_records])
            r2_arr    = np.array([r['r2']   for r in window_records])
            dates     = [r['date'] for r in window_records]
            n_windows = len(beta_arr)

            beta_mean     = float(np.nanmean(beta_arr))
            beta_std      = float(np.nanstd (beta_arr, ddof=1))
            r2_mean       = float(np.nanmean(r2_arr))
            t_stat        = (beta_mean * np.sqrt(n_windows) / beta_std
                             if beta_std > 1e-12 else 0.0)
            p_value       = float(2 * (1 - stats.t.cdf(abs(t_stat), df=n_windows - 1)))
            beta_pos_rate = float(np.mean(beta_arr > 0))

            results[period] = {
                'beta_mean':     beta_mean,
                'beta_std':      beta_std,
                'r2_mean':       r2_mean,
                't_stat':        t_stat,
                'p_value':       p_value,
                'beta_pos_rate': beta_pos_rate,
                'n_windows':     n_windows,
                'beta_series':   beta_arr,
                'r2_series':     r2_arr,
                'dates':         dates,
            }

        if not results:
            return {}

        # ── 保存 OLS 衰减 CSV（批量模式）────────────────────────────
        decay_rows = []
        for period, stat in results.items():
            decay_rows.append({
                'forward_bars':    period,
                'forward_minutes': period * 5,
                'beta_mean':       round(stat['beta_mean'],     6),
                'beta_std':        round(stat['beta_std'],      6),
                'r2_mean':         round(stat['r2_mean'],       6),
                't_stat':          round(stat['t_stat'],        4),
                'p_value':         round(stat['p_value'],       6),
                'beta_pos_rate':   round(stat['beta_pos_rate'], 4),
                'n_windows':       stat['n_windows'],
                'window_days':     self.window_days,
                'norm_window':     self.norm_window,
                'ror_type':        'ror_norm' if self.use_ror_norm else 'ror_raw',
            })
        if self.output_dir is not None:
            pl.DataFrame(decay_rows).write_csv(
                self.ols_dir / f'ols_decay_{factor}.csv')

        self._plot_ols(factor, results)

        key = self.label_bars if self.label_bars in results else list(results.keys())[0]
        return {
            'factor': factor,
            **{f'period_{p}_beta':         v['beta_mean']     for p, v in results.items()},
            **{f'period_{p}_r2':           v['r2_mean']       for p, v in results.items()},
            **{f'period_{p}_tstat':        v['t_stat']        for p, v in results.items()},
            **{f'period_{p}_pvalue':       v['p_value']       for p, v in results.items()},
            **{f'period_{p}_beta_posrate': v['beta_pos_rate'] for p, v in results.items()},
            'main_beta':         results[key]['beta_mean'],
            'main_r2':           results[key]['r2_mean'],
            'main_t_stat':       results[key]['t_stat'],
            'main_p_value':      results[key]['p_value'],
            'main_beta_posrate': results[key]['beta_pos_rate'],
            'main_n_windows':    results[key]['n_windows'],
        }

    def _plot_ols(self, factor: str, results: Dict):
        """
        绘制窗口级 OLS Beta 时序图（所有预测周期） + Beta 衰减柱状图。

        布局（n_periods + 1 行）：
        - 前 n_periods 行：各预测周期的窗口 beta 柱状图（颜色区分正负）
                          + 窗口 R² 折线（右轴，辅助观察拟合质量）
        - 最后一行：beta 均值衰减图（左轴柱 + 右轴 t 统计量）
        """
        n_periods = len(results)
        if n_periods == 0:
            return

        fig, axes = plt.subplots(n_periods + 1, 1,
                                 figsize=(14, 3.5 * (n_periods + 1)))
        if n_periods == 1:
            plt.close(fig)
            fig, axes = plt.subplots(n_periods + 1, 1,
                                     figsize=(14, 3.5 * (n_periods + 1)))

        ror_label = 'ror_norm (波动率(ATR)校正)' if self.use_ror_norm else 'ror_raw (原始收益)'

        # ── 逐预测周期：窗口 beta 时序图 ──────────────────────────
        for i, (period, stat) in enumerate(sorted(results.items())):
            ax = axes[i] if hasattr(axes, '__len__') else axes

            dates_dt = []
            for d in stat['dates']:
                if isinstance(d, datetime.date):
                    dates_dt.append(d)
                else:
                    try:
                        dates_dt.append(datetime.date.fromisoformat(str(d)))
                    except Exception:
                        dates_dt.append(d)

            beta_arr = stat['beta_series']
            r2_arr   = stat['r2_series']

            # 左轴：裸柱状图（颜色区分正负），不画滚动均值
            bar_width = max(1.0, self.window_days * 0.6)
            colors = ['#d62728' if v < 0 else '#1f77b4' for v in beta_arr]
            ax.bar(dates_dt, beta_arr, color=colors, alpha=0.55, width=bar_width)
            ax.axhline(0, color='black', lw=0.8, ls='--')
            ax.set_ylabel('窗口 Beta')

            # 右轴：窗口 R²（散点，辅助观察）
            ax_r2 = ax.twinx()
            ax_r2.scatter(dates_dt, r2_arr, color='#ff7f0e', s=12,
                          alpha=0.7, zorder=3)
            ax_r2.set_ylabel('R²', fontsize=8)
            ax_r2.set_ylim(bottom=0)

            p = stat['p_value']
            sig = '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else ''))

            ax.set_title(
                f'{self.symbol_name} | {factor} | 预测{period}棒({period * 5}分钟)\n'
                f'Beta均值={stat["beta_mean"]:.5f}  '
                f'R²均值={stat["r2_mean"]:.5f}  '
                f't={stat["t_stat"]:.2f}  '
                f'p={stat["p_value"]:.4f}{sig}  '
                f'Beta>0占比={stat["beta_pos_rate"]:.1%}  '
                f'N={stat["n_windows"]}个窗口 × {self.window_days}交易日\n'
                f'[{ror_label}，z-score(window={self.norm_window})，'
                f'非重叠OLS，每窗口≥{self.window_days * 30}根bar]',
                fontsize=8)

            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

        # ── 最后一行：Beta 衰减图 ──────────────────────────────────
        ax_decay = axes[-1] if hasattr(axes, '__len__') else axes
        periods_sorted = sorted(results.keys())
        beta_vals = [results[p]['beta_mean'] for p in periods_sorted]
        t_vals    = [results[p]['t_stat']    for p in periods_sorted]
        x = np.arange(len(periods_sorted))

        bar_colors = ['#2ca02c' if v > 0 else '#d62728' for v in beta_vals]
        ax_decay.bar(x - 0.2, beta_vals, width=0.35,
                     color=bar_colors, alpha=0.7, label='Beta均值（左轴）')

        ax_t = ax_decay.twinx()
        ax_t.bar(x + 0.2, t_vals, width=0.35,
                 color='#9467bd', alpha=0.6, label='t统计量（右轴）')
        ax_t.axhline( 2.0, color='#9467bd', lw=0.8, ls='--', alpha=0.6)
        ax_t.axhline(-2.0, color='#9467bd', lw=0.8, ls='--', alpha=0.6)

        ax_decay.set_xticks(x)
        ax_decay.set_xticklabels([f'{p}棒\n{p * 5}min' for p in periods_sorted])
        ax_decay.axhline(0, color='black', lw=0.8, ls='--')
        ax_decay.set_title(
            f'{self.symbol_name} | {factor} | OLS Beta 衰减分析（窗口={self.window_days}交易日）\n'
            f'紫色虚线 = t = ±2（5% 显著性参考）',
            fontsize=9)
        ax_decay.set_ylabel('Beta均值')
        ax_t.set_ylabel('t 统计量')

        lines1, labs1 = ax_decay.get_legend_handles_labels()
        lines2, labs2 = ax_t.get_legend_handles_labels()
        ax_decay.legend(lines1 + lines2, labs1 + labs2, fontsize=7, loc='upper right')

        plt.tight_layout()
        if self.output_dir is not None:
            fig.savefig(self.ols_dir / 'figures' / f'ols_timeseries_{factor}.png',
                        bbox_inches='tight')
        if self.inline:
            plt.show()
        else:
            plt.close(fig)

    # ──────────────────────────────────────────────────────────
    # 因子分层收益分析（Quantile Analysis）
    # ──────────────────────────────────────────────────────────

    def run_quantile_analysis(
        self,
        df: pl.DataFrame,
        factor: str,
        n_quantiles: int = 5,
        rolling_window: int = 500,
        min_samples: int = 20,
        session_template: Optional[str] = None,
        session_slots: Optional[List[Dict]] = None,
        use_ror_norm: bool = False,
        correct_autocorr: bool = True,
    ) -> Dict:
        """
        因子分层收益分析。

        维度一：全样本分层收益分布（单调性验证）
        -----------------------------------------
        使用 Polars rolling_quantile 预计算各层边界（向量化，无未来信息），
        将每根 bar 分配到 Q1~Qn 层，统计每层的均值、中位数、std、胜率。

        维度二：时段 × 分层 交叉热力图（时段效应验证）
        ------------------------------------------------
        从 datetime 提取 time，将每根 bar 映射到 session_slots/session_template
        定义的时段，交叉统计各时段 × 各分层的均值收益，绘制热力图。

        滚动分位数说明
        --------------
        分层边界基于过去 rolling_window 根 bar 的滚动分位数：
          Q=0.2, 0.4, 0.6, 0.8（对应 n_quantiles=5 的四条分界线）
        前 rolling_window 根 bar 为预热期，不参与分析（置 null 后 drop）。

        Parameters
        ----------
        df               : 含 factor 列、ror_norm_Nb / ror_future_Nb、datetime 列的 DataFrame
        factor           : 因子列名
        n_quantiles      : 分层数（默认5）
        rolling_window   : 滚动分位数回望窗口（默认500根bar）
        min_samples      : 滚动分位数最小样本数（默认20）
        session_template : 内置模板名（如 "night_0230"）或自定义 list[dict]；
                           优先级高于 session_slots。
                           内置模板：day_only / night_2300 / night_0100 /
                                     night_0230 / stock_index / bond
        session_slots    : 时段定义列表，每项 {"name": str, "start": "HH:MM", "end": "HH:MM"}
                           session_template 为 None 时生效；两者均为 None 则跳过时段分析
        use_ror_norm     : False（默认）使用原始收益率 ror_future_Nb，统计量和图表以基点(bp)
                           为单位展示；True 使用 ATR 归一化收益率 ror_norm_Nb，无量纲。
        correct_autocorr : True（默认）使用日度聚合 t 检验校正日内自相关：
                           先按 trading_date 聚合每日均值，再对日均值序列做单样本 t 检验，
                           有效自由度 = 交易日数（n_days），与 Alphalens 等主流库一致。
                           False 使用朴素 t 检验（所有 bar 直接参与，自相关未校正），
                           p 值偏低，显著性可能虚高。

        Returns
        -------
        dict 包含：
          quantile_stats   : list of dict（各层全样本统计，use_ror_norm=False 时单位为 bp）
          timeslot_stats   : list of dict（各层 × 各时段统计），空列表如果无时段配置
          n_bars_analyzed  : 分析用到的有效 bar 数
          use_bp           : bool，是否以基点为单位
          ls_spread_unit   : 'bp' 或 '（无量纲）'
        """
        if factor not in df.columns:
            return {}

        # ── session_template 优先于 session_slots ─────────────────
        if session_template is not None:
            if isinstance(session_template, list):
                # 用户直接传入自定义 slots 列表
                session_slots = session_template
            elif isinstance(session_template, str):
                resolved = SESSION_TEMPLATES.get(session_template)
                if resolved is None:
                    warnings.warn(
                        f"run_quantile_analysis: session_template="
                        f"'{session_template}' 不在内置模板中，忽略时段分析。\n"
                        f"可选内置模板：{list(SESSION_TEMPLATES.keys())}"
                    )
                    # session_slots 保持原值（None 或调用方已传入的值）
                else:
                    session_slots = resolved

        # 确定收益列（主周期）
        # use_ror_norm=False（默认）→ 原始收益率，以基点(bp)展示
        # use_ror_norm=True         → ATR归一化收益率，无量纲
        use_bp  = not use_ror_norm
        ror_col = (f'ror_norm_{self.label_bars}b'
                   if use_ror_norm
                   else f'ror_future_{self.label_bars}b')
        if ror_col not in df.columns:
            # 回退到第一个可用周期
            for p in self.ic_forward_periods:
                c = f'ror_norm_{p}b' if use_ror_norm else f'ror_future_{p}b'
                if c in df.columns:
                    ror_col = c
                    break
            else:
                return {}

        # ── 1. 计算滚动分位数边界 ──────────────────────────────
        quantile_probs = [i / n_quantiles for i in range(1, n_quantiles)]
        # e.g. n_quantiles=5 → [0.2, 0.4, 0.6, 0.8]

        factor_series = df[factor]
        boundaries = {}
        for q in quantile_probs:
            rq = factor_series.rolling_quantile(
                q,
                window_size=rolling_window,
                min_samples=min_samples,
            )
            boundaries[q] = rq

        # ── 2. 逐行分配分层标签 ───────────────────────────────
        # 构建辅助列：rolling_q0.2, rolling_q0.4, ...
        # trading_date 用于时段统计中按交易日聚合做自相关校正（correct_autocorr=True）
        select_cols = ['datetime', 'trading_date', factor, ror_col] \
            if 'trading_date' in df.columns else ['datetime', factor, ror_col]
        work_df = df.select(select_cols).clone()
        for q, series in boundaries.items():
            work_df = work_df.with_columns(
                series.alias(f'__rq_{int(q * 100):02d}__')
            )

        # Polars 表达式：逐层判断
        # Q1: factor <= rq20
        # Q2: factor <= rq40
        # ...
        # Qn: factor > rq(n-1)/n
        q_col_names = [f'__rq_{int(q * 100):02d}__' for q in quantile_probs]

        # 构建条件链：倒序遍历，使最外层条件为最小分位数边界（rq20 → Q1）
        # Polars when/then/otherwise 从最外层开始求值，倒序构建确保：
        #   最外层：factor <= rq20 → Q1（最先判断，约 20% bar）
        #   次外层：factor <= rq40 → Q2（rq20 < factor <= rq40，约 20% bar）
        #   ...
        #   兜底：factor > rq80 → Q5（约 20% bar）
        # 正序构建时最外层为 factor <= rq80 → Q4，导致 ~80% bar 落入 Q4，Q1~Q3 为空。
        expr = pl.lit(n_quantiles)   # 默认最高层（兜底）
        for i in range(len(quantile_probs) - 1, -1, -1):
            col   = q_col_names[i]
            layer = i + 1   # Q1=1, Q2=2, ...
            expr  = (
                pl.when(pl.col(factor) <= pl.col(col))
                .then(pl.lit(layer))
                .otherwise(expr)
            )
        work_df = work_df.with_columns(expr.alias('__quantile_layer__'))

        # 丢掉预热期（任何滚动分位数为 null 的行）
        null_filter = pl.all_horizontal(
            [pl.col(c).is_not_null() for c in q_col_names]
        )
        work_df = (
            work_df
            .filter(null_filter)
            .filter(pl.col(factor).is_not_null())
            .filter(pl.col(ror_col).is_not_null())
        )

        if len(work_df) < n_quantiles * 30:
            return {}

        n_bars_analyzed = len(work_df)

        # ── 3. 全样本分层统计 ─────────────────────────────────
        # use_bp=True 时将原始收益率乘以 10000 转为基点(bp)，方便阅读
        BP = 10000.0 if use_bp else 1.0

        quantile_stats = []
        ror_arr_by_layer = {}

        for layer in range(1, n_quantiles + 1):
            grp = work_df.filter(pl.col('__quantile_layer__') == layer)[ror_col]
            arr = grp.to_numpy().astype(float)
            arr = arr[np.isfinite(arr)]
            if len(arr) < 10:
                continue
            arr_disp = arr * BP   # 展示用数组（bp 或无量纲）
            ror_arr_by_layer[layer] = arr_disp
            quantile_stats.append({
                'layer':       layer,
                'label':       f'Q{layer}',
                'n_bars':      int(len(arr_disp)),
                'mean':        float(np.mean(arr_disp)),
                'median':      float(np.median(arr_disp)),
                'std':         float(np.std(arr_disp, ddof=1)),
                'win_rate':    float(np.mean(arr > 0)),   # 胜率基于原始值方向
                'skew':        float(stats.skew(arr_disp)),
                'kurt':        float(stats.kurtosis(arr_disp)),
                'q10':         float(np.percentile(arr_disp, 10)),
                'q25':         float(np.percentile(arr_disp, 25)),
                'q75':         float(np.percentile(arr_disp, 75)),
                'q90':         float(np.percentile(arr_disp, 90)),
            })

        if not quantile_stats:
            return {}

        # ── 4. 时段 × 分层 交叉统计 ──────────────────────────
        timeslot_stats = []
        if session_slots:
            # 提取时间（分钟数，便于跨日比较）
            # datetime 列格式：Polars Datetime
            # 显式 cast 为 Int32，避免 dt.hour()*60 在某些 Polars 版本中
            # 产生 Int8 中间类型，导致 21*60=1260 溢出为负值，使时段过滤全部失效
            time_minutes = (
                work_df['datetime'].dt.hour().cast(pl.Int32) * 60
                + work_df['datetime'].dt.minute().cast(pl.Int32)
            )
            work_df = work_df.with_columns(
                time_minutes.alias('__time_minutes__')
            )

            def _parse_hhmm(s: str) -> int:
                """将 "HH:MM" 转为分钟数"""
                h, m = s.split(':')
                return int(h) * 60 + int(m)

            for slot in session_slots:
                slot_name  = slot['name']
                start_min  = _parse_hhmm(slot['start'])
                end_min    = _parse_hhmm(slot['end'])

                # 处理跨日时段（如夜盘中段 22:00~02:30）
                if end_min < start_min:
                    # 跨日：time >= start OR time < end
                    slot_mask = (
                        (pl.col('__time_minutes__') >= start_min) |
                        (pl.col('__time_minutes__') < end_min)
                    )
                else:
                    slot_mask = (
                        (pl.col('__time_minutes__') >= start_min) &
                        (pl.col('__time_minutes__') < end_min)
                    )

                slot_df = work_df.filter(slot_mask)
                if len(slot_df) < n_quantiles * 5:
                    continue

                for layer in range(1, n_quantiles + 1):
                    grp = slot_df.filter(
                        pl.col('__quantile_layer__') == layer
                    )
                    arr = grp[ror_col].to_numpy().astype(float)
                    arr = arr[np.isfinite(arr)]
                    arr_disp = arr * BP   # bp 或无量纲
                    n_d      = len(arr_disp)
                    mean_val = float(np.mean(arr_disp))        if n_d > 0 else np.nan
                    std_val  = float(np.std(arr_disp, ddof=1)) if n_d > 1 else np.nan

                    # ── 显著性检验 ────────────────────────────────────
                    t_stat  = np.nan
                    p_value = np.nan
                    n_days  = 0

                    has_date = 'trading_date' in grp.columns

                    if correct_autocorr and has_date and n_d >= 2:
                        # 方法 A（默认）：日度聚合 t 检验
                        # 假设：同一交易日内的bar相互自相关，不同交易日之间独立。
                        # 先按 trading_date 聚合为日均值，再对日均值序列做单样本 t 检验。
                        # 有效自由度 = n_days - 1，与 Alphalens 等主流库一致。
                        daily = (
                            grp.filter(pl.col(ror_col).is_not_null())
                               .group_by('trading_date')
                               .agg(pl.col(ror_col).mean().alias('_dm'))
                               .drop_nulls()
                        )
                        daily_vals = daily['_dm'].to_numpy().astype(float) * BP
                        daily_vals = daily_vals[np.isfinite(daily_vals)]
                        n_days = len(daily_vals)
                        if n_days >= 20:   # 至少20个独立交易日才计算显著性
                            d_mean = float(np.mean(daily_vals))
                            d_std  = float(np.std(daily_vals, ddof=1))
                            if d_std > 1e-12:
                                t_stat  = d_mean / (d_std / np.sqrt(n_days))
                                p_value = float(
                                    2 * (1 - stats.t.cdf(abs(t_stat), df=n_days - 1))
                                )
                    elif not correct_autocorr and n_d >= 2:
                        # 方法 B（朴素）：直接对所有 bar 做单样本 t 检验，未校正自相关
                        if np.isfinite(std_val) and std_val > 1e-12:
                            t_stat  = mean_val / (std_val / np.sqrt(n_d))
                            p_value = float(
                                2 * (1 - stats.t.cdf(abs(t_stat), df=n_d - 1))
                            )

                    timeslot_stats.append({
                        'slot':     slot_name,
                        'layer':    layer,
                        'label':    f'Q{layer}',
                        'n_bars':   n_d,
                        'n_days':   n_days,
                        'mean':     mean_val,
                        'win_rate': float(np.mean(arr > 0)) if len(arr) > 0 else np.nan,
                        'std':      std_val,
                        't_stat':   round(float(t_stat),  4) if np.isfinite(t_stat)  else np.nan,
                        'p_value':  round(float(p_value), 6) if np.isfinite(p_value) else np.nan,
                    })

        # ── 5. 保存 CSV（批量模式）────────────────────────────
        if self.output_dir is not None:
            qa_dir = self.output_dir / 'quantile'
            (qa_dir / 'figures').mkdir(parents=True, exist_ok=True)

            pl.DataFrame(quantile_stats).write_csv(
                qa_dir / f'quantile_stats_{factor}.csv'
            )
            if timeslot_stats:
                pl.DataFrame(timeslot_stats).write_csv(
                    qa_dir / f'quantile_timeslot_{factor}.csv'
                )

        # ── 6. 绘图 ───────────────────────────────────────────
        self._plot_quantile_bar(factor, quantile_stats, ror_col, n_bars_analyzed,
                                use_bp=use_bp)
        self._plot_quantile_boxplot(factor, ror_arr_by_layer, ror_col,
                                    use_bp=use_bp)
        if timeslot_stats and session_slots:
            self._plot_quantile_timeslot_heatmap(
                factor, timeslot_stats, n_quantiles, session_slots,
                use_bp=use_bp,
                ror_col=ror_col,
                rolling_window=rolling_window,
                correct_autocorr=correct_autocorr)

        # ── 7. 返回摘要统计 ──────────────────────────────────
        # 多空价差：最高层 - 最低层（已转换为展示单位：bp 或无量纲）
        top_layer    = quantile_stats[-1] if quantile_stats else {}
        bottom_layer = quantile_stats[0]  if quantile_stats else {}
        ls_spread    = top_layer.get('mean', np.nan) - bottom_layer.get('mean', np.nan)

        # 单调性检验（相邻层均值的 Spearman 相关）
        layer_means = [s['mean'] for s in quantile_stats]
        layer_ids   = list(range(1, len(layer_means) + 1))
        monotone_r  = float(stats.spearmanr(layer_ids, layer_means).correlation) \
            if len(layer_means) >= 3 else np.nan

        return {
            'factor':           factor,
            'n_quantiles':      n_quantiles,
            'rolling_window':   rolling_window,
            'n_bars_analyzed':  n_bars_analyzed,
            'ror_col':          ror_col,
            'use_bp':           use_bp,
            'ls_spread':        round(ls_spread, 4),
            'ls_spread_unit':   'bp' if use_bp else '（无量纲）',
            'monotone_r':       round(monotone_r, 4),
            'quantile_stats':   quantile_stats,
            'timeslot_stats':   timeslot_stats,
        }

    def _plot_quantile_bar(self, factor: str,
                           quantile_stats: List[Dict],
                           ror_col: str,
                           n_bars_analyzed: int,
                           use_bp: bool = False):
        """
        分层收益柱状图：
        - 左轴：各层均值收益柱状图（误差棒 = 1 std / sqrt(n)）
        - 右轴：各层胜率折线
        - 标注：各层样本量 n
        use_bp=True 时，左轴单位为基点(bp)，数值已在统计阶段完成转换。
        """
        if not quantile_stats:
            return

        labels   = [s['label']    for s in quantile_stats]
        means    = np.array([s['mean']     for s in quantile_stats])
        stds     = np.array([s['std']      for s in quantile_stats])
        ns       = np.array([s['n_bars']   for s in quantile_stats], dtype=float)
        win_rates= np.array([s['win_rate'] for s in quantile_stats])

        se = stds / np.sqrt(np.maximum(ns, 1))   # 标准误

        fig, ax1 = plt.subplots(figsize=(8, 5))

        x      = np.arange(len(labels))
        colors = ['#d62728' if v < 0 else '#1f77b4' for v in means]
        y_label   = '均值收益（bp）' if use_bp else '均值收益'
        leg_label = '均值收益（bp，左轴）' if use_bp else '均值收益（左轴）'
        ax1.bar(x, means, color=colors, alpha=0.7, yerr=se,
                capsize=4, error_kw={'lw': 1.2},
                label=leg_label)
        ax1.axhline(0, color='black', lw=0.8, ls='--')
        ax1.set_xticks(x)
        ax1.set_xticklabels(labels)
        ax1.set_ylabel(y_label)

        # 标注样本量
        for xi, (n, m) in enumerate(zip(ns, means)):
            ax1.text(xi, m + se[xi] * 1.1 + abs(means).max() * 0.02,
                     f'n={int(n):,}',
                     ha='center', va='bottom', fontsize=7)

        # 右轴：胜率
        ax2 = ax1.twinx()
        ax2.plot(x, win_rates, color='#ff7f0e', marker='o',
                 lw=1.8, ms=6, label='胜率（右轴）')
        ax2.axhline(0.5, color='#ff7f0e', lw=0.8, ls='--', alpha=0.5)
        ax2.set_ylabel('胜率')
        ax2.set_ylim(0, 1)

        # 多空价差
        ls_spread = means[-1] - means[0]
        spread_str = f'{ls_spread:.2f} bp' if use_bp else f'{ls_spread:.5f}'

        ax1.set_title(
            f'{self.symbol_name} | {factor} | 因子分层收益分布\n'
            f'[{ror_col}，滚动分位数分层，n_bars={n_bars_analyzed:,}，'
            f'多空价差(Q{len(labels)}-Q1)={spread_str}]',
            fontsize=9
        )

        lines1, labs1 = ax1.get_legend_handles_labels()
        lines2, labs2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labs1 + labs2, fontsize=8, loc='upper left')

        plt.tight_layout()
        if self.output_dir is not None:
            qa_dir = self.output_dir / 'quantile' / 'figures'
            fig.savefig(qa_dir / f'quantile_bar_{factor}.png', bbox_inches='tight')
        if self.inline:
            plt.show()
        else:
            plt.close(fig)

    def _plot_quantile_boxplot(self, factor: str,
                               ror_arr_by_layer: Dict[int, np.ndarray],
                               ror_col: str,
                               use_bp: bool = False):
        """
        分层收益箱线图：直观展示收益分布的形态、偏态和厚尾情况。
        use_bp=True 时，纵轴单位为基点(bp)，数组已在统计阶段完成转换。
        """
        if not ror_arr_by_layer:
            return

        layers   = sorted(ror_arr_by_layer.keys())
        data     = [ror_arr_by_layer[l] for l in layers]
        labels   = [f'Q{l}' for l in layers]

        fig, ax = plt.subplots(figsize=(8, 5))

        bp = ax.boxplot(data, labels=labels, patch_artist=True,
                        showfliers=False,   # 隐藏离群点，避免遮挡
                        medianprops={'color': 'black', 'lw': 2},
                        whiskerprops={'lw': 1.2},
                        capprops={'lw': 1.2})

        # 按层着色（从冷到暖）
        colors_bp = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(layers)))
        for patch, color in zip(bp['boxes'], colors_bp):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        # 均值标记
        means = [np.mean(ror_arr_by_layer[l]) for l in layers]
        ax.scatter(range(1, len(layers) + 1), means,
                   color='navy', zorder=5, s=40, marker='D',
                   label='均值')

        ax.axhline(0, color='black', lw=0.8, ls='--')
        ax.set_ylabel('收益率（bp）' if use_bp else '收益率')
        ax.set_title(
            f'{self.symbol_name} | {factor} | 因子分层收益箱线图\n'
            f'[{ror_col}，菱形=均值，箱体=IQR，须=1.5×IQR，隐藏极端离群点]',
            fontsize=9
        )
        ax.legend(fontsize=8)

        plt.tight_layout()
        if self.output_dir is not None:
            qa_dir = self.output_dir / 'quantile' / 'figures'
            fig.savefig(qa_dir / f'quantile_boxplot_{factor}.png', bbox_inches='tight')
        if self.inline:
            plt.show()
        else:
            plt.close(fig)

    def _plot_quantile_timeslot_heatmap(self, factor: str,
                                        timeslot_stats: List[Dict],
                                        n_quantiles: int,
                                        session_slots: List[Dict],
                                        use_bp: bool = False,
                                        ror_col: str = '',
                                        rolling_window: int = 500,
                                        correct_autocorr: bool = True):
        """
        时段 × 分层 热力图：
        - 行：时段（按 session_slots 顺序）
        - 列：分层（Q1 ~ Qn）
        - 颜色：均值收益（红蓝色板，0 为白）
        - 格子内标注：均值收益（上）+ 样本量 n（下）
        - 样本量 < 30 的格子打灰色遮罩（数据不足）
        use_bp=True 时，颜色和标注均以基点(bp)为单位，数值已在统计阶段完成转换。
        """
        if not timeslot_stats:
            return

        slot_names = [s['name'] for s in session_slots]
        layer_labels = [f'Q{l}' for l in range(1, n_quantiles + 1)]

        # 构建矩阵
        mean_mat   = np.full((len(slot_names), n_quantiles), np.nan)
        n_mat      = np.zeros((len(slot_names), n_quantiles), dtype=int)
        p_mat      = np.full((len(slot_names), n_quantiles), np.nan)
        win_mat    = np.full((len(slot_names), n_quantiles), np.nan)
        n_days_mat = np.zeros((len(slot_names), n_quantiles), dtype=int)

        slot_idx = {name: i for i, name in enumerate(slot_names)}
        for row in timeslot_stats:
            si = slot_idx.get(row['slot'])
            li = row['layer'] - 1   # 0-indexed
            if si is None or li < 0 or li >= n_quantiles:
                continue
            mean_mat[si, li]   = row['mean']
            n_mat[si, li]      = row['n_bars']
            p_mat[si, li]      = row.get('p_value', np.nan)
            win_mat[si, li]    = row.get('win_rate', np.nan)
            n_days_mat[si, li] = row.get('n_days', 0)

        fig, ax = plt.subplots(figsize=(max(6, n_quantiles * 1.4),
                                        max(4.5, len(slot_names) * 1.1 + 1.5)))

        # 色板范围：对称于 0
        vmax = np.nanmax(np.abs(mean_mat))
        vmax = vmax if np.isfinite(vmax) and vmax > 0 else 1e-5

        im = ax.imshow(mean_mat, cmap='RdBu_r', vmin=-vmax, vmax=vmax,
                       aspect='auto')
        cbar_label = '均值收益（bp）' if use_bp else '均值收益'
        plt.colorbar(im, ax=ax, label=cbar_label, shrink=0.6)

        # 灰色遮罩（样本量不足）
        min_n = 30
        for i in range(len(slot_names)):
            for j in range(n_quantiles):
                if n_mat[i, j] < min_n:
                    ax.add_patch(plt.Rectangle(
                        (j - 0.5, i - 0.5), 1, 1,
                        fill=True, facecolor='lightgray', alpha=0.6, zorder=2
                    ))

        # 显著性星号辅助函数
        def _pval_stars(p: float) -> str:
            if not np.isfinite(p): return ''
            if p < 0.001: return '***'
            if p < 0.01:  return '**'
            if p < 0.05:  return '*'
            return ''

        # 格子内标注：4 行布局
        #   行1（上）：均值
        #   行2：显著性星号（不显著时跳过渲染）
        #   行3：胜率
        #   行4（下）：样本量（correct_autocorr=True 时优先显示 n_days，否则 n_bars）
        for i in range(len(slot_names)):
            for j in range(n_quantiles):
                mv = mean_mat[i, j]
                nv = n_mat[i, j]
                pv = p_mat[i, j]
                wr = win_mat[i, j]
                nd = n_days_mat[i, j]
                if np.isfinite(mv):
                    txt_color = 'white' if abs(mv) > vmax * 0.6 else 'black'
                    gray      = txt_color if abs(mv) > vmax * 0.6 else 'gray'
                    mv_str    = f'{mv:.1f}' if use_bp else f'{mv:.4f}'
                    stars     = _pval_stars(pv)

                    # 行1：均值
                    ax.text(j, i - 0.30, mv_str,
                            ha='center', va='center', fontsize=7,
                            color=txt_color, zorder=3)
                    # 行2：显著性星号（无星号时不渲染，避免空行占位）
                    if stars:
                        ax.text(j, i - 0.08, stars,
                                ha='center', va='center', fontsize=8,
                                fontweight='bold', color=txt_color, zorder=3)
                    # 行3：胜率
                    if np.isfinite(wr):
                        ax.text(j, i + 0.13, f'胜率{wr*100:.1f}%',
                                ha='center', va='center', fontsize=6,
                                color=gray, zorder=3)
                    # 行4：样本量
                    # correct_autocorr=True 时显示 n_days（有效独立观测数）
                    # correct_autocorr=False 时显示 n_bars（原始 bar 数）
                    if correct_autocorr and nd > 0:
                        n_str = f'n={nd}日'
                    else:
                        n_str = f'n={nv:,}bar'
                    ax.text(j, i + 0.33, n_str,
                            ha='center', va='center', fontsize=6,
                            color=gray, zorder=3)
                else:
                    ax.text(j, i, 'N/A', ha='center', va='center',
                            fontsize=7, color='gray', zorder=3)

        ax.set_xticks(range(n_quantiles))
        ax.set_xticklabels(layer_labels)
        ax.set_yticks(range(len(slot_names)))
        ax.set_yticklabels(slot_names)
        ax.set_xlabel('因子分层')
        ax.set_ylabel('交易时段')
        unit_str     = 'bp' if use_bp else '无量纲'
        label_period = f'{self.label_bars}棒/{self.label_bars * 5}分钟'
        if correct_autocorr:
            sig_note = ('显著性：日度聚合t检验（n=交易日数，≥20日才计算）'
                        '，已校正日内自相关')
        else:
            sig_note = '显著性：朴素t检验（n=bar数），未校正日内自相关，显著性可能虚高'
        ax.set_title(
            f'{self.symbol_name} | {factor} | 时段 × 因子分层 均值收益热力图\n'
            f'[分箱：因子值滚动{rolling_window}棒分位数，Q1最低~Q{n_quantiles}最高  |  '
            f'收益率：{ror_col}（{label_period}，{unit_str}）  |  '
            f'灰色格子：样本量 < {min_n}]\n'
            f'[* p<0.05  ** p<0.01  *** p<0.001  |  {sig_note}]',
            fontsize=8
        )

        plt.tight_layout()
        if self.output_dir is not None:
            qa_dir = self.output_dir / 'quantile' / 'figures'
            fig.savefig(qa_dir / f'quantile_timeslot_heatmap_{factor}.png',
                        bbox_inches='tight')
        if self.inline:
            plt.show()
        else:
            plt.close(fig)

    # ──────────────────────────────────────────────────────────
    # Horizon 分析：因子预测能力随持仓期的演化
    # ──────────────────────────────────────────────────────────

    def run_horizon_analysis(
        self,
        df: pl.DataFrame,
        factor: str,
        horizons: List[int],
        n_quantiles: int = 5,
        rolling_window: int = 500,
        min_samples: int = 20,
        correct_autocorr: bool = True,
        use_daily_mean: bool = False,
    ) -> Dict:
        """
        因子预测能力随持仓 Horizon 的演化分析。

        将因子值按滚动分位数分为 Q1~Qn 层，对每个持仓期 h（bar数），
        统计各层的均值收益率（bp）、胜率和显著性。

        显著性检验
        ----------
        correct_autocorr=True（默认）：日度聚合 t 检验（与 quantile_only 一致）
          - 每日日均值作为独立观测，n_eff = 交易日数
          - 多空价差显著性：Q5 vs Q1 日均值序列做双样本独立 t 检验
        correct_autocorr=False：朴素 t 检验，未校正日内自相关

        Parameters
        ----------
        df           : 含 factor 列、ror_future_Nb 系列列、trading_date 的 DataFrame
        factor       : 因子列名
        horizons     : 持仓期列表（bar数），如 [1,2,...,12]
        n_quantiles  : 分层数，默认5
        rolling_window : 滚动分位数回望窗口，默认500
        min_samples  : 滚动分位数最小样本数，默认20
        correct_autocorr : 是否做日度聚合自相关校正，默认True
        use_daily_mean   : False（默认）ax1 折线使用 bar 等权均值；
                           True 使用日等权均值（mean(daily_means)），与
                           cumreturn_only 的计算口径完全一致。
                           多空价差（ax2）自动跟随此开关。

        Returns
        -------
        dict 包含：
          horizons        : 有效的 horizon 列表
          minutes         : 对应分钟数
          n_quantiles     : 分层数
          use_daily_mean  : 当前使用的均值口径
          layer_stats     : Dict[horizon → Dict[layer → stats]]
          ls_spread       : Dict[horizon → float]（Q_n - Q_1 均值价差，bp）
          ls_p_value      : Dict[horizon → float]（双样本 t 检验 p 值）
          ls_t_stat       : Dict[horizon → float]
          peak_ls_horizon_bars  : L/S 价差绝对值最大的 horizon
          peak_ls_horizon_min   : 对应分钟数
        """
        if factor not in df.columns:
            return {}

        # ── 1. 因子分层（与 run_quantile_analysis 完全一致）────────
        quantile_probs = [i / n_quantiles for i in range(1, n_quantiles)]
        factor_series  = df[factor]
        boundaries     = {}
        for q in quantile_probs:
            boundaries[q] = factor_series.rolling_quantile(
                q, window_size=rolling_window, min_samples=min_samples)

        select_cols = ['trading_date', factor] \
            if 'trading_date' in df.columns else [factor]
        # 加入所有可用的 ror_future_hb 列
        ror_cols_available = [f'ror_future_{h}b' for h in horizons
                              if f'ror_future_{h}b' in df.columns]
        valid_horizons = [h for h in horizons
                          if f'ror_future_{h}b' in df.columns]
        if not valid_horizons:
            return {}

        work_df = df.select(list(set(select_cols + ror_cols_available + [factor]))).clone()

        # 分层赋值（倒序构建 when/then/otherwise）
        q_col_names = [f'__rq_{int(q * 100):02d}__' for q in quantile_probs]
        for q, col in zip(quantile_probs, q_col_names):
            work_df = work_df.with_columns(boundaries[q].alias(col))

        expr = pl.lit(n_quantiles)
        for i in range(len(quantile_probs) - 1, -1, -1):
            col   = q_col_names[i]
            layer = i + 1
            expr  = (pl.when(pl.col(factor) <= pl.col(col))
                     .then(pl.lit(layer)).otherwise(expr))
        work_df = work_df.with_columns(expr.alias('__layer__'))

        # 丢弃预热期
        null_filter = pl.all_horizontal(
            [pl.col(c).is_not_null() for c in q_col_names])
        work_df = (work_df
                   .filter(null_filter)
                   .filter(pl.col(factor).is_not_null()))
        work_df = work_df.drop(q_col_names)

        if len(work_df) < n_quantiles * 30:
            return {}

        BP = 10000.0
        has_date = 'trading_date' in work_df.columns
        MIN_DAYS = 20

        layer_stats: Dict = {}
        ls_spread:   Dict = {}
        ls_p_value:  Dict = {}
        ls_t_stat:   Dict = {}

        for h in valid_horizons:
            ror_col = f'ror_future_{h}b'
            h_df = work_df.filter(pl.col(ror_col).is_not_null())
            layer_stats[h] = {}

            for layer in range(1, n_quantiles + 1):
                grp = h_df.filter(pl.col('__layer__') == layer)
                arr = grp[ror_col].to_numpy().astype(float)
                arr = arr[np.isfinite(arr)]
                arr_bp = arr * BP
                n_d    = len(arr_bp)

                std_val  = float(np.std(arr_bp, ddof=1)) if n_d > 1 else np.nan
                win_rate = float(np.mean(arr > 0))       if n_d > 0 else np.nan

                t_stat     = np.nan
                p_value    = np.nan
                n_days     = 0
                daily_vals = None

                # 日均值计算：use_daily_mean=True 或 correct_autocorr=True 时均需要
                if has_date and n_d >= 2 and (use_daily_mean or correct_autocorr):
                    _daily = (grp.filter(pl.col(ror_col).is_not_null())
                                 .group_by('trading_date')
                                 .agg(pl.col(ror_col).mean().alias('_dm'))
                                 .drop_nulls())
                    dv = _daily['_dm'].to_numpy().astype(float) * BP
                    daily_vals = dv[np.isfinite(dv)]
                    n_days = len(daily_vals)

                # mean_val：ax1 折线的 y 值，由 use_daily_mean 开关控制
                if use_daily_mean and daily_vals is not None and n_days > 0:
                    mean_val = float(np.mean(daily_vals))   # 日等权均值
                else:
                    mean_val = float(np.mean(arr_bp)) if n_d > 0 else np.nan  # bar等权均值

                # 显著性检验（沿用 correct_autocorr 逻辑，与 mean_val 口径无关）
                if correct_autocorr and daily_vals is not None and n_days >= MIN_DAYS:
                    d_mean = float(np.mean(daily_vals))
                    d_std  = float(np.std(daily_vals, ddof=1))
                    if d_std > 1e-12:
                        t_stat  = d_mean / (d_std / np.sqrt(n_days))
                        p_value = float(2 * (1 - stats.t.cdf(
                            abs(t_stat), df=n_days - 1)))
                elif not correct_autocorr and n_d >= 2 and \
                        np.isfinite(std_val) and std_val > 1e-12:
                    t_stat  = mean_val / (std_val / np.sqrt(n_d))
                    p_value = float(2 * (1 - stats.t.cdf(abs(t_stat), df=n_d - 1)))

                layer_stats[h][layer] = {
                    'mean':     mean_val,
                    'std':      std_val,
                    'win_rate': win_rate,
                    't_stat':   round(float(t_stat),  4) if np.isfinite(t_stat)  else np.nan,
                    'p_value':  round(float(p_value), 6) if np.isfinite(p_value) else np.nan,
                    'n_bars':   n_d,
                    'n_days':   n_days,
                }

            # ── 多空价差 ────────────────────────────────────────────────────
            # 价差值 = ax1 两条折线的垂直差（与图表视觉完全一致）
            # 直接从已计算好的 layer_stats 中读取，不再重新聚合
            q5_mean = layer_stats[h].get(n_quantiles, {}).get('mean', np.nan)
            q1_mean = layer_stats[h].get(1, {}).get('mean', np.nan)
            spread_mean = q5_mean - q1_mean \
                if (np.isfinite(q5_mean) and np.isfinite(q1_mean)) else np.nan

            spread_t = np.nan
            spread_p = np.nan

            top_grp = h_df.filter(pl.col('__layer__') == n_quantiles)
            bot_grp = h_df.filter(pl.col('__layer__') == 1)

            if correct_autocorr and has_date:
                # 方案 B：日度差值序列的单样本 t 检验（已校正日内自相关）
                # 每天：当日 Q_n 均值 - 当日 Q_1 均值 → 日度价差序列
                # 对日度价差序列做单样本 t 检验：E[日度价差] = 0？
                # 通过 inner join on trading_date 保证 Q_n 和 Q_1 在同一天配对
                def _day_means_df(grp_df, rc):
                    return (grp_df.filter(pl.col(rc).is_not_null())
                                  .group_by('trading_date')
                                  .agg(pl.col(rc).mean().alias('_dm'))
                                  .drop_nulls())

                top_day = _day_means_df(top_grp, ror_col)
                bot_day = _day_means_df(bot_grp, ror_col)

                # inner join：只保留 Q_n 和 Q_1 当天都有数据的日期
                paired = top_day.join(bot_day, on='trading_date',
                                      how='inner', suffix='_q1')
                if len(paired) >= MIN_DAYS:
                    q5_vals      = paired['_dm'].to_numpy().astype(float)    * BP
                    q1_vals      = paired['_dm_q1'].to_numpy().astype(float) * BP
                    daily_spread = (q5_vals - q1_vals)
                    daily_spread = daily_spread[np.isfinite(daily_spread)]
                    n_paired     = len(daily_spread)
                    if n_paired >= MIN_DAYS:
                        d_std = float(np.std(daily_spread, ddof=1))
                        if d_std > 1e-12:
                            spread_t = float(np.mean(daily_spread)) / \
                                       (d_std / np.sqrt(n_paired))
                            spread_p = float(2 * (1 - stats.t.cdf(
                                abs(spread_t), df=n_paired - 1)))
            else:
                # correct_autocorr=False：朴素 Welch 双样本 t 检验（bar 级别）
                top_arr = top_grp[ror_col].to_numpy().astype(float)
                bot_arr = bot_grp[ror_col].to_numpy().astype(float)
                top_arr = top_arr[np.isfinite(top_arr)] * BP
                bot_arr = bot_arr[np.isfinite(bot_arr)] * BP
                if len(top_arr) >= 2 and len(bot_arr) >= 2:
                    t_res    = stats.ttest_ind(top_arr, bot_arr, equal_var=False)
                    spread_t = float(t_res.statistic)
                    spread_p = float(t_res.pvalue)

            ls_spread[h]  = round(float(spread_mean), 4) if np.isfinite(spread_mean) else np.nan
            ls_t_stat[h]  = round(spread_t, 4) if np.isfinite(spread_t)  else np.nan
            ls_p_value[h] = round(spread_p, 6) if np.isfinite(spread_p)  else np.nan

        # ── 绘图 ───────────────────────────────────────────────
        self._plot_horizon_analysis(
            factor, valid_horizons, layer_stats, ls_spread, ls_p_value,
            n_quantiles, correct_autocorr, use_daily_mean)

        # ── 峰值 L/S horizon ───────────────────────────────────
        abs_spreads = {h: abs(v) for h, v in ls_spread.items() if np.isfinite(v)}
        peak_h = max(abs_spreads, key=abs_spreads.get) if abs_spreads else None

        return {
            'horizons':              valid_horizons,
            'minutes':               [h * 5 for h in valid_horizons],
            'n_quantiles':           n_quantiles,
            'correct_autocorr':      correct_autocorr,
            'use_daily_mean':        use_daily_mean,
            'layer_stats':           layer_stats,
            'ls_spread':             ls_spread,
            'ls_t_stat':             ls_t_stat,
            'ls_p_value':            ls_p_value,
            'peak_ls_horizon_bars':  peak_h,
            'peak_ls_horizon_min':   peak_h * 5 if peak_h else None,
        }

    def _plot_horizon_analysis(
        self,
        factor: str,
        horizons: List[int],
        layer_stats: Dict,
        ls_spread: Dict,
        ls_p_value: Dict,
        n_quantiles: int,
        correct_autocorr: bool,
        use_daily_mean: bool = False,
    ):
        """
        绘制 Horizon 分析图（2 子图）：

        子图1：各分层均值收益 by Horizon
          - Q1~Qn 折线，带 ±1 SE 误差带
          - 每个点标注显著性星号

        子图2：多空价差 (Qn-Q1) by Horizon
          - 柱状图，正蓝负红
          - 每个柱顶标注双样本 t 检验显著性
        """
        if not horizons or not layer_stats:
            return

        minutes = [h * 5 for h in horizons]
        x       = np.arange(len(horizons))

        def _stars(p):
            if not np.isfinite(p): return ''
            if p < 0.001: return '***'
            if p < 0.01:  return '**'
            if p < 0.05:  return '*'
            return ''

        # ── 颜色方案：Q1(红) ~ Qn(绿)，符合金融惯例（高分层=绿=期望优）──
        cmap   = plt.cm.RdYlGn
        colors = [cmap(i / max(n_quantiles - 1, 1)) for i in range(n_quantiles)]

        fig, (ax1, ax2) = plt.subplots(
            2, 1, figsize=(max(10, len(horizons) * 0.9), 9),
            gridspec_kw={'height_ratios': [2.5, 1]}, sharex=True)

        # ── 子图1：各层均值收益折线 ──────────────────────────────
        for layer in range(1, n_quantiles + 1):
            means = []
            ses   = []
            for h in horizons:
                s = layer_stats.get(h, {}).get(layer, {})
                means.append(s.get('mean', np.nan))
                n = s.get('n_days', 0) if correct_autocorr else s.get('n_bars', 0)
                std = s.get('std', np.nan)
                ses.append(std / np.sqrt(max(n, 1)) if np.isfinite(std) and n > 0
                           else np.nan)

            means = np.array(means)
            ses   = np.array(ses)
            color = colors[layer - 1]
            label = f'Q{layer}'

            ax1.plot(x, means, color=color, lw=2.0, marker='o', ms=5, label=label)
            # 误差带 ±1 SE
            valid = np.isfinite(means) & np.isfinite(ses)
            if valid.sum() > 1:
                ax1.fill_between(x[valid],
                                 (means - ses)[valid],
                                 (means + ses)[valid],
                                 color=color, alpha=0.12)

            # 显著性星号（标在最高误差棒上方）
            for xi, h in enumerate(horizons):
                s  = layer_stats.get(h, {}).get(layer, {})
                pv = s.get('p_value', np.nan)
                st = _stars(pv)
                if st and np.isfinite(means[xi]):
                    offset = (ses[xi] if np.isfinite(ses[xi]) else 0) + \
                             abs(np.nanmax(np.abs(means))) * 0.04
                    ax1.text(xi, means[xi] + offset, st,
                             ha='center', va='bottom', fontsize=7,
                             color=color, fontweight='bold')

        ax1.axhline(0, color='black', lw=0.8, ls='--')
        ax1.set_ylabel('均值收益（bp）')
        mean_note = ('日等权均值（与 cumreturn_only 一致）' if use_daily_mean
                     else 'bar等权均值')
        sig_note  = ('日度聚合t检验，已校正日内自相关' if correct_autocorr
                     else '朴素t检验，未校正自相关')
        ax1.set_title(
            f'{self.symbol_name} | {factor} | 各分层均值收益 by Horizon\n'
            f'[Q1最低~Q{n_quantiles}最高  |  均值口径：{mean_note}  |  '
            f'显著性：* p<0.05  ** p<0.01  *** p<0.001  |  {sig_note}]',
            fontsize=9)
        ax1.legend(fontsize=8, loc='upper right', ncol=n_quantiles)

        # ── 子图2：多空价差柱状图 ─────────────────────────────────
        spreads = [ls_spread.get(h, np.nan) for h in horizons]
        pvals   = [ls_p_value.get(h, np.nan) for h in horizons]
        bar_colors = ['#1f77b4' if (np.isfinite(v) and v >= 0) else '#d62728'
                      for v in spreads]
        bars = ax2.bar(x, spreads, color=bar_colors, alpha=0.75, width=0.6)
        ax2.axhline(0, color='black', lw=0.8, ls='--')

        for xi, (sv, pv) in enumerate(zip(spreads, pvals)):
            st = _stars(pv)
            if st and np.isfinite(sv):
                yoff = sv + abs(sv) * 0.08 + abs(max(
                    (abs(s) for s in spreads if np.isfinite(s)), default=1)) * 0.04
                ax2.text(xi, yoff, st,
                         ha='center', va='bottom', fontsize=8, fontweight='bold',
                         color='black')

        ax2.set_ylabel('多空价差\n(bp)')
        ax2.set_xticks(x)
        ax2.set_xticklabels([f'{h}棒\n{h*5}min' for h in horizons], fontsize=8)
        ax2.set_title(
            f'多空价差 (Q{n_quantiles}-Q1) by Horizon  |  '
            f'显著性：{"日度差值单样本t检验（已校正自相关）" if correct_autocorr else "朴素Welch双样本t检验（未校正）"}',
            fontsize=9)

        plt.tight_layout()
        if self.output_dir is not None:
            fig.savefig(
                self.ic_dir / 'figures' / f'horizon_analysis_{factor}.png',
                bbox_inches='tight')
        if self.inline:
            plt.show()
        else:
            plt.close(fig)

    # ──────────────────────────────────────────────────────────
    # 因子预测力纯积分（因子分层收益累加）- Overlapping 模式
    # ──────────────────────────────────────────────────────────
    def run_cumreturn(
        self,
        df: pl.DataFrame,
        factor: str,
        horizon: int,
        n_quantiles: int = 5,
        rolling_window: int = 500,
        min_samples: int = 20,
        use_daily_mean: bool = True,
    ) -> Dict:
        """
        Qn / Q1 / Qn-Q1 分层累计收益率曲线（纯预测力积分）。
        核心理念
        --------
        本方法不包含任何人造的交易规则（如资金管理、扣除手续费等），
        通过直接累加多期预测收益（ror_future_hb），计算因子在不同历史时期的毛预测贡献。
        这反映的是因子的“预测能力积分”，不等同于实盘可落地的资金曲线。
        Parameters
        ----------
        df             : 含 factor 列、trading_date、ror_future_hb 的 DataFrame
        factor         : 因子列名
        horizon        : 持仓期（bar 数），决定使用 ror_future_{horizon}b
        n_quantiles    : 分层数，默认5
        rolling_window : 滚动分位数回望窗口，默认500
        min_samples    : 滚动分位数最小样本数，默认20
        use_daily_mean : True（默认）日等权，inner join 配对日度聚合；
                         False bar等权，所有 bar 直接按时间顺序累加。
        Returns
        -------
        dict 包含累加曲线及年度统计结果。
        """
        ror_col = f'ror_future_{horizon}b'
        if factor not in df.columns or ror_col not in df.columns:
            return {}
        # ── 1. 因子分层 ────────
        quantile_probs = [i / n_quantiles for i in range(1, n_quantiles)]
        factor_series  = df[factor]
        select_cols = (['trading_date', factor, ror_col]
                       if 'trading_date' in df.columns
                       else [factor, ror_col])
        work_df = df.select(select_cols).clone()
        for q in quantile_probs:
            rq = factor_series.rolling_quantile(
                q, window_size=rolling_window, min_samples=min_samples)
            work_df = work_df.with_columns(rq.alias(f'__rq_{int(q*100):02d}__'))
        q_col_names = [f'__rq_{int(q*100):02d}__' for q in quantile_probs]
        expr = pl.lit(n_quantiles)
        for i in range(len(quantile_probs) - 1, -1, -1):
            expr = (pl.when(pl.col(factor) <= pl.col(q_col_names[i]))
                    .then(pl.lit(i + 1)).otherwise(expr))
        work_df = work_df.with_columns(expr.alias('__layer__'))
        null_filter = pl.all_horizontal([pl.col(c).is_not_null() for c in q_col_names])
        work_df = (work_df
                   .filter(null_filter)
                   .filter(pl.col(factor).is_not_null())
                   .filter(pl.col(ror_col).is_not_null())
                   .drop(q_col_names))
        BP = 10000.0
        if use_daily_mean:
            # ── 2a. 日等权聚合 ─────────────
            if 'trading_date' not in work_df.columns:
                return {}
            def _daily_layer_mean(layer_id: int) -> pl.DataFrame:
                return (work_df
                        .filter(pl.col('__layer__') == layer_id)
                        .group_by('trading_date')
                        .agg(pl.col(ror_col).mean().alias('ret'))
                        .sort('trading_date'))
            q5_daily = _daily_layer_mean(n_quantiles)
            q1_daily = _daily_layer_mean(1)
            paired = q5_daily.join(q1_daily, on='trading_date',
                                   how='inner', suffix='_q1')
            paired = paired.sort('trading_date')
            if len(paired) < 5:
                return {}
            dates = paired['trading_date'].to_list()
            r_q5  = paired['ret'].to_numpy().astype(float)    * BP
            r_q1  = paired['ret_q1'].to_numpy().astype(float) * BP
            n_obs = len(dates)
        else:
            # ── 2b. bar 等权累加 ────
            if 'trading_date' not in work_df.columns:
                q5_arr = (work_df.filter(pl.col('__layer__') == n_quantiles)
                                 [ror_col].to_numpy().astype(float)) * BP
                q1_arr = (work_df.filter(pl.col('__layer__') == 1)
                                 [ror_col].to_numpy().astype(float)) * BP
                n_min = min(len(q5_arr), len(q1_arr))
                r_q5  = q5_arr[:n_min]
                r_q1  = q1_arr[:n_min]
                dates = list(range(n_min))
            else:
                def _daily_layer_mean_bar(layer_id: int) -> pl.DataFrame:
                    return (work_df
                            .filter(pl.col('__layer__') == layer_id)
                            .group_by('trading_date')
                            .agg(pl.col(ror_col).sum().alias('ret'))
                            .sort('trading_date'))
                q5_d = _daily_layer_mean_bar(n_quantiles)
                q1_d = _daily_layer_mean_bar(1)
                combined = q5_d.join(q1_d, on='trading_date',
                                     how='outer', coalesce=True,
                                     suffix='_q1')
                combined = combined.sort('trading_date').fill_null(0.0)
                if len(combined) < 5:
                    return {}
                dates = combined['trading_date'].to_list()
                r_q5  = combined['ret'].to_numpy().astype(float)    * BP
                r_q1  = combined['ret_q1'].to_numpy().astype(float) * BP
            n_obs = len(dates)
        # ── 3. 预测力单利累加 ─────────────────────────────────────
        r_ls    = r_q5 - r_q1
        r_q1_sh = -r_q1
        cumret_q5   = np.cumsum(r_q5)
        cumret_q1sh = np.cumsum(r_q1_sh)
        cumret_ls   = np.cumsum(r_ls)
        # ── 4. 胜率计算 ───────────────────────────────────────────
        q5_arr_all = work_df.filter(pl.col('__layer__') == n_quantiles
                                    )[ror_col].to_numpy().astype(float)
        q1_arr_all = work_df.filter(pl.col('__layer__') == 1
                                    )[ror_col].to_numpy().astype(float)
        q5_fin = q5_arr_all[np.isfinite(q5_arr_all)]
        q1_fin = q1_arr_all[np.isfinite(q1_arr_all)]
        win_rate_q5 = float(np.mean(q5_fin > 0)) if len(q5_fin) > 0 else np.nan
        win_rate_q1 = float(np.mean(q1_fin > 0)) if len(q1_fin) > 0 else np.nan
        # ── 5. 逐年统计 ───────────────────────────────────────────
        annual_stats: Dict = {}
        years = sorted(set(
            (str(d)[:4] if isinstance(d, str) else str(d.year)
             if hasattr(d, 'year') else str(d))
            for d in dates
        ))
        for yr in years:
            mask = np.array([
                (str(d)[:4] if isinstance(d, str) else str(d.year)
                 if hasattr(d, 'year') else str(d)) == yr
                for d in dates
            ])
            annual_stats[int(yr)] = {
                'q5':       round(float(r_q5[mask].sum()),    2),
                'q1_short': round(float(r_q1_sh[mask].sum()), 2),
                'ls':       round(float(r_ls[mask].sum()),    2),
            }
        # ── 6. 绘图 ───────────────────────────────────────────────
        self._plot_cumreturn(
            factor, dates, cumret_q5, cumret_q1sh, cumret_ls,
            annual_stats, horizon, n_quantiles,
            use_daily_mean=use_daily_mean,
            win_rate_q5=win_rate_q5,
            win_rate_q1=win_rate_q1)
        return {
            'horizon':          horizon,
            'n_quantiles':      n_quantiles,
            'use_daily_mean':   use_daily_mean,
            'trading_dates':    dates,
            'cumret_q5':        cumret_q5.tolist(),
            'cumret_q1_short':  cumret_q1sh.tolist(),
            'cumret_ls':        cumret_ls.tolist(),
            'daily_ret_q5':     r_q5.tolist(),
            'daily_ret_ls':     r_ls.tolist(),
            'annual_stats':     annual_stats,
            'win_rate_q5':      round(win_rate_q5, 4),
            'win_rate_q1':      round(win_rate_q1, 4),
            'n_days' if use_daily_mean else 'n_bars': n_obs,
        }
    def _plot_cumreturn(
        self,
        factor: str,
        dates: list,
        cumret_q5:    np.ndarray,
        cumret_q1sh:  np.ndarray,
        cumret_ls:    np.ndarray,
        annual_stats: Dict,
        horizon: int,
        n_quantiles: int,
        use_daily_mean: bool = True,
        win_rate_q5: float = np.nan,
        win_rate_q1: float = np.nan,
    ):
        """
        因子预测积分累计图（2 子图）：
        子图1：Qn（多头）、Q1（空头，取负）、Qn-Q1（多空）毛累计曲线
          - 标题标注胜率与总贡献 bp
          - 年份边界垂直虚线
        子图2：逐年收益贡献柱状图（Qn / Q1_short / Qn-Q1 并排）
        """
        if not dates:
            return
        # ── 日期轴转换 ────────────────────────────────────────────
        try:
            dt_dates = [datetime.date.fromisoformat(str(d)) for d in dates]
        except Exception:
            dt_dates = list(range(len(dates)))
        years = sorted(annual_stats.keys())
        year_boundaries = []
        for yr in years[1:]:
            for i, d in enumerate(dt_dates):
                if isinstance(d, datetime.date) and d.year == yr:
                    year_boundaries.append((i, yr))
                    break
        # 调整为 2 个子图，高度比例 2.5 : 1
        fig, (ax1, ax2) = plt.subplots(
            2, 1, figsize=(14, 9),
            gridspec_kw={'height_ratios': [2.5, 1]})
        # ── 子图1：无摩擦累计预测积分曲线 ────────────────────────────
        ax1.plot(dt_dates, cumret_q5,  color='#1f77b4', lw=1.6,
                 label=f'Q{n_quantiles}（多头组合）')
        ax1.plot(dt_dates, cumret_q1sh, color='#d62728', lw=1.6,
                 label='Q1（空头组合，取负）')
        ax1.plot(dt_dates, cumret_ls,  color='#2ca02c', lw=2.0,
                 label=f'Q{n_quantiles}-Q1（多空预测积分）')
        ax1.axhline(0, color='black', lw=0.8, ls='--', alpha=0.6)
        for xi, yr in year_boundaries:
            if 0 <= xi < len(dt_dates):
                ax1.axvline(dt_dates[xi], color='gray', lw=0.8, ls=':', alpha=0.7)
                ax1.text(dt_dates[xi], min(cumret_ls) * 1.02 if cumret_ls.min() < 0
                         else 0, str(yr), fontsize=7, color='gray',
                         ha='left', va='bottom')
        wr_q5_str = f'{win_rate_q5:.1%}' if np.isfinite(win_rate_q5) else 'N/A'
        wr_q1_str = f'{win_rate_q1:.1%}' if np.isfinite(win_rate_q1) else 'N/A'
        mode_str  = '日等权' if use_daily_mean else 'bar等权（重叠累加）'
        
        ax1.set_ylabel('因子累计预测贡献（bp）')
        ax1.set_title(
            f'{self.symbol_name} | {factor} | 因子绝对预测力积分（{mode_str}）\n'
            f'[horizon={horizon}棒/{horizon*5}分钟  |  '
            f'总积分={cumret_ls[-1]:.1f}bp  |  '
            f'Q{n_quantiles}胜率={wr_q5_str}  Q1胜率={wr_q1_str}]',
            fontsize=9)
        ax1.legend(fontsize=8, loc='upper left')
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
        # ── 子图2：逐年预测积分贡献柱状图 ─────────────────────────────────
        yr_list = sorted(annual_stats.keys())
        n_yr    = len(yr_list)
        x_yr    = np.arange(n_yr)
        w       = 0.25
        q5_vals  = [annual_stats[yr]['q5']       for yr in yr_list]
        q1s_vals = [annual_stats[yr]['q1_short']  for yr in yr_list]
        ls_vals  = [annual_stats[yr]['ls']        for yr in yr_list]
        def _bar_colors(vals):
            return ['#1f77b4' if v >= 0 else '#d62728' for v in vals]
        ax2.bar(x_yr - w, q5_vals,  width=w, color=_bar_colors(q5_vals),
                alpha=0.75, label=f'Q{n_quantiles}（多头贡献）')
        ax2.bar(x_yr,     q1s_vals, width=w, color=_bar_colors(q1s_vals),
                alpha=0.75, label='Q1（空头贡献）')
        ax2.bar(x_yr + w, ls_vals,  width=w, color=_bar_colors(ls_vals),
                alpha=0.75, label='多空总贡献')
        
        ax2.axhline(0, color='black', lw=0.8, ls='--')
        ax2.set_xticks(x_yr)
        ax2.set_xticklabels([str(yr) for yr in yr_list], fontsize=9)
        ax2.set_ylabel('年度预测积分（bp）')
        ax2.set_title('因子预测力跨期稳定性 (Annual Contribution)', fontsize=9)
        ax2.legend(fontsize=7, loc='upper right', ncol=3)
        plt.tight_layout()
        if self.output_dir is not None:
            fig.savefig(
                self.ic_dir / 'figures' / f'factor_payout_{factor}.png',
                bbox_inches='tight')
        if self.inline:
            plt.show()
        else:
            plt.close(fig)

    # ──────────────────────────────────────────────────────────
    # 因子诊断：自相关、分布、平稳性
    # ──────────────────────────────────────────────────────────

    def run_factor_properties(
        self,
        df: pl.DataFrame,
        factor: str,
        max_lags: int = 40,
    ) -> Dict:
        """
        对因子值本身（不依赖收益率）进行三类统计检验。

        方法概述
        --------
        不依赖任何收益率列，仅使用因子列自身，独立于 IC/OLS/分层分析运行。

        三类检验
        --------
        **① 自相关结构**
          - 按 window_days（默认20个交易日，约1260根bar/窗口）的非重叠窗口分箱，
            与 IC/OLS 分析保持完全一致的窗口粒度。
          - 在每个窗口内对因子序列手动计算滞后 1~max_lags 的 Pearson 自相关；
            跨窗口取均值（窗口均值 ACF）和标准差（窗口间 σ），反映 ACF 的 regime 变化。
          - 同时计算全量 ACF（不分窗口，statsmodels fft）作为图中对比标注。
          - 由窗口均值 ACF(1) 估计 AR(1) 半衰期：-log(2)/log(|acf1|)，单位为 bar。
          - Ljung-Box 检验（滞后10，基于全量序列）。
          - ESS/N 估计：1 / (1 + 2 * sum(acf_win_mean[0:10]))。

          窗口大小的精度对比（以 max_lags=40 为例）：
            每日 (~63 bar)   → lag=40 仅 23 对，SE ≈ 0.22（不可靠）
            window_days=20   → lag=40 约 1220 对，SE ≈ 0.029（可靠）

        **② 分布形态**
          - 基础描述统计：mean / std / skew / kurt（excess）
          - Jarque-Bera 正态性检验
          - 分位数（1%/5%/25%/50%/75%/95%/99%）
          - Null 率

        **③ 平稳性（日均 ADF + KPSS）**
          - 保持每日粒度：对每个交易日的日内因子序列（~63 bar）分别做
            ADF（autolag='AIC'）和 KPSS（regression='c'），跨日取平均统计量和
            平均 p 值。日内序列 60 bar 对 ADF 的 autolag 选阶足够。
          - 综合判断逻辑：
              ADF p_mean < 0.05 → 拒绝单位根（偏平稳）
              KPSS p_mean < 0.05 → 拒绝平稳假设（偏非平稳）
              两者均支持平稳 → "✅ 平稳"
              两者均支持非平稳 → "❌ 非平稳"
              结论冲突 → "⚠️ 不确定"
          - 同时输出逐日均值/标准差时序及逐年稳定性统计。

        Parameters
        ----------
        df        : 含 factor 列和 trading_date 列的 DataFrame
        factor    : 因子列名
        max_lags  : ACF 最大滞后阶数，默认 40（= 200 分钟）

        Returns
        -------
        dict，包含自相关、分布、平稳性的全量统计结果
        """
        from statsmodels.tsa.stattools import acf as sm_acf, adfuller, kpss
        from statsmodels.stats.diagnostic import acorr_ljungbox
        from scipy import stats as sp_stats

        if factor not in df.columns:
            return {}

        has_date = 'trading_date' in df.columns

        # ── ① 全量因子序列（drop null） ──────────────────────────
        factor_series = df[factor].drop_nulls().to_numpy().astype(float)
        factor_series = factor_series[np.isfinite(factor_series)]
        n_valid = len(factor_series)
        null_rate = float(df[factor].is_null().sum() + df[factor].is_nan().sum() if
                          df[factor].dtype in (pl.Float64, pl.Float32) else
                          df[factor].is_null().sum()) / max(len(df), 1)

        if n_valid < max_lags + 10:
            return {}

        # ── ② 分布统计 ──────────────────────────────────────────
        mean_val  = float(np.mean(factor_series))
        std_val   = float(np.std(factor_series, ddof=1))
        skew_val  = float(sp_stats.skew(factor_series))
        kurt_val  = float(sp_stats.kurtosis(factor_series))   # excess kurtosis
        jb_stat, jb_p = sp_stats.jarque_bera(factor_series)
        pcts = np.percentile(factor_series, [1, 5, 25, 50, 75, 95, 99])
        percentiles = {
            'p1': float(pcts[0]), 'p5': float(pcts[1]),
            'p25': float(pcts[2]), 'p50': float(pcts[3]),
            'p75': float(pcts[4]), 'p95': float(pcts[5]),
            'p99': float(pcts[6]),
        }

        # ── ③ 全量 ACF + Ljung-Box ────────────────────────────
        acf_full_arr, confint_full = sm_acf(
            factor_series, nlags=max_lags, fft=True, alpha=0.05)
        acf_full = [float(v) for v in acf_full_arr[1:]]   # lag 1..max_lags
        # 全量 95% 置信区间（对称，取正半侧宽度）
        acf_full_ci95 = float(1.96 / np.sqrt(n_valid))

        lb_result  = acorr_ljungbox(factor_series, lags=[10], return_df=True)
        lb_stat    = float(lb_result['lb_stat'].iloc[0])
        lb_p       = float(lb_result['lb_pvalue'].iloc[0])

        # ── ④ 窗口 ACF（非重叠 window_days 窗口，与 IC/OLS 粒度一致）──────
        # 每窗口约 window_days × 63 ≈ 1260 bar，lag=40 时仍有约 1220 对有效样本，
        # Pearson SE ≈ 0.029（vs 每日粒度的 0.22），精度提升约 7 倍。
        lags = list(range(1, max_lags + 1))
        window_acf_matrix = []   # shape: (n_windows, max_lags)
        window_last_dates  = []  # 每个窗口的最后交易日（用作时间坐标）

        # 同时按日收集均值/标准差，供平稳性图使用
        daily_means  = []
        daily_stds   = []
        daily_dates  = []

        if has_date:
            # 复用 _assign_window_ids 将每根 bar 打上窗口 ID
            df_w, wid_to_last_date = self._assign_window_ids(df)

            # 逐窗口计算 ACF
            complete_wids = sorted(
                wid for wid in wid_to_last_date.keys()
            )
            # 每窗口最少 bar 数：至少为 max_lags 的 10 倍，确保最高阶 ACF 可靠
            min_obs_win = max(max_lags * 10, self.window_days * 30)

            for wid in complete_wids:
                win_arr = (
                    df_w.filter(pl.col('__window_id__') == wid)[factor]
                        .drop_nulls()
                        .to_numpy().astype(float)
                )
                win_arr = win_arr[np.isfinite(win_arr)]
                if len(win_arr) < min_obs_win:
                    continue
                win_acf = []
                for lag in lags:
                    x = win_arr[:-lag]
                    y = win_arr[lag:]
                    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
                        win_acf.append(np.nan)
                    else:
                        win_acf.append(float(np.corrcoef(x, y)[0, 1]))
                window_acf_matrix.append(win_acf)
                window_last_dates.append(wid_to_last_date[wid])

            # 按日收集均值/标准差（平稳性图用，保持日粒度）
            sorted_dates = sorted(df['trading_date'].unique().to_list())
            for d in sorted_dates:
                day_arr = (df.filter(pl.col('trading_date') == d)[factor]
                             .drop_nulls().to_numpy().astype(float))
                day_arr = day_arr[np.isfinite(day_arr)]
                if len(day_arr) < 5:
                    continue
                daily_means.append(float(np.mean(day_arr)))
                daily_stds.append(float(np.std(day_arr, ddof=1)))
                daily_dates.append(d)

        n_windows_acf = len(window_acf_matrix)

        if n_windows_acf >= 3:
            mat = np.array(window_acf_matrix)   # (n_windows, max_lags)
            acf_win_mean = [float(np.nanmean(mat[:, i])) for i in range(max_lags)]
            acf_win_std  = [float(np.nanstd (mat[:, i], ddof=1)) for i in range(max_lags)]
        else:
            # 回退到全量 ACF（窗口数不足时）
            acf_win_mean = acf_full[:]
            acf_win_std  = [0.0] * max_lags

        acf1_win_mean = float(acf_win_mean[0]) if acf_win_mean else float('nan')
        acf1_full_val = float(acf_full[0])      if acf_full      else float('nan')

        # AR(1) 半衰期（基于窗口均值 ACF(1)）
        phi = acf1_win_mean
        if np.isfinite(phi) and abs(phi) > 1e-6 and abs(phi) < 1.0:
            half_life_bars = float(-np.log(2) / np.log(abs(phi)))
        else:
            half_life_bars = float('nan')
        half_life_minutes = half_life_bars * 5 if np.isfinite(half_life_bars) else float('nan')

        # ESS/N（基于窗口均值 ACF 的前 10 个滞后）
        k_ess = min(10, len(acf_win_mean))
        acf_sum = sum(acf_win_mean[:k_ess])
        ess_ratio = float(1.0 / (1.0 + 2.0 * acf_sum)) if (1.0 + 2.0 * acf_sum) > 0 else float('nan')

        # ── ⑤ 平稳性（日均 ADF + KPSS，保持日粒度） ─────────────────
        adf_stats, adf_ps   = [], []
        kpss_stats, kpss_ps = [], []
        n_days_stat = 0

        if has_date and daily_dates:
            for d in daily_dates:
                day_arr = (df.filter(pl.col('trading_date') == d)[factor]
                             .drop_nulls().to_numpy().astype(float))
                day_arr = day_arr[np.isfinite(day_arr)]
                if len(day_arr) < 10:
                    continue
                try:
                    adf_r = adfuller(day_arr, autolag='AIC')
                    adf_stats.append(float(adf_r[0]))
                    adf_ps.append(float(adf_r[1]))
                except Exception:
                    pass
                try:
                    kpss_r = kpss(day_arr, regression='c', nlags='auto')
                    kpss_stats.append(float(kpss_r[0]))
                    kpss_ps.append(float(kpss_r[1]))
                except Exception:
                    pass
            n_days_stat = len(adf_stats)

        adf_stat_mean  = float(np.mean(adf_stats))  if adf_stats  else float('nan')
        adf_p_mean     = float(np.mean(adf_ps))     if adf_ps     else float('nan')
        kpss_stat_mean = float(np.mean(kpss_stats)) if kpss_stats else float('nan')
        kpss_p_mean    = float(np.mean(kpss_ps))    if kpss_ps    else float('nan')

        # 综合平稳性结论
        if np.isfinite(adf_p_mean) and np.isfinite(kpss_p_mean):
            adf_stationary  = adf_p_mean  < 0.05
            kpss_stationary = kpss_p_mean >= 0.05
            if adf_stationary and kpss_stationary:
                stationarity = '✅ 平稳'
            elif not adf_stationary and not kpss_stationary:
                stationarity = '❌ 非平稳'
            else:
                stationarity = '⚠️ 不确定（ADF 与 KPSS 结论不一致）'
        else:
            stationarity = '样本不足，跳过'

        # ── ⑥ 跨年稳定性（基于日均值/标准差） ──────────────────────
        annual_stats = []
        if daily_dates:
            year_data: Dict = {}
            for d, mu, sigma in zip(daily_dates, daily_means, daily_stds):
                yr = d.year if hasattr(d, 'year') else int(str(d)[:4])
                year_data.setdefault(yr, {'means': [], 'stds': [], 'n': 0})
                year_data[yr]['means'].append(mu)
                year_data[yr]['stds'].append(sigma)
                year_data[yr]['n'] += 1
            for yr in sorted(year_data):
                annual_stats.append({
                    'year':   yr,
                    'mean':   float(np.mean(year_data[yr]['means'])),
                    'std':    float(np.mean(year_data[yr]['stds'])),
                    'n_days': year_data[yr]['n'],
                })

        # ── ⑦ 汇总结果 ──────────────────────────────────────────
        result = {
            'factor':               factor,
            # 自相关（窗口粒度）
            'acf_lags':             lags,
            'acf_win_mean':         acf_win_mean,    # 窗口均值 ACF
            'acf_win_std':          acf_win_std,     # 窗口间标准差
            'acf_full':             acf_full,        # 全量 ACF（对比）
            'acf_full_ci95':        acf_full_ci95,
            'acf1_win_mean':        round(acf1_win_mean, 6),
            'acf1_full':            round(acf1_full_val, 6),
            'half_life_bars':       round(half_life_bars,    2) if np.isfinite(half_life_bars)    else float('nan'),
            'half_life_minutes':    round(half_life_minutes, 1) if np.isfinite(half_life_minutes) else float('nan'),
            'ess_ratio':            round(ess_ratio, 4) if np.isfinite(ess_ratio) else float('nan'),
            'lb_stat':              round(lb_stat, 4),
            'lb_p':                 round(lb_p,    6),
            'n_windows_acf':        n_windows_acf,
            'window_days':          self.window_days,
            # 分布
            'n_valid':              n_valid,
            'null_rate':            round(null_rate, 4),
            'mean':                 round(mean_val, 6),
            'std':                  round(std_val,  6),
            'skew':                 round(skew_val, 4),
            'kurt':                 round(kurt_val, 4),
            'jb_stat':              round(float(jb_stat), 4),
            'jb_p':                 round(float(jb_p),    6),
            'percentiles':          percentiles,
            # 平稳性（日粒度）
            'adf_stat_mean':        round(adf_stat_mean,  4) if np.isfinite(adf_stat_mean)  else float('nan'),
            'adf_p_mean':           round(adf_p_mean,     4) if np.isfinite(adf_p_mean)     else float('nan'),
            'kpss_stat_mean':       round(kpss_stat_mean, 4) if np.isfinite(kpss_stat_mean) else float('nan'),
            'kpss_p_mean':          round(kpss_p_mean,    4) if np.isfinite(kpss_p_mean)    else float('nan'),
            'stationarity':         stationarity,
            'n_days_stationarity':  n_days_stat,
            # 跨年
            'annual_stats':         annual_stats,
            # 私有绘图字段（前缀 _ 的字段不写入 CSV、不出现在最终返回值中）
            '_window_last_dates':   window_last_dates,
            '_daily_dates':         daily_dates,
            '_daily_means':         daily_means,
            '_daily_stds':          daily_stds,
            '_factor_series':       factor_series,
        }

        # ── ⑧ 保存 CSV（批量模式，去掉绘图用的私有字段） ──────────
        if self.output_dir is not None:
            prop_dir = self.output_dir / 'properties'
            prop_dir.mkdir(parents=True, exist_ok=True)
            csv_row = {k: v for k, v in result.items()
                       if not k.startswith('_') and
                       not isinstance(v, (list, dict))}
            pl.DataFrame([csv_row]).write_csv(
                prop_dir / f'properties_{factor}.csv')

        # ── ⑨ 绘图 ───────────────────────────────────────────────
        self._plot_factor_properties(factor, result)

        # 移除私有绘图字段后返回
        return {k: v for k, v in result.items() if not k.startswith('_')}

    def _plot_factor_properties(self, factor: str, result: Dict):
        """
        绘制因子诊断图表（2 张图，共 5 个子图）。

        图 1（自相关 + 分布）：3 行 × 1 列
          子图 0：窗口均值 ACF 柱状图（±1σ 窗口间误差带 + 全量95%置信线）
          子图 1：因子值分布直方图 + 正态拟合曲线
          子图 2：Q-Q 图（与标准正态比较）

        图 2（平稳性 + 跨年稳定性）：2 行 × 1 列
          子图 0：逐日均值 & 标准差时序双轴折线图（日粒度）
          子图 1：逐年均值（左轴柱）+ 逐年标准差（右轴折线）
        """
        lags             = result['acf_lags']
        acf_win_mean     = np.array(result['acf_win_mean'])
        acf_win_std      = np.array(result['acf_win_std'])
        acf_full         = np.array(result['acf_full'])
        ci95             = result['acf_full_ci95']
        factor_series    = result['_factor_series']
        window_last_dates = result['_window_last_dates']
        daily_dates      = result['_daily_dates']
        daily_means      = result['_daily_means']
        daily_stds       = result['_daily_stds']
        annual_stats     = result['annual_stats']
        n_windows        = result['n_windows_acf']
        window_days      = result['window_days']

        # ── 图 1：自相关 + 分布 ───────────────────────────────────
        fig1, axes1 = plt.subplots(3, 1, figsize=(12, 11))
        fig1.suptitle(
            f'{self.symbol_name} | {factor} | 因子诊断（自相关 + 分布）',
            fontsize=10)

        # 子图 0：窗口均值 ACF
        ax = axes1[0]
        x  = np.arange(1, len(lags) + 1)
        colors_acf = ['#1f77b4' if v >= 0 else '#d62728' for v in acf_win_mean]
        ax.bar(x, acf_win_mean, color=colors_acf, alpha=0.65,
               label='窗口均值 ACF', zorder=3)
        ax.errorbar(x, acf_win_mean, yerr=acf_win_std,
                    fmt='none', color='gray', alpha=0.5, capsize=2, lw=0.8,
                    label=f'±1σ（窗口间，N={n_windows}）')
        ax.axhline( ci95, color='#2ca02c', lw=1.0, ls='--', alpha=0.8,
                    label=f'全量95%置信线 ±{ci95:.3f}')
        ax.axhline(-ci95, color='#2ca02c', lw=1.0, ls='--', alpha=0.8)
        ax.axhline(0, color='black', lw=0.6)

        # 标注文字
        acf1w = result['acf1_win_mean']
        acf1f = result['acf1_full']
        hl    = result['half_life_bars']
        hlm   = result['half_life_minutes']
        ess   = result['ess_ratio']
        lb_p  = result['lb_p']
        hl_str  = f'{hl:.1f} bar / {hlm:.0f} min' if np.isfinite(hl) else 'N/A'
        ess_str = f'{ess:.3f}' if np.isfinite(ess) else 'N/A'
        info = (f'全量 ACF(1) = {acf1f:.4f}\n'
                f'窗口均值 ACF(1) = {acf1w:.4f} ± {acf_win_std[0]:.4f}（窗口间σ）\n'
                f'AR(1) 半衰期 = {hl_str}\n'
                f'ESS/N = {ess_str}（实际独立样本占比）\n'
                f'Ljung-Box(lag=10) p = {lb_p:.4f}')
        ax.text(0.98, 0.97, info, transform=ax.transAxes,
                fontsize=7, va='top', ha='right',
                bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.8))
        ax.set_xlabel('滞后阶数（bar）')
        ax.set_ylabel('自相关系数')
        ax.set_title(
            f'窗口均值 ACF（非重叠 {window_days} 交易日窗口，N_windows={n_windows}，'
            f'与 IC/OLS 分析粒度一致）',
            fontsize=8)
        ax.legend(fontsize=7, loc='upper left')

        # 子图 1：分布直方图 + 正态拟合
        ax = axes1[1]
        from scipy.stats import norm as sp_norm
        ax.hist(factor_series, bins=100, density=True,
                color='#1f77b4', alpha=0.6, label='因子值')
        mu, sigma = np.mean(factor_series), np.std(factor_series)
        x_range = np.linspace(factor_series.min(), factor_series.max(), 300)
        ax.plot(x_range, sp_norm.pdf(x_range, mu, sigma),
                color='#d62728', lw=1.8, ls='--', label='正态拟合')
        jb_str = f'{result["jb_p"]:.4f}'
        dist_info = (f'均值 = {result["mean"]:.4f}    标准差 = {result["std"]:.4f}\n'
                     f'偏度 = {result["skew"]:.4f}    超额峰度 = {result["kurt"]:.4f}\n'
                     f'Jarque-Bera p = {jb_str}（正态性检验）\n'
                     f'Null 率 = {result["null_rate"]:.1%}')
        ax.text(0.98, 0.97, dist_info, transform=ax.transAxes,
                fontsize=7, va='top', ha='right',
                bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.8))
        ax.set_xlabel('因子值')
        ax.set_ylabel('密度')
        ax.set_title('因子值分布（直方图 + 正态拟合）', fontsize=8)
        ax.legend(fontsize=7)

        # 子图 2：Q-Q 图
        ax = axes1[2]
        from scipy import stats as sp_stats_qqplot
        (quantiles, values), (slope, intercept, r) = sp_stats_qqplot.probplot(
            factor_series, dist='norm')
        ax.scatter(quantiles, values, s=4, alpha=0.4, color='#1f77b4',
                   label='样本分位数')
        ax.plot(quantiles, slope * np.array(quantiles) + intercept,
                color='#d62728', lw=1.5, label='参考线（正态）')
        ax.set_xlabel('理论分位数（标准正态）')
        ax.set_ylabel('样本分位数')
        ax.set_title('Q-Q 图（vs 标准正态）—— 偏离参考线越多，厚尾越严重', fontsize=8)
        ax.legend(fontsize=7)

        plt.tight_layout()
        if self.output_dir is not None:
            prop_dir = self.output_dir / 'properties'
            prop_dir.mkdir(parents=True, exist_ok=True)
            fig1.savefig(prop_dir / f'properties_acf_dist_{factor}.png',
                         bbox_inches='tight')
        if self.inline:
            plt.show()
        else:
            plt.close(fig1)

        # ── 图 2：平稳性 + 跨年稳定性 ────────────────────────────
        fig2, axes2 = plt.subplots(2, 1, figsize=(14, 8))
        fig2.suptitle(
            f'{self.symbol_name} | {factor} | 因子诊断（平稳性 + 跨年稳定性）',
            fontsize=10)

        # 子图 0：逐日均值 & 标准差时序双轴
        ax0 = axes2[0]
        if daily_dates:
            try:
                dt_dates = [datetime.date.fromisoformat(str(d)) for d in daily_dates]
            except Exception:
                dt_dates = list(range(len(daily_dates)))
            ax0.plot(dt_dates, daily_means, color='#1f77b4', lw=1.0,
                     alpha=0.8, label='逐日均值（左轴）')
            ax0.axhline(0, color='black', lw=0.6, ls='--', alpha=0.5)
            ax0.set_ylabel('日均值', color='#1f77b4')
            ax0_r = ax0.twinx()
            ax0_r.plot(dt_dates, daily_stds, color='#ff7f0e', lw=1.0,
                       alpha=0.7, label='逐日标准差（右轴）')
            ax0_r.set_ylabel('日内标准差', color='#ff7f0e')
            ax0.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            ax0.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
            plt.setp(ax0.xaxis.get_majorticklabels(), rotation=45, ha='right')
            # 图例合并
            lines1, labs1 = ax0.get_legend_handles_labels()
            lines2, labs2 = ax0_r.get_legend_handles_labels()
            ax0.legend(lines1 + lines2, labs1 + labs2, fontsize=7, loc='upper left')

        # 平稳性结论标注
        adf_p  = result['adf_p_mean']
        kpss_p = result['kpss_p_mean']
        stat_str = result['stationarity']
        adf_p_str  = f'{adf_p:.4f}'  if np.isfinite(adf_p)  else 'N/A'
        kpss_p_str = f'{kpss_p:.4f}' if np.isfinite(kpss_p) else 'N/A'
        stat_info = (f'日均 ADF  p = {adf_p_str}\n'
                     f'日均 KPSS p = {kpss_p_str}\n'
                     f'综合结论：{stat_str}')
        ax0.text(0.01, 0.97, stat_info, transform=ax0.transAxes,
                 fontsize=7.5, va='top', ha='left',
                 bbox=dict(boxstyle='round,pad=0.3', fc='lightyellow', alpha=0.9))
        ax0.set_title(
            f'逐日因子均值 & 标准差时序（N_days={result["n_days_stationarity"]}）',
            fontsize=8)

        # 子图 1：逐年均值 + 标准差并排柱
        ax1 = axes2[1]
        if annual_stats:
            years   = [s['year']   for s in annual_stats]
            yr_mean = [s['mean']   for s in annual_stats]
            yr_std  = [s['std']    for s in annual_stats]
            yr_nday = [s['n_days'] for s in annual_stats]
            x_yr = np.arange(len(years))
            bar_colors = ['#1f77b4' if v >= 0 else '#d62728' for v in yr_mean]
            ax1.bar(x_yr, yr_mean, color=bar_colors, alpha=0.7,
                    label='年均值（左轴）')
            ax1.axhline(0, color='black', lw=0.6, ls='--')
            ax1.set_xticks(x_yr)
            ax1.set_xticklabels([str(y) for y in years])
            ax1.set_ylabel('年均值')
            # 标注 n_days
            for xi, (nd, mv) in enumerate(zip(yr_nday, yr_mean)):
                offset = abs(max(yr_mean, key=abs)) * 0.04 if yr_mean else 0
                ax1.text(xi, mv + (offset if mv >= 0 else -offset * 2),
                         f'n={nd}日', ha='center', va='bottom', fontsize=7)
            ax1_r = ax1.twinx()
            ax1_r.plot(x_yr, yr_std, color='#ff7f0e', marker='D',
                       ms=6, lw=1.6, label='年标准差（右轴）')
            ax1_r.set_ylabel('年内标准差', color='#ff7f0e')
            lines1, labs1 = ax1.get_legend_handles_labels()
            lines2, labs2 = ax1_r.get_legend_handles_labels()
            ax1.legend(lines1 + lines2, labs1 + labs2, fontsize=7, loc='upper left')
        ax1.set_title('逐年均值与标准差（因子分布跨年稳定性）', fontsize=8)

        plt.tight_layout()
        if self.output_dir is not None:
            prop_dir = self.output_dir / 'properties'
            prop_dir.mkdir(parents=True, exist_ok=True)
            fig2.savefig(prop_dir / f'properties_stationarity_{factor}.png',
                         bbox_inches='tight')
        if self.inline:
            plt.show()
        else:
            plt.close(fig2)

    # ──────────────────────────────────────────────────────────
    # 单因子回测（持仓状态机）
    # ──────────────────────────────────────────────────────────

    def _run_single_threshold_backtest(
        self,
        factor_arr:   np.ndarray,
        next_open_arr: np.ndarray,
        ror_arr:      np.ndarray,
        session_arr:  np.ndarray,
        trading_date_arr: np.ndarray,
        rq_lo_arr:    np.ndarray,
        rq_hi_arr:    np.ndarray,
        slippage_ticks: float,
    ) -> List[Dict]:
        """
        持仓状态机核心（纯 numpy 循环，约 45000 bar / 品种可在秒内完成）。

        状态变量
        --------
        position  : 0 = 空仓，+1 = 持多，-1 = 持空
        entry_idx : 开仓信号产生的 bar 索引（不是成交 bar）
        entry_price: 开仓成交价（= next_open[entry_idx]）

        成交规则
        --------
        所有交易均在信号产生后的**下一根 bar 开盘价**成交，
        与 ror_future_Nb 的 start_price = open.shift(-1) 完全对齐。

        成本计算
        --------
        cost_one_side = slippage_ticks * tick_size / entry_price
        每笔平仓（含反手/到期/单独平仓）均在平仓记录中计入往返成本：
          round_trip_cost = 2 × cost_one_side
        这样开仓时不单独记录，平仓时一次性扣除完整往返成本。

        Returns
        -------
        trades : list of dict，每笔已完成交易的记录
          entry_bar_idx  : 信号 bar 索引
          exit_bar_idx   : 平仓 bar 索引
          exit_date      : 平仓所在交易日
          direction      : +1 / -1
          raw_pnl_bp     : 无成本毛收益（bp）
          net_pnl_bp     : 扣成本净收益（bp）
          bars_held      : 实际持仓 bar 数
        """
        n         = len(factor_arr)
        label_bars = self.label_bars
        tick_size  = self.tick_size

        position    = 0
        entry_idx   = -1
        entry_price = np.nan

        trades: List[Dict] = []

        for i in range(n):
            # ── 跳过预热期（分位数尚未就绪）或无效数据 ──────────
            if (not np.isfinite(factor_arr[i]) or
                    not np.isfinite(rq_lo_arr[i]) or
                    not np.isfinite(rq_hi_arr[i])):
                continue

            # 下一根bar开盘价（入场/出场价）
            nxt = next_open_arr[i]
            if not np.isfinite(nxt) or nxt <= 0:
                # next_open 无效（session末最后一根bar），禁止新开仓
                # 若有持仓且已到期，正常平仓
                if position != 0 and (i - entry_idx) >= label_bars:
                    # 用当前 bar 的 next_open 仍无效，跳过（下一个 bar 处理）
                    pass
                continue

            # ── 信号计算 ──────────────────────────────────────────
            fv = factor_arr[i]
            if fv > rq_hi_arr[i]:
                signal = 1
            elif fv < rq_lo_arr[i]:
                signal = -1
            else:
                signal = 0

            # can_enter：ror_future 非 null 且同 session 内有足够空间
            # ror_arr[i] 非 nan 保证了从 bar i 信号出发持有 label_bars 根后
            # 还在同一 session 内（因为 ror_future 是 over(session_id) 计算的）
            can_enter = (signal != 0 and np.isfinite(ror_arr[i]))

            # ── 有仓位 ────────────────────────────────────────────
            if position != 0:
                bars_held = i - entry_idx

                # Case A：到期
                if bars_held >= label_bars:
                    exit_price = nxt
                    gross = position * (exit_price / entry_price - 1)
                    cost  = 2.0 * slippage_ticks * tick_size / entry_price
                    trades.append({
                        'entry_bar_idx': entry_idx,
                        'exit_bar_idx':  i,
                        'exit_date':     trading_date_arr[i],
                        'direction':     position,
                        'raw_pnl_bp':    gross * 10000,
                        'net_pnl_bp':    (gross - cost) * 10000,
                        'bars_held':     bars_held,
                    })
                    # 到期后处理新信号（同价平旧开新）
                    if can_enter:
                        position    = signal
                        entry_idx   = i
                        entry_price = exit_price
                    else:
                        position  = 0
                        entry_idx = -1
                        entry_price = np.nan

                # Case B：未到期，反向信号
                elif signal == -position:
                    exit_price = nxt
                    gross = position * (exit_price / entry_price - 1)
                    cost  = 2.0 * slippage_ticks * tick_size / entry_price
                    trades.append({
                        'entry_bar_idx': entry_idx,
                        'exit_bar_idx':  i,
                        'exit_date':     trading_date_arr[i],
                        'direction':     position,
                        'raw_pnl_bp':    gross * 10000,
                        'net_pnl_bp':    (gross - cost) * 10000,
                        'bars_held':     bars_held,
                    })
                    # 反手开新仓（can_enter 必须为 True，否则仅平仓）
                    if can_enter:
                        position    = signal
                        entry_idx   = i
                        entry_price = exit_price
                    else:
                        position    = 0
                        entry_idx   = -1
                        entry_price = np.nan

                # Case B2：同向或无信号 → 持仓不动
                # else: pass

            # ── 空仓 ──────────────────────────────────────────────
            else:
                if can_enter:
                    position    = signal
                    entry_idx   = i
                    entry_price = nxt

        return trades

    def _calc_metrics(
        self,
        trades:       List[Dict],
        all_dates:    np.ndarray,
    ) -> Dict:
        """
        从 trades 列表计算全套绩效指标，同时处理 raw（cost=0）和 net（有成本）两套。

        日度聚合方式
        ------------
        按 exit_date 分组汇总当日所有已完成交易的 pnl（sum），
        再与全样本所有交易日做 outer join，无交易日填 0，
        确保 Sharpe 分母包含所有空仓日（更保守、更真实）。

        Returns
        -------
        dict 包含 raw_* 和 net_* 两组指标，以及 n_trades / n_long / n_short。
        """
        if not trades:
            return {}

        raw_pnl = np.array([t['raw_pnl_bp'] for t in trades])
        net_pnl = np.array([t['net_pnl_bp'] for t in trades])
        dirs    = np.array([t['direction']   for t in trades])
        dates   = np.array([t['exit_date']   for t in trades])

        n_trades = len(trades)
        n_long   = int(np.sum(dirs > 0))
        n_short  = int(np.sum(dirs < 0))

        # ── 日度聚合（含全部0收益日）──────────────────────────────
        # 用字典先聚合有交易日
        date_raw: Dict = {}
        date_net: Dict = {}
        for d, r, ne in zip(dates, raw_pnl, net_pnl):
            date_raw[d] = date_raw.get(d, 0.0) + r
            date_net[d] = date_net.get(d, 0.0) + ne

        n_all_days = len(all_dates)
        daily_raw = np.array([date_raw.get(d, 0.0) for d in all_dates])
        daily_net = np.array([date_net.get(d, 0.0) for d in all_dates])

        def _metrics_from_daily(daily: np.ndarray, pnl_arr: np.ndarray) -> Dict:
            cumret = np.cumsum(daily)
            total  = float(cumret[-1]) if len(cumret) > 0 else 0.0
            annual = total / n_all_days * 252 if n_all_days > 0 else 0.0

            mean_d = float(np.mean(daily))
            std_d  = float(np.std(daily, ddof=1)) if len(daily) > 1 else 0.0
            sharpe = mean_d / std_d * np.sqrt(252) if std_d > 1e-12 else 0.0

            # 最大回撤（水下深度）
            running_max = np.maximum.accumulate(cumret)
            drawdown    = cumret - running_max
            max_dd      = float(np.min(drawdown))

            # 逐笔胜率与盈亏比
            win  = pnl_arr[pnl_arr > 0]
            loss = pnl_arr[pnl_arr < 0]
            win_rate = float(len(win) / len(pnl_arr)) if len(pnl_arr) > 0 else 0.0
            if len(loss) > 0 and len(win) > 0:
                pf = float(np.mean(win) / abs(np.mean(loss)))
            else:
                pf = float('nan')

            return {
                'total_bp':     round(total,  2),
                'annual_bp':    round(annual, 2),
                'sharpe':       round(sharpe, 4),
                'max_dd_bp':    round(max_dd, 2),
                'win_rate':     round(win_rate, 4),
                'profit_factor': round(pf, 4) if np.isfinite(pf) else float('nan'),
                'daily_pnl':    daily,
                'cumret':       cumret,
                'drawdown':     drawdown,
            }

        raw_m = _metrics_from_daily(daily_raw, raw_pnl)
        net_m = _metrics_from_daily(daily_net, net_pnl)

        return {
            'n_trades':            n_trades,
            'n_long':              n_long,
            'n_short':             n_short,
            # raw（无成本）
            'total_raw_bp':        raw_m['total_bp'],
            'annual_raw_bp':       raw_m['annual_bp'],
            'sharpe_raw':          raw_m['sharpe'],
            'max_dd_raw_bp':       raw_m['max_dd_bp'],
            'win_rate_raw':        raw_m['win_rate'],
            'profit_factor_raw':   raw_m['profit_factor'],
            # net（有成本）
            'total_net_bp':        net_m['total_bp'],
            'annual_net_bp':       net_m['annual_bp'],
            'sharpe_net':          net_m['sharpe'],
            'max_dd_net_bp':       net_m['max_dd_bp'],
            'win_rate_net':        net_m['win_rate'],
            'profit_factor_net':   net_m['profit_factor'],
            # 时序数据（供绘图用）
            '_daily_raw':          raw_m['daily_pnl'],
            '_daily_net':          net_m['daily_pnl'],
            '_cumret_raw':         raw_m['cumret'],
            '_cumret_net':         net_m['cumret'],
            '_drawdown_raw':       raw_m['drawdown'],
            '_drawdown_net':       net_m['drawdown'],
        }

    def run_backtest(
        self,
        df:             pl.DataFrame,
        factor:         str,
        thresholds:     List[Tuple[float, float]] = None,
        rolling_window: int   = 500,
        min_samples:    int   = 20,
        slippage_ticks: float = 1.5,
    ) -> Dict:
        """
        单因子持仓状态机回测。

        方法
        ----
        基于滚动历史分位数（无未来信息）生成方向信号，通过持仓状态机
        模拟真实 CTA 回测逻辑，同时输出 cost=0 和 with_cost 两条净值曲线。

        信号规则
        --------
        factor > 滚动上分位数（hi_q）→ 做多（+1）
        factor < 滚动下分位数（lo_q）→ 做空（-1）
        其余 → 空仓等待

        持仓状态机规则
        --------------
        - 持仓同向：不重复开仓
        - 持仓反向：下一bar开盘反手（平旧 + 开新，扣双边成本）
        - 持仓满 label_bars 到期 + 有新信号：同价平旧开新（续仓或反手）
        - 持仓满 label_bars 到期 + 无有效新信号：仅平仓
        - 靠近 session 末（ror_future_Nb 为 null）：禁止开新仓

        成本计算
        --------
        单边成本 = slippage_ticks × tick_size / 入场价（开仓和平仓各一次）

        Parameters
        ----------
        df             : features_df，需包含 factor、open、continuous_session_id、
                         ror_future_{label_bars}b、trading_date 列
        factor         : 因子列名
        thresholds     : 阈值对列表，如 [(0.2,0.8),(0.25,0.75)]；
                         None 时默认 [(0.2,0.8),(0.25,0.75),(0.3,0.7)]
        rolling_window : 滚动分位数回望窗口（bar数）
        min_samples    : 滚动分位数最小样本数
        slippage_ticks : 单边滑点（跳数）

        Returns
        -------
        dict 包含：
          factor          : 因子名
          best_threshold  : (lo_q, hi_q)，按 sharpe_net 最高
          best_sharpe_net : float
          best_annual_net_bp : float
          best_max_dd_net_bp : float
          all_metrics     : {(lo_q, hi_q): metrics_dict}
          all_dates       : 分析期所有交易日列表
        """
        # ── 前置检查 ──────────────────────────────────────────────
        ror_col = f'ror_future_{self.label_bars}b'
        required = [factor, 'open', 'continuous_session_id', ror_col, 'trading_date']
        missing  = [c for c in required if c not in df.columns]
        if missing:
            warnings.warn(f'run_backtest: 缺少必要列 {missing}，跳过回测。')
            return {}

        if thresholds is None:
            thresholds = [(0.2, 0.8), (0.25, 0.75), (0.3, 0.7)]

        # ── 过滤年份范围内的数据，按时间排序 ─────────────────────
        df = df.sort('datetime') if 'datetime' in df.columns else df

        # ── 预提取 numpy 数组（一次性，避免循环中反复 to_numpy）───
        factor_arr      = df[factor].to_numpy().astype(float)
        next_open_arr   = (
            df['open'].shift(-1).over('continuous_session_id')
            .to_numpy().astype(float)
        )
        ror_arr         = df[ror_col].to_numpy().astype(float)
        trading_date_arr = df['trading_date'].to_numpy()

        # 全样本所有交易日（用于日度聚合填0）
        all_dates = np.array(sorted(df['trading_date'].unique().to_list()))

        all_results: Dict[Tuple, Dict] = {}

        for lo_q, hi_q in thresholds:
            # 滚动分位数（向量化，与 run_quantile_analysis 一致）
            rq_lo = (
                df[factor]
                .rolling_quantile(lo_q,
                                  window_size=rolling_window,
                                  min_samples=min_samples)
                .to_numpy().astype(float)
            )
            rq_hi = (
                df[factor]
                .rolling_quantile(hi_q,
                                  window_size=rolling_window,
                                  min_samples=min_samples)
                .to_numpy().astype(float)
            )

            trades = self._run_single_threshold_backtest(
                factor_arr, next_open_arr, ror_arr,
                df['continuous_session_id'].to_numpy(),
                trading_date_arr,
                rq_lo, rq_hi,
                slippage_ticks,
            )

            metrics = self._calc_metrics(trades, all_dates)
            if metrics:
                all_results[(lo_q, hi_q)] = metrics

        if not all_results:
            return {}

        # ── 保存 CSV（批量模式）────────────────────────────────────
        if self.bt_dir is not None:
            rows = []
            for (lo_q, hi_q), m in all_results.items():
                rows.append({
                    'factor':          factor,
                    'lo_q':            lo_q,
                    'hi_q':            hi_q,
                    'slippage_ticks':  slippage_ticks,
                    'tick_size':       self.tick_size,
                    'label_bars':      self.label_bars,
                    'n_trades':        m['n_trades'],
                    'n_long':          m['n_long'],
                    'n_short':         m['n_short'],
                    'total_raw_bp':    m['total_raw_bp'],
                    'annual_raw_bp':   m['annual_raw_bp'],
                    'sharpe_raw':      m['sharpe_raw'],
                    'max_dd_raw_bp':   m['max_dd_raw_bp'],
                    'win_rate_raw':    m['win_rate_raw'],
                    'profit_factor_raw': m['profit_factor_raw'],
                    'total_net_bp':    m['total_net_bp'],
                    'annual_net_bp':   m['annual_net_bp'],
                    'sharpe_net':      m['sharpe_net'],
                    'max_dd_net_bp':   m['max_dd_net_bp'],
                    'win_rate_net':    m['win_rate_net'],
                    'profit_factor_net': m['profit_factor_net'],
                })
            pl.DataFrame(rows).write_csv(
                self.bt_dir / f'backtest_{factor}.csv')

        # ── 绘图 ──────────────────────────────────────────────────
        self._plot_backtest(factor, all_results, all_dates,
                            slippage_ticks, rolling_window)

        # ── 返回摘要 ──────────────────────────────────────────────
        best_thr = max(all_results,
                       key=lambda k: all_results[k].get('sharpe_net', -np.inf))
        best_m   = all_results[best_thr]

        # 去掉绘图私有字段后返回
        clean_results = {
            k: {kk: vv for kk, vv in v.items() if not kk.startswith('_')}
            for k, v in all_results.items()
        }

        return {
            'factor':              factor,
            'best_threshold':      best_thr,
            'best_sharpe_net':     best_m['sharpe_net'],
            'best_annual_net_bp':  best_m['annual_net_bp'],
            'best_max_dd_net_bp':  best_m['max_dd_net_bp'],
            'all_metrics':         clean_results,
            'all_dates':           all_dates.tolist(),
        }

    def _plot_backtest(
        self,
        factor:         str,
        all_results:    Dict[Tuple, Dict],
        all_dates:      np.ndarray,
        slippage_ticks: float,
        rolling_window: int,
    ):
        """
        单因子回测绘图，共两张图。

        图1：净值曲线 + 回撤（文件名：backtest_{factor}.png）
        -------------------------------------------------------
        上子图：多阈值的 cost=0（虚线）和 with_cost（实线）净值曲线叠加，
                颜色区分阈值，年份边界竖向虚线
        下子图：各阈值的 with_cost 最大回撤曲线（fill_between）

        图2：绩效指标汇总表（文件名：backtest_metrics_{factor}.png）
        --------------------------------------------------------------
        每阈值两行（无成本 / 有成本），9列绩效指标
        """
        if not all_results:
            return

        # ── 日期轴转换 ────────────────────────────────────────────
        try:
            dt_dates = [datetime.date.fromisoformat(str(d)) for d in all_dates]
        except Exception:
            dt_dates = list(range(len(all_dates)))

        # 年份边界
        year_boundaries = []
        seen_years: set = set()
        for xi, d in enumerate(dt_dates):
            yr = d.year if isinstance(d, datetime.date) else None
            if yr is not None and yr not in seen_years:
                if seen_years:  # 第一年不画线
                    year_boundaries.append((xi, yr))
                seen_years.add(yr)

        # ── 颜色方案（最多5种阈值）───────────────────────────────
        palette = ['#1f77b4', '#d62728', '#2ca02c', '#9467bd', '#ff7f0e']
        thr_list = list(all_results.keys())
        thr_color = {thr: palette[i % len(palette)]
                     for i, thr in enumerate(thr_list)}

        # ══════════════════════════════════════════════════════════
        # 图1：净值 + 回撤
        # ══════════════════════════════════════════════════════════
        fig1, (ax_equity, ax_dd) = plt.subplots(
            2, 1,
            figsize=(14, 9),
            gridspec_kw={'height_ratios': [3, 1]},
            sharex=True
        )

        for thr, m in all_results.items():
            lo_q, hi_q = thr
            color = thr_color[thr]
            cumret_raw = m['_cumret_raw']
            cumret_net = m['_cumret_net']
            drawdown_net = m['_drawdown_net']

            # 数组长度与 all_dates 对齐
            x = dt_dates[:len(cumret_net)]

            lbl_raw = (f'({lo_q:.0%}/{hi_q:.0%}) cost=0'
                       f'  S={m["sharpe_raw"]:.2f}')
            lbl_net = (f'({lo_q:.0%}/{hi_q:.0%}) with cost'
                       f'  S={m["sharpe_net"]:.2f}')

            ax_equity.plot(x, cumret_raw[:len(x)],
                           color=color, lw=1.2, ls='--', alpha=0.55,
                           label=lbl_raw)
            ax_equity.plot(x, cumret_net[:len(x)],
                           color=color, lw=1.8, ls='-', alpha=0.90,
                           label=lbl_net)

            # 回撤（有成本）
            ax_dd.fill_between(x, drawdown_net[:len(x)], 0,
                                color=color, alpha=0.30)
            ax_dd.plot(x, drawdown_net[:len(x)],
                       color=color, lw=0.8, ls='-', alpha=0.70)

        ax_equity.axhline(0, color='black', lw=0.8, ls='--', alpha=0.6)
        for xi, yr in year_boundaries:
            if xi < len(dt_dates):
                ax_equity.axvline(dt_dates[xi], color='gray',
                                  lw=0.8, ls=':', alpha=0.6)
                ax_equity.text(dt_dates[xi],
                               ax_equity.get_ylim()[0],
                               str(yr), fontsize=7, color='gray',
                               ha='left', va='bottom')

        ax_equity.set_ylabel('累计收益（bp）')
        ax_equity.set_title(
            f'{self.symbol_name} | {factor} | 单因子回测净值曲线\n'
            f'[持仓状态机  |  持仓期={self.label_bars}棒/{self.label_bars*5}分钟  |  '
            f'滚动分位数窗口={rolling_window}棒  |  '
            f'slippage={slippage_ticks}跳 × tick={self.tick_size}\n'
            f'虚线=无成本(cost=0)  实线=扣成本(with cost)  '
            f'颜色区分阈值对]',
            fontsize=8
        )
        ax_equity.legend(fontsize=7, loc='upper left',
                         ncol=max(1, len(thr_list)))

        ax_dd.axhline(0, color='black', lw=0.6, ls='--', alpha=0.5)
        ax_dd.set_ylabel('回撤（bp）')
        ax_dd.set_title('回撤曲线（有成本）', fontsize=8)
        ax_dd.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax_dd.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        plt.setp(ax_dd.xaxis.get_majorticklabels(), rotation=45, ha='right')

        plt.tight_layout()
        if self.bt_dir is not None:
            fig1.savefig(self.bt_dir / 'figures' / f'backtest_{factor}.png',
                         bbox_inches='tight')
        if self.inline:
            plt.show()
        else:
            plt.close(fig1)

        # ══════════════════════════════════════════════════════════
        # 图2：绩效指标汇总表
        # ══════════════════════════════════════════════════════════
        col_labels = ['策略', '年化收益(bp)', 'Sharpe',
                      '最大回撤(bp)', '胜率', '盈亏比',
                      '总笔数', '多头', '空头']

        table_data = []
        row_colors = []

        for thr, m in all_results.items():
            lo_q, hi_q = thr
            color_base = thr_color[thr]

            def _fmt(v, pct=False):
                if not np.isfinite(v):
                    return 'N/A'
                return f'{v:.1%}' if pct else f'{v:.2f}'

            # 无成本行
            row_raw = [
                f'({lo_q:.0%}/{hi_q:.0%}) cost=0',
                f'{m["annual_raw_bp"]:.1f}',
                f'{m["sharpe_raw"]:.3f}',
                f'{m["max_dd_raw_bp"]:.1f}',
                f'{m["win_rate_raw"]:.1%}',
                _fmt(m['profit_factor_raw']),
                str(m['n_trades']),
                str(m['n_long']),
                str(m['n_short']),
            ]
            # 有成本行
            row_net = [
                f'({lo_q:.0%}/{hi_q:.0%}) with cost',
                f'{m["annual_net_bp"]:.1f}',
                f'{m["sharpe_net"]:.3f}',
                f'{m["max_dd_net_bp"]:.1f}',
                f'{m["win_rate_net"]:.1%}',
                _fmt(m['profit_factor_net']),
                str(m['n_trades']),
                str(m['n_long']),
                str(m['n_short']),
            ]

            table_data.append(row_raw)
            table_data.append(row_net)

            # 行颜色：有成本行用阈值颜色（浅），无成本行更浅
            import matplotlib.colors as mcolors
            rgba_base = mcolors.to_rgba(color_base)
            row_colors.append([(*rgba_base[:3], 0.10)] * len(col_labels))
            row_colors.append([(*rgba_base[:3], 0.25)] * len(col_labels))

        n_rows = len(table_data)
        fig2_h = max(3.5, 0.55 * n_rows + 1.5)
        fig2, ax_t = plt.subplots(figsize=(16, fig2_h))
        ax_t.axis('off')

        tbl = ax_t.table(
            cellText=table_data,
            colLabels=col_labels,
            cellLoc='center',
            loc='center',
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(9)
        tbl.auto_set_column_width(col=list(range(len(col_labels))))

        # 表头颜色
        for j in range(len(col_labels)):
            tbl[0, j].set_facecolor('#404040')
            tbl[0, j].set_text_props(color='white', fontweight='bold')

        # 行颜色 + 有成本行加粗
        for row_idx, (row_data, rc) in enumerate(zip(table_data, row_colors)):
            is_net = 'with cost' in row_data[0]
            for col_idx in range(len(col_labels)):
                cell = tbl[row_idx + 1, col_idx]
                cell.set_facecolor(rc[col_idx])
                if is_net:
                    cell.set_text_props(fontweight='bold')

        ax_t.set_title(
            f'{self.symbol_name} | {factor} | 单因子回测绩效汇总\n'
            f'[slippage={slippage_ticks}跳 × tick_size={self.tick_size}  |  '
            f'持仓期={self.label_bars}棒  |  滚动分位数窗口={rolling_window}棒]',
            fontsize=9, pad=12
        )

        plt.tight_layout()
        if self.bt_dir is not None:
            fig2.savefig(
                self.bt_dir / 'figures' / f'backtest_metrics_{factor}.png',
                bbox_inches='tight')
        if self.inline:
            plt.show()
        else:
            plt.close(fig2)


# ==============================================================
# 2. CrossFactorAnalyzer
# ==============================================================

class CrossFactorAnalyzer:
    """
    多因子相关性分析：Spearman 相关矩阵 + 热度图 + 层次聚类树
    """

    def __init__(self, output_dir: Path, correlation_threshold: float = 0.7,
                 method: str = 'spearman'):
        self.output_dir = output_dir
        self.threshold  = correlation_threshold
        self.method     = method
        (output_dir / 'figures').mkdir(parents=True, exist_ok=True)

    def run(self, df: pl.DataFrame, factors: List[str],
            feature_groups_info: Dict,
            label_bars: int) -> Dict:
        """
        计算因子相关矩阵并出图。

        注：相关性矩阵的间隔采样目的与 IC 不同。
        这里是为了减少因子值序列本身的时间自相关（因子通常有很强的 persistence），
        使相关性估计不受高自相关因子的过度影响。
        每隔 label_bars 取一个样本，保留约 N/label_bars 个近似独立的截面观测。
        """
        available = [f for f in factors if f in df.columns]
        if len(available) < 2:
            return {}

        # 非重叠采样
        sampled = (
            df
            .with_row_index('_row_idx')
            .filter(pl.col('_row_idx') % label_bars == 0)
            .drop('_row_idx')
            .select(available)
            .drop_nulls()
        )

        mat = sampled.to_numpy().astype(float)
        n_factors = len(available)

        # Spearman 相关矩阵
        if self.method == 'spearman':
            corr_mat = np.zeros((n_factors, n_factors))
            for i in range(n_factors):
                for j in range(i, n_factors):
                    mask = np.isfinite(mat[:, i]) & np.isfinite(mat[:, j])
                    if mask.sum() > 10:
                        r, _ = stats.pearsonr(mat[mask, i], mat[mask, j])
                        corr_mat[i, j] = corr_mat[j, i] = r
                    else:
                        corr_mat[i, j] = corr_mat[j, i] = np.nan
            np.fill_diagonal(corr_mat, 1.0)
        else:
            corr_mat = np.corrcoef(mat.T)

        # 按特征组排序
        ordered_factors, group_boundaries, group_labels = \
            self._order_by_group(available, feature_groups_info)

        # 重排矩阵
        idx_map = {f: i for i, f in enumerate(available)}
        new_order = [idx_map[f] for f in ordered_factors if f in idx_map]
        corr_reordered = corr_mat[np.ix_(new_order, new_order)]

        # 保存 CSV
        corr_df = pl.DataFrame(
            {f: corr_reordered[:, j].tolist()
             for j, f in enumerate(ordered_factors)}
        ).with_columns(pl.Series('factor', ordered_factors))
        corr_df.write_csv(self.output_dir / 'correlation_matrix.csv')

        # 高相关因子对
        redundant = []
        for i in range(len(ordered_factors)):
            for j in range(i + 1, len(ordered_factors)):
                r = corr_reordered[i, j]
                if np.isfinite(r) and abs(r) >= self.threshold:
                    redundant.append({
                        'factor_a': ordered_factors[i],
                        'factor_b': ordered_factors[j],
                        'correlation': round(r, 4),
                    })
        if redundant:
            pl.DataFrame(redundant).sort('correlation', descending=True)\
              .write_csv(self.output_dir / 'redundant_pairs.csv')

        # 热度图
        self._plot_heatmap(corr_reordered, ordered_factors,
                           group_boundaries, group_labels)
        # 聚类树
        self._plot_dendrogram(corr_reordered, ordered_factors)

        return {'n_factors': len(ordered_factors),
                'n_redundant_pairs': len(redundant)}

    def _order_by_group(self, factors: List[str],
                        feature_groups_info: Dict
                        ) -> Tuple[List[str], List[int], List[str]]:
        """按特征组对因子排序，返回排序后因子列表、组边界和组名"""
        group_of = {}
        for g_name, info in feature_groups_info.items():
            for f in info.get('features', []):
                group_of[f] = g_name

        groups_seen = []
        ordered = []
        for g_name in feature_groups_info:
            grp_factors = [f for f in factors
                           if group_of.get(f) == g_name]
            if grp_factors:
                ordered.extend(grp_factors)
                groups_seen.append((g_name, len(grp_factors)))

        # 未分组的放最后
        ungrouped = [f for f in factors if f not in group_of]
        if ungrouped:
            ordered.extend(ungrouped)
            groups_seen.append(('其他', len(ungrouped)))

        # 组边界
        boundaries = []
        pos = 0
        labels = []
        for g_name, cnt in groups_seen:
            boundaries.append(pos)
            labels.append(g_name)
            pos += cnt

        return ordered, boundaries, labels

    def _plot_heatmap(self, corr: np.ndarray, factors: List[str],
                      boundaries: List[int], group_labels: List[str]):
        n = len(factors)
        fig_size = max(16, n * 0.35)
        fig, ax = plt.subplots(figsize=(fig_size, fig_size * 0.85))

        mask = np.zeros_like(corr, dtype=bool)
        np.fill_diagonal(mask, True)  # 对角线留白

        sns.heatmap(corr, ax=ax,
                    mask=False,
                    cmap='RdBu_r', center=0, vmin=-1, vmax=1,
                    xticklabels=factors, yticklabels=factors,
                    linewidths=0.3, linecolor='#eeeeee',
                    cbar_kws={'shrink': 0.6, 'label': 'Spearman 相关系数'})

        ax.set_xticklabels(ax.get_xticklabels(), fontsize=6, rotation=90)
        ax.set_yticklabels(ax.get_yticklabels(), fontsize=6, rotation=0)

        # 特征组边框线
        colors_group = plt.cm.tab20.colors
        for idx, (pos, glabel) in enumerate(zip(boundaries, group_labels)):
            next_pos = boundaries[idx + 1] if idx + 1 < len(boundaries) else n
            color = colors_group[idx % len(colors_group)]
            for spine_pos in [pos, next_pos]:
                ax.axvline(spine_pos, color=color, lw=1.5, alpha=0.8)
                ax.axhline(spine_pos, color=color, lw=1.5, alpha=0.8)
            mid = (pos + next_pos) / 2
            ax.text(mid, -0.8, glabel, ha='center', va='bottom',
                    fontsize=7, color=color, rotation=30)

        ax.set_title(f'因子相关性热度图（Spearman，非重叠采样）\n'
                     f'红色=正相关，蓝色=负相关，|r|>{self.threshold:.1f} 视为高度冗余',
                     fontsize=10)
        fig.tight_layout()
        fig.savefig(self.output_dir / 'figures' / 'heatmap.png',
                    bbox_inches='tight', dpi=150)
        plt.close(fig)

    def _plot_dendrogram(self, corr: np.ndarray, factors: List[str]):
        # 距离矩阵：1 - |corr|
        dist = 1.0 - np.abs(np.nan_to_num(corr))
        np.fill_diagonal(dist, 0)
        # 压缩为 condensed 格式
        from scipy.spatial.distance import squareform
        condensed = squareform(dist, checks=False)
        Z = linkage(condensed, method='ward')

        n = len(factors)
        fig, ax = plt.subplots(figsize=(max(14, n * 0.3), 7))
        dendrogram(Z, labels=factors, ax=ax,
                   leaf_rotation=90, leaf_font_size=7,
                   color_threshold=0.7 * max(Z[:, 2]))
        ax.set_title('因子层次聚类树（距离 = 1 - |Spearman 相关系数|）', fontsize=10)
        ax.set_ylabel('距离')
        ax.axhline(y=1 - self.threshold, color='red', ls='--', lw=1,
                   label=f'相关阈值 |r|={self.threshold}')
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(self.output_dir / 'figures' / 'dendrogram.png',
                    bbox_inches='tight', dpi=150)
        plt.close(fig)


# ==============================================================
# 3. FactorAnalysisPipeline
# ==============================================================

class FactorAnalysisPipeline:
    """
    单因子分析总流程管理器

    步骤：
    1. 加载数据（复用 data_loader）
    2. 特征工程（复用 FeatureEngineer）
    3. 数据过滤（只保留 train_start_year ~ train_end_year）
    4. 逐品种、逐因子运行 IC + 回测
    5. 跨品种汇总排名
    6. 跨品种合并后的相关性分析
    7. 生成汇总报告
    """

    def __init__(self,
                 config_path: str = 'config.json',
                 factors_override: Optional[List[str]] = None,
                 symbols_override: Optional[List[str]] = None):

        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = json.load(f)

        self.symbols  = parse_symbols(self.config)
        self.fa_cfg   = self.config.get('factor_analysis', {})
        self.ds_cfg   = self.config.get('data_split', {})
        self.fe_cfg   = self.config.get('feature_engineering', {})

        # 因子列表
        self.factors = factors_override or self.fa_cfg.get('factors', [])

        # 品种过滤
        if symbols_override:
            self.symbols = [s for s in self.symbols
                            if get_symbol_name(s['code']) in symbols_override]

        # 输出目录
        self.output_dir = Path(self.fa_cfg.get('output_dir', './factor_result'))
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 临时数据目录（复用 train_result 的 temp_data）
        self.temp_dir = Path(self.config['output']['temp_data_dir'])
        self.temp_dir.mkdir(parents=True, exist_ok=True)

        self.timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

        # 结果容器
        self.ic_summary    = []   # list of dict（逐因子 × 品种）
        self.ols_summary   = []   # list of dict（逐因子 × 品种，OLS Beta）
        self.qa_summary    = []   # list of dict（逐因子 × 品种，分层分析）
        self.prop_summary  = []   # list of dict（逐因子 × 品种，因子诊断）
        self.bt_summary    = []   # list of dict（逐因子 × 品种，回测）
        self.all_features_dfs = {}  # symbol_name → features_df（用于跨品种相关性分析）
        self.feature_groups_info = {}

        print(f"\n{'='*80}")
        print(f"单因子分析流程")
        print(f"{'='*80}")
        print(f"分析因子 ({len(self.factors)}): {self.factors}")
        print(f"分析品种 ({len(self.symbols)}): "
              f"{[get_symbol_name(s['code']) for s in self.symbols]}")
        train_start = self.ds_cfg.get('train_start_year', 2022)
        train_end   = self.ds_cfg.get('train_end_year',   2024)
        print(f"数据范围: {train_start} ~ {train_end}（样本内，不含测试年份）")
        print(f"输出目录: {self.output_dir}")
        print(f"{'='*80}\n")

    # ──────────────────────────────────────────────────────────

    def run(self):
        # Step 1: 加载数据
        print("Step 1: 加载数据...")
        data_dict = self._load_data()
        if not data_dict:
            print("⚠ 没有可用数据，流程终止")
            return

        # Step 2~4: 逐品种处理
        print("\nStep 2: 逐品种特征工程 + 因子分析...")
        for sym in self.symbols:
            symbol_name = get_symbol_name(sym['code'])
            tick_size   = sym.get('tick_size', 1.0)
            if symbol_name not in data_dict:
                continue
            print(f"\n  [{symbol_name}]")
            try:
                self._process_symbol(symbol_name, data_dict[symbol_name], tick_size, sym)
            except Exception as e:
                import traceback; traceback.print_exc()
                print(f"  ✗ {symbol_name} 处理异常: {e}")

        # Step 5: 跨品种汇总排名
        print("\nStep 3: 汇总因子排名...")
        self._aggregate_summary()

        # Step 6: 跨品种合并相关性分析
        print("\nStep 4: 因子相关性分析...")
        self._run_cross_factor()

        # Step 7: 报告
        print("\nStep 5: 生成汇总报告...")
        self._generate_report()

        print(f"\n{'='*80}")
        print(f"因子分析完成！输出目录: {self.output_dir}")
        print(f"{'='*80}")

    # ──────────────────────────────────────────────────────────

    def _load_data(self) -> Dict[str, pl.DataFrame]:
        start_date = datetime.datetime.strptime(
            self.config['data_source']['start_date'], '%Y-%m-%d').date()
        end_date = datetime.datetime.strptime(
            self.config['data_source']['end_date'], '%Y-%m-%d').date()

        symbol_codes = [s['code'] for s in self.symbols]
        batch_download(
            symbols=symbol_codes,
            start_date=start_date,
            end_date=end_date,
            periods=['1m'],
            save_format='parquet',
            save_dir=str(self.temp_dir),
            source_dir=self.config['data_source']['source_dir']
        )

        data_dict = {}
        for sym in self.symbols:
            code  = sym['code']
            safe  = code.replace(".", "_").replace("@", "_")
            name  = get_symbol_name(code)
            files = list(self.temp_dir.glob(f"{safe}_1m_*.parquet"))
            if files:
                data_dict[name] = pl.read_parquet(files[0])
                print(f"  ✓ {name} ({len(data_dict[name]):,} rows)")
            else:
                print(f"  ⚠ 未找到: {name}")
        return data_dict

    def _process_symbol(self, symbol_name: str,
                        bar1min: pl.DataFrame,
                        tick_size: float,
                        sym_cfg: dict = None):
        """特征工程 + 过滤年份 + 运行 IC + OLS + 回测 + 分层分析"""
        if sym_cfg is None:
            sym_cfg = {}

        # 特征工程
        engineer = FeatureEngineer(
            label_bars=self.fe_cfg.get('label_bars', 6),
            label_threshold=self.fe_cfg.get('label_threshold', 0.5),
            atr_long_window=self.fe_cfg.get('atr_long_window', 400),
        )
        features_df = engineer.create_features_and_labels(bar1min, verbose=False)
        self.feature_groups_info = engineer.feature_groups_info  # 保存最新品种的组信息

        # 过滤：只保留 train_start_year ~ train_end_year
        train_start = self.ds_cfg.get('train_start_year', 2022)
        train_end   = self.ds_cfg.get('train_end_year',   2024)
        df = features_df.filter(
            (pl.col('datetime').dt.year() >= train_start) &
            (pl.col('datetime').dt.year() <= train_end)
        )
        print(f"    数据过滤后: {len(df):,} 行 "
              f"({train_start}~{train_end})")

        # 确定实际可分析的因子
        factors_to_analyze = [f for f in self.factors if f in df.columns]
        if not factors_to_analyze:
            print(f"    ⚠ 无可用因子，跳过")
            return

        # 缓存（用于跨品种相关性分析）
        self.all_features_dfs[symbol_name] = df

        # 品种输出目录
        sym_dir = self.output_dir / symbol_name
        sym_dir.mkdir(exist_ok=True)

        label_bars         = self.fe_cfg.get('label_bars', 6)
        ic_forward_periods = self.fa_cfg.get('ic_forward_periods', [1, 3, 6, 12])
        use_ror_norm       = self.fa_cfg.get('use_ror_norm', True)
        norm_window        = self.fa_cfg.get('ols_norm_window', 200)
        window_days        = self.fa_cfg.get('window_days', 20)

        # 分层分析配置
        qa_cfg         = self.fa_cfg.get('quantile_analysis', {})
        qa_n_quantiles = qa_cfg.get('n_quantiles',  5)
        qa_rolling_win = qa_cfg.get('rolling_window', 500)
        qa_min_samples = qa_cfg.get('min_samples',  20)

        # ── 时段模板解析（品种级优先，支持内置名称或自定义 list） ──
        raw_template = sym_cfg.get('session_template', None)

        if raw_template is None:
            warnings.warn(
                f"[{symbol_name}] 未配置 session_template，"
                f"时段分析将使用默认模板 'night_2300'。\n"
                f"可选内置模板：{list(SESSION_TEMPLATES.keys())}\n"
                f"也可在 symbols 配置中直接传入自定义 list[dict]。"
            )
            qa_slots = SESSION_TEMPLATES['night_2300']

        elif isinstance(raw_template, list):
            # 用户直接在 config 中传入自定义 slots 列表
            qa_slots = raw_template

        elif isinstance(raw_template, str):
            qa_slots = SESSION_TEMPLATES.get(raw_template)
            if qa_slots is None:
                warnings.warn(
                    f"[{symbol_name}] session_template='{raw_template}' "
                    f"不在内置模板中，时段分析将使用默认模板 'night_2300'。\n"
                    f"可选内置模板：{list(SESSION_TEMPLATES.keys())}"
                )
                qa_slots = SESSION_TEMPLATES['night_2300']

        else:
            warnings.warn(
                f"[{symbol_name}] session_template 类型非法"
                f"（需为 str 或 list，实际为 {type(raw_template).__name__}），"
                f"时段分析将使用默认模板 'night_2300'。"
            )
            qa_slots = SESSION_TEMPLATES['night_2300']

        analyzer = SingleFactorAnalyzer(
            symbol_name=symbol_name,
            label_bars=label_bars,
            ic_forward_periods=ic_forward_periods,
            use_ror_norm=use_ror_norm,
            tick_size=tick_size,
            output_dir=sym_dir,
            norm_window=norm_window,
            window_days=window_days,
            winsorize_ror=self.fa_cfg.get('winsorize_ror', True),
            winsorize_quantiles=tuple(self.fa_cfg.get('winsorize_quantiles', [0.01, 0.99])),
        )

        ic_rows  = []
        ols_rows = []
        qa_rows  = []
        prop_rows = []
        bt_rows  = []

        for factor in factors_to_analyze:
            print(f"    分析因子: {factor} ...", end=' ', flush=True)

            # IC 分析
            ic_stat = analyzer.run_ic(df, factor)
            if ic_stat:
                ic_stat['symbol'] = symbol_name
                ic_rows.append(ic_stat)

            # OLS Beta 分析
            ols_stat = analyzer.run_ols(df, factor)
            if ols_stat:
                ols_stat['symbol'] = symbol_name
                ols_rows.append(ols_stat)

            # 因子分层分析
            qa_result = analyzer.run_quantile_analysis(
                df, factor,
                n_quantiles    = qa_n_quantiles,
                rolling_window = qa_rolling_win,
                min_samples    = qa_min_samples,
                session_slots  = qa_slots,
            )
            qa_monotone = float('nan')
            qa_spread   = float('nan')
            if qa_result:
                qa_monotone = qa_result.get('monotone_r', float('nan'))
                qa_spread   = qa_result.get('ls_spread',  float('nan'))
                qa_rows.append({
                    'symbol':          symbol_name,
                    'factor':          factor,
                    'ls_spread':       qa_spread,
                    'monotone_r':      qa_monotone,
                    'n_bars_analyzed': qa_result.get('n_bars_analyzed', 0),
                    'n_quantiles':     qa_n_quantiles,
                    'rolling_window':  qa_rolling_win,
                    'ror_col':         qa_result.get('ror_col', ''),
                })

            # 因子诊断
            prop_result = analyzer.run_factor_properties(df, factor)
            acf1_val   = float('nan')
            hl_val     = float('nan')
            if prop_result:
                acf1_val = prop_result.get('acf1_win_mean', float('nan'))
                hl_val   = prop_result.get('half_life_bars',  float('nan'))
                prop_rows.append({
                    'symbol':          symbol_name,
                    'factor':          factor,
                    'acf1_win_mean':  acf1_val,
                    'half_life_bars':  hl_val,
                    'half_life_minutes': prop_result.get('half_life_minutes', float('nan')),
                    'ess_ratio':       prop_result.get('ess_ratio',       float('nan')),
                    'lb_p':            prop_result.get('lb_p',            float('nan')),
                    'jb_p':            prop_result.get('jb_p',            float('nan')),
                    'skew':            prop_result.get('skew',            float('nan')),
                    'kurt':            prop_result.get('kurt',            float('nan')),
                    'adf_p_mean':      prop_result.get('adf_p_mean',      float('nan')),
                    'kpss_p_mean':     prop_result.get('kpss_p_mean',     float('nan')),
                    'stationarity':    prop_result.get('stationarity',    ''),
                    'n_valid':         prop_result.get('n_valid',         0),
                    'null_rate':       prop_result.get('null_rate',       float('nan')),
                })

            # 单因子回测
            bt_result = {}
            bt_cfg = self.fa_cfg.get('backtest', {})
            if bt_cfg.get('enabled', False):
                bt_result = analyzer.run_backtest(
                    df, factor,
                    thresholds     = [tuple(t) for t in bt_cfg.get(
                                          'thresholds', [[0.2, 0.8]])],
                    rolling_window = bt_cfg.get('rolling_window', 500),
                    min_samples    = bt_cfg.get('min_samples',    20),
                    slippage_ticks = bt_cfg.get('slippage_ticks', 1.5),
                )
                if bt_result:
                    bt_rows.append({
                        'symbol':          symbol_name,
                        'factor':          factor,
                        'best_threshold':  str(bt_result['best_threshold']),
                        'sharpe_net':      round(bt_result['best_sharpe_net'],    4),
                        'annual_net_bp':   round(bt_result['best_annual_net_bp'], 2),
                        'max_dd_net_bp':   round(bt_result['best_max_dd_net_bp'], 2),
                    })

            hl_str   = f'{hl_val:.1f}bar' if np.isfinite(hl_val) else 'N/A'
            bt_sharpe = bt_result.get('best_sharpe_net', float('nan'))
            bt_s_str  = f'{bt_sharpe:.3f}' if np.isfinite(bt_sharpe) else 'N/A'
            print(f"IC={ic_stat.get('main_ic_mean', float('nan')):.4f}  "
                  f"Beta={ols_stat.get('main_beta', float('nan')):.5f}  "
                  f"QA_spread={qa_spread:.5f}  "
                  f"QA_mono={qa_monotone:.3f}  "
                  f"ACF1={acf1_val:.3f}  HL={hl_str}  "
                  f"BT_Sharpe={bt_s_str}")

        # 保存品种级汇总
        if ic_rows:
            pl.DataFrame(ic_rows).write_csv(sym_dir / 'ic' / 'ic_stats_all_factors.csv')
        if ols_rows:
            pl.DataFrame(ols_rows).write_csv(sym_dir / 'ols' / 'ols_stats_all_factors.csv')
        if qa_rows:
            qa_dir = sym_dir / 'quantile'
            qa_dir.mkdir(exist_ok=True)
            pl.DataFrame(qa_rows).write_csv(qa_dir / 'quantile_stats_all_factors.csv')
        if prop_rows:
            prop_dir = sym_dir / 'properties'
            prop_dir.mkdir(exist_ok=True)
            pl.DataFrame(prop_rows).write_csv(prop_dir / 'factor_properties_all.csv')
        if bt_rows:
            bt_dir = sym_dir / 'backtest'
            bt_dir.mkdir(exist_ok=True)
            pl.DataFrame(bt_rows).write_csv(bt_dir / 'backtest_all_factors.csv')

        # 绘制品种级 IC 热图（所有因子 × 所有预测周期）
        if ic_rows:
            self._plot_ic_heatmap(ic_rows, ic_forward_periods,
                                  sym_dir / 'ic' / 'figures' / 'ic_heatmap.png',
                                  symbol_name)

        # 汇总到全局
        self.ic_summary.extend(ic_rows)
        self.ols_summary.extend(ols_rows)
        self.qa_summary.extend(qa_rows)
        self.prop_summary.extend(prop_rows)
        self.bt_summary.extend(bt_rows)

    def _plot_ic_heatmap(self, ic_rows: list,
                         periods: List[int],
                         save_path: Path,
                         symbol_name: str):
        """绘制 因子(行) × 预测周期(列) 的 ICIR 热图"""
        factors = [r['factor'] for r in ic_rows]
        cols    = [f'period_{p}_icir' for p in periods]
        data = np.array([[r.get(c, np.nan) for c in cols] for r in ic_rows])

        fig, ax = plt.subplots(figsize=(max(6, len(periods) * 1.2),
                                        max(4, len(factors) * 0.4)))
        sns.heatmap(data, ax=ax,
                    annot=True, fmt='.2f', cmap='RdYlGn', center=0,
                    linewidths=0.5,
                    xticklabels=[f'{p}棒\n{p*5}min' for p in periods],
                    yticklabels=factors,
                    cbar_kws={'label': '年化ICIR', 'shrink': 0.8},
                    annot_kws={'size': 8})
        ax.set_title(f'{symbol_name} | 因子 × 预测周期 ICIR 热图\n'
                     f'（日度全量采样，年化ICIR，含t统计量）', fontsize=9)
        fig.tight_layout()
        fig.savefig(save_path, bbox_inches='tight', dpi=150)
        plt.close(fig)

    # ──────────────────────────────────────────────────────────

    def _aggregate_summary(self):
        """汇总所有品种的 IC、OLS 和分层分析结果，输出跨品种排名表"""
        if not self.ic_summary and not self.ols_summary:
            return

        # IC 汇总
        if self.ic_summary:
            ic_df = pl.DataFrame(self.ic_summary)
            # 按因子分组，计算跨品种均值
            group_cols = [c for c in ic_df.columns
                          if c not in ('factor', 'symbol')]
            agg_exprs = [pl.col(c).mean().alias(f'avg_{c}')
                         for c in group_cols if ic_df[c].dtype in
                         (pl.Float64, pl.Float32, pl.Int64, pl.Int32)]
            if agg_exprs:
                ic_agg = (ic_df.group_by('factor')
                          .agg(agg_exprs +
                               [pl.len().alias('n_symbols_analyzed')])
                          .sort('avg_main_icir', descending=True,
                                nulls_last=True))
                ic_agg.write_csv(self.output_dir / 'ic_summary_rank.csv')
                print(f"  IC排名已保存: {self.output_dir / 'ic_summary_rank.csv'}")

        # OLS Beta 汇总：按因子分组，跨品种取均值
        if self.ols_summary:
            ols_df = pl.DataFrame(self.ols_summary)
            ols_group_cols = [c for c in ols_df.columns
                              if c not in ('factor', 'symbol')]
            ols_agg_exprs = [pl.col(c).mean().alias(f'avg_{c}')
                             for c in ols_group_cols if ols_df[c].dtype in
                             (pl.Float64, pl.Float32, pl.Int64, pl.Int32)]
            if ols_agg_exprs:
                ols_agg = (ols_df.group_by('factor')
                           .agg(ols_agg_exprs +
                                [pl.len().alias('n_symbols_analyzed')])
                           .sort('avg_main_t_stat', descending=True,
                                 nulls_last=True))
                ols_agg.write_csv(self.output_dir / 'ols_summary_rank.csv')
                print(f"  OLS排名已保存: {self.output_dir / 'ols_summary_rank.csv'}")

        # 合并汇总排名（IC + OLS）
        if self.ic_summary and self.ols_summary:
            try:
                ic_rank  = pl.read_csv(self.output_dir / 'ic_summary_rank.csv')
                ols_rank = pl.read_csv(self.output_dir / 'ols_summary_rank.csv')
                merged   = ic_rank.join(ols_rank, on='factor', how='outer', coalesce=True)
                if self.qa_summary:
                    qa_rank = pl.read_csv(self.output_dir / 'quantile_summary_rank.csv')
                    merged  = merged.join(qa_rank, on='factor', how='outer', coalesce=True)
                merged.write_csv(self.output_dir / 'factor_summary_rank.csv')
                print(f"  综合排名已保存: {self.output_dir / 'factor_summary_rank.csv'}")
            except Exception:
                pass

        # 分层分析跨品种汇总
        if self.qa_summary:
            qa_df = pl.DataFrame(self.qa_summary)
            qa_agg_exprs = [
                pl.col(c).mean().alias(f'avg_{c}')
                for c in ['ls_spread', 'monotone_r', 'n_bars_analyzed']
                if c in qa_df.columns
            ]
            if qa_agg_exprs:
                qa_agg = (
                    qa_df.group_by('factor')
                    .agg(qa_agg_exprs + [pl.len().alias('n_symbols_analyzed')])
                    .sort('avg_monotone_r', descending=True, nulls_last=True)
                )
                qa_agg.write_csv(self.output_dir / 'quantile_summary_rank.csv')
                print(f"  分层分析排名已保存: {self.output_dir / 'quantile_summary_rank.csv'}")

        # 因子诊断跨品种汇总
        if self.prop_summary:
            prop_df = pl.DataFrame(self.prop_summary)
            prop_num_cols = [c for c in prop_df.columns
                             if c not in ('factor', 'symbol', 'stationarity')
                             and prop_df[c].dtype in
                             (pl.Float64, pl.Float32, pl.Int64, pl.Int32)]
            prop_agg_exprs = [pl.col(c).mean().alias(f'avg_{c}')
                              for c in prop_num_cols]
            # 平稳性结论：取最常见的值
            prop_agg_exprs.append(
                pl.col('stationarity').mode().first().alias('mode_stationarity'))
            if prop_agg_exprs:
                prop_agg = (
                    prop_df.group_by('factor')
                    .agg(prop_agg_exprs + [pl.len().alias('n_symbols_analyzed')])
                    .sort('avg_acf1_win_mean', descending=False, nulls_last=True)
                )
                prop_agg.write_csv(self.output_dir / 'prop_summary_rank.csv')
                print(f"  因子诊断排名已保存: {self.output_dir / 'prop_summary_rank.csv'}")

        # 回测跨品种汇总
        if self.bt_summary:
            bt_df = pl.DataFrame(self.bt_summary)
            bt_agg = (
                bt_df.group_by('factor')
                .agg([
                    pl.col('sharpe_net').mean().alias('avg_sharpe_net'),
                    pl.col('annual_net_bp').mean().alias('avg_annual_net_bp'),
                    pl.col('max_dd_net_bp').mean().alias('avg_max_dd_net_bp'),
                    pl.len().alias('n_symbols_analyzed'),
                ])
                .sort('avg_sharpe_net', descending=True, nulls_last=True)
            )
            bt_agg.write_csv(self.output_dir / 'backtest_summary_rank.csv')
            print(f"  回测排名已保存: {self.output_dir / 'backtest_summary_rank.csv'}")

    # ──────────────────────────────────────────────────────────

    def _run_cross_factor(self):
        """合并所有品种的数据，运行跨因子相关性分析"""
        if not self.all_features_dfs:
            return

        # 所有品种数据纵向拼接（非重叠采样后合并，统计量更稳健）
        all_dfs = []
        for sym_name, df in self.all_features_dfs.items():
            available = [f for f in self.factors if f in df.columns]
            if available:
                all_dfs.append(df.select(available))

        if not all_dfs:
            return

        combined = pl.concat(all_dfs, how='diagonal')

        cross_dir = self.output_dir / 'cross_factor'
        cross_dir.mkdir(exist_ok=True)

        analyzer = CrossFactorAnalyzer(
            output_dir=cross_dir,
            correlation_threshold=self.fa_cfg.get('correlation_threshold', 0.7),
            method=self.fa_cfg.get('correlation_method', 'spearman'),
        )
        result = analyzer.run(
            combined,
            self.factors,
            self.feature_groups_info,
            label_bars=self.fe_cfg.get('label_bars', 6),
        )
        print(f"  相关性分析完成: {result.get('n_factors', 0)} 个因子, "
              f"{result.get('n_redundant_pairs', 0)} 对高相关因子")

    # ──────────────────────────────────────────────────────────

    def _generate_report(self):
        """生成文字汇总报告"""
        label_bars = self.fe_cfg.get('label_bars', 6)
        report = f"""
{'='*80}
单因子分析汇总报告
{'='*80}
分析时间: {self.timestamp}
分析因子: {self.factors}
分析品种: {[get_symbol_name(s['code']) for s in self.symbols]}
数据范围: {self.ds_cfg.get('train_start_year')} ~ {self.ds_cfg.get('train_end_year')}
IC方法: 日度全量采样（每日全部bar参与，不间隔），年化ICIR + t统计量
        有效N_eff ≈ 63/{label_bars} ≈ {63/label_bars:.1f}/日
{'='*80}
"""
        # IC 排名
        ic_rank_file = self.output_dir / 'ic_summary_rank.csv'
        if ic_rank_file.exists():
            ic_rank = pl.read_csv(ic_rank_file)
            report += "\nIC 排名（按平均年化ICIR降序）:\n"
            report += f"{'因子':<25} {'平均ICIR':>10} {'平均t值':>10} {'平均IC均值':>12} {'平均胜率':>10}\n"
            report += "-" * 70 + "\n"
            for row in ic_rank.head(20).iter_rows(named=True):
                report += (
                    f"{row.get('factor',''):<25} "
                    f"{row.get('avg_main_icir',   float('nan')):>10.4f} "
                    f"{row.get('avg_main_t_stat', float('nan')):>10.3f} "
                    f"{row.get('avg_main_ic_mean',float('nan')):>12.6f} "
                    f"{row.get('avg_main_win_rate',float('nan')):>10.2%}\n"
                )

        # 分层分析排名
        qa_rank_file = self.output_dir / 'quantile_summary_rank.csv'
        if qa_rank_file.exists():
            qa_rank = pl.read_csv(qa_rank_file)
            qa_cfg  = self.fa_cfg.get('quantile_analysis', {})
            report += (
                f"\n因子分层分析排名（按单调性系数降序，n_quantiles="
                f"{qa_cfg.get('n_quantiles', 5)}，"
                f"rolling_window={qa_cfg.get('rolling_window', 500)}）:\n"
            )
            report += f"{'因子':<25} {'单调性(Spearman)':>18} {'多空价差(Q高-Q低)':>18} {'品种数':>8}\n"
            report += "-" * 72 + "\n"
            for row in qa_rank.iter_rows(named=True):
                report += (
                    f"{row.get('factor',''):<25} "
                    f"{row.get('avg_monotone_r',    float('nan')):>18.4f} "
                    f"{row.get('avg_ls_spread',     float('nan')):>18.6f} "
                    f"{row.get('n_symbols_analyzed', 0):>8}\n"
                )

        # 因子诊断概览
        prop_rank_file = self.output_dir / 'prop_summary_rank.csv'
        if prop_rank_file.exists():
            prop_rank = pl.read_csv(prop_rank_file)
            report += "\n因子诊断概览（按日均ACF(1)升序，ACF越小信号越抖）:\n"
            report += (f"{'因子':<25} {'日均ACF(1)':>12} {'半衰期(bar)':>12}"
                       f" {'ESS/N':>8} {'JB-p':>10} {'平稳性':>20} {'品种数':>8}\n")
            report += "-" * 100 + "\n"
            for row in prop_rank.iter_rows(named=True):
                hl  = row.get('avg_half_life_bars', float('nan'))
                ess = row.get('avg_ess_ratio',      float('nan'))
                jbp = row.get('avg_jb_p',           float('nan'))
                report += (
                    f"{row.get('factor',''):<25} "
                    f"{row.get('avg_acf1_win_mean', float('nan')):>12.4f} "
                    f"{hl:>12.1f} "
                    f"{ess:>8.3f} "
                    f"{jbp:>10.4f} "
                    f"{row.get('mode_stationarity', 'N/A'):>20} "
                    f"{row.get('n_symbols_analyzed', 0):>8}\n"
                )

        # 回测排名
        bt_rank_file = self.output_dir / 'backtest_summary_rank.csv'
        if bt_rank_file.exists():
            bt_rank = pl.read_csv(bt_rank_file)
            bt_cfg  = self.fa_cfg.get('backtest', {})
            report += (
                f"\n回测排名（按平均Sharpe降序，最优阈值，"
                f"slippage={bt_cfg.get('slippage_ticks', 1.5)}跳）:\n"
            )
            report += (f"{'因子':<25} {'平均Sharpe':>12} "
                       f"{'平均年化收益(bp)':>18} {'平均最大回撤(bp)':>18} {'品种数':>8}\n")
            report += "-" * 84 + "\n"
            for row in bt_rank.iter_rows(named=True):
                report += (
                    f"{row.get('factor',''):<25} "
                    f"{row.get('avg_sharpe_net',    float('nan')):>12.4f} "
                    f"{row.get('avg_annual_net_bp', float('nan')):>18.1f} "
                    f"{row.get('avg_max_dd_net_bp', float('nan')):>18.1f} "
                    f"{row.get('n_symbols_analyzed', 0):>8}\n"
                )

        report += f"\n{'='*80}\n输出目录: {self.output_dir}\n{'='*80}\n"
        report_file = self.output_dir / f'report_{self.timestamp}.txt'
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)
        print(report)
        print(f"✓ 报告已保存: {report_file}")


# ==============================================================
# 4. FactorLab
# ==============================================================

class FactorLab:
    """
    Notebook 交互式因子实验室

    设计原则
    --------
    - 只做单因子测试，完全隔绝 train_end_year 之后的数据
    - 只向用户提供原始多周期 K 线（bars 字典），由用户自行设计因子函数
    - 复用 SingleFactorAnalyzer 的 IC 分析和回测基础设施（inline 模式）
    - 初始化只走最小依赖链（跳过特征计算），速度快

    数据隔离
    --------
    self.bars 包含：year <= train_end_year 的数据（含预热区，屏蔽测试集）
    IC / 回测分析：仅在 [train_start_year, train_end_year] 上运行

    快速使用示例
    ------------
    lab = FactorLab("SHFE_au")            # 初始化一次

    def my_factor(bars: dict) -> pl.Series:
        bar5m = bars["5m"]
        # 利用 continuous_session_id 做 session 内 rolling，避免跨夜污染
        return bar5m["close"].rolling_mean(20).over("continuous_session_id")

    stats = lab.ic_only(my_factor, "ma20")        # IC 分析
    stats = lab.ols_only(my_factor, "ma20")        # OLS Beta 分析
    stats = lab.quantile_only(my_factor, "ma20")   # 分层收益分析
    stats = lab.horizon_only(my_factor, "ma20")    # Horizon 演化分析
    stats = lab.cumreturn_only(my_factor, "ma20")  # 累计收益曲线

    bars["5m"] 列说明
    -----------------
    原始 OHLCV : open, high, low, close, volume, open_oi, close_oi
    时间字段   : datetime, datetime_nano, trading_date, bar_end_ts
    结构性元数据（FactorLab 新增）:
        continuous_session_id : 连续交易时段编号，用于 .over() session 内计算
        cum_price_gap         : 累计跳空金额，用于去跳空价格：close - cum_price_gap
        bar_idx_in_session    : 时段内 bar 序号（从 1 开始），用于开收盘逻辑

    factor_func 协议
    ----------------
    def factor_func(bars: dict) -> pl.Series:
        # bars = {"1m": df, "5m": df, "30m": df, "1d": df}
        # 返回 pl.Series，长度必须等于 len(bars["5m"])，按行号对齐
        ...
    """

    def __init__(self, symbol: str, config_path: str = "config.json"):
        """
        初始化 FactorLab。

        Parameters
        ----------
        symbol      : 品种简称，如 "SHFE_au"（与 config.json 中 get_symbol_name 结果一致）
        config_path : 配置文件路径
        """
        # ── 1. 读取配置 ───────────────────────────────────────────
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)

        symbols      = parse_symbols(config)
        ds_cfg       = config.get('data_split', {})
        fe_cfg       = config.get('feature_engineering', {})
        fa_cfg       = config.get('factor_analysis', {})
        data_cfg     = config.get('data_source', {})
        output_cfg   = config.get('output', {})

        self._train_start_year   = ds_cfg.get('train_start_year', 2022)
        self._train_end_year     = ds_cfg.get('train_end_year',   2024)
        self._label_bars         = fe_cfg.get('label_bars',       6)
        self._ic_periods         = fa_cfg.get('ic_forward_periods', [1, 3, 6, 12])
        self._use_ror_norm       = fa_cfg.get('use_ror_norm',     True)
        self._winsorize_ror      = fa_cfg.get('winsorize_ror',    True)
        self._winsorize_quantiles= tuple(fa_cfg.get('winsorize_quantiles', [0.01, 0.99]))
        self._fe_cfg           = fe_cfg   # 保存，供 check_leakage 重建 FeatureEngineer 使用
        self._fa_cfg           = fa_cfg   # 保存，供 backtest_only 读取 backtest 配置

        # ── 2. 找到当前 symbol 的 tick_size 与 session_template ──
        self._symbol_name     = symbol
        self._tick_size       = 1.0   # 默认值
        self._session_template = None  # 默认值，从 config.json 品种配置读取
        for sym in symbols:
            if get_symbol_name(sym['code']) == symbol:
                self._tick_size        = sym.get('tick_size', 1.0)
                self._symbol_code      = sym['code']
                self._session_template = sym.get('session_template', None)
                break
        else:
            raise ValueError(
                f"品种 '{symbol}' 在 config.json 的 symbols 列表中未找到。\n"
                f"可用品种: {[get_symbol_name(s['code']) for s in symbols]}")

        # ── 3. 加载 1m parquet（优先从缓存读取）──────────────────
        temp_dir = Path(output_cfg.get('temp_data_dir', './temp_data'))
        temp_dir.mkdir(parents=True, exist_ok=True)

        safe_code = self._symbol_code.replace(".", "_").replace("@", "_")
        cached    = list(temp_dir.glob(f"{safe_code}_1m_*.parquet"))

        if cached:
            print(f"加载缓存数据: {cached[0].name}")
            bar1min = pl.read_parquet(cached[0])
        else:
            print(f"未找到缓存，下载数据中...")
            import datetime as _dt
            start_date = _dt.datetime.strptime(
                data_cfg.get('start_date', '2021-01-01'), '%Y-%m-%d').date()
            end_date = _dt.datetime.strptime(
                data_cfg.get('end_date', '2025-12-31'), '%Y-%m-%d').date()
            batch_download(
                symbols=[self._symbol_code],
                start_date=start_date,
                end_date=end_date,
                periods=['1m'],
                save_format='parquet',
                save_dir=str(temp_dir),
                source_dir=data_cfg.get('source_dir', '')
            )
            cached = list(temp_dir.glob(f"{safe_code}_1m_*.parquet"))
            if not cached:
                raise FileNotFoundError(f"数据下载失败，未找到 {safe_code}_1m_*.parquet")
            bar1min = pl.read_parquet(cached[0])

        # ── 4. 轻量特征工程（跳过特征，只算基础设施列）────
        engineer = FeatureEngineer(
            label_bars      = self._label_bars,
            label_threshold = fe_cfg.get('label_threshold', 0.5),
            atr_long_window = fe_cfg.get('atr_long_window', 400),
        )
        engineer._build_lab_infra(bar1min, verbose=True)

        # ── 5. 过滤：屏蔽 train_end_year 之后的数据 ───────────────
        def _filter_bars(df: pl.DataFrame) -> pl.DataFrame:
            """保留 year <= train_end_year 的行（含预热区，屏蔽测试集）"""
            if 'datetime' in df.columns:
                return df.filter(
                    pl.col('datetime').dt.year() <= self._train_end_year)
            elif 'trading_date' in df.columns:
                return df.filter(
                    pl.col('trading_date').cast(pl.Date).dt.year()
                    <= self._train_end_year)
            return df

        self._bars = {
            tf: _filter_bars(df)
            for tf, df in engineer.bars.items()
        }
        self._infra_df = engineer.infra_df.filter(
            pl.col('datetime').dt.year() <= self._train_end_year
        )

        # ── 6. 一致性校验：bars["5m"] 行数必须等于 infra_df 行数 ─
        n_bars   = len(self._bars["5m"])
        n_infra  = len(self._infra_df)
        if n_bars != n_infra:
            raise RuntimeError(
                f"内部错误：bars['5m'] 行数 ({n_bars}) ≠ infra_df 行数 ({n_infra})。\n"
                f"请检查数据管线是否存在行数不一致问题。")

        print(f"\nFactorLab 初始化完成")
        print(f"  品种         : {self._symbol_name}")
        print(f"  bars['5m']   : {n_bars:,} 行（含预热区，year <= {self._train_end_year}）")
        print(f"  分析样本     : year in [{self._train_start_year}, {self._train_end_year}]")
        print(f"  tick_size    : {self._tick_size}")
        print(f"  label_bars   : {self._label_bars} 根（{self._label_bars * 5} 分钟）")
        tpl = self._session_template or '未配置（quantile_only 时段热力图将跳过）'
        print(f"  session_template: {tpl}")

        # 保存原始1m数据，供 check_leakage 泄露检测使用
        self._bar1min = bar1min

    # ──────────────────────────────────────────────────────────
    # 公开接口
    # ──────────────────────────────────────────────────────────

    @property
    def bars(self) -> Dict[str, pl.DataFrame]:
        """
        多周期 K 线字典，已屏蔽 train_end_year 之后的数据，含预热区。

        bars["5m"] 额外包含结构性元数据：
            continuous_session_id, cum_price_gap, bar_idx_in_session
        """
        return self._bars

    def ic_only(self, factor_func, name: str,
                ic_periods: Optional[List[int]] = None,
                window_days: int = 20) -> Dict:
        """
        只运行 IC 分析（非重叠窗口 + ICIR + t统计量），inline 显示图表。

        window_days : 每个非重叠窗口的交易日数，默认20（约1个月）
        """
        az = self._make_analyzer(ic_periods=ic_periods, window_days=window_days)
        df = self._apply_factor(factor_func, name)
        return az.run_ic(df, name)

    def ols_only(self, factor_func, name: str,
                 norm_window: int = 200,
                 window_days: int = 20) -> Dict:
        """
        只运行 Rolling OLS Beta 分析，inline 显示窗口 Beta 时序图 + 衰减图。

        norm_window : 滚动 z-score 回望窗口（bar数），默认200（≈2个交易日）
        window_days : 每个非重叠 OLS 窗口的交易日数，默认20（约1个月）
        """
        df = self._apply_factor(factor_func, name)
        return self._make_analyzer(norm_window=norm_window,
                                   window_days=window_days).run_ols(df, name)

    def quantile_only(self, factor_func, name: str,
                      n_quantiles: int = 5,
                      rolling_window: int = 500,
                      min_samples: int = 20,
                      session_template: Optional[str] = None,
                      session_slots: Optional[List[Dict]] = None,
                      use_ror_norm: bool = False,
                      correct_autocorr: bool = True) -> Dict:
        """
        只运行因子分层收益分析，inline 显示分层柱状图、箱线图和时段热力图。

        Parameters
        ----------
        factor_func      : callable (bars: dict) -> pl.Series
        name             : 因子名称
        n_quantiles      : 分层数，默认5
        rolling_window   : 滚动分位数回望窗口（bar数），默认500
        min_samples      : 滚动分位数最小样本数，默认20
        session_template : 内置模板名（如 "night_0230" / "stock_index" / "bond"）
                           或自定义 list[dict]；优先级高于 session_slots。
                           内置模板：day_only / night_2300 / night_0100 /
                                     night_0230 / stock_index / bond
                           未传入时自动从 config.json 品种配置读取。
        session_slots    : 时段定义列表，session_template 为 None 时生效；
                           两者均为 None 则跳过时段热力图。
                           例如：[{"name": "开盘段", "start": "09:00", "end": "10:00"}, ...]
        use_ror_norm     : False（默认）使用原始收益率，图表以基点(bp)为单位；
                           True 使用 ATR 归一化收益率，无量纲。
        correct_autocorr : True（默认）使用日度聚合 t 检验：先按交易日聚合均值，
                           再对日均值序列做 t 检验，有效自由度 = 交易日数（≥20才计算）。
                           False 使用朴素 t 检验（bar级别），未校正日内自相关。
        """
        # 未显式指定 session_template 时，回退到品种级配置（来自 config.json）
        effective_template = (session_template if session_template is not None
                              else self._session_template)
        df = self._apply_factor(factor_func, name)
        return self._make_analyzer().run_quantile_analysis(
            df, name,
            n_quantiles=n_quantiles,
            rolling_window=rolling_window,
            min_samples=min_samples,
            session_template=effective_template,
            session_slots=session_slots,
            use_ror_norm=use_ror_norm,
            correct_autocorr=correct_autocorr,
        )

    def horizon_only(self, factor_func, name: str,
                     horizons: Optional[List[int]] = None,
                     n_quantiles: int = 5,
                     rolling_window: int = 500,
                     min_samples: int = 20,
                     correct_autocorr: bool = True,
                     use_daily_mean: bool = False) -> Dict:
        """
        因子预测能力随持仓 Horizon 的演化分析。

        将因子值按滚动分位数分为 Q1~Qn 层，对每个持仓期 h（bar 数），
        统计各层均值收益（bp）、胜率、显著性，以及多空价差的双样本 t 检验。

        Parameters
        ----------
        factor_func      : callable (bars: dict) -> pl.Series
        name             : 因子名称
        horizons         : 持仓期列表（bar数）。
                           None（默认）→ 自动从 infra_df 中检测所有可用的
                           ror_future_Nb 列，提取其中的 N 序列。
                           示例：[1, 2, 3, 6, 12] 或 list(range(1, 13))。
        n_quantiles      : 分层数，默认5
        rolling_window   : 滚动分位数回望窗口（bar数），默认500
        min_samples      : 滚动分位数最小样本数，默认20
        correct_autocorr : True（默认）日度聚合 t 检验；
                           False 朴素 t 检验（未校正日内自相关）。
        use_daily_mean   : False（默认）ax1 折线使用 bar 等权均值；
                           True 使用日等权均值，与 cumreturn_only 的口径完全一致。

        Returns
        -------
        dict 包含：
          horizons              : 有效 horizon 列表（bar数）
          minutes               : 对应分钟数
          use_daily_mean        : 当前使用的均值口径
          layer_stats           : Dict[horizon → Dict[layer → stats]]
          ls_spread             : Dict[horizon → float]（Q_n - Q_1，bp）
          ls_t_stat             : Dict[horizon → float]（双样本 Welch t）
          ls_p_value            : Dict[horizon → float]
          peak_ls_horizon_bars  : 多空价差绝对值最大的 horizon
          peak_ls_horizon_min   : 对应分钟数
        """
        df = self._apply_factor(factor_func, name)

        # 自动检测可用 horizon
        if horizons is None:
            import re
            horizons = sorted([
                int(m.group(1))
                for c in df.columns
                if (m := re.match(r'ror_future_(\d+)b', c))
            ])
        if not horizons:
            print(f"⚠️  infra_df 中未找到任何 ror_future_Nb 列，"
                  f"请确认 FeatureEngineer 的 horizon_bars 配置。")
            return {}

        return self._make_analyzer().run_horizon_analysis(
            df, name,
            horizons=horizons,
            n_quantiles=n_quantiles,
            rolling_window=rolling_window,
            min_samples=min_samples,
            correct_autocorr=correct_autocorr,
            use_daily_mean=use_daily_mean,
        )

    def cumreturn_only(self, factor_func, name: str,
                       horizon: Optional[int] = None,
                       n_quantiles: int = 5,
                       rolling_window: int = 500,
                       min_samples: int = 20,
                       use_daily_mean: bool = True) -> Dict:
        """
        Q5 / Q1 / Q5-Q1 分层累计收益率曲线（单利）。

        展示因子在不同年份的预测能力：
        - 子图1：Q5（多头）、Q1（空头取负）、Q5-Q1（多空组合）的累积收益曲线（bp）
                 附有年份边界线和多空组合的最大回撤填充区域
        - 子图2：逐年收益贡献柱状图

        Parameters
        ----------
        factor_func    : callable (bars: dict) -> pl.Series
        name           : 因子名称
        horizon        : 持仓期（bar数），None → 使用 label_bars（默认）
        n_quantiles    : 分层数，默认5
        rolling_window : 滚动分位数回望窗口（bar数），默认500
        min_samples    : 滚动分位数最小样本数，默认20
        use_daily_mean : True（默认）日等权，inner join 配对日度聚合，
                         与 horizon_only(use_daily_mean=True) 口径一致；
                         False bar等权，outer join 保留所有天（缺失填0），
                         与 horizon_only(use_daily_mean=False) 口径一致。

        Returns
        -------
        dict 包含：
          use_daily_mean   : 当前使用的均值口径
          trading_dates    : 交易日列表
          cumret_q5        : Q5 单利累积收益（bp）
          cumret_q1_short  : Q1 做空单利累积收益（bp）
          cumret_ls        : 多空组合单利累积收益（无摩擦，bp）
          cumret_ls_net    : 多空组合单利累积收益（扣摩擦后，bp）
          annual_stats     : Dict[year → {'q5': bp, 'q1_short': bp, 'ls': bp}]
          win_rate_q5      : Q5 层 bar 等权胜率
          win_rate_q1      : Q1 层 bar 等权胜率
          friction_bp      : 每期往返摩擦成本（bp）
          max_drawdown_ls  : 多空组合最大回撤（无摩擦，bp）
          sharpe_annual_ls : 多空组合年化 Sharpe（无摩擦）
          n_days / n_bars  : 有效观测数
        """
        h = horizon if horizon is not None else self._label_bars
        df = self._apply_factor(factor_func, name)
        return self._make_analyzer().run_cumreturn(
            df, name,
            horizon=h,
            n_quantiles=n_quantiles,
            rolling_window=rolling_window,
            min_samples=min_samples,
            use_daily_mean=use_daily_mean,
        )

    # ──────────────────────────────────────────────────────────
    # 内部方法
    # ──────────────────────────────────────────────────────────

    def _apply_factor(self, factor_func, name: str) -> pl.DataFrame:
        """
        调用用户函数，校验返回值，将因子列注入 infra_df，
        过滤到分析样本范围 [train_start_year, train_end_year]。
        """
        # 调用用户函数
        try:
            result = factor_func(self._bars)
        except Exception as e:
            raise RuntimeError(f"调用 factor_func 时出错: {e}") from e

        # 类型检查
        if not isinstance(result, pl.Series):
            raise TypeError(
                f"factor_func 必须返回 pl.Series，"
                f"实际返回 {type(result).__name__}。\n"
                f"示例：return bars['5m']['close'].rolling_mean(20)")

        # 长度检查（行号对齐的前提）
        expected = len(self._bars["5m"])
        if len(result) != expected:
            raise ValueError(
                f"返回的 Series 长度 ({len(result)}) ≠ bars['5m'] 行数 ({expected})。\n"
                f"factor_func 必须基于 bars['5m'] 按行逐一计算并返回，不能过滤或采样。")

        # 将因子列注入 infra_df（行号对齐），过滤到样本内年份
        analysis_df = self._infra_df.with_columns(result.rename(name))
        analysis_df = analysis_df.filter(
            (pl.col('datetime').dt.year() >= self._train_start_year) &
            (pl.col('datetime').dt.year() <= self._train_end_year)
        )

        n_valid = analysis_df[name].drop_nulls().len()
        n_total = len(analysis_df)
        if n_valid < 100:
            print(f"⚠️  警告：因子 '{name}' 在分析样本内有效值仅 {n_valid} 个（共 {n_total} 行），"
                  f"可能是预热窗口过长或因子本身产生了大量 null。")

        return analysis_df

    def _make_analyzer(self,
                       ic_periods: Optional[List[int]] = None,
                       norm_window: Optional[int] = None,
                       window_days: Optional[int] = None,
                       ) -> 'SingleFactorAnalyzer':
        """创建 inline=True、output_dir=None 的 SingleFactorAnalyzer"""
        return SingleFactorAnalyzer(
            symbol_name        = self._symbol_name,
            label_bars         = self._label_bars,
            ic_forward_periods = ic_periods or self._ic_periods,
            use_ror_norm       = self._use_ror_norm,
            tick_size          = self._tick_size,
            output_dir         = None,    # 不写文件
            inline             = True,    # plt.show() inline
            norm_window        = norm_window  if norm_window  is not None else 200,
            window_days        = window_days  if window_days  is not None else 20,
            winsorize_ror      = self._winsorize_ror,
            winsorize_quantiles= self._winsorize_quantiles,
        )

    def properties_only(
        self,
        factor_func,
        name: str,
        max_lags: int = 40,
    ) -> Dict:
        """
        只运行因子诊断分析，inline 显示自相关、分布和平稳性图表。

        不依赖收益率列，独立于 IC/OLS/分层分析运行。
        建议在 ic_only / ols_only 之前先运行，用于：
          - 评估因子自相关强度（半衰期）→ 指导滚动 z-score 窗口长度
          - 检验分布形态（正态性、厚尾）→ 决定是否需要截尾预处理
          - 验证平稳性 → 判断因子是否需要差分
          - 观察跨年稳定性 → 发现因子均值/方差的结构性漂移

        Parameters
        ----------
        factor_func : callable (bars: dict) -> pl.Series
        name        : 因子名称
        max_lags    : ACF 最大滞后阶数，默认 40 棒（200 分钟）

        Returns
        -------
        dict，参见 SingleFactorAnalyzer.run_factor_properties 的返回值说明
        """
        df = self._apply_factor(factor_func, name)
        return self._make_analyzer().run_factor_properties(df, name, max_lags=max_lags)

    def backtest_only(
        self,
        factor_func,
        name:           str,
        thresholds:     Optional[List[Tuple[float, float]]] = None,
        rolling_window: Optional[int]   = None,
        slippage_ticks: Optional[float] = None,
        min_samples:    Optional[int]   = None,
    ) -> Dict:
        """
        单因子持仓状态机回测，inline 模式显示净值曲线和绩效指标表。

        参数优先级：函数调用时显式传入 > config backtest 配置 > 内置默认值。

        Parameters
        ----------
        factor_func    : callable (bars: dict) -> pl.Series
        name           : 因子名称
        thresholds     : 阈值对列表，如 [(0.2,0.8),(0.25,0.75),(0.3,0.7)]；
                         None → 从 config backtest.thresholds 读取
        rolling_window : 滚动分位数回望窗口（bar数）；None → 从 config 读取
        slippage_ticks : 单边滑点（跳数）；None → 从 config 读取
        min_samples    : 滚动分位数最小样本数；None → 从 config 读取

        Returns
        -------
        dict，参见 SingleFactorAnalyzer.run_backtest 的返回值说明
        """
        bt_cfg = self._fa_cfg.get('backtest', {})

        if thresholds is None:
            thresholds = [tuple(t) for t in bt_cfg.get(
                'thresholds', [(0.2, 0.8), (0.25, 0.75), (0.3, 0.7)])]
        if rolling_window is None:
            rolling_window = bt_cfg.get('rolling_window', 500)
        if slippage_ticks is None:
            slippage_ticks = bt_cfg.get('slippage_ticks', 1.5)
        if min_samples is None:
            min_samples = bt_cfg.get('min_samples', 20)

        df = self._apply_factor(factor_func, name)
        return self._make_analyzer().run_backtest(
            df, name,
            thresholds     = thresholds,
            rolling_window = rolling_window,
            min_samples    = min_samples,
            slippage_ticks = slippage_ticks,
        )

    # ──────────────────────────────────────────────────────────
    # 数据泄露检测
    # ──────────────────────────────────────────────────────────

    def check_leakage(
        self,
        factor_func,
        name:         str            = 'factor',
        inject_idx:   Optional[int]  = None,
        inject_delta: float          = 1e6,
        check_bars:   bool           = True,
        verbose:      bool           = True,
    ) -> Dict:
        """
        数据泄露检测（注入测试 / Perturbation Test）。

        原理
        ----
        在原始 bar1min 中选定一个注入点 inject_idx，将该点及之后的
        **所有1m bar** 的 OHLC、volume、open_oi、close_oi 全部叠加
        inject_delta，然后对比注入前后的两次完整 Pipeline 输出：

          - 多周期K线（5m / 30m / 日线）的所有列
          - 用户定义的因子值

        安全区定义
        ----------
        inject_idx 之前的1m bar（行号 < inject_idx）对应的多周期K线和
        因子值属于「安全区」：
          bar_end_ts <= inject_nano（注入点1m bar 的 datetime_nano）

        安全区内任何列发生变化 → 存在数据泄露（look-ahead bias）。
        安全区内无任何变化   → 未发现泄露。

        注意
        ----
        · inject_delta 默认 1e6（+100万），远超任何归一化范围。
        · 同时注入 OHLC / volume / OI，确保所有聚合统计量（max/min/sum 等）
          都受影响，避免漏判。
        · datetime / datetime_nano / bar_end_ts 不修改（不破坏时序结构）。

        Parameters
        ----------
        factor_func  : callable (bars: dict) -> pl.Series
        name         : 因子名称（报告显示用）
        inject_idx   : 注入起始的1m bar 行索引；None = 数据集中间位置
        inject_delta : 叠加到各字段的偏移量（价格字段同时 +delta，
                       volume / OI 字段乘以2——这样0值也能被扰动）
        check_bars   : 是否同时检查多周期K线（True）
        verbose      : 是否打印详细报告

        Returns
        -------
        dict
            'leaked'         : bool  任何地方有泄露则为 True
            'inject_1m_idx'  : int   注入起始行索引
            'inject_1m_time' : 注入起始时间
            'inject_n_rows'  : 实际注入的1m bar 数量（= n_1m - inject_idx）
            'inject_delta'   : float
            'cutoff_5m_time' : 安全区最后一根5m bar 的时间
            'n_safe_5m_bars' : 安全区内5m bar 数量
            'factor'         : {'leaked', 'n_leaked_bars', 'first_leak_time'}
            '5m_bars'        : {'leaked', 'leaked_columns': [...]}
            '30m_bars'       : {'leaked', 'leaked_columns': [...]}
            '1d_bars'        : {'leaked', 'leaked_columns': [...]}
        """
        bar1min = self._bar1min
        n_1m    = len(bar1min)

        # ── Step 1：确定注入点 ────────────────────────────────
        if inject_idx is None:
            inject_idx = n_1m // 2
        if not (0 <= inject_idx < n_1m):
            raise ValueError(
                f"inject_idx={inject_idx} 超出范围 [0, {n_1m - 1}]")

        inject_nano = int(bar1min['datetime_nano'][inject_idx])
        inject_time = bar1min['datetime'][inject_idx]
        n_injected  = n_1m - inject_idx

        # ── Step 2：构建注入版 bar1min_mod ───────────────────
        # inject_idx 及之后的所有行：
        #   · OHLC    → +inject_delta
        #   · volume  → ×2（乘法确保0值也被扰动）
        #   · open_oi / close_oi → ×2
        # datetime / datetime_nano / bar_end_ts 不改动（保持时序结构）
        bar1min_mod = (
            bar1min
            .with_row_index('__row__')
            .with_columns([
                pl.when(pl.col('__row__') >= inject_idx)
                  .then(pl.col('open')     + inject_delta)
                  .otherwise(pl.col('open')).alias('open'),
                pl.when(pl.col('__row__') >= inject_idx)
                  .then(pl.col('high')     + inject_delta)
                  .otherwise(pl.col('high')).alias('high'),
                pl.when(pl.col('__row__') >= inject_idx)
                  .then(pl.col('low')      + inject_delta)
                  .otherwise(pl.col('low')).alias('low'),
                pl.when(pl.col('__row__') >= inject_idx)
                  .then(pl.col('close')    + inject_delta)
                  .otherwise(pl.col('close')).alias('close'),
                pl.when(pl.col('__row__') >= inject_idx)
                  .then(pl.col('volume')   * 2)
                  .otherwise(pl.col('volume')).alias('volume'),
                pl.when(pl.col('__row__') >= inject_idx)
                  .then(pl.col('open_oi')  * 2)
                  .otherwise(pl.col('open_oi')).alias('open_oi'),
                pl.when(pl.col('__row__') >= inject_idx)
                  .then(pl.col('close_oi') * 2)
                  .otherwise(pl.col('close_oi')).alias('close_oi'),
            ])
            .drop('__row__')
        )

        # ── Step 3：Pipeline A（原始数据） ───────────────────
        fe_A = FeatureEngineer(
            label_bars      = self._label_bars,
            label_threshold = self._fe_cfg.get('label_threshold', 0.5),
            atr_long_window = self._fe_cfg.get('atr_long_window', 400),
        )
        fe_A._build_lab_infra(bar1min, verbose=False)
        bars_A = fe_A.bars

        try:
            factor_A = factor_func(bars_A)
        except Exception as e:
            raise RuntimeError(f"Pipeline A 调用 factor_func 失败: {e}") from e

        # ── Step 4：Pipeline B（注入数据） ───────────────────
        fe_B = FeatureEngineer(
            label_bars      = self._label_bars,
            label_threshold = self._fe_cfg.get('label_threshold', 0.5),
            atr_long_window = self._fe_cfg.get('atr_long_window', 400),
        )
        fe_B._build_lab_infra(bar1min_mod, verbose=False)
        bars_B = fe_B.bars

        try:
            factor_B = factor_func(bars_B)
        except Exception as e:
            raise RuntimeError(f"Pipeline B 调用 factor_func 失败: {e}") from e

        # ── Step 5：确定安全区 ────────────────────────────────
        # 安全区 = 多周期bar 的 bar_end_ts <= inject_nano
        # 含义：该多周期bar 的最后一根1m bar 在 inject_idx 开始之前结束
        safe_5m  = bars_A["5m"] ['bar_end_ts'] <= inject_nano
        safe_30m = bars_A["30m"]['bar_end_ts'] <= inject_nano
        safe_1d  = bars_A["1d"] ['bar_end_ts'] <= inject_nano

        n_safe_5m = int(safe_5m.sum())
        cutoff_5m_time = (
            bars_A["5m"].filter(safe_5m)['datetime'].max()
            if n_safe_5m > 0 else None
        )

        # ── Step 6：逐列对比安全区 ────────────────────────────
        def _find_leaks(df_a, df_b, mask, time_col='datetime'):
            """在 mask 选出的安全行中，找出 A 与 B 有差异的列"""
            sub_a = df_a.filter(mask)
            sub_b = df_b.filter(mask)
            if len(sub_a) == 0:
                return []
            leaked = []
            skip_cols = {time_col, 'datetime_nano', 'trading_date', 'bar_end_ts'}
            for col in sub_a.columns:
                if col in skip_cols:
                    continue
                try:
                    arr_a = sub_a[col].cast(pl.Float64).to_numpy()
                    arr_b = sub_b[col].cast(pl.Float64).to_numpy()
                except Exception:
                    continue
                nan_both  = np.isnan(arr_a) & np.isnan(arr_b)
                inf_same  = (np.isinf(arr_a) & np.isinf(arr_b)
                             & (np.sign(arr_a) == np.sign(arr_b)))
                diff_mask = ~np.equal(arr_a, arr_b) & ~nan_both & ~inf_same
                diff_idx  = np.where(diff_mask)[0]
                if len(diff_idx) > 0:
                    leaked.append({
                        'col':             col,
                        'n_leaked_bars':   int(len(diff_idx)),
                        'first_leak_time': sub_a[time_col][int(diff_idx[0])],
                    })
            return leaked

        leaked_5m  = _find_leaks(bars_A["5m"],  bars_B["5m"],  safe_5m)  if check_bars else []
        leaked_30m = _find_leaks(bars_A["30m"], bars_B["30m"], safe_30m) if check_bars else []
        leaked_1d  = _find_leaks(bars_A["1d"],  bars_B["1d"],  safe_1d,
                                 time_col='trading_date')                 if check_bars else []

        # 因子值检查（factor 序列与 bars["5m"] 行对齐）
        fa_arr     = factor_A.to_numpy().astype(float)
        fb_arr     = factor_B.to_numpy().astype(float)
        safe_np    = safe_5m.to_numpy()
        fa_safe    = fa_arr[safe_np]
        fb_safe    = fb_arr[safe_np]
        nan_both_f = np.isnan(fa_safe) & np.isnan(fb_safe)
        inf_same_f = (np.isinf(fa_safe) & np.isinf(fb_safe)
                      & (np.sign(fa_safe) == np.sign(fb_safe)))
        diff_f     = ~np.equal(fa_safe, fb_safe) & ~nan_both_f & ~inf_same_f
        diff_idx_f = np.where(diff_f)[0]
        factor_leak_time = (
            bars_A["5m"].filter(safe_5m)['datetime'][int(diff_idx_f[0])]
            if len(diff_idx_f) > 0 else None
        )

        # ── Step 7：组装报告 ──────────────────────────────────
        any_leaked = (
            len(leaked_5m) > 0 or len(leaked_30m) > 0
            or len(leaked_1d) > 0 or len(diff_idx_f) > 0
        )

        report = {
            'leaked':         any_leaked,
            'inject_1m_idx':  inject_idx,
            'inject_1m_time': inject_time,
            'inject_n_rows':  n_injected,
            'inject_delta':   inject_delta,
            'cutoff_5m_time': cutoff_5m_time,
            'n_safe_5m_bars': n_safe_5m,
            'factor': {
                'leaked':          len(diff_idx_f) > 0,
                'n_leaked_bars':   int(len(diff_idx_f)),
                'first_leak_time': factor_leak_time,
            },
            '5m_bars':  {'leaked': len(leaked_5m)  > 0, 'leaked_columns': leaked_5m},
            '30m_bars': {'leaked': len(leaked_30m) > 0, 'leaked_columns': leaked_30m},
            '1d_bars':  {'leaked': len(leaked_1d)  > 0, 'leaked_columns': leaked_1d},
        }

        if verbose:
            self._print_leakage_report(report, name)

        return report

    def _print_leakage_report(self, report: Dict, name: str):
        """将 check_leakage 的结果格式化打印到控制台"""
        W = 66
        bar = '═' * W

        verdict = '❌ 发现泄露！' if report['leaked'] else '✅ 未发现泄露'
        print(f"\n╔{bar}╗")
        print(f"║  数据泄露检测报告  │  因子: {name}  │  品种: {self._symbol_name}")
        print(f"╠{bar}╣")
        print(f"  注入点  : 1m bar #{report['inject_1m_idx']}  "
              f"时间: {report['inject_1m_time']}  "
              f"共注入{report.get('inject_n_rows', '?')}根1m bar  "
              f"delta=+{report['inject_delta']:,.0f}")
        n_safe = report['n_safe_5m_bars']
        print(f"  安全区  : 5m bar ≤ {report['cutoff_5m_time']}  "
              f"（共 {n_safe:,} 根5m bar）")
        print(f"╠{bar}╣")

        # 因子结果
        f = report['factor']
        if f['leaked']:
            print(f"  因子 ({name:<20})  ❌ 泄露  "
                  f"{f['n_leaked_bars']} 根bar，最早: {f['first_leak_time']}")
        else:
            print(f"  因子 ({name:<20})  ✅ 无泄露  "
                  f"（{n_safe:,} 根安全bar 全部一致）")

        # 多周期K线结果
        for tf, key in [('5m', '5m_bars'), ('30m', '30m_bars'), ('1d', '1d_bars')]:
            blk = report[key]
            leaked_cols = blk.get('leaked_columns')
            if leaked_cols is None:
                print(f"  {tf} K线{' ' * (4 - len(tf))}                        （未检测）")
                continue
            if blk['leaked']:
                print(f"  {tf} K线  ❌ 泄露！")
                for c in leaked_cols:
                    print(f"    └─ {c['col']}: "
                          f"{c['n_leaked_bars']} 根bar泄露，"
                          f"最早 {c['first_leak_time']}")
            else:
                print(f"  {tf} K线  ✅ 无泄露")

        print(f"╠{bar}╣")
        print(f"  结论：{verdict}")
        print(f"╚{bar}╝\n")
