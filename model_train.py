"""
模型训练模块（增强版）
支持SHAP分析、参数扫描、完整可视化
"""

import numpy as np
import polars as pl
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from pathlib import Path
from typing import Dict, Tuple, List
from sklearn.metrics import confusion_matrix
from tqdm import tqdm

# 设置绘图参数
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False
sns.set_style('whitegrid')


class ModelTrainer:
    """LightGBM模型训练器（增强版）"""
    
    def __init__(self, config: dict, symbol_name: str, output_dir: Path):
        """
        初始化训练器
        
        Parameters:
        -----------
        config : dict
            配置字典
        symbol_name : str
            品种名称
        output_dir : Path
            输出目录
        """
        self.config = config
        self.symbol_name = symbol_name
        self.output_dir = output_dir
        
        # 创建子目录
        self.model_dir = output_dir / 'models'
        self.figure_dir = output_dir / 'figures'
        self.result_dir = output_dir / 'results'
        
        for d in [self.model_dir, self.figure_dir, self.result_dir]:
            d.mkdir(parents=True, exist_ok=True)
        
        # 模型容器
        self.pretrain_model = None
        self.final_model = None
        self.baseline_model = None
        self.selected_features = None
        self.dropped_features = None
        
        # 结果容器
        self.train_metrics = {}
        self.valid_metrics = {}
        self.test_metrics = {}

        # 类别特征
        self.categorical_features = []

    def prepare_datasets(self, 
                        features_df: pl.DataFrame,
                        labels_df: pl.DataFrame,
                        categorical_features: List[str] = []) -> Dict:
        """准备训练/验证/测试数据集"""

        # 筛选年份
        train_start_year = self.config.get('data_split', {}).get('train_start_year', 2022)
        train_end_year = self.config.get('data_split', {}).get('train_end_year', 2024)
        
        features_df = features_df.filter(pl.col('datetime').dt.year() >= train_start_year)
        labels_df = labels_df.filter(pl.col('datetime').dt.year() >= train_start_year)
        
        # 划分训练+验证 vs 测试
        datetime_train_valid = features_df.filter(
            pl.col('datetime').dt.year() <= train_end_year
        ).select('datetime')
        X_train_valid = features_df.filter(
            pl.col('datetime').dt.year() <= train_end_year
        ).drop(['datetime'])
        labels_train_valid = labels_df.filter(
            pl.col('datetime').dt.year() <= train_end_year
        )
        y_train_valid = labels_train_valid.select('label')

        datetime_test = features_df.filter(
            pl.col('datetime').dt.year() > train_end_year
        ).select('datetime')
        X_test = features_df.filter(
            pl.col('datetime').dt.year() > train_end_year
        ).drop(['datetime'])
        labels_test = labels_df.filter(
            pl.col('datetime').dt.year() > train_end_year
        )
        y_test = labels_test.select('label')
        
        # 划分训练 vs 验证
        valid_ratio = self.config.get('data_split', {}).get('valid_ratio', 0.3)
        embargo = self.config.get('data_split', {}).get('embargo', 10)
        
        Ndata = X_train_valid.shape[0]
        train_end = int(Ndata * (1 - valid_ratio)) - embargo
        valid_start = int(Ndata * (1 - valid_ratio))
        
        datetime_train = datetime_train_valid[:train_end]
        X_train = X_train_valid[:train_end, :]
        y_train = y_train_valid[:train_end]
        
        datetime_valid = datetime_train_valid[valid_start:]
        X_valid = X_train_valid[valid_start:, :]
        y_valid = y_train_valid[valid_start:]
        
        # 删除空值
        datetime_train, X_train, y_train = self._remove_nulls(datetime_train, X_train, y_train)
        datetime_valid, X_valid, y_valid = self._remove_nulls(datetime_valid, X_valid, y_valid)
        datetime_test, X_test, y_test = self._remove_nulls(datetime_test, X_test, y_test)
        
        # 添加Dummy特征
        original_features = X_train.columns
        if self.config.get('feature_selection', {}).get('add_dummy', False):
            np.random.seed(42)
            
            y_train_np = y_train.to_numpy().ravel()
            y_valid_np = y_valid.to_numpy().ravel()
            y_test_np = y_test.to_numpy().ravel()
            
            X_train = X_train.with_columns([
                pl.Series('DUMMY_shuffled_label', np.random.permutation(y_train_np))
            ])
            X_valid = X_valid.with_columns([
                pl.Series('DUMMY_shuffled_label', np.random.permutation(y_valid_np))
            ])
            X_test = X_test.with_columns([
                pl.Series('DUMMY_shuffled_label', np.random.permutation(y_test_np))
            ])
            
            print(f"      已添加Dummy特征")
        
        # 标签转换
        y_train_lgb = (y_train.to_numpy().ravel() + 1).astype(int)
        y_valid_lgb = (y_valid.to_numpy().ravel() + 1).astype(int)
        y_test_lgb = (y_test.to_numpy().ravel() + 1).astype(int)
        
        print(f"    数据集划分:")
        print(f"      训练集: {X_train.shape}")
        print(f"      验证集: {X_valid.shape}")
        print(f"      测试集: {X_test.shape}")
        
        # 保存类别特征
        self.categorical_features = categorical_features

        return {
            'datetime_train': datetime_train,
            'X_train': X_train,
            'y_train_lgb': y_train_lgb,
            'datetime_valid': datetime_valid,
            'X_valid': X_valid,
            'y_valid_lgb': y_valid_lgb,
            'datetime_test': datetime_test,
            'X_test': X_test,
            'y_test_lgb': y_test_lgb,
            'original_features': original_features,
            'labels_train': labels_train_valid[:train_end, :],
            'labels_valid': labels_train_valid[valid_start:, :],
            'labels_test': labels_test,
        }
    
    def pretrain_and_select_features(self, X_train, y_train, X_valid, y_valid):
        """预训练并进行特征筛选（支持SHAP）"""
        
        if not self.config.get('feature_selection', {}).get('enabled', True):
            print("    特征筛选: 禁用")
            self.selected_features = X_train.columns
            return None
        
        print("    预训练中...")

        # 准备类别特征
        categorical_feature_indices = self._get_categorical_indices(X_train.columns)
        X_train_np = X_train.to_numpy()
        X_valid_np = X_valid.to_numpy()
        
        pretrain_set = lgb.Dataset(X_train_np, 
                                   label=y_train,
                                   categorical_feature=categorical_feature_indices)
        
        prevalid_set = lgb.Dataset(X_valid_np, label=y_valid, reference=pretrain_set)
        
        lgb_params = self.config.get('lgb_params', {})
        num_boost_round = self.config.get('training', {}).get('num_boost_round', 500)
        
        self.pretrain_model = lgb.train(
            lgb_params,
            pretrain_set,
            num_boost_round=num_boost_round,
            valid_sets=[pretrain_set, prevalid_set],
            valid_names=['train', 'valid'],
            callbacks=[lgb.log_evaluation(period=100)]
        )
        
        # 获取特征重要性（Gain）
        feature_names = X_train.columns
        importance_gain = self.pretrain_model.feature_importance(importance_type='gain')
        importance_split = self.pretrain_model.feature_importance(importance_type='split')
        
        importance_df = pl.DataFrame({
            'feature': feature_names,
            'importance_gain': importance_gain,
            'importance_split': importance_split
        }).with_columns([
            pl.col('feature').str.starts_with('DUMMY_').alias('is_dummy')
        ]).sort('importance_gain', descending=True)
        
        # 计算SHAP值（如果启用）
        compute_shap = self.config.get('feature_selection', {}).get('compute_shap', False)
        if compute_shap:
            print("    计算SHAP值...")
            shap_importance = self._compute_shap_values(X_train_np, feature_names)
            importance_df = importance_df.with_columns([
                pl.Series('importance_shap', shap_importance)
            ])
        
        importance_df.write_csv(self.result_dir / 'pretrain_feature_importance.csv')
        
        # 确定使用哪种方法进行特征筛选
        selection_method = self.config.get('feature_selection', {}).get('method', 'gain')
        
        if selection_method == 'shap' and compute_shap:
            importance_df = importance_df.sort('importance_shap', descending=True)
            print(f"    使用SHAP值进行特征筛选")
        else:
            print(f"    使用Gain值进行特征筛选")
        
        # 特征筛选
        real_features_df = importance_df.filter(~pl.col('is_dummy'))
        n_real = len(real_features_df)
        drop_ratio = self.config.get('feature_selection', {}).get('drop_ratio', 0.2)
        n_keep = int(n_real * (1 - drop_ratio))
        
        self.selected_features = real_features_df.head(n_keep)['feature'].to_list()
        self.dropped_features = real_features_df.tail(n_real - n_keep)['feature'].to_list()

        print(f"    特征筛选: {n_real} -> {n_keep}")
        
        # 生成可视化
        self._plot_pretrain_analysis(importance_df, X_train_np, feature_names)
        
        return importance_df
    
    def _compute_shap_values(self, X_sample_full, feature_names) -> np.ndarray:
        """计算SHAP值"""
        
        sample_size = min(
            self.config.get('feature_selection', {}).get('shap_sample_size', 1000),
            len(X_sample_full)
        )
        
        sample_indices = np.random.choice(len(X_sample_full), sample_size, replace=False)
        X_sample = X_sample_full[sample_indices]
        
        print(f"      在{sample_size}个样本上计算SHAP值...")
        
        explainer = shap.TreeExplainer(self.pretrain_model)
        shap_values = explainer.shap_values(X_sample)
        
        # shap_values是列表，每个类别一个数组
        # 转换为 (n_samples, n_features, n_classes)
        shap_array = np.array(shap_values)
        
        # 计算平均绝对SHAP值
        shap_abs = np.abs(shap_array)
        shap_mean_over_classes = shap_abs.mean(axis=2)  # (n_samples, n_features)
        mean_shap_importance = shap_mean_over_classes.mean(axis=0)  # (n_features,)
        
        # 保存SHAP数据用于后续绘图
        self.shap_values_for_plot = shap_mean_over_classes
        self.X_sample_for_shap = X_sample
        self.feature_names_for_shap = feature_names
        
        print(f"      ✓ SHAP计算完成")
        
        return mean_shap_importance
    
    def _plot_pretrain_analysis(self, importance_df, X_train_np, feature_names):
        """生成预训练分析图表"""
        
        # 1. 特征重要性图（Top 30）
        self._plot_feature_importance(
            importance_df.head(30),
            'pretrain_feature_importance.png',
            'Top 30 Feature Importance (Pretrain)'
        )
        
        # 2. SHAP相关图表
        compute_shap = self.config.get('feature_selection', {}).get('compute_shap', False)
        if compute_shap and hasattr(self, 'shap_values_for_plot'):
            # 2.1 SHAP summary plot
            self._plot_shap_summary()
            
            # 2.2 Gain vs SHAP对比图
            self._plot_gain_vs_shap(importance_df)
        
        # 3. 特征筛选分析图
        add_dummy = self.config.get('feature_selection', {}).get('add_dummy', False)
        if add_dummy:
            self._plot_feature_selection_analysis(importance_df)
    
    def _plot_shap_summary(self):
        """绘制SHAP summary图"""
        
        try:
            fig, ax = plt.subplots(figsize=(10, 8))
            
            shap.summary_plot(
                self.shap_values_for_plot,
                self.X_sample_for_shap,
                feature_names=list(self.feature_names_for_shap),
                max_display=20,
                show=False,
                plot_type='bar'
            )
            
            plt.title('SHAP Feature Importance (Mean Absolute)', 
                     fontsize=14, fontweight='bold')
            plt.tight_layout()
            plt.savefig(self.figure_dir / 'shap_summary.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"      ✓ 已保存: shap_summary.png")
            
        except Exception as e:
            print(f"      ⚠ SHAP summary图生成失败: {e}")
    
    def _plot_gain_vs_shap(self, importance_df):
        """绘制Gain vs SHAP对比图"""
        
        try:
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # 获取Top 20特征（按Gain排序）
            top20 = importance_df.filter(~pl.col('is_dummy')).head(20)
            
            x = np.arange(len(top20))
            width = 0.35
            
            # 归一化
            gain_values = top20['importance_gain'].to_numpy()
            gain_norm = gain_values / gain_values.max()
            
            shap_values = top20['importance_shap'].to_numpy()
            shap_norm = shap_values / shap_values.max()
            
            ax.barh(x - width/2, gain_norm, width, label='Gain Importance', 
                   alpha=0.8, color='steelblue')
            ax.barh(x + width/2, shap_norm, width, label='SHAP Importance', 
                   alpha=0.8, color='coral')
            
            ax.set_yticks(x)
            ax.set_yticklabels(top20['feature'].to_list())
            ax.set_xlabel('Normalized Importance', fontsize=12)
            ax.set_title('Feature Importance: Gain vs SHAP (Top 20)', 
                        fontsize=14, fontweight='bold')
            ax.legend()
            ax.invert_yaxis()
            plt.tight_layout()
            plt.savefig(self.figure_dir / 'gain_vs_shap.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"      ✓ 已保存: gain_vs_shap.png")
            
        except Exception as e:
            print(f"      ⚠ Gain vs SHAP图生成失败: {e}")
    
    def _plot_feature_selection_analysis(self, importance_df):
        """绘制特征筛选分析图"""
        
        try:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 12))
            
            # 左图：Top 30特征
            top_features = importance_df.head(30)
            importances = top_features['importance_gain'].to_list()
            features = top_features['feature'].to_list()
            is_dummy = top_features['is_dummy'].to_list()
            
            colors = ['red' if d else 'steelblue' for d in is_dummy]
            ax1.barh(range(len(importances)), importances, color=colors)
            ax1.set_yticks(range(len(features)))
            ax1.set_yticklabels(features)
            ax1.set_xlabel('Feature Importance (Gain)', fontsize=12)
            ax1.set_title('Top 30 Features (Pretrain)', fontsize=14, fontweight='bold')
            ax1.invert_yaxis()
            
            # Dummy基线
            dummy_importance = importance_df.filter(pl.col('is_dummy'))['importance_gain'].to_numpy()[0]
            ax1.axvline(x=dummy_importance, color='red', linestyle='--', linewidth=2,
                       label=f'Dummy Baseline ({dummy_importance:.1f})')
            ax1.legend()
            
            # 右图：Bottom 30特征（筛选决策）
            real_features = importance_df.filter(~pl.col('is_dummy'))
            bottom_features = real_features.tail(30)
            importances_bottom = bottom_features['importance_gain'].to_list()
            features_bottom = bottom_features['feature'].to_list()
            
            colors_bottom = ['red' if f in self.dropped_features else 'green' 
                           for f in features_bottom]
            ax2.barh(range(len(importances_bottom)), importances_bottom, color=colors_bottom)
            ax2.set_yticks(range(len(features_bottom)))
            ax2.set_yticklabels(features_bottom)
            ax2.set_xlabel('Feature Importance (Gain)', fontsize=12)
            ax2.set_title('Bottom 30 Features (Red = Dropped)', fontsize=14, fontweight='bold')
            ax2.invert_yaxis()
            
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor='green', label='Kept'),
                Patch(facecolor='red', label='Dropped')
            ]
            ax2.legend(handles=legend_elements, loc='lower right')
            
            plt.tight_layout()
            plt.savefig(self.figure_dir / 'feature_selection_analysis.png', 
                       dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"      ✓ 已保存: feature_selection_analysis.png")
            
        except Exception as e:
            print(f"      ⚠ 特征筛选分析图生成失败: {e}")
    
    def train_final_model(self, X_train, y_train, X_valid, y_valid):
        """训练最终模型"""
        
        print("    训练最终模型...")
        
        X_train_selected = X_train.select(self.selected_features)
        X_valid_selected = X_valid.select(self.selected_features)
        
        X_train_np = X_train_selected.to_numpy()
        X_valid_np = X_valid_selected.to_numpy()
        
        categorical_feature_indices = self._get_categorical_indices(
            X_train_selected.columns
        )

        train_set = lgb.Dataset(X_train_np, 
                                label=y_train,
                                categorical_feature=categorical_feature_indices)
        
        valid_set = lgb.Dataset(X_valid_np, label=y_valid, reference=train_set)
        
        lgb_params = self.config.get('lgb_params', {})
        num_boost_round = self.config.get('training', {}).get('num_boost_round', 500)
        
        self.final_model = lgb.train(
            lgb_params,
            train_set,
            num_boost_round=num_boost_round,
            valid_sets=[train_set, valid_set],
            valid_names=['train', 'valid'],
            callbacks=[lgb.log_evaluation(period=100)]
        )
        
        # 保存模型
        model_path = self.model_dir / f'{self.symbol_name}_model.txt'
        self.final_model.save_model(str(model_path))
        print(f"      ✓ 模型已保存: {model_path.name}")
        
        # 特征重要性
        importance_gain = self.final_model.feature_importance(importance_type='gain')
        importance_df = pl.DataFrame({
            'feature': self.selected_features,
            'importance_gain': importance_gain
        }).sort('importance_gain', descending=True)
        
        importance_df.write_csv(self.result_dir / 'final_feature_importance.csv')
        
        # 绘制特征重要性
        self._plot_feature_importance(
            importance_df.head(30),
            'final_feature_importance.png',
            'Top 30 Feature Importance (Final Model)'
        )
        
        return importance_df
    
    def _get_categorical_indices(self, feature_names: List[str]) -> List[int]:
        """
        获取类别特征的索引位置
        
        Parameters:
        -----------
        feature_names : List[str]
            特征名称列表
        
        Returns:
        --------
        indices : List[int]
            类别特征的索引列表
        """
        indices = []
        for i, name in enumerate(feature_names):
            if name in self.categorical_features:
                indices.append(i)
        return indices
    
    def evaluate_model(self, X, y_true, dataset_name: str, 
                      window: int = None, quantile: float = None) -> Tuple:
        """
        评估模型
        
        Parameters:
        -----------
        X : np.ndarray
            特征数据
        y_true : np.ndarray
            真实标签
        dataset_name : str
            数据集名称
        window : int, optional
            滚动窗口大小（用于参数扫描）
        quantile : float, optional
            分位数阈值（用于参数扫描）
        """
        
        y_pred_proba = self.final_model.predict(X, num_iteration=self.final_model.best_iteration)
        y_pred = np.argmax(y_pred_proba, axis=1)
        
        # 应用滚动分位数过滤
        if self.config.get('signal_filter', {}).get('use_rolling_quantile', True):
            # 如果提供了window和quantile，使用它们；否则使用默认值
            if window is None:
                window_val = self.config.get('signal_filter', {}).get('rolling_window', 100)
                if isinstance(window_val, list):
                    window_val = window_val[0]  # 使用第一个值
                window = window_val
            
            if quantile is None:
                quantile_val = self.config.get('signal_filter', {}).get('signal_quantile', 0.9)
                if isinstance(quantile_val, list):
                    quantile_val = quantile_val[0]  # 使用第一个值
                quantile = quantile_val
            
            y_pred = self._apply_rolling_quantile_filter(y_pred_proba, y_pred, window, quantile)
        
        y_pred_original = y_pred - 1
        y_true_original = y_true - 1
        
        # 计算指标
        cm = confusion_matrix(y_true_original, y_pred_original, labels=[-1, 0, 1])
        
        n_pred_down = (y_pred_original == -1).sum()
        n_pred_up = (y_pred_original == 1).sum()
        n_signals = n_pred_down + n_pred_up
        
        short_precision = cm[0][0] / n_pred_down if n_pred_down > 0 else 0
        long_precision = cm[2][2] / n_pred_up if n_pred_up > 0 else 0
        
        if n_signals > 0:
            trading_mask = y_pred_original != 0
            win_rate = ((y_pred_original[trading_mask] * y_true_original[trading_mask]) > 0).sum() / n_signals
            long_short_ratio = n_pred_up / (n_pred_down + 1e-9)
        else:
            win_rate = 0
            long_short_ratio = 0
        
        metrics = {
            'short_precision': short_precision,
            'long_precision': long_precision,
            'n_signals': n_signals,
            'n_long': n_pred_up,
            'n_short': n_pred_down,
            'long_short_ratio': long_short_ratio,
            'win_rate': win_rate,
        }
        
        # 只在非扫描模式下打印
        if window is None or quantile is None:
            print(f"      {dataset_name}:")
            print(f"        做多精确率: {long_precision:.4f}")
            print(f"        做空精确率: {short_precision:.4f}")
            print(f"        方向胜率: {win_rate:.4f}")
            print(f"        信号数: {n_signals} ({n_signals/len(y_pred_original)*100:.1f}%)")
            print(f"        多空比: {long_short_ratio:.2f}")
        
        return y_pred_proba, y_pred_original, metrics
    
    def parameter_scan(self, X_test, y_test):
        """参数扫描：不同window和quantile组合"""
        
        window_values = self.config.get('signal_filter', {}).get('rolling_window', [100])
        quantile_values = self.config.get('signal_filter', {}).get('signal_quantile', [0.9])
        
        # 确保是列表
        if not isinstance(window_values, list):
            window_values = [window_values]
        if not isinstance(quantile_values, list):
            quantile_values = [quantile_values]
        
        if len(window_values) == 1 and len(quantile_values) == 1:
            print("      参数扫描: 禁用（只有一组参数）")
            return
        
        print(f"      参数扫描: {len(window_values)} 个窗口 × {len(quantile_values)} 个分位数")
        
        scan_results = []
        
        total = len(window_values) * len(quantile_values)
        with tqdm(total=total, desc="      扫描进度") as pbar:
            for window in window_values:
                for quantile in quantile_values:
                    _, y_pred, metrics = self.evaluate_model(
                        X_test, y_test, 
                        dataset_name='Scan',
                        window=window,
                        quantile=quantile
                    )
                    
                    scan_results.append({
                        'window': window,
                        'quantile': quantile,
                        'n_signals': metrics['n_signals'],
                        'signal_pct': metrics['n_signals'] / len(y_test) * 100,
                        'n_long': metrics['n_long'],
                        'n_short': metrics['n_short'],
                        'long_short_ratio': metrics['long_short_ratio'],
                        'win_rate': metrics['win_rate'],
                        'long_prec': metrics['long_precision'],
                        'short_prec': metrics['short_precision'],
                    })
                    
                    pbar.update(1)
        
        # 保存扫描结果
        scan_df = pl.DataFrame(scan_results)
        scan_df.write_csv(self.result_dir / 'parameter_scan.csv')
        
        print(f"      ✓ 参数扫描完成，已保存: parameter_scan.csv")
        
        # 绘制扫描结果
        self._plot_parameter_scan(scan_df)
    
    def _plot_parameter_scan(self, scan_df):
        """绘制参数扫描结果"""
        
        import pandas as pd
        scan_pd = scan_df.to_pandas()
        
        window_values = scan_pd['window'].unique()
        
        # 1. 四合一图：Win Rate, Signal Count, Long Precision, Short Precision
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        for window in window_values:
            subset = scan_pd[scan_pd['window'] == window]
            quantiles = subset['quantile'].values
            win_rates = subset['win_rate'].values
            n_sigs = subset['n_signals'].values
            long_precs = subset['long_prec'].values
            short_precs = subset['short_prec'].values
            
            axes[0, 0].plot(quantiles, win_rates, 'o-', label=f'Window={window}', linewidth=2)
            axes[0, 1].plot(quantiles, n_sigs, 's-', label=f'Window={window}', linewidth=2)
            axes[1, 0].plot(quantiles, long_precs, '^-', label=f'Window={window}', linewidth=2)
            axes[1, 1].plot(quantiles, short_precs, 'v-', label=f'Window={window}', linewidth=2)
        
        axes[0, 0].set_xlabel('Signal Quantile', fontsize=11)
        axes[0, 0].set_ylabel('Win Rate', fontsize=11)
        axes[0, 0].set_title('Win Rate vs Quantile', fontsize=12, fontweight='bold')
        axes[0, 0].axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='50% baseline')
        axes[0, 0].legend()
        axes[0, 0].grid(alpha=0.3)
        
        axes[0, 1].set_xlabel('Signal Quantile', fontsize=11)
        axes[0, 1].set_ylabel('Number of Signals', fontsize=11)
        axes[0, 1].set_title('Signal Count vs Quantile', fontsize=12, fontweight='bold')
        axes[0, 1].legend()
        axes[0, 1].grid(alpha=0.3)
        
        axes[1, 0].set_xlabel('Signal Quantile', fontsize=11)
        axes[1, 0].set_ylabel('Long Precision', fontsize=11)
        axes[1, 0].set_title('Long Precision vs Quantile', fontsize=12, fontweight='bold')
        axes[1, 0].legend()
        axes[1, 0].grid(alpha=0.3)
        
        axes[1, 1].set_xlabel('Signal Quantile', fontsize=11)
        axes[1, 1].set_ylabel('Short Precision', fontsize=11)
        axes[1, 1].set_title('Short Precision vs Quantile', fontsize=12, fontweight='bold')
        axes[1, 1].legend()
        axes[1, 1].grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.figure_dir / 'rolling_quantile_scan.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"      ✓ 已保存: rolling_quantile_scan.png")
        
        # 2. 多空平衡度图
        fig, ax = plt.subplots(figsize=(12, 6))
        
        for window in window_values:
            subset = scan_pd[scan_pd['window'] == window]
            quantiles = subset['quantile'].values
            ratios = subset['long_short_ratio'].values
            ax.plot(quantiles, ratios, 'o-', label=f'Window={window}', 
                   linewidth=2, markersize=8)
        
        ax.axhline(y=1.0, color='green', linestyle='--', linewidth=2, 
                  label='Perfect Balance')
        ax.fill_between([quantiles.min(), quantiles.max()], 0.8, 1.2, 
                       alpha=0.2, color='green', label='Acceptable Range')
        ax.set_xlabel('Signal Quantile', fontsize=12)
        ax.set_ylabel('Long/Short Ratio', fontsize=12)
        ax.set_title('Long-Short Balance vs Quantile', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.figure_dir / 'long_short_balance.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"      ✓ 已保存: long_short_balance.png")
    
    def save_predictions(self, datetime_test, y_test, y_pred, y_pred_proba):
        """保存预测结果"""
        
        test_results = pl.DataFrame({
            'datetime': datetime_test.to_series(),
            'y_true': y_test - 1,
            'y_pred': y_pred,
            'prob_down': y_pred_proba[:, 0],
            'prob_neutral': y_pred_proba[:, 1],
            'prob_up': y_pred_proba[:, 2],
        })
        
        result_file = self.result_dir / f'{self.symbol_name}_predictions.csv'
        test_results.write_csv(result_file)
        print(f"      ✓ 预测结果已保存: {result_file.name}")
    
    def _apply_rolling_quantile_filter(self, y_pred_proba, y_pred, window, quantile):
        """应用滚动分位数过滤"""
        
        prob_down = y_pred_proba[:, 0]
        prob_neutral = y_pred_proba[:, 1]
        prob_up = y_pred_proba[:, 2]
        
        long_strength = prob_up - np.maximum(prob_down, prob_neutral)
        short_strength = prob_down - np.maximum(prob_up, prob_neutral)
        
        def rolling_quantile_threshold(signal, window, q):
            thresholds = np.full(len(signal), np.nan)
            for i in range(len(signal)):
                if i < window:
                    thresholds[i] = np.quantile(signal[:i], q) if i > 0 else -np.inf
                else:
                    thresholds[i] = np.quantile(signal[i-window:i], q)
            return thresholds
        
        long_threshold = rolling_quantile_threshold(long_strength, window, quantile)
        short_threshold = rolling_quantile_threshold(short_strength, window, quantile)
        
        y_pred_filtered = y_pred.copy()
        y_pred_filtered[(y_pred == 2) & (long_strength < long_threshold)] = 1
        y_pred_filtered[(y_pred == 0) & (short_strength < short_threshold)] = 1
        
        return y_pred_filtered
    
    def _plot_feature_importance(self, importance_df, filename, title):
        """绘制特征重要性"""
        
        plt.figure(figsize=(10, 12))
        
        importances = importance_df['importance_gain'].to_list()
        features = importance_df['feature'].to_list()
        
        plt.barh(range(len(importances)), importances, color='steelblue')
        plt.yticks(range(len(features)), features)
        plt.xlabel('Importance (Gain)', fontsize=12)
        plt.title(title, fontsize=14, fontweight='bold')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig(self.figure_dir / filename, dpi=300, bbox_inches='tight')
        plt.close()
    
    @staticmethod
    def _remove_nulls(datetime_df, X, y):
        """删除空值"""
        X_mask = X.with_columns([
            pl.col(colname).is_not_null().alias(colname) for colname in X.columns
        ])
        X_all_not_null = pl.all_horizontal(X_mask)
        
        y_mask = y.with_columns([
            pl.col(colname).is_not_null().alias(colname) for colname in y.columns
        ])
        y_all_not_null = pl.all_horizontal(y_mask)
        
        final_mask = X_all_not_null & y_all_not_null
        
        return datetime_df.filter(final_mask), X.filter(final_mask), y.filter(final_mask)


    def analyze_label_distribution(self, data_dict: Dict):
        """
        分析标签分布
        
        Parameters:
        -----------
        data_dict : Dict
            包含训练/验证/测试数据的字典
        """
        print("\n    标签分布分析:")
        
        # 收集标签统计（转换回原始空间 -1, 0, 1）
        y_train_original = data_dict['y_train_lgb'] - 1
        labels_train = data_dict['labels_train']
        y_valid_original = data_dict['y_valid_lgb'] - 1
        labels_valid = data_dict['labels_valid']
        y_test_original = data_dict['y_test_lgb'] - 1
        labels_test = data_dict['labels_test']
        
        # 计算各数据集的标签分布
        train_dist = self._calculate_label_stats(y_train_original, labels_train, 'Train')
        valid_dist = self._calculate_label_stats(y_valid_original, labels_valid, 'Valid')
        test_dist = self._calculate_label_stats(y_test_original, labels_test, 'Test')
        
        # 保存统计数据
        label_stats = {
            'train': train_dist,
            'valid': valid_dist,
            'test': test_dist
        }
        
        # 生成可视化
        self._plot_label_comparison(label_stats)
        
        # 保存为CSV
        self._save_label_stats_csv(label_stats)
        
        return label_stats

    def _calculate_label_stats(self, y: np.ndarray, labels_df: pl.DataFrame, dataset_name: str) -> Dict:
        """
        计算单个数据集的标签统计
        
        Returns:
        --------
        stats : Dict
            包含计数和比例的统计信息
        """
        n_total = len(y)
        n_down = (y == -1).sum()
        n_neutral = (y == 0).sum()
        n_up = (y == 1).sum()
        
        # 统计非零标签对应的绝对价格变化
        price_change_pos = labels_df.filter(pl.col('label') == 1)['price_change'].to_numpy()
        price_change_neg = labels_df.filter(pl.col('label') == -1)['price_change'].abs().to_numpy() # 加了绝对值
        pos_pct25 = np.nanquantile(price_change_pos, 0.25)
        neg_pct25 = np.nanquantile(price_change_neg, 0.25)
        pos_pct75 = np.nanquantile(price_change_pos, 0.75)
        neg_pct75 = np.nanquantile(price_change_neg, 0.75)

        stats = {
            'dataset': dataset_name,
            'total': n_total,
            'n_down': n_down,
            'n_neutral': n_neutral,
            'n_up': n_up,
            'pct_down': n_down / n_total * 100,
            'pct_neutral': n_neutral / n_total * 100,
            'pct_up': n_up / n_total * 100,
            'ratio_up_down': n_up / (n_down + 1e-9),
            'pos_label_price_change_pct25(ticks)': pos_pct25, # 正标签对应价格变化的25分位数
            'neg_label_price_change_pct25(ticks)': neg_pct25, # ...
            'pos_label_price_change_pct75(ticks)': pos_pct75,
            'neg_label_price_change_pct75(ticks)': neg_pct75,
        }
        
        print(f"\n      {dataset_name}集 (n={n_total}):")
        print(f"        做空(-1): {n_down:6d} ({stats['pct_down']:5.2f}%)")
        print(f"        中性( 0): {n_neutral:6d} ({stats['pct_neutral']:5.2f}%)")
        print(f"        做多(+1): {n_up:6d} ({stats['pct_up']:5.2f}%)")
        print(f"        多空比:   {stats['ratio_up_down']:.3f}")

        return stats

    def _plot_label_comparison(self, label_stats: Dict):
        """绘制训练集vs测试集标签分布对比"""
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # 左图：百分比堆叠条形图
        datasets = ['Train', 'Valid', 'Test']
        down_pcts = [label_stats['train']['pct_down'], 
                    label_stats['valid']['pct_down'],
                    label_stats['test']['pct_down']]
        neutral_pcts = [label_stats['train']['pct_neutral'],
                    label_stats['valid']['pct_neutral'],
                    label_stats['test']['pct_neutral']]
        up_pcts = [label_stats['train']['pct_up'],
                label_stats['valid']['pct_up'],
                label_stats['test']['pct_up']]
        
        x = np.arange(len(datasets))
        width = 0.6
        
        p1 = axes[0].bar(x, down_pcts, width, label='Down (-1)', color='#e74c3c', alpha=0.7)
        p2 = axes[0].bar(x, neutral_pcts, width, bottom=down_pcts, 
                        label='Neutral (0)', color='#95a5a6', alpha=0.7)
        p3 = axes[0].bar(x, up_pcts, width, 
                        bottom=np.array(down_pcts) + np.array(neutral_pcts),
                        label='Up (+1)', color='#2ecc71', alpha=0.7)
        
        axes[0].set_ylabel('Percentage (%)', fontsize=11)
        axes[0].set_title('Label Distribution Comparison', fontsize=12, fontweight='bold')
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(datasets)
        axes[0].legend(loc='upper right')
        axes[0].grid(axis='y', alpha=0.3)
        
        # 添加百分比标签
        for i, (d, n, u) in enumerate(zip(down_pcts, neutral_pcts, up_pcts)):
            axes[0].text(i, d/2, f'{d:.1f}%', ha='center', va='center', fontsize=9)
            axes[0].text(i, d + n/2, f'{n:.1f}%', ha='center', va='center', fontsize=9)
            axes[0].text(i, d + n + u/2, f'{u:.1f}%', ha='center', va='center', fontsize=9)
        
        # 右图：多空比对比
        ratios = [label_stats['train']['ratio_up_down'],
                label_stats['valid']['ratio_up_down'],
                label_stats['test']['ratio_up_down']]
        
        bars = axes[1].bar(x, ratios, width, color='steelblue', alpha=0.7, edgecolor='black')
        axes[1].axhline(y=1.0, color='red', linestyle='--', linewidth=2, label='Balanced (1.0)')
        
        # 添加数值标签
        for bar, ratio in zip(bars, ratios):
            height = bar.get_height()
            axes[1].text(bar.get_x() + bar.get_width()/2., height,
                    f'{ratio:.3f}',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        axes[1].set_ylabel('Long/Short Ratio', fontsize=11)
        axes[1].set_title('Long/Short Ratio Comparison', fontsize=12, fontweight='bold')
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(datasets)
        axes[1].legend()
        axes[1].grid(axis='y', alpha=0.3)
        
        plt.suptitle(f'{self.symbol_name} - Label Distribution Stability', 
                    fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(self.figure_dir / 'label_distribution_comparison.png', 
                dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"      ✓ 已保存: label_distribution_comparison.png")

    def _save_label_stats_csv(self, label_stats: Dict):
        """保存标签统计为CSV"""
        
        stats_data = []
        for dataset in ['train', 'valid', 'test']:
            stats = label_stats[dataset]
            stats_data.append({
                'dataset': stats['dataset'],
                'total': stats['total'],
                'n_down': stats['n_down'],
                'n_neutral': stats['n_neutral'],
                'n_up': stats['n_up'],
                'pct_down': f"{stats['pct_down']:.2f}%",
                'pct_neutral': f"{stats['pct_neutral']:.2f}%",
                'pct_up': f"{stats['pct_up']:.2f}%",
                'ratio_up_down': f"{stats['ratio_up_down']:.3f}",
                'pos_label_price_change_pct25': f"{stats['pos_label_price_change_pct25(ticks)']:.3f}",
                'neg_label_price_change_pct25': f"{stats['neg_label_price_change_pct25(ticks)']:.3f}",
                'pos_label_price_change_pct75': f"{stats['pos_label_price_change_pct75(ticks)']:.3f}",
                'neg_label_price_change_pct75': f"{stats['neg_label_price_change_pct75(ticks)']:.3f}",
            })
        
        stats_df = pl.DataFrame(stats_data)
        stats_df.write_csv(self.result_dir / 'label_statistics.csv')
        print(f"      ✓ 已保存: label_statistics.csv")

    def calculate_model_improvement(self, train_metrics: Dict, 
                                valid_metrics: Dict, 
                                test_metrics: Dict,
                                label_stats: Dict):
        """
        计算模型相对于基线的提升
        
        基线：
        1. 随机猜测（33.3%准确率）
        2. 多数类预测（预测最多的类别）
        3. 标签分布预测（按标签比例随机预测）
        """
        print("\n    模型提升分析:")
        
        improvements = {}
        
        for dataset_name, metrics, stats in [
            ('train', train_metrics, label_stats['train']),
            ('valid', valid_metrics, label_stats['valid']),
            ('test', test_metrics, label_stats['test'])
        ]:
            # 基线1：随机猜测
            random_baseline = 1.0 / 3.0
            
            # 基线2：多数类预测
            majority_baseline = max(
                stats['pct_down'],
                stats['pct_neutral'],
                stats['pct_up']
            ) / 100.0
            
            # 基线3：按标签分布预测（期望准确率）
            distribution_baseline = (
                (stats['pct_down']/100)**2 +
                (stats['pct_neutral']/100)**2 +
                (stats['pct_up']/100)**2
            )
            
            # 模型实际性能（信号准确率）
            if metrics['n_signals'] > 0:
                # 只考虑交易信号（不含中性）
                model_accuracy = (
                    metrics['long_precision'] * metrics['n_long'] +
                    metrics['short_precision'] * metrics['n_short']
                ) / metrics['n_signals']
                
                # 模型的方向胜率
                model_win_rate = metrics['win_rate']
            else:
                model_accuracy = 0
                model_win_rate = 0
            
            improvements[dataset_name] = {
                'random_baseline': random_baseline,
                'majority_baseline': majority_baseline,
                'distribution_baseline': distribution_baseline,
                'model_accuracy': model_accuracy,
                'model_win_rate': model_win_rate,
                'improvement_vs_random': (model_win_rate - random_baseline) / random_baseline * 100,
                'improvement_vs_majority': (model_win_rate - majority_baseline) / majority_baseline * 100,
                'improvement_vs_distribution': (model_win_rate - distribution_baseline) / distribution_baseline * 100,
            }
            
            print(f"\n      {dataset_name.capitalize()}集:")
            print(f"        基线性能:")
            print(f"          随机猜测:     {random_baseline*100:.2f}%")
            print(f"          多数类预测:   {majority_baseline*100:.2f}%")
            print(f"          分布预测:     {distribution_baseline*100:.2f}%")
            print(f"        模型性能:")
            print(f"          信号准确率:   {model_accuracy*100:.2f}%")
            print(f"          方向胜率:     {model_win_rate*100:.2f}%")
            print(f"        相对提升:")
            print(f"          vs 随机:      {improvements[dataset_name]['improvement_vs_random']:+.1f}%")
            print(f"          vs 多数类:    {improvements[dataset_name]['improvement_vs_majority']:+.1f}%")
            print(f"          vs 分布:      {improvements[dataset_name]['improvement_vs_distribution']:+.1f}%")
        
        # 生成提升效果可视化
        self._plot_model_improvement(improvements)
        
        # 保存提升统计
        self._save_improvement_stats(improvements)
        
        return improvements

    def _plot_model_improvement(self, improvements: Dict):
        """绘制模型提升效果图"""
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        datasets = ['Train', 'Valid', 'Test']
        x = np.arange(len(datasets))
        width = 0.15
        
        # 左图：绝对性能对比
        random_baseline = [improvements['train']['random_baseline']*100,
                        improvements['valid']['random_baseline']*100,
                        improvements['test']['random_baseline']*100]
        majority_baseline = [improvements['train']['majority_baseline']*100,
                            improvements['valid']['majority_baseline']*100,
                            improvements['test']['majority_baseline']*100]
        distribution_baseline = [improvements['train']['distribution_baseline']*100,
                            improvements['valid']['distribution_baseline']*100,
                            improvements['test']['distribution_baseline']*100]
        model_win_rate = [improvements['train']['model_win_rate']*100,
                        improvements['valid']['model_win_rate']*100,
                        improvements['test']['model_win_rate']*100]
        
        axes[0].bar(x - 1.5*width, random_baseline, width, label='Random (33.3%)', 
                color='lightgray', alpha=0.7)
        axes[0].bar(x - 0.5*width, majority_baseline, width, label='Majority Class', 
                color='lightblue', alpha=0.7)
        axes[0].bar(x + 0.5*width, distribution_baseline, width, label='Distribution', 
                color='lightgreen', alpha=0.7)
        axes[0].bar(x + 1.5*width, model_win_rate, width, label='Model Win Rate', 
                color='red', alpha=0.7)
        
        axes[0].set_ylabel('Accuracy (%)', fontsize=11)
        axes[0].set_title('Model Performance vs Baselines', fontsize=12, fontweight='bold')
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(datasets)
        axes[0].legend()
        axes[0].grid(axis='y', alpha=0.3)
        
        # 右图：相对提升
        improvement_random = [improvements['train']['improvement_vs_random'],
                            improvements['valid']['improvement_vs_random'],
                            improvements['test']['improvement_vs_random']]
        improvement_majority = [improvements['train']['improvement_vs_majority'],
                            improvements['valid']['improvement_vs_majority'],
                            improvements['test']['improvement_vs_majority']]
        improvement_distribution = [improvements['train']['improvement_vs_distribution'],
                                improvements['valid']['improvement_vs_distribution'],
                                improvements['test']['improvement_vs_distribution']]
        
        axes[1].bar(x - width, improvement_random, width, label='vs Random', 
                color='steelblue', alpha=0.7)
        axes[1].bar(x, improvement_majority, width, label='vs Majority Class', 
                color='coral', alpha=0.7)
        axes[1].bar(x + width, improvement_distribution, width, label='vs Distribution', 
                color='green', alpha=0.7)
        
        axes[1].axhline(y=0, color='black', linestyle='-', linewidth=1)
        axes[1].set_ylabel('Improvement (%)', fontsize=11)
        axes[1].set_title('Relative Improvement over Baselines', fontsize=12, fontweight='bold')
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(datasets)
        axes[1].legend()
        axes[1].grid(axis='y', alpha=0.3)
        
        plt.suptitle(f'{self.symbol_name} - Model Improvement Analysis', 
                    fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(self.figure_dir / 'model_improvement.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"      ✓ 已保存: model_improvement.png")

    def _save_improvement_stats(self, improvements: Dict):
        """保存提升统计为CSV"""
        
        stats_data = []
        for dataset in ['train', 'valid', 'test']:
            imp = improvements[dataset]
            stats_data.append({
                'dataset': dataset.capitalize(),
                'random_baseline': f"{imp['random_baseline']*100:.2f}%",
                'majority_baseline': f"{imp['majority_baseline']*100:.2f}%",
                'distribution_baseline': f"{imp['distribution_baseline']*100:.2f}%",
                'model_win_rate': f"{imp['model_win_rate']*100:.2f}%",
                'improvement_vs_random': f"{imp['improvement_vs_random']:+.1f}%",
                'improvement_vs_majority': f"{imp['improvement_vs_majority']:+.1f}%",
                'improvement_vs_distribution': f"{imp['improvement_vs_distribution']:+.1f}%",
            })
        
        stats_df = pl.DataFrame(stats_data)
        stats_df.write_csv(self.result_dir / 'model_improvement_stats.csv')
        print(f"      ✓ 已保存: model_improvement_stats.csv")
    