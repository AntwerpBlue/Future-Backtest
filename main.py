"""
主控制脚本
协调数据加载、特征工程、模型训练的完整流程
支持 --mode train / factor_analysis 两种运行模式
"""

import json
import datetime
import warnings
from pathlib import Path
from typing import Dict
import polars as pl

warnings.filterwarnings('ignore')

import sys
sys.path.append(str(Path(__file__).parent))

from data_loader import batch_download
from feature_engineering import FeatureEngineer
from model_train import ModelTrainer


# ============================================================
# 工具函数：统一解析 symbols 配置（兼容旧格式字符串数组）
# ============================================================

def parse_symbols(config: dict) -> list:
    """
    返回 list of dict: [{"code": "KQ.m@SHFE.au", "tick_size": 0.02}, ...]
    兼容旧格式（字符串列表）
    """
    raw = config.get('symbols', [])
    result = []
    for item in raw:
        if isinstance(item, str):
            result.append({"code": item, "tick_size": 1.0})
        else:
            result.append(item)
    return result


def get_symbol_name(symbol_code: str) -> str:
    """从完整品种代码提取简化名称，如 KQ.m@SHFE.au → SHFE_au"""
    if '@' in symbol_code:
        return symbol_code.split('@')[1].replace('.', '_')
    return symbol_code.replace('.', '_').replace('@', '_')


# ============================================================
# 训练流程
# ============================================================

class Pipeline:
    """多品种训练流程管理器"""

    def __init__(self, config_path: str = 'config.json'):
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = json.load(f)

        self.symbols = parse_symbols(self.config)

        self.output_dir = Path(self.config['output']['output_dir'])
        self.temp_dir   = Path(self.config['output']['temp_data_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.temp_dir.mkdir(parents=True, exist_ok=True)

        self.timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        self.results = {}

        print(f"\n{'='*80}")
        print(f"多品种训练流程")
        print(f"{'='*80}")
        print(f"配置文件: {config_path}")
        print(f"输出目录: {self.output_dir}")
        print(f"品种数量: {len(self.symbols)}")
        print(f"品种列表: {', '.join(s['code'] for s in self.symbols)}")
        print(f"{'='*80}\n")

    def run(self):
        try:
            print(f"\n{'='*80}")
            print("步骤1: 数据加载")
            print(f"{'='*80}")
            data_dict = self._load_data()

            if not data_dict:
                print("⚠ 没有成功加载任何品种数据，流程终止")
                return

            print(f"\n{'='*80}")
            print("步骤2: 特征工程与模型训练")
            print(f"{'='*80}")

            for i, sym in enumerate(self.symbols, 1):
                symbol_name = get_symbol_name(sym['code'])
                if symbol_name not in data_dict:
                    continue
                print(f"\n[{i}/{len(self.symbols)}] 处理品种: {symbol_name}")
                print("-" * 80)
                try:
                    result = self._process_symbol(symbol_name, data_dict[symbol_name])
                    self.results[symbol_name] = result if result else {'status': 'failed'}
                    print(f"{'✓' if result else '✗'} {symbol_name} 处理{'完成' if result else '失败'}")
                except Exception as e:
                    import traceback; traceback.print_exc()
                    self.results[symbol_name] = {'status': 'failed', 'error': str(e)}

            print(f"\n{'='*80}")
            print("步骤3: 生成汇总报告")
            print(f"{'='*80}")
            self._generate_summary()

        except Exception as e:
            import traceback; traceback.print_exc()
            raise

    def _load_data(self) -> Dict[str, pl.DataFrame]:
        start_date = datetime.datetime.strptime(
            self.config['data_source']['start_date'], '%Y-%m-%d').date()
        end_date = datetime.datetime.strptime(
            self.config['data_source']['end_date'], '%Y-%m-%d').date()

        symbol_codes = [s['code'] for s in self.symbols]

        saved_files = batch_download(
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
            code = sym['code']
            safe  = code.replace(".", "_").replace("@", "_")
            name  = get_symbol_name(code)
            files = list(self.temp_dir.glob(f"{safe}_1m_*.parquet"))
            if files:
                print(f"  加载: {name} <- {files[0].name}")
                data_dict[name] = pl.read_parquet(files[0])
            else:
                print(f"  ⚠ 未找到数据: {name}")

        print(f"\n✓ 成功加载 {len(data_dict)} 个品种的数据")
        return data_dict

    def _process_symbol(self, symbol_name: str, bar1min: pl.DataFrame) -> dict:
        symbol_dir = self.output_dir / symbol_name
        symbol_dir.mkdir(exist_ok=True)

        print(f"  [1/3] 特征工程...")
        engineer = FeatureEngineer(
            label_bars=self.config['feature_engineering']['label_bars'],
            label_threshold=self.config['feature_engineering']['label_threshold'],
            atr_long_window=self.config['feature_engineering'].get('atr_long_window', 400),
        )
        try:
            features_df = engineer.create_features_and_labels(bar1min, verbose=False)
        except Exception as e:
            print(f"    ✗ 特征工程失败: {e}")
            return None

        feature_dir = symbol_dir / 'features'
        engineer.save_features(feature_dir, 'parquet')

        print(f"  [2/3] 模型训练...")
        trainer = ModelTrainer(self.config, symbol_name, symbol_dir)
        try:
            feature_cols = ['datetime'] + engineer.feature_cols
            label_cols   = ['datetime', 'label', 'ror_future', 'price_change']

            features_only = features_df.select(feature_cols)
            labels_only   = features_df.select(label_cols)

            data_dict = trainer.prepare_datasets(
                features_only, labels_only,
                categorical_features=engineer.categorical_features
            )
            label_stats      = trainer.analyze_label_distribution(data_dict)
            importance_df    = trainer.pretrain_and_select_features(
                data_dict['X_train'], data_dict['y_train_lgb'],
                data_dict['X_valid'], data_dict['y_valid_lgb'])
            final_importance = trainer.train_final_model(
                data_dict['X_train'], data_dict['y_train_lgb'],
                data_dict['X_valid'], data_dict['y_valid_lgb'])
        except Exception as e:
            print(f"    ✗ 模型训练失败: {e}")
            return None

        print(f"  [3/3] 模型评估...")
        sel = trainer.selected_features
        X_tr = data_dict['X_train'].select(sel).to_numpy()
        X_va = data_dict['X_valid'].select(sel).to_numpy()
        X_te = data_dict['X_test'].select(sel).to_numpy()

        _, _, train_metrics = trainer.evaluate_model(X_tr, data_dict['y_train_lgb'], 'Train')
        _, _, valid_metrics = trainer.evaluate_model(X_va, data_dict['y_valid_lgb'], 'Valid')
        y_proba, y_pred, test_metrics = trainer.evaluate_model(X_te, data_dict['y_test_lgb'], 'Test')

        trainer.save_predictions(data_dict['datetime_test'], data_dict['y_test_lgb'], y_pred, y_proba)
        trainer.parameter_scan(X_te, data_dict['y_test_lgb'])
        improvement_stats = trainer.calculate_model_improvement(
            train_metrics, valid_metrics, test_metrics, label_stats)

        return {
            'status': 'success',
            'symbol_name': symbol_name,
            'n_features': len(trainer.selected_features),
            'n_train': len(data_dict['X_train']),
            'n_valid': len(data_dict['X_valid']),
            'n_test':  len(data_dict['X_test']),
            'train_metrics': train_metrics,
            'valid_metrics': valid_metrics,
            'test_metrics':  test_metrics,
            'label_stats':   label_stats,
            'improvement_stats': improvement_stats,
        }

    def _generate_summary(self):
        success_symbols = [s for s, r in self.results.items() if r.get('status') == 'success']
        failed_symbols  = [s for s, r in self.results.items() if r.get('status') == 'failed']

        print(f"\n总结:  成功 {len(success_symbols)} 个品种  失败 {len(failed_symbols)} 个品种")
        if not success_symbols:
            return

        report = f"""
{'='*80}
多品种训练汇总报告
{'='*80}
训练时间: {self.timestamp}
总品种数: {len(self.symbols)}  成功: {len(success_symbols)}  失败: {len(failed_symbols)}
"""
        for symbol_name in success_symbols:
            r   = self.results[symbol_name]
            tm  = r['test_metrics']
            ls  = r.get('label_stats', {}).get('test', {})
            imp = r.get('improvement_stats', {}).get('test', {})
            report += f"""
{'-'*80}
品种: {symbol_name}
  数据集: 训练{r['n_train']:,}  验证{r['n_valid']:,}  测试{r['n_test']:,}
  标签分布(测试): 空{ls.get('pct_down',0):.1f}%  中性{ls.get('pct_neutral',0):.1f}%  多{ls.get('pct_up',0):.1f}%
  模型性能(测试): 多头精确率{tm['long_precision']:.4f}  空头精确率{tm['short_precision']:.4f}  胜率{tm['win_rate']:.4f}
  信号统计: {tm['n_signals']}个({tm['n_signals']/r['n_test']*100:.1f}%)  多空比{tm['long_short_ratio']:.2f}
  提升(vs随机): {imp.get('improvement_vs_random',0):+.1f}%  (vs分布): {imp.get('improvement_vs_distribution',0):+.1f}%
"""
        if failed_symbols:
            report += "\n失败品种:\n" + "\n".join(
                f"  - {s}: {self.results[s].get('error','unknown')}"
                for s in failed_symbols)
        report += f"\n{'='*80}\n输出目录: {self.output_dir}\n{'='*80}\n"

        report_file = self.output_dir / f'report_{self.timestamp}.txt'
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)
        print(report)
        print(f"✓ 详细报告已保存: {report_file}")


# ============================================================
# 主函数
# ============================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(description='期货多品种训练/因子分析流程')
    parser.add_argument('--config',  type=str, default='config.json')
    parser.add_argument('--mode',    type=str, default='train',
                        choices=['train', 'factor_analysis'],
                        help='运行模式: train | factor_analysis')
    parser.add_argument('--factors', type=str, default=None,
                        help='指定因子（逗号分隔），覆盖config中的设置')
    parser.add_argument('--symbols', type=str, default=None,
                        help='指定品种简称（逗号分隔），如 SHFE_au,SHFE_cu')
    args = parser.parse_args()

    if args.mode == 'train':
        Pipeline(config_path=args.config).run()

    elif args.mode == 'factor_analysis':
        from factor_analysis import FactorAnalysisPipeline
        fa = FactorAnalysisPipeline(
            config_path=args.config,
            factors_override=args.factors.split(',') if args.factors else None,
            symbols_override=args.symbols.split(',') if args.symbols else None,
        )
        fa.run()


if __name__ == '__main__':
    main()
