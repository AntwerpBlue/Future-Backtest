import datetime
import polars as pl
from pathlib import Path


def download_complete_data(symbol, start_date, end_date, 
                          periods=['1m', '1h', '1d'],
                          save_format='both',
                          save_dir="./data",
                          source_dir="/program1/tqcta_data/kline/main"):
    """
    从本地CSV文件读取并处理数据
    
    Args:
        symbol: 合约代码，如 "KQ.m@SHFE.rb"
        start_date: 开始日期
        end_date: 结束日期
        periods: 要拉取的周期列表 ['1m', '1h', '1d']
        save_format: 保存格式 ('csv', 'parquet', 'both')
        save_dir: 保存目录
        source_dir: 源数据目录
    
    Returns:
        saved_files: 保存的文件路径列表
    """
    print(f"\n开始处理 {symbol} 的数据")
    print(f"时间范围：{start_date} 至 {end_date}")
    print(f"周期：{periods}")
    print(f"源目录：{source_dir}\n")
    
    # 周期映射：周期名称 -> 秒数
    period_to_seconds = {
        '1m': 60,
        '3m': 180,
        '5m': 300,
        '15m': 900,
        '30m': 1800,
        '1h': 3600,
        '2h': 7200,
        '4h': 14400,
        '1d': 86400,
    }
    
    source_dir = Path(source_dir)
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True)
    
    # 时间范围（扩展到完整的一天）
    start_datetime = datetime.datetime.combine(start_date, datetime.time.min)
    end_datetime = datetime.datetime.combine(end_date, datetime.time.max)
    
    saved_files = []
    
    for period in periods:
        if period not in period_to_seconds:
            print(f"警告：不支持的周期 {period}，跳过")
            continue
        
        seconds = period_to_seconds[period]
        
        # 构造源文件路径
        source_file = source_dir / f"{symbol}_{seconds}.csv"
        
        if not source_file.exists():
            print(f"⚠ 警告：文件不存在 {source_file.name}，跳过")
            continue
        
        print(f"\n处理 {period} 周期数据...")
        print(f"  读取文件: {source_file.name}")
        
        try:
            # 读取CSV
            df = pl.read_csv(source_file)
            print(f"  原始数据: {len(df)} 条")
            
            # 清洗列名（去掉品种前缀）
            df = _clean_column_names(df)
            
            # 过滤时间范围
            df = _filter_by_time(df, start_datetime, end_datetime)
            
            if len(df) == 0:
                print(f"过滤后无数据")
                continue
            
            print(f"  处理后数据: {len(df)} 条")
            
            # 显示时间范围（使用datetime或datetime_nano）
            time_col = 'datetime_nano' if 'datetime_nano' in df.columns else 'datetime'
            if time_col in df.columns:
                df_temp = df.with_columns([
                    pl.col(time_col).cast(pl.UInt64).cast(pl.Datetime('ns')).alias('_dt')
                ])
                print(f"  时间范围: {df_temp['_dt'].min()} ~ {df_temp['_dt'].max()}")
            
            # 保存文件
            files = _save_data(df, symbol, start_date, end_date, period, save_format, save_dir)
            saved_files.extend(files)
            
        except Exception as e:
            print(f"处理失败: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"\n{'='*60}")
    print(f"数据处理完成！共保存 {len(saved_files)} 个文件到 {save_dir}")
    print(f"{'='*60}\n")
    
    return saved_files


def _clean_column_names(df: pl.DataFrame) -> pl.DataFrame:
    """
    清洗列名，去掉品种前缀
    
    例如：
    - KQ.m@CFFEX.IC.open -> open
    - KQ.m@CFFEX.IC.high -> high
    - datetime -> datetime (保持不变)
    - datetime_nano -> datetime_nano (保持不变)
    """
    rename_map = {}
    
    for col in df.columns:
        # 如果列名包含品种前缀（包含"KQ.m@"）
        if 'KQ.m@' in col:
            # 提取最后一个点号之后的内容
            new_name = col.split('.')[-1]
            rename_map[col] = new_name
    
    # 重命名列
    if rename_map:
        df = df.rename(rename_map)
        print(f"  清洗列名: {len(rename_map)} 个列")
    
    return df


def _filter_by_time(df: pl.DataFrame, start_datetime: datetime.datetime, 
                   end_datetime: datetime.datetime) -> pl.DataFrame:
    """
    按时间范围过滤数据
    """
    # 确定使用哪个时间列
    if 'datetime_nano' in df.columns:
        time_col = 'datetime_nano'
    elif 'datetime' in df.columns:
        time_col = 'datetime'
    else:
        print("  警告：没有找到时间列，跳过时间过滤")
        return df
    
    # 将时间列转换为datetime类型进行过滤
    df = df.with_columns([
        pl.col(time_col).cast(pl.UInt64).cast(pl.Datetime('ns')).alias('_temp_dt')
    ])
    
    # 过滤时间范围
    df = df.filter(
        (pl.col('_temp_dt') >= start_datetime) & 
        (pl.col('_temp_dt') <= end_datetime)
    )
    
    # 删除临时列
    df = df.drop('_temp_dt')
    
    return df


def _save_data(df: pl.DataFrame, symbol: str, start_date: datetime.date, 
               end_date: datetime.date, period: str, save_format: str, 
               save_dir: Path) -> list:
    """
    保存数据为CSV和/或Parquet格式
    """
    # 处理文件名：KQ.m@SHFE.rb -> KQ_m_SHFE_rb
    safe_symbol = symbol.replace(".", "_").replace("@", "_")
    base_name = f"{safe_symbol}_{period}_{start_date}_{end_date}"
    
    saved_files = []
    
    # 保存CSV
    if save_format in ['csv', 'both']:
        csv_file = save_dir / f"{base_name}.csv"
        df.write_csv(csv_file)
        saved_files.append(csv_file)
        print(f"  ✓ CSV: {csv_file.name}")
    
    # 保存Parquet
    if save_format in ['parquet', 'both']:
        parquet_file = save_dir / f"{base_name}.parquet"
        df.write_parquet(parquet_file)
        saved_files.append(parquet_file)
        print(f"  ✓ Parquet: {parquet_file.name}")
    
    return saved_files


def batch_download(symbols: list, start_date: datetime.date, end_date: datetime.date,
                  periods: list = ['1m', '1h', '1d'],
                  save_format: str = 'both',
                  save_dir: str = "./data",
                  source_dir: str = "/program1/tqcta_data/kline/main"):
    """
    批量下载多个品种的数据
    
    Args:
        symbols: 品种列表，如 ["KQ.m@SHFE.rb", "KQ.m@SHFE.cu"]
        其他参数同download_complete_data
    """
    print(f"\n{'='*60}")
    print(f"批量处理 {len(symbols)} 个品种")
    print(f"{'='*60}")
    
    all_saved_files = []
    success_count = 0
    fail_count = 0
    
    for i, symbol in enumerate(symbols, 1):
        print(f"\n[{i}/{len(symbols)}] 处理品种: {symbol}")
        print("-" * 60)
        
        try:
            saved_files = download_complete_data(
                symbol=symbol,
                start_date=start_date,
                end_date=end_date,
                periods=periods,
                save_format=save_format,
                save_dir=save_dir,
                source_dir=source_dir
            )
            
            if saved_files:
                all_saved_files.extend(saved_files)
                success_count += 1
            else:
                fail_count += 1
                
        except Exception as e:
            print(f"处理失败: {e}")
            fail_count += 1
    
    print(f"\n{'='*60}")
    print(f"批量处理完成")
    print(f"成功: {success_count} 个品种")
    print(f"失败: {fail_count} 个品种")
    print(f"总文件: {len(all_saved_files)} 个")
    print(f"{'='*60}\n")
    
    return all_saved_files


if __name__ == "__main__":
    # 示例1：单个品种
    download_complete_data(
        symbol="KQ.m@CFFEX.IC",
        start_date=datetime.date(2021, 1, 1),
        end_date=datetime.date(2021, 6, 30),
        periods=['1m', '1h', '1d'],
        save_format='both',
        save_dir="./data"
    )
    
    # 示例2：批量处理多个品种
    # symbols = [
    #     "KQ.m@SHFE.rb",
    #     "KQ.m@SHFE.cu",
    #     "KQ.m@DCE.i",
    #     "KQ.m@CZCE.MA",
    # ]
    # batch_download(
    #     symbols=symbols,
    #     start_date=datetime.date(2021, 1, 1),
    #     end_date=datetime.date(2021, 6, 30),
    #     periods=['1m', '1h', '1d'],
    #     save_format='parquet'
    # )
