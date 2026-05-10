# 行业滚动选股并保存权重 CSV
from src.strategies.base_strategy import StrategyConfig
from src.strategies.ml_strategy import MLStockSelectionStrategy, SectorNeutralMLStrategy
from src.data.data_fetcher import fetch_price_data
import os
import pandas as pd

# 读取清洁基本面数据（若外部提供了 sector/gsector 列，将启用行业中性版本）
fundamentals = pd.read_csv('data/fundamentals.csv')
# prices_df = fetch_price_data(tickers, start_date, end_date)

config = StrategyConfig(
    name="ML Stock Selection",
    description="Machine learning based stock selection"
)
base_strategy = MLStockSelectionStrategy(config)
sector_config = StrategyConfig(
    name="Sector Neutral ML",
    description="Sector-neutral ML strategy"
)
sector_strategy = SectorNeutralMLStrategy(sector_config)

# data_dict = { 'fundamentals': fundamentals_clean, 'prices': prices_df }
data_dict = { 'fundamentals': fundamentals}

# 采用滚动模式：每个季度独立训练/验证并预测
test_quarters = 4
top_quantile = 0.9

# 如果存在行业列，使用行业滚动选股，否则回退基础版本
# sector_col_exists = ('sector' in fundamentals_clean.columns) or ('gsector' in fundamentals_clean.columns)
sector_col_exists = False
if sector_col_exists:
    res = sector_strategy.generate_weights(
        data_dict,
        test_quarters=test_quarters,
        top_quantile=top_quantile,
        prediction_mode='rolling'
    )
else:
    # res = base_strategy.generate_weights(
    #     data_dict,
    #     test_quarters=test_quarters,
    #     top_quantile=top_quantile,
    #     prediction_mode='rolling'
    # )
    res = base_strategy.generate_weights(
        data=data_dict,
        prediction_mode='single',
        test_quarters=test_quarters,
        top_quantile=top_quantile,
        weight_method='equal',
        confirm_mode='today',
        execution_date='2025-10-12'  # 下单日期（或希望确认的日期）
    )

weights = res.weights.copy()
print(f"weights rows: {len(weights)}")

# 保存权重
out_dir = 'data'
os.makedirs(out_dir, exist_ok=True)
if sector_col_exists:
    out_path = os.path.join(out_dir, 'ml_weights_sector.csv')
else:
    out_path = os.path.join(out_dir, 'ml_weights_today.csv')
weights.to_csv(out_path, index=False)
print(f"Saved weights to {out_path}")