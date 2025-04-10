import pandas as pd
import numpy as np

print("Loading data...")
df = pd.read_csv('jackpot/query/query_result/jackpot_criteria_3822.csv')
print(f"Data loaded with {len(df)} rows")

metrics = ['EXPECTED_ROI', 'ROI_STANDARD_DEVIATION', 'SHARPE_RATIO', 'WIN_LOSS_RATIO', 'MAX_TRADE_PROPORTION']
stats = pd.DataFrame(index=metrics)

print("Calculating statistics...")
for metric in metrics:
    print(f"Processing {metric}...")
    stats.loc[metric, 'Mean'] = df[metric].mean()
    stats.loc[metric, 'Median'] = df[metric].median()
    stats.loc[metric, 'Std Dev'] = df[metric].std()
    stats.loc[metric, 'Min'] = df[metric].min()
    stats.loc[metric, 'Max'] = df[metric].max()
    stats.loc[metric, 'Q1'] = df[metric].quantile(0.25)
    stats.loc[metric, 'Q3'] = df[metric].quantile(0.75)
    stats.loc[metric, 'Skewness'] = df[metric].skew()
    stats.loc[metric, 'Kurtosis'] = df[metric].kurt()
    
    # 이상치 계산 (IQR 방식)
    Q1 = df[metric].quantile(0.25)
    Q3 = df[metric].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df[(df[metric] < lower_bound) | (df[metric] > upper_bound)]
    stats.loc[metric, 'Outliers'] = len(outliers)

# 소수점 셋째 자리에서 반올림
stats = stats.round(3)

print("\n영문 칼럼명을 가진 통계 결과:")
print(stats) 