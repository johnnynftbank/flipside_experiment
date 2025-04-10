import pandas as pd
import numpy as np

# 데이터 로드
df = pd.read_csv('jackpot/query/query_result/jackpot_criteria_3822.csv')

print('=== 지표별 Z-score > 3인 데이터 개수 확인 ===')
cols = ['EXPECTED_ROI', 'ROI_STANDARD_DEVIATION', 'SHARPE_RATIO', 'WIN_LOSS_RATIO', 'MAX_TRADE_PROPORTION']

for col in cols:
    z_scores = (df[col] - df[col].mean()) / df[col].std()
    extreme_count = sum(z_scores > 3)
    percentage = extreme_count / len(df) * 100
    print(f'{col}: {extreme_count}개 ({percentage:.2f}%)')
    
    # 극단값이 있다면 예시 출력
    if extreme_count > 0:
        extreme_examples = df[z_scores > 3].sort_values(col, ascending=False).head(3)
        print("  극단값 예시:")
        for idx, row in extreme_examples.iterrows():
            print(f"  - {row['SWAPPER'][:10]}...: {row[col]}")
            
print('\n=== 지표별 상위 10% 값 확인 ===')
for col in cols:
    percentile_90 = df[col].quantile(0.9)
    print(f'{col}: {percentile_90:.4f}') 