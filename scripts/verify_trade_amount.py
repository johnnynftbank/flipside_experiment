import pandas as pd
import numpy as np

# 데이터 로드
df = pd.read_csv('../data/results/chain_meme_volume.csv')

# 1. averageVolumePerTradeUSD 컬럼 평균값
print("=== averageVolumePerTradeUSD 컬럼 평균 ===")
avg_by_chain = df.groupby('blockchain')['averageVolumePerTradeUSD'].mean().sort_values(ascending=False)
for chain, avg in avg_by_chain.items():
    print(f"{chain}: ${avg:.2f}")

# 2. volumeUSD / numberOfTrades로 직접 계산
print("\n=== volumeUSD/numberOfTrades 직접 계산 ===")
chain_metrics = df.groupby('blockchain').agg({
    'volumeUSD': 'sum',
    'numberOfTrades': 'sum'
}).reset_index()
chain_metrics['calculated_avg'] = chain_metrics['volumeUSD'] / chain_metrics['numberOfTrades']
chain_metrics = chain_metrics.sort_values('calculated_avg', ascending=False)

for _, row in chain_metrics.iterrows():
    print(f"{row['blockchain']}: ${row['calculated_avg']:.2f} (총거래량: {row['volumeUSD']:,.0f}, 총거래수: {row['numberOfTrades']:,})") 