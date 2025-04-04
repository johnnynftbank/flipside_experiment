import pandas as pd
import numpy as np

# 데이터 로드
df = pd.read_csv('../data/results/chain_meme_volume.csv')

# 1. 체인별 총 거래량 계산
volume_by_chain = df.groupby('blockchain')['volumeUSD'].sum().sort_values(ascending=False)
total_volume = volume_by_chain.sum()

print("=== 체인별 거래량 분석 ===")
for chain, volume in volume_by_chain.items():
    print(f"{chain}: {volume:,.0f} ({volume/total_volume:.1%})")
print(f"총 거래량: {total_volume:,.0f}\n")

# 2. 체인별 총 수익 계산
revenue_by_chain = df.groupby('blockchain')['botRevenueUSD'].sum().sort_values(ascending=False)
total_revenue = revenue_by_chain.sum()

print("=== 체인별 수익 분석 ===")
for chain, revenue in revenue_by_chain.items():
    print(f"{chain}: {revenue:,.0f} ({revenue/total_revenue:.1%})")
print(f"총 수익: {total_revenue:,.0f}\n")

# 3. 체인별 평균 일일 사용자 계산
users_by_chain = df.groupby('blockchain')['numberOfUsers'].mean().sort_values(ascending=False)
total_users = users_by_chain.sum()

print("=== 체인별 일평균 사용자 분석 ===")
for chain, users in users_by_chain.items():
    print(f"{chain}: {users:,.0f} ({users/total_users:.1%})")
print(f"총 일평균 사용자: {total_users:,.0f}\n")

# 4. 체인별 거래당 평균 금액 계산
avg_trade_by_chain = df.groupby('blockchain')['averageVolumePerTradeUSD'].mean().sort_values(ascending=False)

print("=== 체인별 거래당 평균 금액 ===")
for chain, avg in avg_trade_by_chain.items():
    print(f"{chain}: ${avg:.2f}")

# 5. 체인별 수익률 계산
chain_metrics = df.groupby('blockchain').agg({
    'volumeUSD': 'sum',
    'botRevenueUSD': 'sum'
}).reset_index()
chain_metrics['return_rate'] = (chain_metrics['botRevenueUSD'] / chain_metrics['volumeUSD']) * 100

print("\n=== 체인별 수익률 ===")
for _, row in chain_metrics.sort_values('return_rate', ascending=False).iterrows():
    print(f"{row['blockchain']}: {row['return_rate']:.3f}%") 