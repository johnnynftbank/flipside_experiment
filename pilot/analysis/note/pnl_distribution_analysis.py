import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

# Set seaborn style
sns.set_style("whitegrid")

# 데이터 로드
df = pd.read_csv('data/results/pnl_distribution_plot.csv')

# 전체 PnL 분포 플롯
plt.figure(figsize=(15, 10))

# 1. 전체 사용자 PnL 분포
plt.subplot(2, 2, 1)
all_data = df[df['USER_GROUP'] == 'all']
sns.histplot(data=all_data, x='TOTAL_PNL', weights='WALLET_COUNT', bins=50, stat='density')
sns.kdeplot(data=all_data, x='TOTAL_PNL', weights='WALLET_COUNT', color='red')
plt.title('Overall PnL Distribution')
plt.xlabel('PnL (SOL)')
plt.ylabel('Density')

# 2. 이탈 vs 활성 사용자 PnL 분포 비교
plt.subplot(2, 2, 2)
churned_data = df[df['USER_GROUP'] == 'churned']
active_data = df[df['USER_GROUP'] == 'active']

sns.kdeplot(data=churned_data, x='TOTAL_PNL', weights='WALLET_COUNT', label='Churned Users', alpha=0.6)
sns.kdeplot(data=active_data, x='TOTAL_PNL', weights='WALLET_COUNT', label='Active Users', alpha=0.6)
plt.title('PnL Distribution: Churned vs Active Users')
plt.xlabel('PnL (SOL)')
plt.ylabel('Density')
plt.legend()

# 3. 이탈 사용자 PnL 분포
plt.subplot(2, 2, 3)
sns.histplot(data=churned_data, x='TOTAL_PNL', weights='WALLET_COUNT', bins=50, stat='density')
sns.kdeplot(data=churned_data, x='TOTAL_PNL', weights='WALLET_COUNT', color='red')
plt.title('Churned Users PnL Distribution')
plt.xlabel('PnL (SOL)')
plt.ylabel('Density')

# 4. 활성 사용자 PnL 분포
plt.subplot(2, 2, 4)
sns.histplot(data=active_data, x='TOTAL_PNL', weights='WALLET_COUNT', bins=50, stat='density')
sns.kdeplot(data=active_data, x='TOTAL_PNL', weights='WALLET_COUNT', color='red')
plt.title('Active Users PnL Distribution')
plt.xlabel('PnL (SOL)')
plt.ylabel('Density')

plt.tight_layout()
plt.savefig('analysis/figures/pnl_distribution.png')
plt.close()

# 통계적 요약
for group in ['all', 'churned', 'active']:
    group_data = df[df['USER_GROUP'] == group]
    weighted_mean = np.average(group_data['TOTAL_PNL'], weights=group_data['WALLET_COUNT'])
    weighted_std = np.sqrt(np.average((group_data['TOTAL_PNL'] - weighted_mean)**2, weights=group_data['WALLET_COUNT']))
    
    print(f"\n{group.upper()} Users Statistics:")
    print(f"Weighted Mean PnL: {weighted_mean:.4f} SOL")
    print(f"Weighted Std Dev: {weighted_std:.4f} SOL")
    print(f"Total Wallets: {group_data['WALLET_COUNT'].sum():,}") 