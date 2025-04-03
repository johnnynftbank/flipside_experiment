import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 데이터 로드
df = pd.read_csv('../../data/samples/solana/fee_distribution_by_range.csv')

# 설정
plt.style.use('seaborn')
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['figure.figsize'] = (15, 10)

# 1. 수수료 구간별 트랜잭션 분포 (막대 그래프)
fig1, ax1 = plt.subplots()
bars = ax1.bar(df['FEE_RANGE'], df['PERCENTAGE'])
ax1.set_title('Solana Transaction Fee Distribution by Range', pad=20)
ax1.set_xlabel('Fee Range')
ax1.set_ylabel('Percentage of Total Transactions (%)')

# 막대 위에 퍼센트 표시
for bar in bars:
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.1f}%',
             ha='center', va='bottom')

plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('../images/fee_distribution_bar.png', dpi=300, bbox_inches='tight')
plt.close()

# 2. 수수료 구간별 누적 분포 (파이 차트)
fig2, ax2 = plt.subplots()
plt.pie(df['PERCENTAGE'], labels=df['FEE_RANGE'], autopct='%1.1f%%',
        startangle=90)
plt.title('Solana Transaction Fee Distribution (Pie Chart)')
plt.axis('equal')
plt.savefig('../images/fee_distribution_pie.png', dpi=300, bbox_inches='tight')
plt.close()

# 3. 수수료 구간별 평균/최소/최대 값 비교 (로그 스케일)
fig3, ax3 = plt.subplots()
x = range(len(df))
ax3.plot(x, df['AVG_FEE_SOL'], 'go-', label='Average Fee')
ax3.plot(x, df['MIN_FEE_SOL'], 'bo-', label='Min Fee')
ax3.plot(x, df['MAX_FEE_SOL'], 'ro-', label='Max Fee')
ax3.set_yscale('log')
ax3.set_xticks(x)
ax3.set_xticklabels(df['FEE_RANGE'], rotation=45, ha='right')
ax3.set_title('Fee Range Analysis (Log Scale)')
ax3.set_ylabel('Fee in SOL (log scale)')
ax3.legend()
plt.tight_layout()
plt.savefig('../images/fee_range_analysis.png', dpi=300, bbox_inches='tight')
plt.close() 