import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.font_manager as fm

# 데이터 로드 및 전처리
df = pd.read_csv('../../data/samples/solana/fee_distribution_by_range.csv')

# 수수료 구간 레이블 업데이트
fee_range_mapping = {
    '1. 5000 (기본 수수료)': '0.000005 SOL (기본 수수료)',
    '2. 5001-5999 (기본 수수료 근접)': '0.000005-0.000006 SOL (기본 수수료 +20% 이내)',
    '3. 6000-9999 (낮은 수수료)': '0.000006-0.00001 SOL (기본 수수료 2배 이내)',
    '4. 10000-49999 (중간 수수료)': '0.00001-0.00005 SOL (기본 수수료 2-10배)',
    '5. 50000-99999 (중고 수수료)': '0.00005-0.0001 SOL (기본 수수료 10-20배)',
    '6. 100000-999999 (높은 수수료)': '0.0001-0.001 SOL (기본 수수료 20-200배)',
    '7. 1000000+ (매우 높은 수수료)': '> 0.001 SOL (기본 수수료 200배 초과)'
}

# 데이터 전처리
df = df.dropna()  # NaN 값 제거
df['FEE_RANGE'] = df['FEE_RANGE'].map(fee_range_mapping)

# 시각화 스타일 설정
plt.rcParams['font.family'] = 'AppleGothic'  # macOS용 한글 폰트
plt.rcParams['figure.figsize'] = (15, 10)
plt.rcParams['axes.unicode_minus'] = False  # 마이너스 기호 문제 해결
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3

# 1. 수수료 구간별 트랜잭션 분포 (막대 그래프)
fig1, ax1 = plt.subplots()
bars = ax1.bar(df['FEE_RANGE'], df['PERCENTAGE'])
ax1.set_title('Solana Transaction Fee Distribution by Range', pad=20, fontsize=14)
ax1.set_xlabel('Fee Range (in SOL)', fontsize=12)
ax1.set_ylabel('Percentage of Total Transactions (%)', fontsize=12)

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
wedges, texts, autotexts = plt.pie(df['PERCENTAGE'], 
                                  labels=df['FEE_RANGE'], 
                                  autopct='%1.1f%%',
                                  startangle=90)
plt.title('Solana Transaction Fee Distribution (Pie Chart)', fontsize=14, pad=20)
plt.setp(autotexts, size=8, weight="bold")
plt.setp(texts, size=8)
plt.axis('equal')
plt.savefig('../images/fee_distribution_pie.png', dpi=300, bbox_inches='tight')
plt.close()

# 3. 수수료 구간별 평균/최소/최대 값 비교 (로그 스케일)
fig3, ax3 = plt.subplots()
x = range(len(df))
ax3.plot(x, df['AVG_FEE_SOL'], 'go-', label='Average Fee', linewidth=2)
ax3.plot(x, df['MIN_FEE_SOL'], 'bo-', label='Min Fee', linewidth=2)
ax3.plot(x, df['MAX_FEE_SOL'], 'ro-', label='Max Fee', linewidth=2)
ax3.set_yscale('log')
ax3.set_xticks(x)
ax3.set_xticklabels(df['FEE_RANGE'], rotation=45, ha='right')
ax3.set_title('Fee Range Analysis (Log Scale)', fontsize=14, pad=20)
ax3.set_ylabel('Fee in SOL (log scale)', fontsize=12)
ax3.legend(fontsize=10)
ax3.grid(True, which="both", ls="-", alpha=0.2)
plt.tight_layout()
plt.savefig('../images/fee_range_analysis.png', dpi=300, bbox_inches='tight')
plt.close() 