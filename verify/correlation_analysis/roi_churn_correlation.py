import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os
import matplotlib.font_manager as fm
import warnings

# 경고 필터링
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# 결과 디렉토리 생성
os.makedirs("verify/correlation_analysis/results", exist_ok=True)

# 데이터 로드
csv_path = "expected_roi_churn_final/query_result/meme_coin_roi_churn_3.csv"
df = pd.read_csv(csv_path)

# NaN 값 확인 및 처리
print("NaN 값 확인:")
print(df.isna().sum())
df = df.dropna()  # NaN 값이 있는 행 제거
print(f"NaN 제거 후 데이터 수: {len(df)}")

# 데이터 탐색
print("\n데이터 기본 정보:")
print(f"전체 데이터 수: {len(df)}")
print(f"변수 목록: {', '.join(df.columns)}")
print(f"이탈 지갑 수: {df[df['WALLET_STATUS'] == 'CHURNED'].shape[0]}")
print(f"활성 지갑 수: {df[df['WALLET_STATUS'] == 'ACTIVE'].shape[0]}")

# 이상치 처리: 기대 수익률 기준으로 상위 1% 제거
roi_upper_limit = np.percentile(df['EXPECTED_ROI'], 99)
print(f"\n이상치 처리: EXPECTED_ROI 상위 1% 이상 제거 (>{roi_upper_limit:.4f})")
df_cleaned = df[df['EXPECTED_ROI'] <= roi_upper_limit]
print(f"이상치 제거 후 데이터 수: {len(df_cleaned)}")

# 이탈 여부를 이진값으로 변환 (CHURNED=1, ACTIVE=0)
df_cleaned['IS_CHURNED'] = (df_cleaned['WALLET_STATUS'] == 'CHURNED').astype(int)

# 기술 통계량 계산
print("\n기술 통계량:")
stats_by_status = df_cleaned.groupby('WALLET_STATUS')['EXPECTED_ROI'].describe()
print(stats_by_status)

# 1. 포인트 바이시리얼 상관계수 계산
# 포인트 바이시리얼 상관계수는 연속변수(EXPECTED_ROI)와 이진변수(IS_CHURNED) 간의 상관관계를 측정
pb_corr, p_value = stats.pointbiserialr(df_cleaned['EXPECTED_ROI'], df_cleaned['IS_CHURNED'])

print("\n포인트 바이시리얼 상관 분석 결과:")
print(f"상관계수: {pb_corr:.4f}")
print(f"p-value: {p_value:.4f}")
print(f"통계적 유의성: {'유의함 (p < 0.05)' if p_value < 0.05 else '유의하지 않음 (p >= 0.05)'}")

# 2. t-test 진행
# 활성 지갑과 이탈 지갑 간의 기대 수익률 차이 검정
active_roi = df_cleaned[df_cleaned['WALLET_STATUS'] == 'ACTIVE']['EXPECTED_ROI']
churned_roi = df_cleaned[df_cleaned['WALLET_STATUS'] == 'CHURNED']['EXPECTED_ROI']

t_stat, t_p_value = stats.ttest_ind(active_roi, churned_roi, equal_var=False)  # 등분산성 가정하지 않음

print("\nt-test 결과:")
print(f"t-통계량: {t_stat:.4f}")
print(f"p-value: {t_p_value:.4f}")
print(f"통계적 유의성: {'유의함 (p < 0.05)' if t_p_value < 0.05 else '유의하지 않음 (p >= 0.05)'}")
print(f"활성 지갑 평균 ROI: {active_roi.mean():.4f}")
print(f"이탈 지갑 평균 ROI: {churned_roi.mean():.4f}")
print(f"평균 ROI 차이: {active_roi.mean() - churned_roi.mean():.4f}")

# 영어로 그래프 제목 설정
plt.rc('font', family='DejaVu Sans')

# 시각화: 박스 플롯
plt.figure(figsize=(10, 6))
sns.set_style("whitegrid")
ax = sns.boxplot(x='WALLET_STATUS', y='EXPECTED_ROI', data=df_cleaned, hue='WALLET_STATUS', legend=False)
ax.set_title('Distribution of Expected ROI by Wallet Status', fontsize=15)
ax.set_xlabel('Wallet Status', fontsize=12)
ax.set_ylabel('Expected ROI', fontsize=12)
plt.tight_layout()
plt.savefig('verify/correlation_analysis/results/roi_wallet_status_boxplot.png', dpi=300)

# 시각화: 히스토그램
plt.figure(figsize=(12, 6))
ax = sns.histplot(data=df_cleaned, x='EXPECTED_ROI', hue='WALLET_STATUS', element='step', stat='density', common_norm=False)
ax.set_title('Distribution of Expected ROI by Wallet Status', fontsize=15)
ax.set_xlabel('Expected ROI', fontsize=12)
ax.set_ylabel('Density', fontsize=12)
plt.tight_layout()
plt.savefig('verify/correlation_analysis/results/roi_wallet_status_histogram.png', dpi=300)

# 시각화: 산점도 (ROI vs 거래 일수)
plt.figure(figsize=(12, 8))
ax = sns.scatterplot(data=df_cleaned, x='EXPECTED_ROI', y='TRADED_DAYS', hue='WALLET_STATUS', alpha=0.6)
ax.set_title('Expected ROI vs Trading Days by Wallet Status', fontsize=15)
ax.set_xlabel('Expected ROI', fontsize=12)
ax.set_ylabel('Trading Days', fontsize=12)
plt.tight_layout()
plt.savefig('verify/correlation_analysis/results/roi_vs_trading_days.png', dpi=300)

# 시각화: 산점도 (ROI vs 거래 횟수)
plt.figure(figsize=(12, 8))
ax = sns.scatterplot(data=df_cleaned, x='EXPECTED_ROI', y='MEME_TRADE_COUNT', hue='WALLET_STATUS', alpha=0.6)
ax.set_title('Expected ROI vs Trade Count by Wallet Status', fontsize=15)
ax.set_xlabel('Expected ROI', fontsize=12)
ax.set_ylabel('Meme Trade Count', fontsize=12)
plt.tight_layout()
plt.savefig('verify/correlation_analysis/results/roi_vs_trade_count.png', dpi=300)

# 결과 요약 파일 생성
with open('verify/correlation_analysis/results/correlation_analysis_summary.txt', 'w') as f:
    f.write("## 기대 수익률과 이탈 지갑 간의 상관 관계 분석 결과\n\n")
    f.write(f"분석 데이터: {csv_path}\n")
    f.write(f"전체 데이터 수: {len(df)}\n")
    f.write(f"이상치 제거 후 데이터 수: {len(df_cleaned)}\n")
    f.write(f"이탈 지갑 수: {df_cleaned[df_cleaned['WALLET_STATUS'] == 'CHURNED'].shape[0]}\n")
    f.write(f"활성 지갑 수: {df_cleaned[df_cleaned['WALLET_STATUS'] == 'ACTIVE'].shape[0]}\n\n")
    
    f.write("### 1. 포인트 바이시리얼 상관 분석 결과\n")
    f.write(f"상관계수: {pb_corr:.4f}\n")
    f.write(f"p-value: {p_value:.4f}\n")
    f.write(f"통계적 유의성: {'유의함 (p < 0.05)' if p_value < 0.05 else '유의하지 않음 (p >= 0.05)'}\n\n")
    
    f.write("### 2. t-test 결과\n")
    f.write(f"t-통계량: {t_stat:.4f}\n")
    f.write(f"p-value: {t_p_value:.4f}\n")
    f.write(f"통계적 유의성: {'유의함 (p < 0.05)' if t_p_value < 0.05 else '유의하지 않음 (p >= 0.05)'}\n")
    f.write(f"활성 지갑 평균 ROI: {active_roi.mean():.4f}\n")
    f.write(f"이탈 지갑 평균 ROI: {churned_roi.mean():.4f}\n")
    f.write(f"평균 ROI 차이: {active_roi.mean() - churned_roi.mean():.4f}\n\n")
    
    f.write("### 3. 기술 통계량\n")
    f.write(f"{stats_by_status.to_string()}\n\n")
    
    if pb_corr > 0:
        f.write("### 해석\n")
        f.write(f"양의 상관관계({pb_corr:.4f})는 낮은 기대 수익률이 활성 상태와 연관되어 있으며, 높은 기대 수익률이 이탈 가능성과 연관되어 있음을 시사합니다.\n")
        f.write("이는 기대 수익률이 높은 지갑일수록 이탈할 가능성이 높다는 것을 의미합니다.\n")
    elif pb_corr < 0:
        f.write("### 해석\n")
        f.write(f"음의 상관관계({pb_corr:.4f})는 높은 기대 수익률이 활성 상태와 연관되어 있으며, 낮은 기대 수익률이 이탈 가능성과 연관되어 있음을 시사합니다.\n")
        f.write("이는 기대 수익률이 낮은 지갑일수록 이탈할 가능성이 높다는 것을 의미합니다.\n")
    else:
        f.write("### 해석\n")
        f.write("기대 수익률과 이탈 사이에 유의미한 상관관계가 발견되지 않았습니다.\n")
    
    if t_p_value < 0.05:
        f.write("\nt-test 결과는 활성 지갑과 이탈 지갑 간의 기대 수익률 평균에 통계적으로 유의미한 차이가 있음을 나타냅니다.\n")
    else:
        f.write("\nt-test 결과는 활성 지갑과 이탈 지갑 간의 기대 수익률 평균에 통계적으로 유의미한 차이가 없음을 나타냅니다.\n")

print("\n분석이 완료되었습니다. 결과는 verify/correlation_analysis/results/ 폴더에 저장되었습니다.") 