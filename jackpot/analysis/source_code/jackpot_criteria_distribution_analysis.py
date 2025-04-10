import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os
import matplotlib.font_manager as fm
import matplotlib as mpl

# 기본 폰트 설정 - 영어만 사용
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False  # 마이너스 기호 깨짐 방지

# 기본 스타일 설정
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("Set2")

# 경로 설정 개선 - 절대 경로 사용
base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
data_path = os.path.join(base_dir, 'query', 'query_result', 'jackpot_criteria_3822.csv')
REPORT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'report', 'image')

print(f"Using data file: {data_path}")
print(f"Saving results to: {REPORT_DIR}")

os.makedirs(REPORT_DIR, exist_ok=True)

# 데이터 로드
try:
    df = pd.read_csv(data_path)
    print(f"Successfully loaded data with shape: {df.shape}")
except FileNotFoundError:
    print(f"Error: Could not find the data file at {data_path}")
    # 대체 경로에서 파일 찾기 시도
    possible_paths = [
        os.path.join('/Users/johnny/Desktop/flipside_experiment/jackpot/query/query_result', 'jackpot_criteria_3822.csv'),
        os.path.join(os.getcwd(), 'jackpot/query/query_result', 'jackpot_criteria_3822.csv')
    ]
    for path in possible_paths:
        try:
            if os.path.exists(path):
                print(f"Found alternative path: {path}")
                df = pd.read_csv(path)
                print(f"Successfully loaded data with shape: {df.shape}")
                break
        except:
            continue
    else:
        raise FileNotFoundError(f"Could not find the data file at any of the expected locations")

# 데이터 탐색
print(f"데이터 크기: {df.shape}")
print("\n기본 통계 정보:")
print(df.describe())

# 분석할 지표 목록 (영어로만 표시)
metrics = ['EXPECTED_ROI', 'ROI_STANDARD_DEVIATION', 'SHARPE_RATIO', 'WIN_LOSS_RATIO', 'MAX_TRADE_PROPORTION']
metrics_labels = {
    'EXPECTED_ROI': 'Expected ROI',
    'ROI_STANDARD_DEVIATION': 'ROI Standard Deviation',
    'SHARPE_RATIO': 'Sharpe Ratio',
    'WIN_LOSS_RATIO': 'Win/Loss Ratio',
    'MAX_TRADE_PROPORTION': 'Max Trade Proportion'
}

# 통계치 저장할 데이터프레임 생성 (영어로만)
stats_df = pd.DataFrame(index=metrics, columns=[
    'Mean', 'Median', 'Std Dev', 'Min', 'Max', 
    '25%', '75%', 'Skewness', 'Kurtosis', 'Outliers'
])

# 이상치 필터링 함수 - 개선된 버전
def filter_outliers(series, method='iqr', n_std=3, quantile_range=(0.01, 0.99)):
    """
    이상치를 필터링하는 함수
    
    Parameters:
    -----------
    series : pandas.Series
        필터링할 데이터 시리즈
    method : str, default='iqr'
        필터링 방식 ('iqr', 'std', 'quantile')
    n_std : float, default=3
        method='std'일 때 사용할 표준편차 배수
    quantile_range : tuple, default=(0.01, 0.99)
        method='quantile'일 때 사용할 분위수 범위
        
    Returns:
    --------
    pandas.Series
        필터링된 데이터
    """
    if method == 'iqr':
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        return series[(series >= lower_bound) & (series <= upper_bound)]
    
    elif method == 'std':
        mean = series.mean()
        std = series.std()
        return series[(series >= mean - n_std * std) & (series <= mean + n_std * std)]
    
    elif method == 'quantile':
        lower = series.quantile(quantile_range[0])
        upper = series.quantile(quantile_range[1])
        return series[(series >= lower) & (series <= upper)]
    
    else:
        raise ValueError("method must be one of 'iqr', 'std', or 'quantile'")

# 각 지표별 히스토그램, 박스플롯 생성 및 통계치 계산
plt.figure(figsize=(20, 15))

for i, metric in enumerate(metrics):
    # IQR 방식으로 이상치 계산
    Q1 = df[metric].quantile(0.25)
    Q3 = df[metric].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    # 이상치 개수 계산
    outliers = df[(df[metric] < lower_bound) | (df[metric] > upper_bound)]
    outlier_count = len(outliers)
    
    # 개선된 이상치 필터링 - 더 엄격한 필터링 적용
    df_filtered = df.copy()
    df_filtered[metric] = filter_outliers(df[metric], method='quantile', quantile_range=(0.01, 0.99))
    df_filtered = df_filtered.dropna(subset=[metric])
    
    # 통계치 계산
    mean_val = df[metric].mean()
    median_val = df[metric].median()
    std_val = df[metric].std()
    min_val = df[metric].min()
    max_val = df[metric].max()
    q25 = df[metric].quantile(0.25)
    q75 = df[metric].quantile(0.75)
    skewness = df[metric].skew()
    kurtosis = df[metric].kurt()
    
    # 통계치 저장
    stats_df.loc[metric] = [mean_val, median_val, std_val, min_val, max_val, 
                           q25, q75, skewness, kurtosis, outlier_count]
    
    # 히스토그램 (이상치 제외)
    plt.subplot(len(metrics), 2, 2*i+1)
    sns.histplot(df_filtered[metric], kde=True, color='steelblue', alpha=0.7)
    plt.axvline(median_val, color='r', linestyle='--', linewidth=2, label=f'Median: {median_val:.2f}')
    plt.title(f'{metrics_labels[metric]} Distribution (Outliers Excluded)', fontsize=14, fontweight='bold')
    plt.xlabel(metrics_labels[metric], fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.legend(loc='best', frameon=True, fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # 박스플롯
    plt.subplot(len(metrics), 2, 2*i+2)
    sns.boxplot(x=df_filtered[metric], color='lightseagreen')
    plt.title(f'{metrics_labels[metric]} Boxplot (Outliers Excluded)', fontsize=14, fontweight='bold')
    plt.xlabel(metrics_labels[metric], fontsize=12)
    plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'{REPORT_DIR}/jackpot_criteria_distributions.png', dpi=300, bbox_inches='tight')
print(f"Saved figure to {REPORT_DIR}/jackpot_criteria_distributions.png")

# 히스토그램 (이상치 제외, 로그 스케일 적용)
plt.figure(figsize=(20, 15))
for i, metric in enumerate(metrics):
    # 개선된 이상치 필터링 적용
    df_filtered = df.copy()
    df_filtered[metric] = filter_outliers(df[metric], method='quantile', quantile_range=(0.01, 0.99))
    df_filtered = df_filtered.dropna(subset=[metric])
    
    plt.subplot(len(metrics), 1, i+1)
    sns.histplot(df_filtered[metric], kde=True, log_scale=(False, True), color='darkslateblue', alpha=0.7)
    plt.axvline(df_filtered[metric].median(), color='r', linestyle='--', linewidth=2, 
                label=f'Median: {df_filtered[metric].median():.2f}')
    plt.title(f'{metrics_labels[metric]} Distribution (Log Scale, Outliers Excluded)', fontsize=14, fontweight='bold')
    plt.xlabel(metrics_labels[metric], fontsize=12)
    plt.ylabel('Frequency (Log)', fontsize=12)
    plt.legend(loc='best', frameon=True, fontsize=10)
    plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'{REPORT_DIR}/jackpot_criteria_distributions_log.png', dpi=300, bbox_inches='tight')
print(f"Saved figure to {REPORT_DIR}/jackpot_criteria_distributions_log.png")

# 각 지표별 상세 분포 시각화 (이상치 제외)
for metric in metrics:
    plt.figure(figsize=(15, 12))
    
    # 개선된 이상치 필터링 적용
    df_filtered = df.copy()
    df_filtered[metric] = filter_outliers(df[metric], method='quantile', quantile_range=(0.01, 0.99))
    df_filtered = df_filtered.dropna(subset=[metric])
    
    # 히스토그램
    plt.subplot(2, 1, 1)
    ax = sns.histplot(df_filtered[metric], kde=True, color='darkblue', alpha=0.6)
    plt.axvline(df_filtered[metric].mean(), color='red', linestyle='--', linewidth=2, 
                label=f'Mean: {df_filtered[metric].mean():.4f}')
    plt.axvline(df_filtered[metric].median(), color='green', linestyle='-.', linewidth=2, 
                label=f'Median: {df_filtered[metric].median():.4f}')
    
    # 중앙값과 평균이 가까워서 겹치는 경우를 방지
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), loc='best', frameon=True, fontsize=10)
    
    plt.title(f'{metrics_labels[metric]} Distribution (Outliers Excluded)', fontsize=16, fontweight='bold')
    plt.xlabel(metrics_labels[metric], fontsize=14)
    plt.ylabel('Frequency', fontsize=14)
    plt.grid(True, alpha=0.3)
    
    # 박스플롯
    plt.subplot(2, 1, 2)
    sns.boxplot(x=df_filtered[metric], color='teal')
    plt.title(f'{metrics_labels[metric]} Boxplot (Outliers Excluded)', fontsize=16, fontweight='bold')
    plt.xlabel(metrics_labels[metric], fontsize=14)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{REPORT_DIR}/{metric}_distribution.png', dpi=300, bbox_inches='tight')
    print(f"Saved figure to {REPORT_DIR}/{metric}_distribution.png")

# 지갑 상태별 지표 분포 비교 (이상치 제외)
plt.figure(figsize=(20, 15))

# 지갑 상태명 변경 (영어로)
df['WALLET_STATUS_LABEL'] = df['WALLET_STATUS'].map({'active': 'Active', 'churned': 'Churned'})

for i, metric in enumerate(metrics):
    plt.subplot(len(metrics), 1, i+1)
    
    # 개선된 이상치 필터링 적용
    df_temp = df.copy()
    df_temp[metric] = filter_outliers(df[metric], method='quantile', quantile_range=(0.01, 0.99))
    df_filtered = df_temp.dropna(subset=[metric])
    
    # 지갑 상태별 박스플롯
    sns.boxplot(x='WALLET_STATUS_LABEL', y=metric, data=df_filtered, 
                palette={"Active": "lightseagreen", "Churned": "coral"})
    plt.title(f'{metrics_labels[metric]} Distribution by Wallet Status (Outliers Excluded)', fontsize=14, fontweight='bold')
    plt.xlabel('Wallet Status', fontsize=12)
    plt.ylabel(metrics_labels[metric], fontsize=12)
    plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'{REPORT_DIR}/jackpot_criteria_by_wallet_status.png', dpi=300, bbox_inches='tight')
print(f"Saved figure to {REPORT_DIR}/jackpot_criteria_by_wallet_status.png")

# 지표 간 상관관계 분석
# 이상치가 제거된 데이터로 상관관계 계산
df_corr = df.copy()
for metric in metrics:
    df_corr[metric] = filter_outliers(df[metric], method='quantile', quantile_range=(0.01, 0.99))
df_corr = df_corr.dropna(subset=metrics)

plt.figure(figsize=(15, 12))
correlation = df_corr[metrics].corr()
mask = np.triu(np.ones_like(correlation, dtype=bool))
cmap = sns.diverging_palette(230, 20, as_cmap=True)

# 상관관계 히트맵 시각화 개선
# 지표 이름을 영어로 변경
correlation.index = [metrics_labels[m] for m in correlation.index]
correlation.columns = [metrics_labels[m] for m in correlation.columns]

sns.heatmap(correlation, mask=mask, annot=True, fmt='.2f', cmap=cmap,
            vmin=-1, vmax=1, linewidths=1, annot_kws={"size": 12},
            cbar_kws={"shrink": .8})
plt.title('Correlation Between Metrics', fontsize=16, fontweight='bold')
plt.xticks(fontsize=12, rotation=45)
plt.yticks(fontsize=12, rotation=0)
plt.tight_layout()
plt.savefig(f'{REPORT_DIR}/jackpot_criteria_correlation.png', dpi=300, bbox_inches='tight')
print(f"Saved figure to {REPORT_DIR}/jackpot_criteria_correlation.png")

# 통계치 테이블 저장
stats_df.to_csv(f'{REPORT_DIR}/../jackpot_criteria_statistics.csv')
print("\n각 지표별 통계치:")
print(stats_df)

# 잭팟 추구형 기준값 분석
# 각 지표의 상위 10% 기준값 계산
percentiles = {}
for metric in metrics:
    threshold = np.percentile(df[metric], 90)
    percentiles[metric] = threshold

print("\n각 지표 상위 10% 기준값:")
for metric, threshold in percentiles.items():
    print(f"{metrics_labels[metric]}: {threshold:.4f}")

# 극단값을 갖는 지갑 비율 분석
extremes = {}
for metric in metrics:
    # Z-score 3 이상인 비율
    z_scores = np.abs(stats.zscore(df[metric], nan_policy='omit'))
    extreme_count = np.sum(z_scores > 3)
    extreme_percentage = (extreme_count / len(df)) * 100
    extremes[metric] = (extreme_count, extreme_percentage)

print("\n각 지표별 극단값(Z-score > 3)을 갖는 지갑 비율:")
for metric, (count, percentage) in extremes.items():
    print(f"{metrics_labels[metric]}: {count}개 ({percentage:.2f}%)") 