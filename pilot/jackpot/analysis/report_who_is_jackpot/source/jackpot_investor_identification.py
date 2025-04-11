import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, confusion_matrix, classification_report
from scipy.stats import mannwhitneyu, ttest_ind
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.cm as cm

# 기본 폰트 설정
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False  # 마이너스 기호 깨짐 방지

# 기본 스타일 설정
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("Set2")

# 경로 설정
base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
data_path = os.path.join(base_dir, 'jackpot', 'query', 'query_result', 'jackpot_criteria_3822.csv')
REPORT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VIZ_DIR = os.path.join(REPORT_DIR, "visualization")
SOURCE_DIR = os.path.join(REPORT_DIR, "source")

print(f"Using data file: {data_path}")
print(f"Saving results to: {REPORT_DIR}")
print(f"Visualization directory: {VIZ_DIR}")

os.makedirs(VIZ_DIR, exist_ok=True)
os.makedirs(SOURCE_DIR, exist_ok=True)

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

# 분석할 지표 목록
metrics = ['EXPECTED_ROI', 'ROI_STANDARD_DEVIATION', 'SHARPE_RATIO', 'WIN_LOSS_RATIO', 'MAX_TRADE_PROPORTION']
metrics_labels = {
    'EXPECTED_ROI': 'Expected ROI',
    'ROI_STANDARD_DEVIATION': 'ROI Standard Deviation',
    'SHARPE_RATIO': 'Sharpe Ratio',
    'WIN_LOSS_RATIO': 'Win/Loss Ratio',
    'MAX_TRADE_PROPORTION': 'Max Trade Proportion'
}

# 이상치 필터링 함수
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

# 데이터 전처리
df_clean = df.copy()
for metric in metrics:
    df_clean[metric] = filter_outliers(df[metric], method='quantile', quantile_range=(0.01, 0.99))
df_clean = df_clean.dropna(subset=metrics)

# 지갑 상태 영어로 변환
df_clean['WALLET_STATUS_LABEL'] = df_clean['WALLET_STATUS'].map({'active': 'Active', 'churned': 'Churned'})

print(f"Clean data shape after outlier removal: {df_clean.shape}")

##############################################
# 1. 기본 통계 분석 - 잭팟 추구자 관점에서 재해석
##############################################

# 각 지표별 기본 통계값
stats_df = pd.DataFrame()
for metric in metrics:
    stats_df[f"{metrics_labels[metric]}"] = [
        df_clean[metric].mean(),
        df_clean[metric].median(),
        df_clean[metric].std(),
        df_clean[metric].min(),
        df_clean[metric].max(),
        df_clean[metric].quantile(0.25),
        df_clean[metric].quantile(0.75),
        stats.skew(df_clean[metric].dropna()),
        stats.kurtosis(df_clean[metric].dropna()),
        len(df_clean[df_clean[metric] > df_clean[metric].mean() + 3*df_clean[metric].std()]) + 
        len(df_clean[df_clean[metric] < df_clean[metric].mean() - 3*df_clean[metric].std()])
    ]
stats_df.index = ['Mean', 'Median', 'Std Dev', 'Min', 'Max', 'Q1', 'Q3', 'Skewness', 'Kurtosis', 'Outliers Count']
stats_df = stats_df.T

# 활성/이탈 그룹별 통계 계산 (간단한 형태로)
active_df = df_clean[df_clean['WALLET_STATUS'] == 'active']
churned_df = df_clean[df_clean['WALLET_STATUS'] == 'churned']

active_stats = {}
churned_stats = {}

for metric in metrics:
    active_stats[f'{metric}_mean'] = active_df[metric].mean()
    active_stats[f'{metric}_median'] = active_df[metric].median()
    active_stats[f'{metric}_std'] = active_df[metric].std()
    
    churned_stats[f'{metric}_mean'] = churned_df[metric].mean()
    churned_stats[f'{metric}_median'] = churned_df[metric].median()
    churned_stats[f'{metric}_std'] = churned_df[metric].std()

# 간단한 데이터프레임으로 변환
status_stats_df = pd.DataFrame({
    'Active': active_stats,
    'Churned': churned_stats
})

# 통계 결과 저장
stats_df.to_csv(os.path.join(SOURCE_DIR, 'metric_statistics.csv'))
status_stats_df.to_csv(os.path.join(SOURCE_DIR, 'status_group_statistics.csv'))

# 이탈률에 영향을 미치는 지표 시각화
plt.figure(figsize=(14, 10))
metric_impact = {}

for i, metric in enumerate(metrics):
    # 상관 관계 및 이탈률 영향 계산
    churn_corr = np.corrcoef(df_clean[metric], df_clean['WALLET_STATUS'] == 'churned')[0, 1]
    metric_impact[metric] = abs(churn_corr)
    
    # 이탈/활성 그룹별 지표 분포
    plt.subplot(2, 3, i+1)
    sns.histplot(
        data=df_clean, 
        x=metric, 
        hue='WALLET_STATUS_LABEL',
        bins=30,
        alpha=0.6,
        palette={'Active': 'lightseagreen', 'Churned': 'coral'}
    )
    plt.title(f'{metrics_labels[metric]}\nChurn Correlation: {churn_corr:.3f}', fontsize=12)
    if abs(churn_corr) > 0.3:
        plt.title(f'{metrics_labels[metric]}\nChurn Correlation: {churn_corr:.3f} ⭐', fontsize=12)

plt.tight_layout()
plt.savefig(os.path.join(VIZ_DIR, 'metric_churn_correlation.png'), dpi=300, bbox_inches='tight')

# 이탈률과 지표의 관계 요약 텍스트 저장
sorted_impact = sorted(metric_impact.items(), key=lambda x: abs(x[1]), reverse=True)
with open(os.path.join(SOURCE_DIR, 'churn_correlation_ranking.txt'), 'w') as f:
    f.write("지표별 이탈률 연관성 순위 (절대값 기준):\n")
    for metric, impact in sorted_impact:
        f.write(f"{metrics_labels[metric]}: {impact:.4f}\n")

##############################################
# 2. 단순 분류 (Percentile-based Classification)
##############################################

# 상위 10% 기준으로 잭팟 추구 그룹 식별
percentiles = {10: {}}
for metric in metrics:
    # 지표의 특성에 따라 상위/하위 기준 설정
    if metric in ['SHARPE_RATIO']:  # 낮을수록 잭팟 추구 성향
        threshold = np.percentile(df_clean[metric], 10)
        mask = df_clean[metric] <= threshold
    else:  # 높을수록 잭팟 추구 성향
        threshold = np.percentile(df_clean[metric], 90)
        mask = df_clean[metric] >= threshold
    
    # 잭팟 추구 그룹 식별 및 저장
    jackpot_group = df_clean[mask].copy()
    non_jackpot_group = df_clean[~mask].copy()
    
    percentiles[10][metric] = {
        'threshold': threshold,
        'jackpot_count': len(jackpot_group),
        'non_jackpot_count': len(non_jackpot_group),
        'jackpot_group': jackpot_group,
        'non_jackpot_group': non_jackpot_group
    }

# 각 그룹의 특성 분석
group_stats = {10: {}}
for metric in metrics:
    group_stats[10][metric] = {}
    jackpot_group = percentiles[10][metric]['jackpot_group']
    non_jackpot_group = percentiles[10][metric]['non_jackpot_group']
    
    # 평균값 계산
    for m in metrics:
        group_stats[10][metric][f'jackpot_mean_{m}'] = jackpot_group[m].mean()
        group_stats[10][metric][f'non_jackpot_mean_{m}'] = non_jackpot_group[m].mean()
    
    # 이탈률 계산
    group_stats[10][metric]['jackpot_churn_rate'] = (jackpot_group['WALLET_STATUS'] == 'churned').mean() * 100
    group_stats[10][metric]['non_jackpot_churn_rate'] = (non_jackpot_group['WALLET_STATUS'] == 'churned').mean() * 100
    
    # 통계적 유의성 검정
    for m in metrics:
        # 정규성 검정 결과에 따라 t-test 또는 Mann-Whitney U test 선택
        _, p_normal_jp = stats.shapiro(jackpot_group[m].dropna())
        _, p_normal_non_jp = stats.shapiro(non_jackpot_group[m].dropna())
        
        if p_normal_jp > 0.05 and p_normal_non_jp > 0.05:  # 두 그룹 모두 정규 분포일 경우
            _, p_val = ttest_ind(jackpot_group[m].dropna(), non_jackpot_group[m].dropna(), equal_var=False)
            test_method = 't-test'
        else:  # 정규 분포가 아닐 경우
            _, p_val = mannwhitneyu(jackpot_group[m].dropna(), non_jackpot_group[m].dropna())
            test_method = 'Mann-Whitney U'
        
        group_stats[10][metric][f'pvalue_{m}'] = p_val
        group_stats[10][metric][f'test_method_{m}'] = test_method
        group_stats[10][metric][f'significant_{m}'] = p_val < 0.05

# 결과 요약 데이터프레임 생성
percentile_summary = pd.DataFrame(index=metrics)
for metric in metrics:
    percentile_summary.loc[metric, 'Threshold'] = percentiles[10][metric]['threshold']
    percentile_summary.loc[metric, 'Jackpot Group Size'] = percentiles[10][metric]['jackpot_count']
    percentile_summary.loc[metric, 'Jackpot Churn Rate (%)'] = group_stats[10][metric]['jackpot_churn_rate']
    percentile_summary.loc[metric, 'Regular Churn Rate (%)'] = group_stats[10][metric]['non_jackpot_churn_rate']
    percentile_summary.loc[metric, 'Churn Rate Diff (%)'] = group_stats[10][metric]['jackpot_churn_rate'] - group_stats[10][metric]['non_jackpot_churn_rate']
    
    # 주요 지표 평균값
    for m in metrics:
        percentile_summary.loc[metric, f'Jackpot Mean {m}'] = group_stats[10][metric][f'jackpot_mean_{m}']

# 지표 이름을 영어로 변경
percentile_summary.index = [metrics_labels[m] for m in percentile_summary.index]

# CSV로 저장
percentile_summary.to_csv(os.path.join(SOURCE_DIR, 'percentile_classification_summary.csv'))

# 이탈률 시각화
plt.figure(figsize=(12, 8))
churn_diff = percentile_summary['Churn Rate Diff (%)'].sort_values(ascending=False)

# 컬러맵 설정 (양수는 빨간색, 음수는 파란색)
colors = ['steelblue' if x < 0 else 'crimson' for x in churn_diff]

ax = sns.barplot(
    x=churn_diff.index,
    y=churn_diff.values,
    palette=colors
)

# 각 막대 위에 이탈률 표시
for i, (idx, val) in enumerate(zip(churn_diff.index, churn_diff.values)):
    jackpot_rate = percentile_summary.loc[idx, 'Jackpot Churn Rate (%)']
    regular_rate = percentile_summary.loc[idx, 'Regular Churn Rate (%)']
    ax.text(i, val + (5 if val > 0 else -10), f'J: {jackpot_rate:.1f}%\nR: {regular_rate:.1f}%', 
            ha='center', va='center', color='black', fontsize=9, fontweight='bold')

plt.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
plt.title('Churn Rate Difference: Jackpot Group vs Regular Group', fontsize=14)
plt.ylabel('Difference in Churn Rate (%)', fontsize=12)
plt.xlabel('Metric Used for Classification', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig(os.path.join(VIZ_DIR, 'jackpot_churn_rate_comparison.png'), dpi=300, bbox_inches='tight')

# 지표별 잭팟/일반 그룹 특성 비교 시각화
metric_numbers = {metrics[i]: f'{i+1}' for i in range(len(metrics))}
metric_mapping = {f'{i+1}': metrics[i] for i in range(len(metrics))}
metric_mapping_full = {f'{i+1}': metrics_labels[metrics[i]] for i in range(len(metrics))}

print("\n지표-번호 매핑:")
for num, metric in metric_mapping_full.items():
    print(f"{num}: {metric}")

# 각 지표별 비교 데이터 준비
comparison_data = []
for metric in metrics:
    jp_means = [group_stats[10][metric][f'jackpot_mean_{m}'] for m in metrics]
    non_jp_means = [group_stats[10][metric][f'non_jackpot_mean_{m}'] for m in metrics]
    
    # 데이터프레임으로 변환하여 저장
    for i, m in enumerate(metrics):
        comparison_data.append({
            'Classification Metric': metrics_labels[metric],
            'Group': 'Jackpot Seeking',
            'Metric': metric_numbers[m],
            'Value': jp_means[i],
            'Original_Metric': metrics_labels[m]
        })
        comparison_data.append({
            'Classification Metric': metrics_labels[metric],
            'Group': 'Regular',
            'Metric': metric_numbers[m],
            'Value': non_jp_means[i],
            'Original_Metric': metrics_labels[m]
        })

comparison_df = pd.DataFrame(comparison_data)

# 바 그래프로 시각화
g = sns.catplot(
    data=comparison_df,
    kind='bar',
    x='Metric',
    y='Value',
    hue='Group',
    col='Classification Metric',
    palette={"Jackpot Seeking": "crimson", "Regular": "steelblue"},
    alpha=0.8,
    height=4,
    aspect=1.2,
    col_wrap=3,
    sharey=False
)

g.set_titles("{col_name} Based Classification")
g.set_axis_labels("Metric Number", "Average Value")

# 범례 추가
for ax in g.axes.flat:
    ax.legend(title='Group')

# 그림 제목 및 범례 설명 추가
g.fig.suptitle('Comparison of Average Metrics: Jackpot Seeking vs Regular Groups (Top 10%)', fontsize=16, y=1.02)

# 그림 아래에 지표 번호 설명 추가
metric_legend = "\nMetric Legend: "
for num, metric in metric_mapping_full.items():
    metric_legend += f"{num}={metric}, "
metric_legend = metric_legend[:-2]

plt.figtext(0.5, -0.02, metric_legend, ha='center', fontsize=11, wrap=True)
plt.tight_layout()
g.savefig(os.path.join(VIZ_DIR, 'jackpot_metric_comparison.png'), dpi=300, bbox_inches='tight')

##############################################
# 3. 복합 조건 기반 분류 (가장 효과적인 식별 방법 탐색)
##############################################

# 잭팟 추구형 식별을 위한 다양한 룰 정의 및 성능 테스트
rules = {
    "Rule 1 (MAX_TRADE_PROPORTION)": df_clean['MAX_TRADE_PROPORTION'] >= percentiles[10]['MAX_TRADE_PROPORTION']['threshold'],
    "Rule 2 (SHARPE_RATIO)": df_clean['SHARPE_RATIO'] <= percentiles[10]['SHARPE_RATIO']['threshold'],
    "Rule 3 (WIN_LOSS_RATIO)": df_clean['WIN_LOSS_RATIO'] <= np.percentile(df_clean['WIN_LOSS_RATIO'], 10),  # 하위 10%
    "Rule 4 (MAX_TRADE_PROPORTION AND SHARPE_RATIO)": (df_clean['MAX_TRADE_PROPORTION'] >= percentiles[10]['MAX_TRADE_PROPORTION']['threshold']) & 
                                                    (df_clean['SHARPE_RATIO'] <= percentiles[10]['SHARPE_RATIO']['threshold']),
    "Rule 5 (MAX_TRADE_PROPORTION OR SHARPE_RATIO)": (df_clean['MAX_TRADE_PROPORTION'] >= percentiles[10]['MAX_TRADE_PROPORTION']['threshold']) | 
                                                    (df_clean['SHARPE_RATIO'] <= percentiles[10]['SHARPE_RATIO']['threshold']),
    "Rule 6 (MAX_TRADE_PROPORTION >= 0.4)": df_clean['MAX_TRADE_PROPORTION'] >= 0.4,
    "Rule 7 (SHARPE_RATIO <= -3.0)": df_clean['SHARPE_RATIO'] <= -3.0,
    "Rule 8 (MAX_TRADE_PROPORTION >= 0.4 AND SHARPE_RATIO <= -3.0)": (df_clean['MAX_TRADE_PROPORTION'] >= 0.4) & (df_clean['SHARPE_RATIO'] <= -3.0),
}

# 각 규칙별 성능 평가
rule_performance = pd.DataFrame(index=rules.keys(), 
                             columns=['True Positive', 'False Positive', 'True Negative', 'False Negative', 
                                     'Precision', 'Recall', 'F1 Score', 'Wallet Count', 'Churn Rate (%)'])

# 실제 이탈 여부
y_true = df_clean['WALLET_STATUS'] == 'churned'

for rule_name, rule_condition in rules.items():
    # 규칙에 따른 예측
    y_pred = rule_condition
    
    # 혼동 행렬 계산
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    # 정밀도, 재현율, F1 점수
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    # 결과 저장
    rule_performance.loc[rule_name, 'True Positive'] = tp
    rule_performance.loc[rule_name, 'False Positive'] = fp
    rule_performance.loc[rule_name, 'True Negative'] = tn
    rule_performance.loc[rule_name, 'False Negative'] = fn
    rule_performance.loc[rule_name, 'Precision'] = precision
    rule_performance.loc[rule_name, 'Recall'] = recall
    rule_performance.loc[rule_name, 'F1 Score'] = f1
    rule_performance.loc[rule_name, 'Wallet Count'] = sum(y_pred)
    rule_performance.loc[rule_name, 'Churn Rate (%)'] = (y_true & y_pred).sum() / sum(y_pred) * 100 if sum(y_pred) > 0 else 0

# 결과 저장
rule_performance.to_csv(os.path.join(SOURCE_DIR, 'rule_performance.csv'))

# 룰별 성능 시각화
plt.figure(figsize=(14, 10))

# F1 점수 기준 정렬
rule_performance_sorted = rule_performance.sort_values('F1 Score', ascending=False)

# F1 점수 시각화
plt.subplot(2, 1, 1)
ax1 = sns.barplot(x=rule_performance_sorted.index, y=rule_performance_sorted['F1 Score'], palette='viridis')
plt.title('Rule Performance Comparison (F1 Score)', fontsize=14)
plt.ylabel('F1 Score', fontsize=12)
plt.xticks(rotation=45, ha='right')

# 정밀도와 재현율 시각화
plt.subplot(2, 1, 2)
ax2 = sns.barplot(x=rule_performance_sorted.index, y=rule_performance_sorted['Churn Rate (%)'], palette='magma')
plt.title('Rule Performance Comparison (Churn Rate)', fontsize=14)
plt.ylabel('Churn Rate (%)', fontsize=12)
plt.xticks(rotation=45, ha='right')

# 룰에 포함된 지갑 수 표시
for i, (idx, val) in enumerate(zip(rule_performance_sorted.index, rule_performance_sorted['Churn Rate (%)'])):
    wallet_count = rule_performance_sorted.loc[idx, 'Wallet Count']
    ax2.text(i, val + 2, f'n={wallet_count}', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig(os.path.join(VIZ_DIR, 'rule_performance_comparison.png'), dpi=300, bbox_inches='tight')

# 최적 룰 분석
best_rules = rule_performance.sort_values('F1 Score', ascending=False).head(3)
with open(os.path.join(SOURCE_DIR, 'best_rules_summary.txt'), 'w') as f:
    f.write("최적 식별 룰 top 3 (F1 Score 기준):\n\n")
    for rule_name in best_rules.index:
        f.write(f"- {rule_name}:\n")
        f.write(f"  Wallet Count: {best_rules.loc[rule_name, 'Wallet Count']}\n")
        f.write(f"  Churn Rate: {best_rules.loc[rule_name, 'Churn Rate (%)']:.2f}%\n")
        f.write(f"  Precision: {best_rules.loc[rule_name, 'Precision']:.4f}\n")
        f.write(f"  Recall: {best_rules.loc[rule_name, 'Recall']:.4f}\n")
        f.write(f"  F1 Score: {best_rules.loc[rule_name, 'F1 Score']:.4f}\n\n")

print("Analysis completed. Results saved to specified directories.")
print(f"Visualization directory: {VIZ_DIR}")
print(f"Source data directory: {SOURCE_DIR}") 