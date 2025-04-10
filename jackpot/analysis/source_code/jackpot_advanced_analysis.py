import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.stats import mannwhitneyu, ttest_ind
import matplotlib.gridspec as gridspec

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

# 분석할 지표 목록 (영어로만 표시)
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

# 데이터 전처리 - 이상치 제거
df_clean = df.copy()
for metric in metrics:
    df_clean[metric] = filter_outliers(df[metric], method='quantile', quantile_range=(0.01, 0.99))
df_clean = df_clean.dropna(subset=metrics)

# 지갑 상태 영어로 변환
df_clean['WALLET_STATUS_LABEL'] = df_clean['WALLET_STATUS'].map({'active': 'Active', 'churned': 'Churned'})

print(f"Clean data shape after outlier removal: {df_clean.shape}")

#################################
# 1. 상관관계 분석 (Correlation Analysis)
#################################

# 피어슨 상관계수와 스피어만 상관계수 계산
pearson_corr = df_clean[metrics].corr(method='pearson')
spearman_corr = df_clean[metrics].corr(method='spearman')

# 상관관계 시각화 함수
def plot_correlation(corr_matrix, title, filename, method_name):
    plt.figure(figsize=(12, 10))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    
    # 지표 이름을 영어로 변경
    corr_matrix.index = [metrics_labels[m] for m in corr_matrix.index]
    corr_matrix.columns = [metrics_labels[m] for m in corr_matrix.columns]
    
    sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', cmap=cmap,
                vmin=-1, vmax=1, linewidths=1, annot_kws={"size": 12},
                cbar_kws={"shrink": .8})
    plt.title(f'{title} ({method_name})', fontsize=16, fontweight='bold')
    plt.xticks(fontsize=12, rotation=45)
    plt.yticks(fontsize=12, rotation=0)
    plt.tight_layout()
    plt.savefig(f'{REPORT_DIR}/{filename}', dpi=300, bbox_inches='tight')
    print(f"Saved correlation heatmap to {REPORT_DIR}/{filename}")

# 피어슨 및 스피어만 상관계수 시각화
plot_correlation(pearson_corr, 'Correlation Between Metrics', 'jackpot_pearson_correlation.png', 'Pearson')
plot_correlation(spearman_corr, 'Correlation Between Metrics', 'jackpot_spearman_correlation.png', 'Spearman')

# 상관관계 분석 결과 요약
print("\n========== Correlation Analysis ==========")
print("Pearson Correlation:")
print(pearson_corr)
print("\nSpearman Correlation:")
print(spearman_corr)

# 산점도 행렬 (주요 지표 중심으로)
selected_metrics = ['EXPECTED_ROI', 'ROI_STANDARD_DEVIATION', 'MAX_TRADE_PROPORTION', 'SHARPE_RATIO']
selected_labels = [metrics_labels[m] for m in selected_metrics]

plt.figure(figsize=(16, 14))
scatter_grid = sns.pairplot(
    df_clean, 
    vars=selected_metrics, 
    hue='WALLET_STATUS_LABEL',
    palette={"Active": "lightseagreen", "Churned": "coral"},
    plot_kws={'alpha': 0.5, 's': 30, 'edgecolor': 'none'},
    diag_kind='kde'
)
scatter_grid.fig.suptitle('Scatter Plot Matrix of Key Metrics', y=1.02, fontsize=18, fontweight='bold')
scatter_grid.set(xlabel='', ylabel='')
for i, var in enumerate(selected_metrics):
    for j, var2 in enumerate(selected_metrics):
        if i >= j:
            ax = scatter_grid.axes[i, j]
            ax.set_xlabel(metrics_labels[var2] if i == len(selected_metrics)-1 else '')
            ax.set_ylabel(metrics_labels[var] if j == 0 else '')
scatter_grid.savefig(f'{REPORT_DIR}/jackpot_scatter_matrix.png', dpi=300, bbox_inches='tight')
print(f"Saved scatter matrix to {REPORT_DIR}/jackpot_scatter_matrix.png")

#################################
# 2. 단순 분류 (Percentile-based Classification)
#################################

# 상위 5%와 10% 기준으로 잭팟 추구 그룹 식별
percentiles = {5: {}, 10: {}}
for metric in metrics:
    for percentile in [5, 10]:
        # 지표의 특성에 따라 상위/하위 기준 설정
        if metric in ['SHARPE_RATIO']:  # 낮을수록 잭팟 추구 성향
            threshold = np.percentile(df_clean[metric], percentile)
            mask = df_clean[metric] <= threshold
        else:  # 높을수록 잭팟 추구 성향
            threshold = np.percentile(df_clean[metric], 100-percentile)
            mask = df_clean[metric] >= threshold
        
        # 잭팟 추구 그룹 식별 및 저장
        jackpot_group = df_clean[mask].copy()
        non_jackpot_group = df_clean[~mask].copy()
        
        percentiles[percentile][metric] = {
            'threshold': threshold,
            'jackpot_count': len(jackpot_group),
            'non_jackpot_count': len(non_jackpot_group),
            'jackpot_group': jackpot_group,
            'non_jackpot_group': non_jackpot_group
        }

# 각 그룹의 특성 분석
group_stats = {}
for percentile in [5, 10]:
    group_stats[percentile] = {}
    
    for metric in metrics:
        group_stats[percentile][metric] = {}
        jackpot_group = percentiles[percentile][metric]['jackpot_group']
        non_jackpot_group = percentiles[percentile][metric]['non_jackpot_group']
        
        # 평균값 계산
        for m in metrics:
            group_stats[percentile][metric][f'jackpot_mean_{m}'] = jackpot_group[m].mean()
            group_stats[percentile][metric][f'non_jackpot_mean_{m}'] = non_jackpot_group[m].mean()
        
        # 이탈률 계산
        group_stats[percentile][metric]['jackpot_churn_rate'] = (jackpot_group['WALLET_STATUS'] == 'churned').mean() * 100
        group_stats[percentile][metric]['non_jackpot_churn_rate'] = (non_jackpot_group['WALLET_STATUS'] == 'churned').mean() * 100
        
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
            
            group_stats[percentile][metric][f'pvalue_{m}'] = p_val
            group_stats[percentile][metric][f'test_method_{m}'] = test_method
            group_stats[percentile][metric][f'significant_{m}'] = p_val < 0.05

# 결과 시각화 - 단순 분류(상위 10% 기준)
plt.figure(figsize=(18, 15))
gs = gridspec.GridSpec(3, 2)

# 각 지표별 잭팟 추구 그룹 vs 일반 그룹 평균값 비교
ax1 = plt.subplot(gs[0, :])
comparison_data = []

for metric in metrics:
    jp_means = [group_stats[10][metric][f'jackpot_mean_{m}'] for m in metrics]
    non_jp_means = [group_stats[10][metric][f'non_jackpot_mean_{m}'] for m in metrics]
    
    # 데이터프레임으로 변환하여 저장
    for i, m in enumerate(metrics):
        comparison_data.append({
            'Classification Metric': metrics_labels[metric],
            'Group': 'Jackpot Seeking',
            'Metric': metrics_labels[m],
            'Value': jp_means[i]
        })
        comparison_data.append({
            'Classification Metric': metrics_labels[metric],
            'Group': 'Regular',
            'Metric': metrics_labels[m],
            'Value': non_jp_means[i]
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
g.set_axis_labels("", "Average Value")
g.fig.suptitle('Comparison of Average Metrics: Jackpot Seeking vs Regular Groups (Top 10%)', fontsize=16, y=1.02)
g.savefig(f'{REPORT_DIR}/jackpot_percentile_classification_comparison.png', dpi=300, bbox_inches='tight')

# 이탈률 비교
plt.figure(figsize=(14, 8))
churn_data = []

for metric in metrics:
    churn_data.append({
        'Classification Metric': metrics_labels[metric],
        'Group': 'Jackpot Seeking',
        'Churn Rate (%)': group_stats[10][metric]['jackpot_churn_rate']
    })
    churn_data.append({
        'Classification Metric': metrics_labels[metric],
        'Group': 'Regular',
        'Churn Rate (%)': group_stats[10][metric]['non_jackpot_churn_rate']
    })

churn_df = pd.DataFrame(churn_data)

sns.barplot(
    data=churn_df,
    x='Classification Metric',
    y='Churn Rate (%)',
    hue='Group',
    palette={"Jackpot Seeking": "crimson", "Regular": "steelblue"},
    alpha=0.8
)

plt.title('Churn Rate Comparison: Jackpot Seeking vs Regular Groups (Top 10%)', fontsize=14)
plt.ylabel('Churn Rate (%)', fontsize=12)
plt.xlabel('Classification Metric', fontsize=12)
plt.xticks(rotation=45)
plt.legend(title='Group')
plt.tight_layout()
plt.savefig(f'{REPORT_DIR}/jackpot_percentile_churn_comparison.png', dpi=300, bbox_inches='tight')

# 결과 요약
print("\n========== Percentile-based Classification Results ==========")
for percentile in [5, 10]:
    print(f"\nTop {percentile}% Classification Results:")
    for metric in metrics:
        print(f"\nUsing {metrics_labels[metric]} as classifier:")
        print(f"  Threshold: {percentiles[percentile][metric]['threshold']:.4f}")
        print(f"  Jackpot group size: {percentiles[percentile][metric]['jackpot_count']} wallets")
        print(f"  Churn rate: Jackpot {group_stats[percentile][metric]['jackpot_churn_rate']:.2f}% vs Regular {group_stats[percentile][metric]['non_jackpot_churn_rate']:.2f}%")
        
        print("  Average metrics for Jackpot group:")
        for m in metrics:
            print(f"    {metrics_labels[m]}: {group_stats[percentile][metric][f'jackpot_mean_{m}']:.4f} (p-value: {group_stats[percentile][metric][f'pvalue_{m}']:.4f}, significant: {group_stats[percentile][metric][f'significant_{m}']}, method: {group_stats[percentile][metric][f'test_method_{m}']}")

#################################
# 3. 군집화 분석 (Clustering Analysis)
#################################

# 데이터 준비 및 전처리
# 스케일링 적용
scaler = StandardScaler()
df_scaled = df_clean.copy()
df_scaled[metrics] = scaler.fit_transform(df_clean[metrics])

# 최적 군집 수 결정 (Elbow 및 Silhouette 방법)
plt.figure(figsize=(16, 6))

# Elbow Method
plt.subplot(1, 2, 1)
inertia = []
silhouette_scores = []
k_range = range(2, 11)

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(df_scaled[metrics])
    inertia.append(kmeans.inertia_)
    
    # Silhouette score 계산 (k가 2 이상일 때만 계산 가능)
    if k > 1:
        silhouette_scores.append(silhouette_score(df_scaled[metrics], kmeans.labels_))

plt.plot(k_range, inertia, 'bo-')
plt.title('Elbow Method for Optimal k', fontsize=14)
plt.xlabel('Number of Clusters (k)', fontsize=12)
plt.ylabel('Inertia', fontsize=12)
plt.grid(True, alpha=0.3)

# Silhouette Method - k_range가 2부터 시작하므로 모든 k에 대해 silhouette_score 계산 가능
plt.subplot(1, 2, 2)
k_range_sil = list(k_range)  # Silhouette score는 k가 2 이상인 경우만 해당됨
plt.plot(k_range_sil, silhouette_scores, 'ro-')
plt.title('Silhouette Method for Optimal k', fontsize=14)
plt.xlabel('Number of Clusters (k)', fontsize=12)
plt.ylabel('Silhouette Score', fontsize=12)
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'{REPORT_DIR}/jackpot_cluster_optimal_k.png', dpi=300, bbox_inches='tight')

# 최적 군집 수 선택 (예: 4개로 가정) - 실제로는 그래프를 보고 결정해야 함
optimal_k = 4  # 그래프 확인 후 결정된 값으로 변경 가능

# K-Means 클러스터링 적용
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
df_clean['cluster'] = kmeans.fit_predict(df_scaled[metrics])

# 군집 중심 계산 (원래 스케일로 변환)
cluster_centers_scaled = kmeans.cluster_centers_
cluster_centers = pd.DataFrame(
    scaler.inverse_transform(cluster_centers_scaled), 
    columns=metrics
)

# 군집별 통계 계산
cluster_stats = df_clean.groupby('cluster').agg({
    **{m: ['mean', 'median', 'std'] for m in metrics},
    'WALLET_STATUS': lambda x: (x == 'churned').mean() * 100  # 이탈률 계산
})

# 간소화된 군집 통계 데이터프레임 생성
cluster_summary = pd.DataFrame(index=range(optimal_k))
for m in metrics:
    cluster_summary[f'{metrics_labels[m]}_mean'] = cluster_centers[m].values
    
cluster_summary['Wallet Count'] = df_clean.groupby('cluster').size().values
cluster_summary['Churn Rate (%)'] = df_clean.groupby('cluster').apply(
    lambda x: (x['WALLET_STATUS'] == 'churned').mean() * 100
).values

# 클러스터 시각화
plt.figure(figsize=(20, 16))

# 선별된 주요 특성 3개 간 3D 산점도
from mpl_toolkits.mplot3d import Axes3D
ax = plt.subplot(2, 2, 1, projection='3d')
scatter = ax.scatter(
    df_scaled['ROI_STANDARD_DEVIATION'], 
    df_scaled['EXPECTED_ROI'], 
    df_scaled['MAX_TRADE_PROPORTION'], 
    c=df_clean['cluster'], 
    cmap='viridis', 
    alpha=0.7,
    s=30
)
ax.set_xlabel(metrics_labels['ROI_STANDARD_DEVIATION'], fontsize=10)
ax.set_ylabel(metrics_labels['EXPECTED_ROI'], fontsize=10)
ax.set_zlabel(metrics_labels['MAX_TRADE_PROPORTION'], fontsize=10)
plt.title('3D Visualization of Clusters', fontsize=14)
plt.colorbar(scatter, ax=ax, label='Cluster')

# 클러스터 별 각 지표 평균값 비교 (방사형 차트)
plt.subplot(2, 2, 2, polar=True)
categories = [metrics_labels[m] for m in metrics]
N = len(categories)
angles = [n / float(N) * 2 * np.pi for n in range(N)]
angles += angles[:1]  # 원형을 완성하기 위해

# 각 클러스터별로 정규화된 평균값 계산
ax = plt.subplot(2, 2, 2, polar=True)
for i in range(optimal_k):
    values = [cluster_centers.iloc[i][m] for m in metrics]
    # 0-1 사이로 정규화
    min_vals = df_clean[metrics].min()
    max_vals = df_clean[metrics].max()
    values_norm = [(val - min_vals[m]) / (max_vals[m] - min_vals[m]) for m, val in zip(metrics, values)]
    values_norm += values_norm[:1]  # 원형을 완성하기 위해
    
    ax.plot(angles, values_norm, linewidth=2, linestyle='solid', label=f'Cluster {i}')
    ax.fill(angles, values_norm, alpha=0.1)

plt.xticks(angles[:-1], categories, fontsize=10)
plt.title('Cluster Profiles (Normalized)', fontsize=14)
plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))

# 클러스터별 지갑 수와 이탈률
ax1 = plt.subplot(2, 2, 3)
cluster_summary['Wallet Count'].plot(kind='bar', ax=ax1, color='skyblue', alpha=0.7)
ax1.set_title('Number of Wallets per Cluster', fontsize=14)
ax1.set_xlabel('Cluster', fontsize=12)
ax1.set_ylabel('Count', fontsize=12)

ax2 = plt.subplot(2, 2, 4)
cluster_summary['Churn Rate (%)'].plot(kind='bar', ax=ax2, color='salmon', alpha=0.7)
ax2.set_title('Churn Rate per Cluster', fontsize=14)
ax2.set_xlabel('Cluster', fontsize=12)
ax2.set_ylabel('Churn Rate (%)', fontsize=12)

plt.tight_layout()
plt.savefig(f'{REPORT_DIR}/jackpot_cluster_analysis.png', dpi=300, bbox_inches='tight')

# 각 클러스터의 특성 확인을 위한 히트맵
plt.figure(figsize=(14, 8))
cluster_mean_df = pd.DataFrame(columns=['Cluster'] + metrics)

for i in range(optimal_k):
    row_data = {'Cluster': f'Cluster {i}'}
    for m in metrics:
        row_data[m] = cluster_centers.iloc[i][m]
    cluster_mean_df = pd.concat([cluster_mean_df, pd.DataFrame([row_data])], ignore_index=True)

# Z-score로 정규화하여 히트맵 생성
cluster_mean_pivot = cluster_mean_df.set_index('Cluster')
cluster_mean_pivot.columns = [metrics_labels[m] for m in metrics]
cluster_mean_norm = (cluster_mean_pivot - cluster_mean_pivot.mean()) / cluster_mean_pivot.std()

sns.heatmap(cluster_mean_norm, cmap='coolwarm', annot=cluster_mean_pivot.round(2), fmt='.2f',
            linewidths=1, annot_kws={"size": 12})
plt.title('Cluster Characteristics Heatmap', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{REPORT_DIR}/jackpot_cluster_heatmap.png', dpi=300, bbox_inches='tight')

# 클러스터 결과 저장
df_with_clusters = df_clean.copy()
df_with_clusters['cluster'] = kmeans.labels_
df_with_clusters.to_csv(f'{REPORT_DIR}/../jackpot_cluster_results.csv', index=False)

# 군집화 결과 요약
print("\n========== Clustering Analysis Results ==========")
print(f"Optimal number of clusters: {optimal_k}")
print("\nCluster centers (original scale):")
for i in range(optimal_k):
    print(f"\nCluster {i}:")
    for m in metrics:
        print(f"  {metrics_labels[m]}: {cluster_centers.iloc[i][m]:.4f}")
    print(f"  Wallet count: {cluster_summary.iloc[i]['Wallet Count']}")
    print(f"  Churn rate: {cluster_summary.iloc[i]['Churn Rate (%)']:.2f}%")

# 잭팟 추구형으로 간주할 수 있는 클러스터 식별
# 예: ROI 표준편차가 높고, 최대 거래 비중이 높고, 샤프 비율이 낮은 클러스터
jackpot_cluster_idx = cluster_centers[['ROI_STANDARD_DEVIATION', 'MAX_TRADE_PROPORTION']].sum(axis=1).idxmax()
print(f"\nPotential Jackpot Seeking Cluster: Cluster {jackpot_cluster_idx}")
print(f"Characteristics:")
for m in metrics:
    print(f"  {metrics_labels[m]}: {cluster_centers.iloc[jackpot_cluster_idx][m]:.4f}")
print(f"  Wallet count: {cluster_summary.iloc[jackpot_cluster_idx]['Wallet Count']}")
print(f"  Churn rate: {cluster_summary.iloc[jackpot_cluster_idx]['Churn Rate (%)']:.2f}%")

# 결과 종합 및 분석 종료
print("\n========== Analysis Complete ==========")
print(f"All results saved to {REPORT_DIR}")
print("Please check the generated visualizations for detailed analysis results.") 