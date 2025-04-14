#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.impute import SimpleImputer
import os

# 한글 폰트 문제 해결을 위한 설정
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

# 작업 디렉토리 설정
os.makedirs('jackpot_status/report/source/figures', exist_ok=True)
os.makedirs('jackpot_status/report/source/stats', exist_ok=True)

# 데이터 로드
file_path = 'jackpot_status/query/query_result/jackpot_roi_analysis_result_7643.csv'
data = pd.read_csv(file_path)

# 데이터 기본 정보 확인
print("데이터 형태:", data.shape)
data_info = data.describe().T
data_info.to_csv('jackpot_status/report/source/stats/data_summary.csv')

# 결측값 확인
missing_values = data.isnull().sum()
missing_values.to_csv('jackpot_status/report/source/stats/missing_values.csv')
print("결측값 개수:", missing_values.sum())

# 분석에 사용할 지표 선정
features = [
    'EXPECTED_ROI', 'ROI_STANDARD_DEVIATION', 'SHARPE_RATIO', 
    'WIN_LOSS_RATIO', 'MAX_TRADE_PROPORTION', 'MAX_TOKEN_ROI', 
    'MIN_TOKEN_ROI', 'ROI_RANGE', 'IQR_ROI', 'UNIQUE_TOKENS_TRADED'
]

# 결측값이 있는 행 처리 - 데이터 수가 충분하므로 결측값이 있는 행 제거
data_cleaned = data.dropna(subset=features)
print(f"원본 데이터: {data.shape[0]}행, 정제 후 데이터: {data_cleaned.shape[0]}행")

# 1. 지표 간 상관성 분석
corr_matrix = data_cleaned[features].corr()
corr_matrix.to_csv('jackpot_status/report/source/stats/correlation_matrix.csv')

# 상관관계 히트맵 시각화
plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', vmin=-1, vmax=1)
plt.title('Pearson Correlation Matrix', fontsize=16)
plt.tight_layout()
plt.savefig('jackpot_status/report/source/figures/correlation_heatmap.png', dpi=300)
plt.close()

# 상관계수가 높은 변수 쌍 찾기 (|r| > 0.7)
high_corr_pairs = []
for i in range(len(features)):
    for j in range(i+1, len(features)):
        if abs(corr_matrix.iloc[i, j]) > 0.7:
            high_corr_pairs.append((features[i], features[j], corr_matrix.iloc[i, j]))

# 상관계수가 높은 변수 쌍 저장
high_corr_df = pd.DataFrame(high_corr_pairs, columns=['Feature1', 'Feature2', 'Correlation'])
high_corr_df.to_csv('jackpot_status/report/source/stats/high_correlation_pairs.csv', index=False)

# 상관성 분석 결과를 바탕으로 중복 지표 제거
# ROI_RANGE는 MAX_TOKEN_ROI와 MIN_TOKEN_ROI의 차이이므로 높은 상관관계를 가짐
# 해석의 용이성을 위해 MAX_TOKEN_ROI, MIN_TOKEN_ROI 대신 ROI_RANGE를 사용
filtered_features = [
    'EXPECTED_ROI', 'ROI_STANDARD_DEVIATION', 'SHARPE_RATIO', 
    'WIN_LOSS_RATIO', 'MAX_TRADE_PROPORTION', 'ROI_RANGE', 
    'IQR_ROI', 'UNIQUE_TOKENS_TRADED'
]

# 산점도 행렬 생성
plt.figure(figsize=(16, 14))
sns.pairplot(data_cleaned[filtered_features], diag_kind='kde', height=2.5)
plt.savefig('jackpot_status/report/source/figures/scatterplot_matrix.png', dpi=300)
plt.close()

# 2. 상위 퍼센타일 기반 분석
percentiles = {}
percentile_stats = {}

for feature in filtered_features:
    # 각 특성별 상위/하위 10% 임계값 계산
    top_10_threshold = data_cleaned[feature].quantile(0.9)
    bottom_10_threshold = data_cleaned[feature].quantile(0.1)
    
    # 상위/하위 10% 그룹 식별
    top_10_mask = data_cleaned[feature] >= top_10_threshold
    bottom_10_mask = data_cleaned[feature] <= bottom_10_threshold
    
    # 각 그룹의 다른 특성 평균값 계산
    top_10_stats = data_cleaned[filtered_features][top_10_mask].mean()
    bottom_10_stats = data_cleaned[filtered_features][bottom_10_mask].mean()
    mid_80_stats = data_cleaned[filtered_features][~(top_10_mask | bottom_10_mask)].mean()
    
    # 결과 저장
    percentiles[feature] = {
        'top_10_threshold': top_10_threshold,
        'bottom_10_threshold': bottom_10_threshold,
        'top_10_count': top_10_mask.sum(),
        'bottom_10_count': bottom_10_mask.sum()
    }
    
    percentile_stats[feature] = {
        'top_10': top_10_stats,
        'bottom_10': bottom_10_stats,
        'mid_80': mid_80_stats
    }

# 퍼센타일 기준값 저장
percentile_thresholds = pd.DataFrame({feature: {
    'top_10_threshold': percentiles[feature]['top_10_threshold'],
    'bottom_10_threshold': percentiles[feature]['bottom_10_threshold'],
    'top_10_count': percentiles[feature]['top_10_count'],
    'bottom_10_count': percentiles[feature]['bottom_10_count']
} for feature in filtered_features}).T

percentile_thresholds.to_csv('jackpot_status/report/source/stats/percentile_thresholds.csv')

# 각 특성별 상위 10% 그룹의 다른 특성 평균
top_10_feature_stats = pd.DataFrame({
    feature: percentile_stats[feature]['top_10'] for feature in filtered_features
})
top_10_feature_stats.to_csv('jackpot_status/report/source/stats/top_10_feature_stats.csv')

# 3. 군집화 분석
# 전처리: 스케일링
X = data_cleaned[filtered_features].copy()
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled_df = pd.DataFrame(X_scaled, columns=filtered_features)

# 최적의 군집 수 찾기
silhouette_scores = []
inertia_values = []
k_range = range(2, 11)

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    silhouette_scores.append(silhouette_score(X_scaled, kmeans.labels_))
    inertia_values.append(kmeans.inertia_)

# 실루엣 점수 및 Elbow 시각화
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# 실루엣 점수 시각화
ax1.plot(k_range, silhouette_scores, 'bo-')
ax1.set_xlabel('Number of Clusters', fontsize=12)
ax1.set_ylabel('Silhouette Score', fontsize=12)
ax1.set_title('Silhouette Score by Cluster Count', fontsize=14)
ax1.grid(True)

# Elbow 곡선 시각화
ax2.plot(k_range, inertia_values, 'ro-')
ax2.set_xlabel('Number of Clusters', fontsize=12)
ax2.set_ylabel('Inertia', fontsize=12)
ax2.set_title('Elbow Method', fontsize=14)
ax2.grid(True)

plt.tight_layout()
plt.savefig('jackpot_status/report/source/figures/kmeans_optimization.png', dpi=300)
plt.close()

# 최적 군집 수 선택 (예: 실루엣 점수 기준)
optimal_k = k_range[np.argmax(silhouette_scores)]
# 만약 Elbow 방법 및 실루엣 점수를 종합적으로 고려해야 한다면, 다른 값으로 조정 가능
print(f"최적 군집 수 (실루엣 점수 기준): {optimal_k}")
print(f"Silhouette Scores: {silhouette_scores}")

# 최종 K-means 모델 적용
final_kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
cluster_labels = final_kmeans.fit_predict(X_scaled)

# 클러스터 결과 원본 데이터에 추가
data_cleaned['CLUSTER'] = cluster_labels

# 클러스터별 특성 평균 계산
cluster_centers = pd.DataFrame(final_kmeans.cluster_centers_, columns=filtered_features)
cluster_centers.index.name = 'CLUSTER'
cluster_centers.to_csv('jackpot_status/report/source/stats/cluster_centers.csv')

# 원래 스케일로 변환
original_scale_centers = pd.DataFrame(
    scaler.inverse_transform(final_kmeans.cluster_centers_),
    columns=filtered_features
)
original_scale_centers.index.name = 'CLUSTER'
original_scale_centers.to_csv('jackpot_status/report/source/stats/cluster_centers_original_scale.csv')

# 클러스터별 데이터 개수
cluster_counts = data_cleaned['CLUSTER'].value_counts().sort_index()
cluster_counts.to_csv('jackpot_status/report/source/stats/cluster_counts.csv')

# 클러스터 시각화 (2D 차원 축소)
from sklearn.decomposition import PCA

# PCA로 2차원 축소
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# 클러스터 시각화
plt.figure(figsize=(12, 10))
for i in range(optimal_k):
    plt.scatter(X_pca[cluster_labels == i, 0], X_pca[cluster_labels == i, 1], 
                label=f'Cluster {i}', s=50, alpha=0.7)

# 중심점 표시
centers_pca = pca.transform(final_kmeans.cluster_centers_)
plt.scatter(centers_pca[:, 0], centers_pca[:, 1], c='black', s=200, alpha=0.7, marker='X')

plt.title('K-means Clustering Results (PCA 2D)', fontsize=16)
plt.xlabel('Principal Component 1', fontsize=12)
plt.ylabel('Principal Component 2', fontsize=12)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('jackpot_status/report/source/figures/kmeans_clusters.png', dpi=300)
plt.close()

# 4. 클러스터 특성 분석 (히트맵)
# 각 클러스터의 특성 평균을 표준화하여 비교
cluster_profile = data_cleaned.groupby('CLUSTER')[filtered_features].mean()
cluster_profile_scaled = (cluster_profile - cluster_profile.mean()) / cluster_profile.std()
cluster_profile_scaled.to_csv('jackpot_status/report/source/stats/cluster_profile_scaled.csv')

# 클러스터 특성 히트맵
plt.figure(figsize=(14, 8))
sns.heatmap(cluster_profile_scaled, annot=True, cmap='coolwarm', fmt='.2f', vmin=-2, vmax=2)
plt.title('Cluster Feature Profile (Z-score)', fontsize=16)
plt.tight_layout()
plt.savefig('jackpot_status/report/source/figures/cluster_heatmap.png', dpi=300)
plt.close()

# 5. 핵심 지표 선정
# 클러스터 분석 및 특성 분석 결과 바탕으로 핵심 지표 선정
# Jackpot 성향이 뚜렷한 클러스터를 식별하고, 해당 클러스터의 지표 평균값 분석

# EXPECTED_ROI와 ROI_STANDARD_DEVIATION이 높은 클러스터가 잭팟 성향일 가능성이 높음
# 두 지표의 Z-score 합계가 가장 높은 클러스터 선택
jackpot_scores = cluster_profile_scaled['EXPECTED_ROI'] + cluster_profile_scaled['ROI_STANDARD_DEVIATION']
jackpot_cluster = jackpot_scores.idxmax()
print(f"Jackpot 성향이 뚜렷한 클러스터: {jackpot_cluster}")

# 해당 클러스터의 특성 프로파일
jackpot_profile = cluster_profile.loc[jackpot_cluster]
jackpot_profile.to_csv('jackpot_status/report/source/stats/jackpot_cluster_profile.csv')

# 핵심 지표 중요도 분석
# 각 지표별로 jackpot 클러스터와 다른 클러스터 간의 차이 계산
other_clusters = [i for i in range(optimal_k) if i != jackpot_cluster]
other_profile = cluster_profile.loc[other_clusters].mean()

indicator_importance = (jackpot_profile - other_profile).abs().sort_values(ascending=False)
indicator_importance.to_csv('jackpot_status/report/source/stats/indicator_importance.csv')

# 상위 4개 핵심 지표 선정
core_indicators = indicator_importance.head(4).index.tolist()
print(f"핵심 지표: {core_indicators}")

# 6. 룰 기반/스코어 기반 식별 기준 설계
# 룰 기반 기준: 각 핵심 지표별 임계값 설정
rule_thresholds = {}
for indicator in core_indicators:
    # 90% 백분위수를 임계값으로 설정 (지표에 따라 조정 가능)
    if indicator in ['EXPECTED_ROI', 'MAX_TOKEN_ROI', 'ROI_RANGE']:
        # 높을수록 Jackpot 성향
        rule_thresholds[indicator] = data_cleaned[indicator].quantile(0.90)
    elif indicator in ['ROI_STANDARD_DEVIATION']:
        # 높을수록 Jackpot 성향
        rule_thresholds[indicator] = data_cleaned[indicator].quantile(0.85)
    else:
        # 지표별 특성에 맞게 임계값 설정
        rule_thresholds[indicator] = data_cleaned[indicator].quantile(0.75)

# 룰 기준 저장
rule_df = pd.DataFrame(rule_thresholds, index=['Threshold']).T
rule_df.to_csv('jackpot_status/report/source/stats/rule_thresholds.csv')

# 스코어 기반 기준: 각 지표별 가중치 설정 및 점수 계산
weights = {}
for i, indicator in enumerate(core_indicators):
    # 중요도 순서에 따라 가중치 설정
    weights[indicator] = (len(core_indicators) - i) / sum(range(1, len(core_indicators) + 1))

# 스코어 가중치 저장
weight_df = pd.DataFrame(weights, index=['Weight']).T
weight_df.to_csv('jackpot_status/report/source/stats/score_weights.csv')

# 스코어 계산 함수 데모
def calculate_score(wallet_data, indicators, weights, scaler=None):
    """
    주어진 지갑 데이터에 대해 Jackpot 성향 점수 계산
    """
    if scaler:
        # 표준화된 데이터 사용
        standardized = scaler.transform(wallet_data[indicators].values.reshape(1, -1))[0]
        score = sum(standardized[i] * weights[indicator] for i, indicator in enumerate(indicators))
    else:
        # 원본 데이터 사용 (min-max 정규화 적용)
        score = 0
        for indicator in indicators:
            min_val = data_cleaned[indicator].min()
            max_val = data_cleaned[indicator].max()
            normalized = (wallet_data[indicator] - min_val) / (max_val - min_val)
            score += normalized * weights[indicator]
    
    return score

# 전체 데이터에 점수 계산 적용
data_cleaned['JACKPOT_SCORE'] = 0
for indicator in core_indicators:
    # 각 지표 정규화
    min_val = data_cleaned[indicator].min()
    max_val = data_cleaned[indicator].max()
    data_cleaned[f'{indicator}_NORM'] = (data_cleaned[indicator] - min_val) / (max_val - min_val)
    
    # 가중치 적용하여 점수 계산
    data_cleaned['JACKPOT_SCORE'] += data_cleaned[f'{indicator}_NORM'] * weights[indicator]

# 점수 분포 저장
score_distribution = data_cleaned['JACKPOT_SCORE'].describe()
score_distribution.to_csv('jackpot_status/report/source/stats/score_distribution.csv')

# 점수 분포 시각화
plt.figure(figsize=(12, 6))
sns.histplot(data_cleaned['JACKPOT_SCORE'], kde=True)
plt.axvline(data_cleaned['JACKPOT_SCORE'].quantile(0.9), color='r', linestyle='--', 
            label='90th Percentile')
plt.title('Jackpot Score Distribution', fontsize=16)
plt.xlabel('Score', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.legend()
plt.tight_layout()
plt.savefig('jackpot_status/report/source/figures/score_distribution.png', dpi=300)
plt.close()

# 7. 룰 기반과 스코어 기반 식별 결과 비교
# 룰 기반 식별
data_cleaned['RULE_BASED'] = False
for indicator in core_indicators:
    if indicator in ['EXPECTED_ROI', 'MAX_TOKEN_ROI', 'ROI_RANGE', 'ROI_STANDARD_DEVIATION']:
        # 높을수록 Jackpot 성향
        data_cleaned.loc[data_cleaned[indicator] >= rule_thresholds[indicator], 'RULE_BASED'] = True
    else:
        # 지표별 특성에 맞게 조건 설정
        # 예시 코드이므로 실제 분석 시 수정 필요
        data_cleaned.loc[data_cleaned[indicator] >= rule_thresholds[indicator], 'RULE_BASED'] = True

# 스코어 기반 식별 (상위 10%)
score_threshold = data_cleaned['JACKPOT_SCORE'].quantile(0.9)
data_cleaned['SCORE_BASED'] = data_cleaned['JACKPOT_SCORE'] >= score_threshold

# 두 방식 비교
comparison = pd.DataFrame({
    'RULE_BASED': data_cleaned['RULE_BASED'].value_counts(),
    'SCORE_BASED': data_cleaned['SCORE_BASED'].value_counts()
})
comparison.to_csv('jackpot_status/report/source/stats/identification_comparison.csv')

# 일치율 계산
agreement = (data_cleaned['RULE_BASED'] & data_cleaned['SCORE_BASED']).sum() / data_cleaned['SCORE_BASED'].sum()
print(f"룰 기반과 스코어 기반 식별 일치율: {agreement:.2%}")

# 결과 저장
data_cleaned[['SWAPPER', 'CLUSTER', 'JACKPOT_SCORE', 'RULE_BASED', 'SCORE_BASED'] + filtered_features].to_csv(
    'jackpot_status/report/source/stats/jackpot_analysis_results.csv', index=False)

print("분석 완료!") 