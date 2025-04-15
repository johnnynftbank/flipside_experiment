#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
전처리 완료된 PCA 데이터에 K-means 클러스터링을 적용하고,
결과를 분석하는 스크립트
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.decomposition import PCA
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D
import os
from datetime import datetime

# 시각화 설정
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.size'] = 12
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12

# 경로 설정
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_FILE = os.path.join(BASE_DIR, 'report', 'prepared_data_for_kmeans.csv')
REPORT_DIR = os.path.join(BASE_DIR, 'report')
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")

# 결과 폴더 생성
RESULTS_DIR = os.path.join(REPORT_DIR, f'kmeans_results_{TIMESTAMP}')
os.makedirs(RESULTS_DIR, exist_ok=True)

# 데이터 로드
print(f"데이터 파일 로드: {INPUT_FILE}")
data = pd.read_csv(INPUT_FILE)
print(f"데이터 크기: {data.shape}")

# 클러스터링에 사용할 특성
features = ['PC1', 'PC2', 'PC3']
X = data[features].values

# 데이터 요약 통계 출력
print("\n데이터 요약 통계:")
print(data[features].describe())

# 데이터 시각화 - PC 분포
fig = plt.figure(figsize=(15, 5))
for i, feature in enumerate(features):
    ax = fig.add_subplot(1, 3, i+1)
    sns.histplot(data=data, x=feature, kde=True, ax=ax)
    ax.set_title(f'{feature} 분포')
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'pc_distributions.png'), dpi=300)
plt.close()

# 3D 산점도
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(data['PC1'], data['PC2'], data['PC3'], 
                    c=data['WALLET_STATUS'].map({'active': 0, 'churned': 1}),
                    cmap='coolwarm', alpha=0.7, s=20)
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_zlabel('PC3')
ax.set_title('PCA 3D 산점도 (WALLET_STATUS)')
legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                            label=status, markerfacecolor=c, markersize=10)
                 for status, c in zip(['active', 'churned'], ['blue', 'red'])]
ax.legend(handles=legend_elements, loc='upper right')
plt.savefig(os.path.join(RESULTS_DIR, 'pca_3d_scatter.png'), dpi=300)
plt.close()

# ---------------------- 최적 클러스터 수(K) 결정 ----------------------
print("\n최적 클러스터 수(K) 결정 중...")

# K 범위 설정
k_range = range(2, 11)

# 엘보우 메소드 (Inertia)
inertias = []
for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X)
    inertias.append(kmeans.inertia_)

# 실루엣 점수
silhouette_avgs = []
for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X)
    silhouette_avg = silhouette_score(X, cluster_labels)
    silhouette_avgs.append(silhouette_avg)
    print(f"k={k}: 실루엣 점수 = {silhouette_avg:.4f}")

# 시각화 - 엘보우 메소드 & 실루엣 점수
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# 엘보우 메소드 플롯
ax1.plot(k_range, inertias, 'o-', linewidth=2, markersize=8)
ax1.set_xlabel('클러스터 수 (k)')
ax1.set_ylabel('Inertia (SSE)')
ax1.set_title('엘보우 메소드')
ax1.grid(True)

# 실루엣 점수 플롯
ax2.plot(k_range, silhouette_avgs, 'o-', linewidth=2, markersize=8, color='red')
ax2.set_xlabel('클러스터 수 (k)')
ax2.set_ylabel('평균 실루엣 점수')
ax2.set_title('실루엣 분석')
ax2.grid(True)

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'optimal_k_analysis.png'), dpi=300)
plt.close()

# 실루엣 시각화 (k=4 예시)
def plot_silhouette_analysis(k):
    # k-means 클러스터링
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X)
    
    # 실루엣 점수 계산
    silhouette_avg = silhouette_score(X, cluster_labels)
    silhouette_values = silhouette_samples(X, cluster_labels)
    
    # 시각화
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    y_lower = 10
    
    for i in range(k):
        # i번째 클러스터에 속한 샘플의 실루엣 점수
        ith_cluster_silhouette_values = silhouette_values[cluster_labels == i]
        ith_cluster_silhouette_values.sort()
        
        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i
        
        color = cm.nipy_spectral(float(i) / k)
        ax.fill_betweenx(np.arange(y_lower, y_upper),
                         0, ith_cluster_silhouette_values,
                         facecolor=color, edgecolor=color, alpha=0.7)
        
        # 클러스터 레이블 달기
        ax.text(-0.05, y_lower + 0.5 * size_cluster_i, f'클러스터 {i}')
        
        # 다음 클러스터를 위한 y_lower 업데이트
        y_lower = y_upper + 10
    
    ax.set_title(f'k={k}에 대한 실루엣 분석')
    ax.set_xlabel('실루엣 계수')
    ax.set_ylabel('클러스터')
    
    # 평균 실루엣 점수 선 추가
    ax.axvline(x=silhouette_avg, color='red', linestyle='--')
    ax.set_yticks([])
    ax.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])
    
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, f'silhouette_analysis_k{k}.png'), dpi=300)
    plt.close()
    
    return silhouette_avg

# k=3, 4, 5에 대한 실루엣 분석
for k in [3, 4, 5]:
    silhouette_avg = plot_silhouette_analysis(k)
    print(f"k={k}: 실루엣 시각화 완료, 평균 실루엣 점수 = {silhouette_avg:.4f}")

# ---------------------- K-means 클러스터링 실행 ----------------------
# 최적 k 선택 (결과 확인 후 수동으로 선택하거나, 자동으로 선택)
best_k = silhouette_avgs.index(max(silhouette_avgs)) + 2  # 실루엣 점수가 최대인 k
print(f"\n실루엣 점수 기준 최적 k: {best_k}")
print(f"분석 결과 및 도메인 지식을 기반으로 클러스터 수를 선택하세요.")

# 사용자가 선택한 k 값 (기본값: 실루엣 점수가 최대인 k)
selected_k = best_k  # 필요에 따라 변경 가능

# K-means 클러스터링 실행
print(f"\nK-means 클러스터링 실행 (k={selected_k})...")
kmeans = KMeans(n_clusters=selected_k, random_state=42, n_init=10)
data['cluster'] = kmeans.fit_predict(X)

# 클러스터 중심점
cluster_centers = pd.DataFrame(kmeans.cluster_centers_, columns=features)
print("\n클러스터 중심점:")
print(cluster_centers)

# 클러스터별 데이터 수
cluster_counts = data['cluster'].value_counts().sort_index()
print("\n클러스터별 데이터 수:")
print(cluster_counts)

# 클러스터별 특성 분석
cluster_stats = data.groupby('cluster')[features].agg(['mean', 'std'])
print("\n클러스터별 특성 통계:")
print(cluster_stats)

# 클러스터별 WALLET_STATUS 분포
wallet_status_by_cluster = pd.crosstab(data['cluster'], data['WALLET_STATUS'], normalize='index') * 100
print("\n클러스터별 WALLET_STATUS 분포 (%):")
print(wallet_status_by_cluster)

# ---------------------- 결과 저장 ----------------------
# 클러스터 결과를 CSV 파일로 저장
data.to_csv(os.path.join(RESULTS_DIR, 'kmeans_clusters.csv'), index=False)
cluster_centers.to_csv(os.path.join(RESULTS_DIR, 'cluster_centers.csv'), index=True)
cluster_stats.to_csv(os.path.join(RESULTS_DIR, 'cluster_statistics.csv'))
wallet_status_by_cluster.to_csv(os.path.join(RESULTS_DIR, 'wallet_status_by_cluster.csv'))

# ---------------------- 결과 시각화 ----------------------
# 클러스터별 색상 매핑
colors = plt.cm.rainbow(np.linspace(0, 1, selected_k))

# 3D 산점도 - 클러스터 시각화
fig = plt.figure(figsize=(14, 12))
ax = fig.add_subplot(111, projection='3d')

# 각 클러스터 데이터 시각화
for i in range(selected_k):
    cluster_data = data[data['cluster'] == i]
    ax.scatter(cluster_data['PC1'], cluster_data['PC2'], cluster_data['PC3'], 
               color=colors[i], s=30, alpha=0.7, label=f'클러스터 {i}')

# 클러스터 중심점 시각화
ax.scatter(cluster_centers['PC1'], cluster_centers['PC2'], cluster_centers['PC3'], 
           c='black', s=200, alpha=1, marker='*', label='중심점')

ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_zlabel('PC3')
ax.set_title(f'K-means 클러스터링 결과 (k={selected_k})')
ax.legend()

plt.savefig(os.path.join(RESULTS_DIR, 'kmeans_3d_visualization.png'), dpi=300)
plt.close()

# 2D 산점도 - 각 PC 쌍별 시각화
pc_pairs = [('PC1', 'PC2'), ('PC1', 'PC3'), ('PC2', 'PC3')]

for pc_pair in pc_pairs:
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # 각 클러스터 데이터 시각화
    for i in range(selected_k):
        cluster_data = data[data['cluster'] == i]
        ax.scatter(cluster_data[pc_pair[0]], cluster_data[pc_pair[1]], 
                   color=colors[i], s=30, alpha=0.7, label=f'클러스터 {i}')
    
    # 클러스터 중심점 시각화
    ax.scatter(cluster_centers[pc_pair[0]], cluster_centers[pc_pair[1]], 
               c='black', s=200, alpha=1, marker='*', label='중심점')
    
    # 클러스터 크기 및 이탈률 annotation 추가
    for i in range(selected_k):
        cluster_data = data[data['cluster'] == i]
        churn_rate = (cluster_data['WALLET_STATUS'] == 'churned').mean() * 100
        center_x = cluster_centers.loc[i, pc_pair[0]]
        center_y = cluster_centers.loc[i, pc_pair[1]]
        
        ax.annotate(f'클러스터 {i}\n'
                   f'크기: {len(cluster_data)}\n'
                   f'이탈률: {churn_rate:.1f}%',
                   (center_x, center_y),
                   textcoords="offset points", 
                   xytext=(0, 10), 
                   ha='center',
                   bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", alpha=0.8))
    
    ax.set_xlabel(pc_pair[0])
    ax.set_ylabel(pc_pair[1])
    ax.set_title(f'K-means 클러스터링 결과 (k={selected_k}): {pc_pair[0]} vs {pc_pair[1]}')
    ax.legend()
    
    plt.savefig(os.path.join(RESULTS_DIR, f'kmeans_2d_{pc_pair[0]}_{pc_pair[1]}.png'), dpi=300)
    plt.close()

# 클러스터별 특성 레이더 차트
def plot_radar_chart(cluster_centers):
    # 데이터 준비
    categories = features
    N = len(categories)
    
    # 각도 계산
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # 원형으로 닫기
    
    # 그림 초기화
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    
    # 각 클러스터에 대해 레이더 차트 그리기
    for i in range(selected_k):
        values = cluster_centers.iloc[i].tolist()
        values += values[:1]  # 원형으로 닫기
        
        ax.plot(angles, values, linewidth=2, linestyle='solid', label=f'클러스터 {i}')
        ax.fill(angles, values, alpha=0.1)
    
    # 축 및 레이블 설정
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)
    ax.set_title(f'클러스터별 특성 비교 (k={selected_k})')
    ax.legend(loc='upper right')
    
    plt.savefig(os.path.join(RESULTS_DIR, 'cluster_radar_chart.png'), dpi=300)
    plt.close()

plot_radar_chart(cluster_centers)

# 클러스터별 WALLET_STATUS 분포 막대 그래프
plt.figure(figsize=(12, 8))
wallet_status_by_cluster.plot(kind='bar', stacked=True, colormap='Blues')
plt.xlabel('클러스터')
plt.ylabel('비율 (%)')
plt.title(f'클러스터별 WALLET_STATUS 분포 (k={selected_k})')
plt.xticks(rotation=0)
plt.legend(title='WALLET_STATUS')
plt.grid(axis='y', alpha=0.3)

# 각 막대 위에 이탈률 표시
for i, (idx, row) in enumerate(wallet_status_by_cluster.iterrows()):
    if 'churned' in row:
        plt.text(i, row['churned']/2, f"{row['churned']:.1f}%", 
                 ha='center', va='center', color='white', fontweight='bold')
    if 'active' in row:
        plt.text(i, row['active'] + row.get('churned', 0)/2, f"{row['active']:.1f}%", 
                 ha='center', va='center', color='black', fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'wallet_status_distribution.png'), dpi=300)
plt.close()

# ---------------------- 보고서 생성 ----------------------
print("\n보고서 생성 중...")

# 클러스터 특성 해석
cluster_interpretations = []
for i in range(selected_k):
    cluster_data = data[data['cluster'] == i]
    cluster_size = len(cluster_data)
    churn_rate = (cluster_data['WALLET_STATUS'] == 'churned').mean() * 100
    
    # PC 평균값 기반 해석
    pc1_mean = cluster_centers.loc[i, 'PC1']
    pc2_mean = cluster_centers.loc[i, 'PC2']
    pc3_mean = cluster_centers.loc[i, 'PC3']
    
    # 간단한 해석 (복잡한 해석은 보고서에서 수동으로 보완)
    interpretation = f"클러스터 {i}: "
    
    if pc1_mean > 1.0:
        interpretation += "높은 PC1, "
    elif pc1_mean < -1.0:
        interpretation += "낮은 PC1, "
    else:
        interpretation += "중간 PC1, "
        
    if pc2_mean > 1.0:
        interpretation += "높은 PC2, "
    elif pc2_mean < -1.0:
        interpretation += "낮은 PC2, "
    else:
        interpretation += "중간 PC2, "
        
    if pc3_mean > 1.0:
        interpretation += "높은 PC3"
    elif pc3_mean < -1.0:
        interpretation += "낮은 PC3"
    else:
        interpretation += "중간 PC3"
    
    cluster_interpretations.append({
        'cluster': i,
        'size': cluster_size,
        'percentage': cluster_size / len(data) * 100,
        'churn_rate': churn_rate,
        'pc1_mean': pc1_mean,
        'pc2_mean': pc2_mean,
        'pc3_mean': pc3_mean,
        'interpretation': interpretation
    })

# 마크다운 보고서 생성
report_content = f"""# K-means 클러스터링 분석 보고서

## 1. 분석 개요

- **분석 데이터**: 이상치가 제거되고 재표준화된 PCA 데이터
- **데이터 크기**: {len(data)}개 관측치
- **사용된 특성**: {', '.join(features)}
- **클러스터 수(k)**: {selected_k}
- **분석 일시**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## 2. 최적 클러스터 수(k) 결정

최적의 클러스터 수를 결정하기 위해 엘보우 메소드와 실루엣 분석을 수행했습니다.

- **엘보우 메소드**: inertia(군집 내 거리 제곱합)의 변화를 분석
- **실루엣 분석**: 클러스터 품질을 나타내는 실루엣 점수를 계산

실루엣 점수가 가장 높은 클러스터 수: **{best_k}**

분석 결과와 도메인 지식을 종합적으로 고려하여 클러스터 수를 **{selected_k}**로 결정했습니다.

## 3. 클러스터 특성 분석

### 3.1 클러스터 기본 정보

| 클러스터 | 크기 | 비율 (%) | 이탈률 (%) | 특성 |
|---------|------|---------|-----------|------|
"""

# 클러스터 기본 정보 표 생성
for ci in cluster_interpretations:
    report_content += f"| 클러스터 {ci['cluster']} | {ci['size']:,} | {ci['percentage']:.1f} | {ci['churn_rate']:.1f} | {ci['interpretation']} |\n"

report_content += """
### 3.2 클러스터 중심점

다음은 각 클러스터의 중심점 좌표입니다:

"""

# 클러스터 중심점 표 추가
report_content += "| 클러스터 | PC1 | PC2 | PC3 |\n"
report_content += "|---------|-----|-----|-----|\n"
for i in range(selected_k):
    report_content += f"| 클러스터 {i} | {cluster_centers.loc[i, 'PC1']:.4f} | {cluster_centers.loc[i, 'PC2']:.4f} | {cluster_centers.loc[i, 'PC3']:.4f} |\n"

report_content += """
### 3.3 클러스터별 WALLET_STATUS 분포

다음은 각 클러스터의 WALLET_STATUS(active/churned) 분포입니다:

"""

# WALLET_STATUS 분포 표 추가
report_content += "| 클러스터 | active (%) | churned (%) |\n"
report_content += "|---------|------------|-------------|\n"
for i in range(selected_k):
    active_pct = wallet_status_by_cluster.loc[i, 'active'] if 'active' in wallet_status_by_cluster.columns else 0
    churned_pct = wallet_status_by_cluster.loc[i, 'churned'] if 'churned' in wallet_status_by_cluster.columns else 0
    report_content += f"| 클러스터 {i} | {active_pct:.1f} | {churned_pct:.1f} |\n"

report_content += """
## 4. 클러스터 해석 및 인사이트

각 클러스터의 특성을 기반으로 한 해석과 비즈니스 인사이트:

"""

# 클러스터별 해석 추가 (간단한 템플릿, 실제로는 분석 결과를 보고 수동으로 작성 필요)
for i in range(selected_k):
    cluster_data = data[data['cluster'] == i]
    churn_rate = (cluster_data['WALLET_STATUS'] == 'churned').mean() * 100
    
    report_content += f"""### 클러스터 {i} (전체의 {cluster_interpretations[i]['percentage']:.1f}%)

- **특성**: {cluster_interpretations[i]['interpretation']}
- **이탈률**: {churn_rate:.1f}%
- **인사이트**: 
  - 이 클러스터의 투자자들은 [특성 해석]
  - 이탈률이 {'높은' if churn_rate > 50 else '중간' if churn_rate > 30 else '낮은'} 편으로, [이탈 원인 추정]
  - [비즈니스 제안]

"""

report_content += """
## 5. 결론 및 추천 사항

클러스터링 분석 결과를 종합하면:

1. [주요 발견 사항 1]
2. [주요 발견 사항 2]
3. [주요 발견 사항 3]

이러한 발견을 바탕으로 다음과 같은 추천 사항을 제시합니다:

1. [추천 사항 1]
2. [추천 사항 2]
3. [추천 사항 3]

## 6. 참조 파일 목록

분석 과정에서 생성된 파일 목록:

1. K-means 클러스터링 결과: `kmeans_clusters.csv`
2. 클러스터 중심점: `cluster_centers.csv`
3. 클러스터 통계: `cluster_statistics.csv`
4. WALLET_STATUS 분포: `wallet_status_by_cluster.csv`
5. 시각화 파일들:
   - 최적 k 분석: `optimal_k_analysis.png`
   - 실루엣 분석: `silhouette_analysis_k*.png`
   - 클러스터 3D 시각화: `kmeans_3d_visualization.png`
   - 클러스터 2D 시각화: `kmeans_2d_*.png`
   - 클러스터 레이더 차트: `cluster_radar_chart.png`
   - WALLET_STATUS 분포: `wallet_status_distribution.png`
"""

# 보고서 파일 저장
report_file = os.path.join(RESULTS_DIR, '03_kmeans_analysis_report.md')
with open(report_file, 'w') as f:
    f.write(report_content)

print(f"\nK-means 클러스터링 분석 완료!")
print(f"결과 파일 위치: {RESULTS_DIR}")
print(f"분석 보고서: {report_file}")
