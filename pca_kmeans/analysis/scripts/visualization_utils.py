#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
K-means 클러스터링 결과를 시각화하기 위한 유틸리티 함수들
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics import silhouette_samples, silhouette_score

# 시각화 설정
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.size'] = 12
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12

def plot_3d_scatter(data, features, color_by=None, output_path=None):
    """
    3D 산점도 플롯
    
    Parameters:
    -----------
    data : pandas.DataFrame
        시각화할 데이터
    features : list
        3D 플롯에 사용할 3개의 특성 (예: ['PC1', 'PC2', 'PC3'])
    color_by : str, optional
        색상 구분에 사용할 컬럼 (예: 'cluster' 또는 'WALLET_STATUS')
    output_path : str, optional
        결과 파일 저장 경로
    """
    if len(features) != 3:
        raise ValueError("3D 산점도에는 정확히 3개의 특성이 필요합니다.")
    
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # 색상 매핑
    if color_by is not None:
        if color_by == 'WALLET_STATUS':
            colors = data[color_by].map({'active': 0, 'churned': 1})
            scatter = ax.scatter(
                data[features[0]], data[features[1]], data[features[2]],
                c=colors, cmap='coolwarm', alpha=0.7, s=20
            )
            legend_elements = [
                plt.Line2D([0], [0], marker='o', color='w', label='active', markerfacecolor='blue', markersize=10),
                plt.Line2D([0], [0], marker='o', color='w', label='churned', markerfacecolor='red', markersize=10)
            ]
            ax.legend(handles=legend_elements, loc='upper right')
        else:
            scatter = ax.scatter(
                data[features[0]], data[features[1]], data[features[2]],
                c=data[color_by], cmap='rainbow', alpha=0.7, s=20
            )
            plt.colorbar(scatter, ax=ax, label=color_by)
    else:
        ax.scatter(
            data[features[0]], data[features[1]], data[features[2]],
            alpha=0.7, s=20
        )
    
    ax.set_xlabel(features[0])
    ax.set_ylabel(features[1])
    ax.set_zlabel(features[2])
    
    title = '3D 산점도'
    if color_by:
        title += f' (색상: {color_by})'
    ax.set_title(title)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def plot_cluster_3d(data, features, cluster_column, cluster_centers=None, output_path=None):
    """
    클러스터별 3D 산점도 플롯
    
    Parameters:
    -----------
    data : pandas.DataFrame
        클러스터링 결과가 포함된 데이터
    features : list
        3D 플롯에 사용할 3개의 특성 (예: ['PC1', 'PC2', 'PC3'])
    cluster_column : str
        클러스터 레이블이 있는 컬럼명
    cluster_centers : pandas.DataFrame, optional
        클러스터 중심점 좌표
    output_path : str, optional
        결과 파일 저장 경로
    """
    if len(features) != 3:
        raise ValueError("3D 산점도에는 정확히 3개의 특성이 필요합니다.")
    
    # 클러스터 수 파악
    n_clusters = data[cluster_column].nunique()
    
    # 클러스터별 색상 매핑
    colors = plt.cm.rainbow(np.linspace(0, 1, n_clusters))
    
    fig = plt.figure(figsize=(14, 12))
    ax = fig.add_subplot(111, projection='3d')
    
    # 각 클러스터 데이터 시각화
    for i in range(n_clusters):
        cluster_data = data[data[cluster_column] == i]
        ax.scatter(
            cluster_data[features[0]], cluster_data[features[1]], cluster_data[features[2]],
            color=colors[i], s=30, alpha=0.7, label=f'클러스터 {i}'
        )
    
    # 클러스터 중심점 시각화
    if cluster_centers is not None:
        ax.scatter(
            cluster_centers[features[0]], cluster_centers[features[1]], cluster_centers[features[2]],
            c='black', s=200, alpha=1, marker='*', label='중심점'
        )
    
    ax.set_xlabel(features[0])
    ax.set_ylabel(features[1])
    ax.set_zlabel(features[2])
    ax.set_title(f'K-means 클러스터링 결과 (k={n_clusters})')
    ax.legend()
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def plot_cluster_2d_pairs(data, features, cluster_column, cluster_centers=None, output_dir=None):
    """
    특성 쌍별 2D 클러스터 산점도 플롯
    
    Parameters:
    -----------
    data : pandas.DataFrame
        클러스터링 결과가 포함된 데이터
    features : list
        2D 플롯에 사용할 특성들 (예: ['PC1', 'PC2', 'PC3'])
    cluster_column : str
        클러스터 레이블이 있는 컬럼명
    cluster_centers : pandas.DataFrame, optional
        클러스터 중심점 좌표
    output_dir : str, optional
        결과 파일 저장 디렉토리 (None이면 화면에 표시)
    """
    # 클러스터 수 파악
    n_clusters = data[cluster_column].nunique()
    
    # 클러스터별 색상 매핑
    colors = plt.cm.rainbow(np.linspace(0, 1, n_clusters))
    
    # 특성 쌍 생성
    if len(features) > 3:
        # 3개 이상 특성이면 주요 특성 쌍만 선택
        feature_pairs = [(features[0], features[1]), (features[0], features[2]), (features[1], features[2])]
    else:
        # 가능한 모든 특성 쌍 생성
        feature_pairs = [(features[i], features[j]) 
                        for i in range(len(features)) 
                        for j in range(i+1, len(features))]
    
    for feature_pair in feature_pairs:
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # 각 클러스터 데이터 시각화
        for i in range(n_clusters):
            cluster_data = data[data[cluster_column] == i]
            ax.scatter(
                cluster_data[feature_pair[0]], cluster_data[feature_pair[1]],
                color=colors[i], s=30, alpha=0.7, label=f'클러스터 {i}'
            )
        
        # 클러스터 중심점 시각화
        if cluster_centers is not None:
            ax.scatter(
                cluster_centers[feature_pair[0]], cluster_centers[feature_pair[1]],
                c='black', s=200, alpha=1, marker='*', label='중심점'
            )
            
            # 클러스터 통계 주석 추가
            for i in range(n_clusters):
                cluster_data = data[data[cluster_column] == i]
                cluster_size = len(cluster_data)
                
                # 이탈률 계산 (WALLET_STATUS 열이 있는 경우)
                churn_rate_text = ""
                if 'WALLET_STATUS' in data.columns:
                    churn_rate = (cluster_data['WALLET_STATUS'] == 'churned').mean() * 100
                    churn_rate_text = f"\n이탈률: {churn_rate:.1f}%"
                
                center_x = cluster_centers.loc[i, feature_pair[0]]
                center_y = cluster_centers.loc[i, feature_pair[1]]
                
                ax.annotate(
                    f'클러스터 {i}\n크기: {cluster_size} ({cluster_size/len(data)*100:.1f}%)' + churn_rate_text,
                    (center_x, center_y),
                    textcoords="offset points", 
                    xytext=(0, 10), 
                    ha='center',
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", alpha=0.8)
                )
        
        ax.set_xlabel(feature_pair[0])
        ax.set_ylabel(feature_pair[1])
        ax.set_title(f'K-means 클러스터링 결과 (k={n_clusters}): {feature_pair[0]} vs {feature_pair[1]}')
        ax.legend()
        
        if output_dir:
            output_path = f"{output_dir}/kmeans_2d_{feature_pair[0]}_{feature_pair[1]}.png"
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

def plot_silhouette_analysis(X, k_range, output_dir=None, random_state=42):
    """
    여러 k값에 대한 실루엣 분석
    
    Parameters:
    -----------
    X : array-like
        클러스터링에 사용할 데이터 (특성 행렬)
    k_range : range or list
        분석할 k 값의 범위
    output_dir : str, optional
        결과 파일 저장 디렉토리
    random_state : int, optional
        재현성을 위한 랜덤 시드
    
    Returns:
    --------
    dict
        각 k에 대한 실루엣 점수
    """
    from sklearn.cluster import KMeans
    
    # 결과 저장용 딕셔너리
    silhouette_scores = {}
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        cluster_labels = kmeans.fit_predict(X)
        
        # 실루엣 점수가 정의되기 위해선 클러스터가 2개 이상이어야 함
        if k > 1:
            silhouette_avg = silhouette_score(X, cluster_labels)
            silhouette_scores[k] = silhouette_avg
            print(f"k={k}: 실루엣 점수 = {silhouette_avg:.4f}")
            
            # 개별 샘플의 실루엣 계수 계산
            silhouette_values = silhouette_samples(X, cluster_labels)
            
            # 시각화
            fig, ax = plt.subplots(1, 1, figsize=(10, 8))
            y_lower = 10
            
            for i in range(k):
                # i번째 클러스터의 실루엣 점수
                ith_cluster_values = silhouette_values[cluster_labels == i]
                ith_cluster_values.sort()
                
                size_cluster_i = ith_cluster_values.shape[0]
                y_upper = y_lower + size_cluster_i
                
                color = cm.nipy_spectral(float(i) / k)
                ax.fill_betweenx(
                    np.arange(y_lower, y_upper),
                    0, ith_cluster_values,
                    facecolor=color, edgecolor=color, alpha=0.7
                )
                
                # 클러스터 레이블 표시
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
            
            if output_dir:
                output_path = f"{output_dir}/silhouette_analysis_k{k}.png"
                plt.savefig(output_path, dpi=300, bbox_inches='tight')
                plt.close()
            else:
                plt.show()
    
    # 실루엣 점수 비교 플롯
    plt.figure(figsize=(10, 6))
    plt.plot(list(silhouette_scores.keys()), list(silhouette_scores.values()), 'o-', markersize=8)
    plt.xlabel('클러스터 수 (k)')
    plt.ylabel('평균 실루엣 점수')
    plt.title('실루엣 점수 비교')
    plt.grid(True)
    
    if output_dir:
        output_path = f"{output_dir}/silhouette_scores.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
    
    return silhouette_scores

def plot_elbow_method(X, k_range, output_path=None, random_state=42):
    """
    엘보우 메소드 분석 및 시각화
    
    Parameters:
    -----------
    X : array-like
        클러스터링에 사용할 데이터 (특성 행렬)
    k_range : range or list
        분석할 k 값의 범위
    output_path : str, optional
        결과 파일 저장 경로
    random_state : int, optional
        재현성을 위한 랜덤 시드
    
    Returns:
    --------
    list
        각 k에 대한 inertia 값 목록
    """
    from sklearn.cluster import KMeans
    
    inertias = []
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        kmeans.fit(X)
        inertias.append(kmeans.inertia_)
    
    plt.figure(figsize=(10, 6))
    plt.plot(k_range, inertias, 'o-', linewidth=2, markersize=8)
    plt.xlabel('클러스터 수 (k)')
    plt.ylabel('Inertia (군집 내 거리 제곱합)')
    plt.title('엘보우 메소드: 최적 클러스터 수 결정')
    plt.grid(True)
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
    
    return inertias

def plot_radar_chart(cluster_centers, features, output_path=None):
    """
    클러스터 중심점 레이더 차트
    
    Parameters:
    -----------
    cluster_centers : pandas.DataFrame
        각 클러스터 중심점 좌표
    features : list
        레이더 차트에 표시할 특성 목록
    output_path : str, optional
        결과 파일 저장 경로
    """
    # 클러스터 수
    n_clusters = len(cluster_centers)
    
    # 특성 수
    n_features = len(features)
    
    # 각도 계산
    angles = [n / float(n_features) * 2 * np.pi for n in range(n_features)]
    angles += angles[:1]  # 원형으로 닫기
    
    # 그림 초기화
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    
    # 각 클러스터에 대해 레이더 차트 그리기
    for i in range(n_clusters):
        values = cluster_centers.iloc[i][features].tolist()
        values += values[:1]  # 원형으로 닫기
        
        ax.plot(angles, values, linewidth=2, linestyle='solid', label=f'클러스터 {i}')
        ax.fill(angles, values, alpha=0.1)
    
    # 축 및 레이블 설정
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(features)
    ax.set_title(f'클러스터별 특성 비교 (k={n_clusters})')
    ax.legend(loc='upper right')
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def plot_wallet_status_distribution(data, cluster_column, output_path=None):
    """
    클러스터별 WALLET_STATUS 분포 시각화
    
    Parameters:
    -----------
    data : pandas.DataFrame
        클러스터링 결과와 WALLET_STATUS가 포함된 데이터
    cluster_column : str
        클러스터 레이블이 있는 컬럼명
    output_path : str, optional
        결과 파일 저장 경로
    """
    if 'WALLET_STATUS' not in data.columns:
        raise ValueError("데이터에 'WALLET_STATUS' 컬럼이 없습니다.")
    
    # 클러스터별 WALLET_STATUS 비율 계산
    status_by_cluster = pd.crosstab(
        data[cluster_column], data['WALLET_STATUS'], 
        normalize='index'
    ) * 100
    
    # 시각화
    plt.figure(figsize=(12, 8))
    status_by_cluster.plot(kind='bar', stacked=True, colormap='Blues')
    plt.xlabel('클러스터')
    plt.ylabel('비율 (%)')
    plt.title(f'클러스터별 WALLET_STATUS 분포')
    plt.xticks(rotation=0)
    plt.legend(title='WALLET_STATUS')
    plt.grid(axis='y', alpha=0.3)
    
    # 각 막대 위에 이탈률 표시
    for i, (idx, row) in enumerate(status_by_cluster.iterrows()):
        if 'churned' in row:
            plt.text(i, row['churned']/2, f"{row['churned']:.1f}%", 
                    ha='center', va='center', color='white', fontweight='bold')
        if 'active' in row:
            plt.text(i, row['active'] + row.get('churned', 0)/2, f"{row['active']:.1f}%", 
                    ha='center', va='center', color='black', fontweight='bold')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
    
    return status_by_cluster
