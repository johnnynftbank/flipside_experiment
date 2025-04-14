#!/usr/bin/env python3
"""
K-means 클러스터링 분석 스크립트
대상 데이터: jackpot_criteria_3822.csv
목적: 잭팟 추구형 투자자 특성 검증
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import seaborn as sns
import os
import warnings
warnings.filterwarnings('ignore')

# 경로 설정
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_FILE = os.path.join(BASE_DIR, 'query/query_result/jackpot_criteria_3822.csv')
OUTPUT_DIR = os.path.join(BASE_DIR, 'analysis/results')

# 결과 디렉토리 생성 (없는 경우)
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_data():
    """
    데이터 로드 및 기본 정보 출력
    """
    print("1. 데이터 로드 중...")
    df = pd.read_csv(DATA_FILE)
    print(f"   - 로드된 데이터 형태: {df.shape}")
    
    # 기본 정보 출력
    print("\n   [데이터 샘플]")
    print(df.head(3))
    
    print("\n   [데이터 컬럼]")
    print(df.columns.tolist())
    
    print("\n   [기본 통계 정보]")
    print(df.describe().T)
    
    return df

def preprocess_data(df):
    """
    데이터 전처리 작업 수행
    - 결측치 처리
    - 이상치 처리
    - 필요 컬럼 선택
    """
    print("\n2. 데이터 전처리 중...")
    
    # 2.1 결측치 확인 및 처리
    print("\n   2.1 결측치 확인 및 처리")
    missing_values = df.isnull().sum()
    print(f"   - 결측치 현황:\n{missing_values[missing_values > 0]}")
    
    if missing_values.sum() > 0:
        df = df.dropna(subset=['EXPECTED_ROI', 'ROI_STANDARD_DEVIATION', 'SHARPE_RATIO', 'WIN_LOSS_RATIO', 'MAX_TRADE_PROPORTION'])
        print(f"   - 결측치 제거 후 데이터 크기: {df.shape}")
    
    # 2.2 입력 지표 선택 (결과 지표는 제외)
    print("\n   2.2 입력 지표 선택")
    input_features = ['EXPECTED_ROI', 'ROI_STANDARD_DEVIATION', 'SHARPE_RATIO', 'WIN_LOSS_RATIO', 'MAX_TRADE_PROPORTION']
    
    # 2.3 이상치 확인 및 처리
    print("\n   2.3 이상치 확인 및 처리")
    for feature in input_features:
        q1 = df[feature].quantile(0.25)
        q3 = df[feature].quantile(0.75)
        iqr = q3 - q1
        upper_bound = q3 + 3 * iqr
        lower_bound = q1 - 3 * iqr
        
        outliers = df[(df[feature] < lower_bound) | (df[feature] > upper_bound)]
        print(f"   - {feature}: {len(outliers)} 이상치 감지됨 (IQR 방식, 3*IQR)")
    
    print("\n   2.4 극단적 이상치 제거 (Z-score > 5)")
    original_len = len(df)
    
    for feature in input_features:
        z_scores = np.abs((df[feature] - df[feature].mean()) / df[feature].std())
        df = df[z_scores < 5]  # Z-score가 5 초과인 극단적 이상치만 제거
    
    print(f"   - 극단적 이상치 제거 전 데이터 크기: {original_len}")
    print(f"   - 극단적 이상치 제거 후 데이터 크기: {len(df)}")
    print(f"   - {original_len - len(df)}개 행 제거됨")
    
    # 2.5 상관관계 분석
    print("\n   2.5 상관관계 분석")
    correlation = df[input_features].corr()
    
    # 높은 상관관계가 있는 변수 쌍 출력
    high_corr = []
    for i in range(len(input_features)):
        for j in range(i+1, len(input_features)):
            if abs(correlation.iloc[i, j]) > 0.7:  # 상관계수 절대값이 0.7 이상
                high_corr.append((input_features[i], input_features[j], correlation.iloc[i, j]))
    
    if high_corr:
        print("   - 높은 상관관계를 가진 변수 쌍:")
        for var1, var2, corr in high_corr:
            print(f"     * {var1} - {var2}: {corr:.4f}")
    else:
        print("   - 변수 간 높은 상관관계(|r| > 0.7)가 발견되지 않았습니다.")
    
    return df, input_features

def normalize_data(df, input_features):
    """
    데이터 정규화 (StandardScaler 사용)
    """
    print("\n3. 데이터 정규화 중...")
    
    # 원본 데이터 백업 (분석용)
    df_original = df.copy()
    
    # 3.1 정규화 전 데이터 분포 출력
    print("\n   3.1 정규화 전 입력 지표 기본 통계")
    print(df[input_features].describe().T[['mean', 'std', 'min', 'max']])
    
    # 3.2 StandardScaler로 정규화
    print("\n   3.2 StandardScaler 적용 (평균 0, 표준편차 1)")
    scaler = StandardScaler()
    df_scaled = pd.DataFrame(
        scaler.fit_transform(df[input_features]),
        columns=input_features
    )
    
    # 3.3 정규화 후 데이터 분포 출력
    print("\n   3.3 정규화 후 입력 지표 기본 통계")
    print(df_scaled.describe().T[['mean', 'std', 'min', 'max']])
    
    return df_scaled, df_original, scaler

def find_optimal_k(df_scaled):
    """
    최적의 클러스터 수 k를 찾는 함수
    - Elbow Method 사용
    - Silhouette Score 사용
    """
    print("\n4. 최적의 클러스터 수(k) 탐색 중...")
    
    # 4.1 Elbow Method
    print("\n   4.1 Elbow Method 실행")
    wcss = []  # Within Cluster Sum of Squares
    k_range = range(2, 11)
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(df_scaled)
        wcss.append(kmeans.inertia_)
        print(f"   - k={k}: WCSS = {kmeans.inertia_:.2f}")
    
    # 4.2 Silhouette Score
    print("\n   4.2 Silhouette Score 계산")
    silhouette_scores = []
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(df_scaled)
        silhouette_avg = silhouette_score(df_scaled, cluster_labels)
        silhouette_scores.append(silhouette_avg)
        print(f"   - k={k}: Silhouette Score = {silhouette_avg:.4f}")
    
    # 4.3 최적의 k 제안
    max_silhouette_idx = np.argmax(silhouette_scores)
    optimal_k_silhouette = k_range[max_silhouette_idx]
    
    # Elbow Method 분석 - 기울기 변화율 계산
    wcss_diffs = np.diff(wcss)
    wcss_diffs_rate = np.diff(wcss_diffs) / wcss_diffs[:-1]
    elbow_idx = np.argmax(np.abs(wcss_diffs_rate)) + 1  # +1 because of double diff
    optimal_k_elbow = k_range[elbow_idx]
    
    print(f"\n   4.3 최적 클러스터 수 분석 결과:")
    print(f"   - Elbow Method 기반 최적 k: {optimal_k_elbow}")
    print(f"   - Silhouette Score 기반 최적 k: {optimal_k_silhouette}")
    
    # 두 방법 모두 고려한 최종 추천
    if optimal_k_elbow == optimal_k_silhouette:
        final_k = optimal_k_elbow
        print(f"   - 최종 추천 k: {final_k} (두 방법에서 동일한 결과)")
    else:
        # 일반적으로 Silhouette Score가 더 신뢰할 수 있음
        final_k = optimal_k_silhouette
        print(f"   - 최종 추천 k: {final_k} (Silhouette Score 기반)")
    
    return final_k, k_range, wcss, silhouette_scores

def perform_kmeans(df_scaled, df_original, optimal_k):
    """
    K-means 클러스터링 수행 및 결과 분석
    """
    print(f"\n5. K-means 클러스터링 수행 (k={optimal_k})...")
    
    # 5.1 K-means 실행
    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(df_scaled)
    
    # 클러스터 센터 저장
    centers = pd.DataFrame(kmeans.cluster_centers_, columns=df_scaled.columns)
    
    # 5.2 원본 데이터에 클러스터 레이블 추가
    df_original['cluster'] = cluster_labels
    
    # 5.3 클러스터별 통계
    print("\n   5.1 클러스터별 데이터 분포")
    cluster_counts = df_original['cluster'].value_counts().sort_index()
    for cluster_id, count in cluster_counts.items():
        percentage = count / len(df_original) * 100
        print(f"   - 클러스터 {cluster_id}: {count}개 데이터 포인트 ({percentage:.2f}%)")
    
    # 5.4 클러스터별 중심점 (센트로이드) 분석
    print("\n   5.2 클러스터별 중심점 (정규화 스케일)")
    print(centers)
    
    # 5.5 원래 스케일로 변환된 중심점
    print("\n   5.3 클러스터별 중심점 (원래 스케일)")
    centers_original_scale = pd.DataFrame(
        scaler.inverse_transform(centers),
        columns=centers.columns
    )
    print(centers_original_scale)
    
    # 5.6 클러스터별 특성 분석
    print("\n   5.4 클러스터별 입력 지표 평균값 (원래 스케일)")
    cluster_means = df_original.groupby('cluster')[input_features].mean()
    print(cluster_means)
    
    # 5.7 결과 지표 (wallet_status)에 따른 분석
    if 'WALLET_STATUS' in df_original.columns:
        print("\n   5.5 클러스터별 지갑 상태 분포")
        wallet_status_by_cluster = pd.crosstab(
            df_original['cluster'], 
            df_original['WALLET_STATUS'], 
            normalize='index'
        ) * 100
        
        cluster_status_counts = pd.crosstab(df_original['cluster'], df_original['WALLET_STATUS'])
        
        for cluster_id in range(optimal_k):
            if cluster_id in cluster_status_counts.index:
                total = cluster_status_counts.loc[cluster_id].sum()
                if 'churned' in cluster_status_counts.columns:
                    churned = cluster_status_counts.loc[cluster_id, 'churned']
                    churn_rate = churned / total * 100
                    print(f"   - 클러스터 {cluster_id}: 이탈률 {churn_rate:.2f}% ({churned}/{total})")
                else:
                    print(f"   - 클러스터 {cluster_id}: 'churned' 상태 없음")
        
        print("\n   [클러스터별 지갑 상태 비율 (%)]")
        print(wallet_status_by_cluster)
    
    return df_original, centers, centers_original_scale

# 메인 실행
if __name__ == "__main__":
    print("========== K-means 클러스터링 분석 시작 ==========")
    
    # 1. 데이터 로드
    df = load_data()
    
    # 2. 데이터 전처리
    df, input_features = preprocess_data(df)
    
    # 3. 데이터 정규화
    df_scaled, df_original, scaler = normalize_data(df, input_features)
    
    # 4. 최적의 k 찾기
    optimal_k, k_range, wcss, silhouette_scores = find_optimal_k(df_scaled)
    
    # 5. K-means 클러스터링 수행
    df_with_clusters, centers, centers_original_scale = perform_kmeans(df_scaled, df_original, optimal_k)
    
    # 6. 결과 저장
    print("\n6. 분석 결과 저장...")
    result_file = os.path.join(OUTPUT_DIR, f'kmeans_analysis_k{optimal_k}.csv')
    df_with_clusters.to_csv(result_file, index=False)
    print(f"   - 클러스터링 결과 저장 완료: {result_file}")
    
    centers_file = os.path.join(OUTPUT_DIR, f'cluster_centers_k{optimal_k}.csv')
    centers_original_scale.to_csv(centers_file, index=True)
    print(f"   - 클러스터 중심점 저장 완료: {centers_file}")
    
    print("\n========== K-means 클러스터링 분석 완료 ==========") 