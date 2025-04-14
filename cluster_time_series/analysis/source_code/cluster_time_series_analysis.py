import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# 데이터 시각화 설정
plt.rcParams['font.family'] = 'AppleGothic'  # 한글 폰트 설정
plt.rcParams['axes.unicode_minus'] = False   # 마이너스 기호 깨짐 방지
sns.set(style="whitegrid")

# 결과 저장 경로
RESULTS_PATH = "../report/"
os.makedirs(RESULTS_PATH, exist_ok=True)

# ============================== 데이터 로드 함수 ==============================
def load_monthly_data(months):
    """
    각 월별 데이터 파일을 로드하고 결합
    
    Parameters:
    -----------
    months : list of tuples
        각 월별 데이터 파일 정보 (year, month, filename)
        
    Returns:
    --------
    pandas.DataFrame
        결합된 월별 데이터
    """
    all_data = []
    
    for year, month, filename in months:
        try:
            file_path = f"../../query/query_result/{filename}"
            monthly_data = pd.read_csv(file_path)
            
            # 월 정보 추가 (이미 쿼리에서 추가했지만 혹시 모르니)
            if 'year' not in monthly_data.columns:
                monthly_data['year'] = year
            if 'month' not in monthly_data.columns:
                monthly_data['month'] = month
                
            all_data.append(monthly_data)
            print(f"{year}년 {month}월 데이터 로드 완료: {len(monthly_data)}개 행")
        except Exception as e:
            print(f"데이터 로드 중 오류 발생 ({year}년 {month}월): {e}")
    
    if not all_data:
        raise ValueError("로드된 데이터가 없습니다. 파일 경로를 확인하세요.")
        
    combined_data = pd.concat(all_data, ignore_index=True)
    print(f"총 {len(combined_data)}개 데이터 로드 완료")
    
    return combined_data

# ============================== 군집 분석 함수 ==============================
def perform_clustering(data, features, n_clusters=5):
    """
    K-means 군집 분석 수행
    
    Parameters:
    -----------
    data : pandas.DataFrame
        분석할 데이터
    features : list
        군집화에 사용할 특성 목록
    n_clusters : int
        군집 수
        
    Returns:
    --------
    pandas.DataFrame
        군집 레이블이 추가된 데이터
    """
    # 결측치가 있는 행 제거
    data_clean = data.dropna(subset=features)
    
    # 특성 추출 및 스케일링
    X = data_clean[features].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # K-means 클러스터링
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    data_clean['cluster'] = kmeans.fit_predict(X_scaled)
    
    # 각 군집의 중심 계산
    centroids = pd.DataFrame(scaler.inverse_transform(kmeans.cluster_centers_), 
                            columns=features)
    
    # 특성 중요도 (중심 간 거리)
    feature_importance = np.std(centroids, axis=0)
    
    print(f"군집 분석 완료: {n_clusters}개 군집")
    print("\n군집 중심:")
    print(centroids)
    print("\n특성 중요도:")
    for i, feat in enumerate(features):
        print(f"{feat}: {feature_importance[i]:.4f}")
    
    return data_clean

# ============================== 군집 시계열 분석 함수 ==============================
def analyze_cluster_over_time(data):
    """
    시간에 따른 군집 변화 분석
    
    Parameters:
    -----------
    data : pandas.DataFrame
        군집 레이블이 포함된 데이터
    """
    # 월별 군집 분포
    monthly_distribution = data.groupby(['year', 'month', 'cluster']).size().unstack(fill_value=0)
    
    # 월별 군집 비율
    monthly_percentage = monthly_distribution.div(monthly_distribution.sum(axis=1), axis=0) * 100
    
    # 군집별 월간 변화 그래프
    plt.figure(figsize=(14, 8))
    
    # 월 레이블 생성
    month_labels = []
    for idx in monthly_distribution.index:
        year, month = idx
        month_labels.append(f"{year}-{month:02d}")
    
    # 절대 수 그래프
    plt.subplot(2, 1, 1)
    monthly_distribution.plot(kind='bar', ax=plt.gca())
    plt.title('군집별 월간 분포 (절대 수)', fontsize=14)
    plt.xlabel('연월', fontsize=12)
    plt.ylabel('지갑 수', fontsize=12)
    plt.xticks(range(len(month_labels)), month_labels, rotation=45)
    plt.legend(title='군집')
    
    # 비율 그래프
    plt.subplot(2, 1, 2)
    monthly_percentage.plot(kind='bar', stacked=True, ax=plt.gca())
    plt.title('군집별 월간 분포 (비율)', fontsize=14)
    plt.xlabel('연월', fontsize=12)
    plt.ylabel('비율 (%)', fontsize=12)
    plt.xticks(range(len(month_labels)), month_labels, rotation=45)
    plt.legend(title='군집')
    
    plt.tight_layout()
    plt.savefig(f"{RESULTS_PATH}monthly_cluster_distribution.png", dpi=300)
    
    # 군집별 추세선 그래프
    plt.figure(figsize=(14, 8))
    
    # 절대 수 추세
    plt.subplot(2, 1, 1)
    for cluster in monthly_distribution.columns:
        plt.plot(range(len(month_labels)), monthly_distribution[cluster], marker='o', label=f'군집 {cluster}')
    plt.title('군집별 월간 추세 (절대 수)', fontsize=14)
    plt.xlabel('연월', fontsize=12)
    plt.ylabel('지갑 수', fontsize=12)
    plt.xticks(range(len(month_labels)), month_labels, rotation=45)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 비율 추세
    plt.subplot(2, 1, 2)
    for cluster in monthly_percentage.columns:
        plt.plot(range(len(month_labels)), monthly_percentage[cluster], marker='o', label=f'군집 {cluster}')
    plt.title('군집별 월간 추세 (비율)', fontsize=14)
    plt.xlabel('연월', fontsize=12)
    plt.ylabel('비율 (%)', fontsize=12)
    plt.xticks(range(len(month_labels)), month_labels, rotation=45)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{RESULTS_PATH}monthly_cluster_trends.png", dpi=300)
    
    # 결과 저장
    monthly_distribution.to_csv(f"{RESULTS_PATH}monthly_cluster_counts.csv")
    monthly_percentage.to_csv(f"{RESULTS_PATH}monthly_cluster_percentages.csv")
    
    return monthly_distribution, monthly_percentage

# ============================== 군집 특성 분석 함수 ==============================
def analyze_cluster_characteristics(data, features):
    """
    각 군집의 특성 분석
    
    Parameters:
    -----------
    data : pandas.DataFrame
        군집 레이블이 포함된 데이터
    features : list
        분석할 특성 목록
    """
    # 군집별 특성 평균
    cluster_means = data.groupby('cluster')[features].mean()
    
    # 특성 분포 시각화 (레이더 차트)
    plt.figure(figsize=(12, 10))
    
    # 특성을 0-1 범위로 정규화
    cluster_means_normalized = (cluster_means - cluster_means.min()) / (cluster_means.max() - cluster_means.min())
    
    # 레이더 차트 각도 설정
    angles = np.linspace(0, 2*np.pi, len(features), endpoint=False).tolist()
    angles += angles[:1]  # 처음으로 돌아가 원을 완성
    
    # 플롯 설정
    ax = plt.subplot(111, polar=True)
    
    # 각 군집에 대한 레이더 차트
    for cluster in cluster_means_normalized.index:
        values = cluster_means_normalized.loc[cluster].values.tolist()
        values += values[:1]  # 처음으로 돌아가 원을 완성
        ax.plot(angles, values, linewidth=2, label=f'군집 {cluster}')
        ax.fill(angles, values, alpha=0.1)
    
    # 레이블 설정
    ax.set_thetagrids(np.degrees(angles[:-1]), features)
    plt.title('군집별 특성 프로파일', fontsize=15)
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    
    plt.tight_layout()
    plt.savefig(f"{RESULTS_PATH}cluster_characteristics_radar.png", dpi=300)
    
    # 군집별 상자 그림
    for feature in features:
        plt.figure(figsize=(10, 6))
        sns.boxplot(x='cluster', y=feature, data=data)
        plt.title(f'군집별 {feature} 분포', fontsize=14)
        plt.xlabel('군집', fontsize=12)
        plt.ylabel(feature, fontsize=12)
        plt.tight_layout()
        plt.savefig(f"{RESULTS_PATH}cluster_{feature}_boxplot.png", dpi=300)
    
    # 결과 저장
    cluster_means.to_csv(f"{RESULTS_PATH}cluster_characteristics.csv")
    
    return cluster_means

# ============================== 메인 실행 코드 ==============================
def main():
    """
    메인 실행 함수
    """
    print("===== 월별 군집 시계열 분석 시작 =====")
    
    # 1. 분석에 사용할 월별 데이터 파일 목록 (예시)
    # 형식: (연도, 월, 파일명)
    months_data = [
        (2024, 9, 'meme_coin_traders_2024_09.csv'),
        (2024, 10, 'meme_coin_traders_2024_10.csv'),
        (2024, 11, 'meme_coin_traders_2024_11.csv'),
        (2024, 12, 'meme_coin_traders_2024_12.csv'),
        (2025, 1, 'meme_coin_traders_2025_01.csv'),
        (2025, 2, 'meme_coin_traders_2025_02.csv'),
        (2025, 3, 'meme_coin_traders_2025_03.csv')
    ]
    
    # 2. 데이터 로드
    print("\n데이터 로드 중...")
    all_data = load_monthly_data(months_data)
    
    # 3. 클러스터링에 사용할 특성 정의
    features = [
        'expected_roi', 
        'roi_standard_deviation', 
        'sharpe_ratio', 
        'win_loss_ratio', 
        'max_trade_proportion',
        'unique_tokens_traded',
        'total_trades',
        'trading_days'
    ]
    
    # 4. 군집 분석 수행
    print("\n군집 분석 수행 중...")
    clustered_data = perform_clustering(all_data, features, n_clusters=5)
    
    # 5. 시간에 따른 군집 변화 분석
    print("\n월별 군집 변화 분석 중...")
    monthly_counts, monthly_percent = analyze_cluster_over_time(clustered_data)
    
    # 6. 군집별 특성 분석
    print("\n군집별 특성 분석 중...")
    cluster_profiles = analyze_cluster_characteristics(clustered_data, features)
    
    print("\n===== 분석 완료 =====")
    print(f"결과가 {RESULTS_PATH} 폴더에 저장되었습니다.")

# 스크립트 실행
if __name__ == '__main__':
    main() 