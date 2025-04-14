#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
군집 분류 분석 스크립트
- jackpot_k-means_report.md의 군집 기준을 정확히 적용
- quarterly_cluster_analysis_25_03.csv 데이터 분류
"""

import pandas as pd
import numpy as np
import os

# 작업 디렉토리 설정
base_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), '')
csv_path = os.path.join(base_dir, 'query/query_result/quarterly_cluster_analysis_25_03.csv')

# 데이터 로드
print(f"데이터 파일 로드 중: {csv_path}")
df = pd.read_csv(csv_path)
print(f"총 {len(df)}개 지갑 데이터 로드 완료")

# 결측치 확인 및 제거
initial_count = len(df)
df = df.dropna(subset=['EXPECTED_ROI', 'ROI_STANDARD_DEVIATION', 'SHARPE_RATIO', 'WIN_LOSS_RATIO', 'MAX_TRADE_PROPORTION'])
dropped_count = initial_count - len(df)
print(f"{dropped_count}개 결측치 행 제거됨, 분석 대상 지갑 수: {len(df)}")

# 데이터 특성 요약 출력
print("\n데이터 특성 요약:")
print("-" * 50)
for col in ['EXPECTED_ROI', 'ROI_STANDARD_DEVIATION', 'SHARPE_RATIO', 'WIN_LOSS_RATIO', 'MAX_TRADE_PROPORTION']:
    print(f"{col} - 최소값: {df[col].min():.4f}, 최대값: {df[col].max():.4f}, 평균: {df[col].mean():.4f}, 중앙값: {df[col].median():.4f}")

# 군집별 중심값 정의 (jackpot_k-means_report.md의 값을 그대로 사용)
cluster_centers = {
    0: {  # 안정적 투자형
        'EXPECTED_ROI': -0.002,
        'ROI_STANDARD_DEVIATION': 0.039,
        'SHARPE_RATIO': -0.085,
        'WIN_LOSS_RATIO': 0.709,
        'MAX_TRADE_PROPORTION': 0.115
    },
    1: {  # 잭팟 추구형
        'EXPECTED_ROI': -0.757,
        'ROI_STANDARD_DEVIATION': 0.179,
        'SHARPE_RATIO': -7.306,
        'WIN_LOSS_RATIO': 0.031,
        'MAX_TRADE_PROPORTION': 0.440
    },
    2: {  # 모험적 투자형
        'EXPECTED_ROI': -0.044,
        'ROI_STANDARD_DEVIATION': 0.610,
        'SHARPE_RATIO': -0.074,
        'WIN_LOSS_RATIO': 0.664,
        'MAX_TRADE_PROPORTION': 0.078
    },
    3: {  # 일반 투자형
        'EXPECTED_ROI': -0.164,
        'ROI_STANDARD_DEVIATION': 0.353,
        'SHARPE_RATIO': -0.468,
        'WIN_LOSS_RATIO': 0.519,
        'MAX_TRADE_PROPORTION': 0.068
    }
}

# 각 군집과의 거리 계산 함수
def calculate_distance(row, center_values):
    # 유클리디안 거리 계산을 위한 특성 정규화
    features = ['EXPECTED_ROI', 'ROI_STANDARD_DEVIATION', 'SHARPE_RATIO', 'WIN_LOSS_RATIO', 'MAX_TRADE_PROPORTION']
    
    # 각 특성별 정규화 스케일 계수 (값의 범위를 고려)
    scale_factors = {
        'EXPECTED_ROI': 1.0,
        'ROI_STANDARD_DEVIATION': 1.0,
        'SHARPE_RATIO': 0.5,  # 값의 범위가 넓어 가중치 조정
        'WIN_LOSS_RATIO': 1.0,
        'MAX_TRADE_PROPORTION': 2.0  # 잭팟 추구형 분류에 중요하므로 가중치 상향
    }
    
    squared_diff_sum = 0
    for feature in features:
        normalized_diff = (row[feature] - center_values[feature]) / scale_factors[feature]
        squared_diff_sum += normalized_diff ** 2
    
    return np.sqrt(squared_diff_sum)

# 군집 할당 함수
def assign_cluster(row):
    distances = {}
    
    # 모든 군집과의 거리 계산
    for cluster_id, center_values in cluster_centers.items():
        distances[cluster_id] = calculate_distance(row, center_values)
    
    # 거리가 임계값 이내인 경우만 군집 할당 (큰 거리는 미분류로 처리)
    min_distance = min(distances.values())
    closest_cluster = min(distances, key=distances.get)
    
    # 거리 임계값 (조정 가능)
    threshold = 2.0
    
    if min_distance < threshold:
        return closest_cluster
    else:
        return -1  # 미분류

# 군집 할당
df['CLUSTER'] = df.apply(assign_cluster, axis=1)
df['NEAREST_CLUSTER_DISTANCE'] = df.apply(lambda row: min([calculate_distance(row, center) for center in cluster_centers.values()]), axis=1)

# 군집별 개수 및 비율 계산
cluster_counts = df['CLUSTER'].value_counts().sort_index()
total_wallets = len(df)
cluster_percentages = (cluster_counts / total_wallets * 100).round(2)

# 결과 출력
cluster_names = {
    -1: "Unclassified wallets",
    0: "Stable investors (Cluster 0)",
    1: "Jackpot seekers (Cluster 1)",
    2: "Adventurous investors (Cluster 2)",
    3: "Regular investors (Cluster 3)"
}

print("\n군집 분석 결과:")
print("-" * 50)
for cluster_id, name in cluster_names.items():
    if cluster_id in cluster_counts:
        count = cluster_counts[cluster_id]
        percent = cluster_percentages[cluster_id]
        if count <= 10:
            print(f"- {name}: {percent}% (only {count} wallets)")
        else:
            print(f"- {name}: {percent}% ({count:,} wallets)")
    else:
        print(f"- {name}: 0% (0 wallets)")

# 분류된 데이터 CSV로 저장
classified_csv_path = os.path.join(base_dir, 'analysis/report/quarterly_classified_wallets.csv')
df.to_csv(classified_csv_path, index=False)
print(f"\n분류 완료된 데이터가 저장됨: {classified_csv_path}")

# 각 군집별 주요 특성의 평균값 출력
print("\n각 군집별 주요 특성 평균:")
print("-" * 50)
cluster_stats = df.groupby('CLUSTER')[['EXPECTED_ROI', 'ROI_STANDARD_DEVIATION', 'SHARPE_RATIO', 'WIN_LOSS_RATIO', 'MAX_TRADE_PROPORTION']].mean().round(4)
print(cluster_stats)

# 각 군집의 가장 잘 맞는 지갑 출력 (군집 중심에 가장 가까운 지갑)
print("\n각 군집의 대표 지갑 (군집 중심에 가장 가까움):")
print("-" * 50)
for cluster_id in range(4):  # 0, 1, 2, 3 군집에 대해
    if cluster_id in df['CLUSTER'].values:
        cluster_wallets = df[df['CLUSTER'] == cluster_id]
        closest_wallet = cluster_wallets.loc[cluster_wallets['NEAREST_CLUSTER_DISTANCE'].idxmin()]
        print(f"군집 {cluster_id} 대표 지갑: {closest_wallet['SWAPPER']}")
        print(f"  EXPECTED_ROI: {closest_wallet['EXPECTED_ROI']:.4f}")
        print(f"  ROI_STANDARD_DEVIATION: {closest_wallet['ROI_STANDARD_DEVIATION']:.4f}")
        print(f"  SHARPE_RATIO: {closest_wallet['SHARPE_RATIO']:.4f}")
        print(f"  WIN_LOSS_RATIO: {closest_wallet['WIN_LOSS_RATIO']:.4f}")
        print(f"  MAX_TRADE_PROPORTION: {closest_wallet['MAX_TRADE_PROPORTION']:.4f}")
        print(f"  거리: {closest_wallet['NEAREST_CLUSTER_DISTANCE']:.4f}")
    else:
        print(f"군집 {cluster_id}에 속한 지갑이 없습니다.") 