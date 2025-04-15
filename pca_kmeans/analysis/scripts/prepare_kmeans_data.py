#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PCA 결과 데이터에서 3시그마 기반 다변량 극단값 이상치를 제거하고,
재표준화한 데이터셋을 생성하는 스크립트
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import os

# 경로 설정
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_FILE = os.path.join(BASE_DIR, 'report', 'transformed_pca_data_for_kmeans.csv')
OUTPUT_FILE = os.path.join(BASE_DIR, 'report', 'prepared_data_for_kmeans.csv')

# 데이터 로드
print(f"데이터 파일 로드: {INPUT_FILE}")
data = pd.read_csv(INPUT_FILE)
print(f"원본 데이터 크기: {data.shape}")

# 3시그마 기반 다변량 극단값 식별
X = data[['PC1', 'PC2', 'PC3']]
extremes = (abs(X - X.mean()) > 3 * X.std()).any(axis=1)
outliers_count = extremes.sum()
print(f"3시그마 초과 극단값: {outliers_count}개 ({outliers_count/len(data)*100:.2f}%)")

# 이상치 제거
data_cleaned = data[~extremes].reset_index(drop=True)
print(f"이상치 제거 후 데이터 크기: {data_cleaned.shape}")

# 재표준화 (PC1, PC2, PC3)
scaler = StandardScaler()
pc_columns = ['PC1', 'PC2', 'PC3']
pc_scaled = scaler.fit_transform(data_cleaned[pc_columns])
data_cleaned[pc_columns] = pc_scaled

# 재표준화 후 분산 확인
print("\n재표준화 후 PC 분산:")
var_pc = data_cleaned[pc_columns].var()
print(var_pc)

print("\n재표준화 후 PC 분산 비율:")
print(f'PC1/PC2 비율: {var_pc[0]/var_pc[1]:.4f}')
print(f'PC1/PC3 비율: {var_pc[0]/var_pc[2]:.4f}')
print(f'PC2/PC3 비율: {var_pc[1]/var_pc[2]:.4f}')

# 처리된 데이터 저장
data_cleaned.to_csv(OUTPUT_FILE, index=False)
print(f"\n전처리 완료된 데이터 저장: {OUTPUT_FILE}")
print(f"최종 데이터 크기: {data_cleaned.shape}")

# 분포 통계 출력
print("\n재표준화 데이터 기본 통계:")
print(data_cleaned[pc_columns].describe())

print("\n처리 완료!")
