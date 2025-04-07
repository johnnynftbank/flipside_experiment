#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ROI 구간별 지갑 분포 분석
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 스타일 설정
plt.style.use('seaborn-v0_8')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 12
plt.rcParams['figure.figsize'] = (12, 8)

# 데이터 로드
data_path = '../../expected_roi_churn_final/query_result/meme_coin_roi_churn_3.csv'
df = pd.read_csv(data_path)

print(f"총 데이터 개수: {len(df)}")
print(f"이탈 지갑 수: {len(df[df['WALLET_STATUS'] == 'CHURNED'])}")
print(f"활성 지갑 수: {len(df[df['WALLET_STATUS'] == 'ACTIVE'])}")

# ROI 분포 확인
print("\nROI 기술 통계:")
print(df['EXPECTED_ROI'].describe())

# ROI 범위 설정 (outlier 처리)
roi_max = 5  # ROI 최대값 설정 (극단값 제외)
filtered_df = df[df['EXPECTED_ROI'] <= roi_max]

print(f"\n필터링 후 데이터 개수: {len(filtered_df)}")
print(f"필터링 후 이탈 지갑 수: {len(filtered_df[filtered_df['WALLET_STATUS'] == 'CHURNED'])}")
print(f"필터링 후 활성 지갑 수: {len(filtered_df[filtered_df['WALLET_STATUS'] == 'ACTIVE'])}")

# ROI 구간 나누기
bins = np.linspace(0, roi_max, 21)  # 0부터 roi_max까지 20개 구간으로 나눔
labels = [f"{bins[i]:.1f}-{bins[i+1]:.1f}" for i in range(len(bins)-1)]

filtered_df['ROI_BUCKET'] = pd.cut(filtered_df['EXPECTED_ROI'], bins=bins, labels=labels, include_lowest=True)

# 구간별 지갑 수 계산
wallet_counts = filtered_df.groupby(['ROI_BUCKET', 'WALLET_STATUS']).size().unstack(fill_value=0)

if 'ACTIVE' not in wallet_counts.columns:
    wallet_counts['ACTIVE'] = 0
if 'CHURNED' not in wallet_counts.columns:
    wallet_counts['CHURNED'] = 0

# 구간별 이탈률 계산
wallet_counts['CHURN_RATE'] = wallet_counts['CHURNED'] / (wallet_counts['ACTIVE'] + wallet_counts['CHURNED']) * 100

# 결과 출력
print("\n구간별 지갑 수:")
print(wallet_counts)

# 시각화 1: 히스토그램 (이탈 지갑과 활성 지갑 구분)
plt.figure(figsize=(14, 8))
ax = sns.barplot(x=wallet_counts.index, y='ACTIVE', data=wallet_counts.reset_index(), color='#4CAF50', label='활성 지갑')
sns.barplot(x=wallet_counts.index, y='CHURNED', data=wallet_counts.reset_index(), color='#F44336', label='이탈 지갑')

plt.title('ROI 구간별 활성 지갑과 이탈 지갑 분포', fontsize=18)
plt.xlabel('Expected ROI 구간', fontsize=14)
plt.ylabel('지갑 수', fontsize=14)
plt.xticks(rotation=45, ha='right')
plt.legend(loc='upper right')
plt.tight_layout()
plt.savefig('../../roi_wallet_distribution.png', dpi=300, bbox_inches='tight')

# 시각화 2: 구간별 이탈률
plt.figure(figsize=(14, 6))
ax = sns.lineplot(x=wallet_counts.index, y='CHURN_RATE', data=wallet_counts.reset_index(), marker='o', color='#FF5722')
plt.title('ROI 구간별 이탈률', fontsize=18)
plt.xlabel('Expected ROI 구간', fontsize=14)
plt.ylabel('이탈률 (%)', fontsize=14)
plt.xticks(rotation=45, ha='right')
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('../../roi_wallet_churn_rate.png', dpi=300, bbox_inches='tight')

print("\n분석 완료. 시각화 이미지가 저장되었습니다.") 