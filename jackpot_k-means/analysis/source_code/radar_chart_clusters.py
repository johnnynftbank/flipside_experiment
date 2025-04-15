#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
레이더 차트 생성: 군집별 특성 비교 시각화
- 목적: 잭팟 추구형 투자자(군집 1)의 특성을 다른 군집과 비교하여 시각화
- 특성: 최대 거래 비중, 기대 수익률, 샤프 비율, 승패 비율, 이탈률
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib as mpl
from matplotlib.path import Path
from matplotlib.spines import Spine
from matplotlib.transforms import Affine2D

# 한글 폰트 문제 방지
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

# 데이터 불러오기
df = pd.read_csv('jackpot_k-means/analysis/report/jackpot_cluster_results.csv')

# 1. 데이터 전처리
# 각 군집별 주요 지표의 평균값 계산
cluster_metrics = df.groupby('cluster').agg({
    'EXPECTED_ROI': 'mean',
    'SHARPE_RATIO': 'mean',
    'WIN_LOSS_RATIO': 'mean',
    'MAX_TRADE_PROPORTION': 'mean'
}).reset_index()

# 군집별 이탈률 계산
churn_rates = df.groupby(['cluster', 'WALLET_STATUS_LABEL']).size().unstack(fill_value=0)
churn_rates['CHURN_RATE'] = churn_rates['Churned'] / (churn_rates['Active'] + churn_rates['Churned'])
cluster_metrics = pd.merge(cluster_metrics, churn_rates[['CHURN_RATE']], left_on='cluster', right_index=True)

# 2. 지표 정규화 (레이더 차트 스케일 조정용)
features = ['MAX_TRADE_PROPORTION', 'EXPECTED_ROI', 'SHARPE_RATIO', 'WIN_LOSS_RATIO', 'CHURN_RATE']

# 정규화를 위한 함수 - 각 지표별 특성에 맞게 조정
def normalize_for_radar(df, features):
    normalized_df = df.copy()
    
    # EXPECTED_ROI: 높을수록 좋음 (값의 범위를 0~1로 조정, 원래 값이 음수가 많으므로 순위를 0~1로 정규화)
    roi_ranks = df['EXPECTED_ROI'].rank(pct=True)
    normalized_df['EXPECTED_ROI_NORM'] = roi_ranks
    
    # SHARPE_RATIO: 높을수록 좋음 (값의 범위를 0~1로 조정)
    sharpe_ranks = df['SHARPE_RATIO'].rank(pct=True)
    normalized_df['SHARPE_RATIO_NORM'] = sharpe_ranks
    
    # WIN_LOSS_RATIO: 높을수록 좋음 (값의 범위를 0~1로 조정)
    normalized_df['WIN_LOSS_RATIO_NORM'] = df['WIN_LOSS_RATIO'] / df['WIN_LOSS_RATIO'].max()
    
    # MAX_TRADE_PROPORTION: 낮을수록 좋음 (값의 범위를 0~1로 조정, 역순)
    normalized_df['MAX_TRADE_PROPORTION_NORM'] = 1 - (df['MAX_TRADE_PROPORTION'] / df['MAX_TRADE_PROPORTION'].max())
    
    # CHURN_RATE: 낮을수록 좋음 (값의 범위를 0~1로 조정, 역순)
    normalized_df['CHURN_RATE_NORM'] = 1 - df['CHURN_RATE']
    
    return normalized_df

normalized_df = normalize_for_radar(cluster_metrics, features)

# 레이더 차트 생성을 위한 함수
def radar_chart(fig, normalized_df, titles, colors):
    # 그래프 설정
    N = len(titles)
    theta = np.linspace(0, 2*np.pi, N, endpoint=False)
    
    # 축 추가
    ax = fig.add_subplot(111, polar=True)
    
    # 각 군집별 레이더 차트 그리기
    for i, row in normalized_df.iterrows():
        cluster = int(row['cluster'])
        values = [
            row['MAX_TRADE_PROPORTION_NORM'],
            row['EXPECTED_ROI_NORM'],
            row['SHARPE_RATIO_NORM'],
            row['WIN_LOSS_RATIO_NORM'],
            row['CHURN_RATE_NORM']
        ]
        values = np.append(values, values[0])  # 폐곡선을 위해 첫 값 반복
        
        # 데이터 포인트
        ax.plot(np.append(theta, theta[0]), values, color=colors[i], linewidth=2, label=f'Cluster {cluster}')
        ax.fill(np.append(theta, theta[0]), values, color=colors[i], alpha=0.25)
    
    # 축 레이블 설정
    ax.set_xticks(theta)
    ax.set_xticklabels([
        'Low Max Trade\nProportion',
        'High Expected\nROI',
        'High Sharpe\nRatio',
        'High Win/Loss\nRatio',
        'Low Churn\nRate'
    ])
    
    # 반지름 눈금 설정
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(['0.25', '0.5', '0.75', '1.0'])
    ax.set_rlim(0, 1)
    
    # 그리드 설정
    ax.grid(True, linestyle='-', alpha=0.7)
    
    # 제목 설정
    plt.title('Cluster Characteristics Comparison\n(Normalized Values, Higher is Better)', y=1.08, fontsize=14)
    
    # 범례 설정
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))

# 레이더 차트 생성
fig = plt.figure(figsize=(10, 8))

titles = [
    'Low Max Trade Proportion',
    'High Expected ROI',
    'High Sharpe Ratio',
    'High Win/Loss Ratio',
    'Low Churn Rate'
]

colors = ['purple', 'blue', 'green', 'orange']

radar_chart(fig, normalized_df, titles, colors)

# 원본 데이터 테이블 추가 (하단에 작은 텍스트로)
original_values = """
Cluster 0: Stable Investor
- Expected ROI: -0.002 (near zero)
- ROI Std Dev: 0.039 (very low)
- Sharpe Ratio: -0.085 (slightly negative)
- Win/Loss Ratio: 0.709 (high)
- Max Trade Proportion: 0.115 (low)
- Churn Rate: 12.4% (very low)

Cluster 1: Jackpot Seeker
- Expected ROI: -0.757 (very low)
- ROI Std Dev: 0.179 (low)
- Sharpe Ratio: -7.306 (extremely low)
- Win/Loss Ratio: 0.031 (extremely low)
- Max Trade Proportion: 0.440 (very high)
- Churn Rate: 95.5% (extremely high)

Cluster 2: Adventurous Investor
- Expected ROI: -0.044 (slightly negative)
- ROI Std Dev: 0.610 (very high)
- Sharpe Ratio: -0.074 (slightly negative)
- Win/Loss Ratio: 0.664 (high)
- Max Trade Proportion: 0.078 (low)
- Churn Rate: 31.3% (medium)

Cluster 3: General Investor
- Expected ROI: -0.164 (somewhat negative)
- ROI Std Dev: 0.353 (medium)
- Sharpe Ratio: -0.468 (negative)
- Win/Loss Ratio: 0.519 (medium)
- Max Trade Proportion: 0.068 (very low)
- Churn Rate: 31.7% (medium)
"""

# plt.figtext(0.1, -0.05, original_values, fontsize=8, wrap=True)

# 이미지 저장
plt.tight_layout()
plt.savefig('jackpot_k-means/analysis/report/images/cluster_radar_chart.png', dpi=300, bbox_inches='tight')
plt.savefig('jackpot_k-means/analysis/report/images/cluster_radar_chart.pdf', bbox_inches='tight')

print("레이더 차트가 성공적으로 생성되었습니다.") 