import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# 결과 디렉토리 확인
os.makedirs("verify/correlation_analysis/results", exist_ok=True)

# 최신 포인트 바이시리얼 상관계수
latest_corr = -0.3202

# 상관관계 행렬 생성
corr_matrix = pd.DataFrame([
    [1.000, latest_corr],
    [latest_corr, 1.000]
], index=['EXPECTED_ROI', 'IS_CHURNED'], columns=['EXPECTED_ROI', 'IS_CHURNED'])

# 히트맵 설정
plt.figure(figsize=(12, 10))
sns.set(font_scale=1.5)
sns.set_style("white")

# 히트맵 생성
mask = np.zeros_like(corr_matrix, dtype=bool)
mask[np.triu_indices_from(mask)] = False

cmap = sns.diverging_palette(220, 10, as_cmap=True)

ax = sns.heatmap(
    corr_matrix, 
    mask=mask,
    cmap=cmap,
    vmax=1.0, 
    vmin=-1.0,
    center=0,
    square=True, 
    linewidths=0.5, 
    cbar_kws={"shrink": 0.8},
    annot=True,
    fmt='.3f',
    annot_kws={"size": 25}
)

# 타이틀 및 라벨 설정
plt.title('Correlation Between Expected ROI and Churn Status', fontsize=20, pad=20)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16, rotation=0)

# 저장
plt.tight_layout()
plt.savefig('verify/correlation_analysis/results/roi_churn_correlation_heatmap_updated.png', dpi=300, bbox_inches='tight')
print("Updated heatmap created with correlation coefficient: -0.3202") 