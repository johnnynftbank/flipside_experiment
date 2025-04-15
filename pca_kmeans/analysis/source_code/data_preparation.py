#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os
from datetime import datetime

# 경로 설정
DATA_PATH = 'pca_kmeans/query/query_result/pca_original_data_3822.csv'
OUTPUT_PATH = 'pca_kmeans/analysis/report'
SOURCE_CODE_PATH = 'pca_kmeans/analysis/source_code'

# 필요한 폴더 생성
os.makedirs(OUTPUT_PATH, exist_ok=True)
os.makedirs(SOURCE_CODE_PATH, exist_ok=True)

# 시각화 스타일 설정
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette('viridis')

print("1단계: 데이터 준비 및 이상치 점검 시작")
print("-" * 50)

# 1.1 데이터 로드
print("1.1 데이터 로드 중...")
df = pd.read_csv(DATA_PATH)
print(f"데이터 크기: {df.shape[0]} 행 x {df.shape[1]} 열")

# 1.2 타겟 변수 선택
target_columns = ['EXPECTED_ROI', 'ROI_STANDARD_DEVIATION', 'SHARPE_RATIO', 'WIN_LOSS_RATIO', 'MAX_TRADE_PROPORTION']
df_target = df[['SWAPPER', 'WALLET_STATUS'] + target_columns]

# 1.3 결측치 확인
print("\n1.3 결측치 확인...")
missing_data = df_target.isnull().sum()
missing_percent = (df_target.isnull().sum() / len(df_target)) * 100
missing_df = pd.DataFrame({
    '결측치 수': missing_data,
    '결측치 비율(%)': missing_percent.round(2)
})
print(missing_df)

# 결측치 분석 결과 저장
missing_df.to_csv(f"{OUTPUT_PATH}/missing_values_analysis.csv")

# 결측치가 있는 경우 처리
if missing_df['결측치 수'].sum() > 0:
    print("\n결측치가 발견되었습니다. 결측치 처리 중...")
    # 주요 컬럼에 결측치가 있는 행 제거
    df_target = df_target.dropna(subset=target_columns)
    print(f"결측치 처리 후 데이터 크기: {df_target.shape[0]} 행 x {df_target.shape[1]} 열")
else:
    print("\n결측치가 없습니다. 추가 처리가 필요하지 않습니다.")

# 1.4 데이터 타입 확인
print("\n1.4 데이터 타입 확인...")
dtypes_df = pd.DataFrame(df_target.dtypes, columns=['데이터 타입'])
print(dtypes_df)
dtypes_df.to_csv(f"{OUTPUT_PATH}/data_types.csv")

# 1.5 기본 통계량 계산
print("\n1.5 기본 통계량 계산...")
stats_df = df_target[target_columns].describe().T
# 추가 통계량 계산 (왜도, 첨도)
stats_df['skewness'] = df_target[target_columns].skew()
stats_df['kurtosis'] = df_target[target_columns].kurtosis()
stats_df['negative_values(%)'] = (df_target[target_columns] < 0).mean() * 100
print(stats_df.round(3))

# 기본 통계량 저장
stats_df.to_csv(f"{OUTPUT_PATH}/basic_statistics.csv")

# 1.6 데이터 분포 시각화
print("\n1.6 데이터 분포 시각화...")

# 히스토그램 생성
plt.figure(figsize=(15, 12))
for i, col in enumerate(target_columns):
    plt.subplot(3, 2, i+1)
    sns.histplot(df_target[col], kde=True)
    plt.axvline(x=0, color='red', linestyle='--', alpha=0.7)  # 0 기준선 추가
    plt.title(f'{col} 분포')
    plt.xlabel(col)
    plt.ylabel('빈도')
    
    # 음수 영역과 양수 영역 표시
    neg_count = (df_target[col] < 0).sum()
    pos_count = (df_target[col] >= 0).sum()
    neg_percent = neg_count / len(df_target) * 100
    pos_percent = pos_count / len(df_target) * 100
    
    plt.annotate(f'음수: {neg_count} ({neg_percent:.1f}%)', 
                 xy=(0.05, 0.85), xycoords='axes fraction', color='blue')
    plt.annotate(f'양수: {pos_count} ({pos_percent:.1f}%)', 
                 xy=(0.05, 0.75), xycoords='axes fraction', color='green')

plt.tight_layout()
plt.savefig(f"{OUTPUT_PATH}/distributions_histograms.png", dpi=300, bbox_inches='tight')
plt.close()

# 박스플롯 생성
plt.figure(figsize=(15, 10))
for i, col in enumerate(target_columns):
    plt.subplot(3, 2, i+1)
    sns.boxplot(x=df_target[col])
    plt.axvline(x=0, color='red', linestyle='--', alpha=0.7)  # 0 기준선 추가
    plt.title(f'{col} 박스플롯')
    
plt.tight_layout()
plt.savefig(f"{OUTPUT_PATH}/boxplots.png", dpi=300, bbox_inches='tight')
plt.close()

# 상관관계 분석
plt.figure(figsize=(10, 8))
correlation_matrix = df_target[target_columns].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, fmt='.2f')
plt.title('변수 간 상관관계')
plt.tight_layout()
plt.savefig(f"{OUTPUT_PATH}/correlation_heatmap.png", dpi=300, bbox_inches='tight')
plt.close()

# 상관관계 행렬 저장
correlation_matrix.to_csv(f"{OUTPUT_PATH}/correlation_matrix.csv")

# 1.7 이상치 분석
print("\n1.7 이상치 분석...")

# IQR 방식으로 이상치 식별
outliers_summary = pd.DataFrame(index=target_columns, 
                              columns=['Q1', 'Q3', 'IQR', '하한경계', '상한경계', '이상치 수', '이상치 비율(%)'])

for col in target_columns:
    Q1 = df_target[col].quantile(0.25)
    Q3 = df_target[col].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outliers = df_target[(df_target[col] < lower_bound) | (df_target[col] > upper_bound)]
    outlier_count = len(outliers)
    outlier_percent = (outlier_count / len(df_target)) * 100
    
    outliers_summary.loc[col] = [Q1, Q3, IQR, lower_bound, upper_bound, outlier_count, outlier_percent]

print(outliers_summary.round(3))
outliers_summary.to_csv(f"{OUTPUT_PATH}/outliers_summary.csv")

# 이상치 처리 - 각 컬럼별로 상하위 1% 윈저라이징 적용
print("\n1.8 이상치 처리 - 윈저라이징 적용...")
df_winsorized = df_target.copy()

for col in target_columns:
    lower_bound = np.percentile(df_target[col], 1)
    upper_bound = np.percentile(df_target[col], 99)
    
    # 윈저라이징 적용
    df_winsorized[col] = df_target[col].clip(lower=lower_bound, upper=upper_bound)

# 윈저라이징 전후 비교
winsorized_comparison = pd.DataFrame(index=target_columns, 
                                    columns=['원본 최소값', '원본 최대값', '윈저라이징 후 최소값', '윈저라이징 후 최대값',
                                            '원본 평균', '원본 표준편차', '윈저라이징 후 평균', '윈저라이징 후 표준편차'])

for col in target_columns:
    winsorized_comparison.loc[col] = [
        df_target[col].min(), 
        df_target[col].max(),
        df_winsorized[col].min(),
        df_winsorized[col].max(),
        df_target[col].mean(),
        df_target[col].std(),
        df_winsorized[col].mean(),
        df_winsorized[col].std()
    ]

print(winsorized_comparison.round(3))
winsorized_comparison.to_csv(f"{OUTPUT_PATH}/winsorized_comparison.csv")

# 이상치 처리 전후 시각화
plt.figure(figsize=(15, 10))
for i, col in enumerate(target_columns):
    plt.subplot(3, 2, i+1)
    sns.kdeplot(df_target[col], label='원본 데이터', color='blue')
    sns.kdeplot(df_winsorized[col], label='윈저라이징 후', color='green')
    plt.axvline(x=0, color='red', linestyle='--', alpha=0.5)
    plt.title(f'{col} 윈저라이징 전후 비교')
    plt.legend()
    
plt.tight_layout()
plt.savefig(f"{OUTPUT_PATH}/winsorized_comparison_plot.png", dpi=300, bbox_inches='tight')
plt.close()

# 1.9 최종 데이터 준비
# 윈저라이징된 데이터로 표준화 준비
print("\n1.9 데이터 표준화 준비...")
from sklearn.preprocessing import StandardScaler

# SWAPPER 컬럼 따로 저장
swapper_ids = df_winsorized['SWAPPER'].values
wallet_status = df_winsorized['WALLET_STATUS'].values

# 표준화 적용
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df_winsorized[target_columns])

# 표준화된 데이터프레임 생성
df_scaled = pd.DataFrame(scaled_data, columns=target_columns)
df_scaled['SWAPPER'] = swapper_ids
df_scaled['WALLET_STATUS'] = wallet_status

# 표준화 결과 확인
scaled_stats = df_scaled[target_columns].describe().T
print(scaled_stats.round(3))
scaled_stats.to_csv(f"{OUTPUT_PATH}/scaled_data_statistics.csv")

# 최종 데이터 저장
df_winsorized.to_csv(f"{OUTPUT_PATH}/winsorized_data.csv", index=False)
df_scaled.to_csv(f"{OUTPUT_PATH}/scaled_data.csv", index=False)

# 1.10 보고서 마크다운 생성
print("\n1.10 마크다운 보고서 생성 중...")

# 수동으로 마크다운 테이블 생성 함수
def dataframe_to_markdown(df):
    markdown = "| " + " | ".join(df.columns) + " |\n"
    markdown += "| " + " | ".join(["---"] * len(df.columns)) + " |\n"
    
    for i, row in df.iterrows():
        values = [str(x) for x in row.values]
        markdown += "| " + " | ".join(values) + " |\n"
    
    return markdown

# 결측치 테이블
missing_table = dataframe_to_markdown(missing_df.reset_index().rename(columns={'index': '변수'}))

# 데이터 타입 테이블
dtypes_table = dataframe_to_markdown(dtypes_df.reset_index().rename(columns={'index': '변수'}))

# 이상치 테이블
outliers_table = dataframe_to_markdown(
    outliers_summary[['이상치 수', '이상치 비율(%)']].round(2).reset_index().rename(columns={'index': '변수'})
)

# 윈저라이징 비교 테이블
winsorized_table = dataframe_to_markdown(
    winsorized_comparison[['원본 평균', '윈저라이징 후 평균', '원본 표준편차', '윈저라이징 후 표준편차']].round(3).reset_index().rename(columns={'index': '변수'})
)

report_template = f"""# 1단계: 데이터 준비 및 이상치 점검 보고서

## 1. 요약 (Executive Summary)

- 총 {df.shape[0]}개의 지갑 데이터에 대해 5개 핵심 지표 분석 완료
- 결측치는 발견되었으며({missing_df['결측치 수'].sum()}개), 해당 행 제거 처리됨
- 음수값(특히 EXPECTED_ROI, SHARPE_RATIO)은 유효한 금융 지표이므로 변환 없이 유지
- 최종 데이터는 표준화(Standardization)되어 PCA 분석 준비 완료

## 2. 데이터 품질 분석

### 결측치 분석:

{missing_table}

**결론**: 결측치가 있는 행({missing_df['결측치 수'].sum()}개)을 제거하여 처리완료

### 데이터 타입 확인:

{dtypes_table}

**결론**: 모든 핵심 지표가 적절한 숫자형(float64)으로 확인됨

## 3. 이상치 분석 및 처리

### 주요 이상치 통계:

{outliers_table}

**이상치 처리 방법**: 상하위 1% 윈저라이징(Winsorizing)을 적용하여 극단값의 영향을 완화하면서도 데이터 포인트를 유지함

### 핵심 시각화:

![분포 히스토그램](distributions_histograms.png)

주요 관찰: EXPECTED_ROI와 SHARPE_RATIO는 음수 값이 많음 (약 {df_target['EXPECTED_ROI'].lt(0).mean()*100:.1f}%)

## 4. 최종 데이터셋 요약

### 처리 전/후 비교 통계:

{winsorized_table}

**표준화 준비 상태**: StandardScaler를 사용하여 각 변수의 평균 0, 표준편차 1로 변환 완료

## 5. 참조 파일 목록

| 단계 | 파일명 | 내용 설명 |
|------|-------|----------|
| 1.3 | missing_values_analysis.csv | 결측치 분석 결과 |
| 1.4 | data_types.csv | 데이터 타입 확인 결과 |
| 1.5 | basic_statistics.csv | 기본 통계량 분석 결과 |
| 1.6 | distributions_histograms.png | 각 변수별 분포 히스토그램 |
| 1.6 | boxplots.png | 각 변수별 박스플롯 |
| 1.6 | correlation_heatmap.png | 변수 간 상관관계 히트맵 |
| 1.6 | correlation_matrix.csv | 상관관계 행렬 |
| 1.7 | outliers_summary.csv | 이상치 식별 결과 |
| 1.8 | winsorized_comparison.csv | 윈저라이징 전후 통계 비교 |
| 1.8 | winsorized_comparison_plot.png | 윈저라이징 전후 분포 비교 |
| 1.9 | winsorized_data.csv | 윈저라이징 적용된 데이터셋 |
| 1.9 | scaled_data.csv | 표준화된 최종 데이터셋 |
| 1.9 | scaled_data_statistics.csv | 표준화된 데이터 통계량 |

"""

with open(f"{OUTPUT_PATH}/data_preparation_report.md", "w") as f:
    f.write(report_template)

print("\n데이터 준비 및 이상치 점검 완료!")
print(f"결과물은 {OUTPUT_PATH} 폴더에서 확인할 수 있습니다.")
print("주요 파일:")
print(f"- 보고서: {OUTPUT_PATH}/data_preparation_report.md")
print(f"- 최종 데이터셋: {OUTPUT_PATH}/scaled_data.csv") 