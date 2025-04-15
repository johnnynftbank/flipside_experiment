#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
주성분 분석(PCA) 실행 모듈
- 표준화된 데이터에 PCA 적용
- 주성분 추출 및 분석
- 결과 시각화 및 해석
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
import os
from pathlib import Path

# 한글 폰트 설정
plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False

# 경로 설정
BASE_DIR = Path("pca_kmeans")
DATA_DIR = BASE_DIR / "analysis" / "report"
RESULT_DIR = DATA_DIR
SOURCE_DIR = BASE_DIR / "analysis" / "source_code"

# 결과 디렉토리가 없으면 생성
if not RESULT_DIR.exists():
    RESULT_DIR.mkdir(parents=True)

# 데이터 로드
def load_data():
    """표준화된 데이터 로드"""
    scaled_data = pd.read_csv(DATA_DIR / "scaled_data.csv")
    print(f"데이터 로드 완료: {scaled_data.shape[0]} 행, {scaled_data.shape[1]} 열")
    return scaled_data

# 주성분 분석 실행
def run_pca(data, n_components=5):
    """PCA 실행"""
    # 분석할 변수 선택 (SWAPPER, WALLET_STATUS 제외)
    features = ['EXPECTED_ROI', 'ROI_STANDARD_DEVIATION', 'SHARPE_RATIO', 
                'WIN_LOSS_RATIO', 'MAX_TRADE_PROPORTION']
    X = data[features]
    
    # PCA 모델 생성
    pca = PCA(n_components=n_components)
    
    # 데이터 적합
    pca.fit(X)
    
    # 주요 결과 저장
    results = {
        'eigenvalues': pca.explained_variance_,
        'explained_variance_ratio': pca.explained_variance_ratio_,
        'cumulative_variance_ratio': np.cumsum(pca.explained_variance_ratio_),
        'components': pca.components_,
        'features': features
    }
    
    return pca, results

# 스크리 플롯 생성
def create_scree_plot(results):
    """스크리 플롯 생성 (고유값 또는 설명된 분산비율)"""
    eigenvalues = results['eigenvalues']
    exp_var_ratio = results['explained_variance_ratio']
    cum_var_ratio = results['cumulative_variance_ratio']
    
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    # 막대 그래프: 개별 설명된 분산비율
    ax1.bar(range(1, len(exp_var_ratio) + 1), exp_var_ratio, 
            alpha=0.8, color='skyblue', label='개별 설명 분산비율')
    ax1.set_xlabel('주성분 번호')
    ax1.set_ylabel('설명된 분산비율')
    ax1.set_xticks(range(1, len(exp_var_ratio) + 1))
    
    # 선 그래프: 누적 설명된 분산비율
    ax2 = ax1.twinx()
    ax2.plot(range(1, len(cum_var_ratio) + 1), cum_var_ratio, 
             marker='o', color='red', linestyle='-', linewidth=2, 
             markersize=8, label='누적 설명 분산비율')
    ax2.set_ylabel('누적 설명 분산비율')
    ax2.axhline(y=0.7, color='gray', linestyle='--', alpha=0.7, label='70% 기준선')
    ax2.axhline(y=0.8, color='gray', linestyle=':', alpha=0.7, label='80% 기준선')
    
    # 타이틀 설정
    plt.title('PCA 스크리 플롯: 설명된 분산 비율 및 누적 분산 비율')
    
    # 범례 조합
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    # 그리드 추가
    ax1.grid(axis='y', linestyle='--', alpha=0.5)
    
    # 파일 저장
    plt.tight_layout()
    plt.savefig(RESULT_DIR / "scree_plot.png", dpi=300)
    plt.close()
    
    print("스크리 플롯 생성 완료")

# 적재량 테이블 저장
def save_loadings_table(results):
    """주성분 적재량 테이블 저장"""
    components = results['components']
    features = results['features']
    
    # 주성분 적재량 테이블 생성
    loadings_df = pd.DataFrame(components.T, 
                               index=features,
                               columns=[f'PC{i+1}' for i in range(components.shape[0])])
    
    # 적재량 저장
    loadings_df.to_csv(RESULT_DIR / "pca_loadings.csv")
    
    print("주성분 적재량 테이블 저장 완료")
    
    return loadings_df

# 적재량 히트맵 생성
def create_loadings_heatmap(loadings_df):
    """주성분 적재량 히트맵"""
    plt.figure(figsize=(10, 8))
    
    # 히트맵 생성
    heatmap = sns.heatmap(loadings_df, annot=True, cmap='coolwarm', fmt='.2f',
                          vmin=-1, vmax=1, center=0, linewidths=.5)
    
    # 타이틀 설정
    plt.title('PCA 성분 적재량 (Component Loadings)')
    
    # 저장
    plt.tight_layout()
    plt.savefig(RESULT_DIR / "pca_loading_heatmap.png", dpi=300)
    plt.close()
    
    print("주성분 적재량 히트맵 생성 완료")

# PCA 결과 테이블 저장
def save_pca_results(results):
    """PCA 결과 테이블 저장"""
    # 고유값, 설명된 분산 비율, 누적 분산 비율 테이블 생성
    summary_df = pd.DataFrame({
        'eigenvalue': results['eigenvalues'],
        'explained_variance_ratio': results['explained_variance_ratio'],
        'cumulative_variance_ratio': results['cumulative_variance_ratio']
    }, index=[f'PC{i+1}' for i in range(len(results['eigenvalues']))])
    
    # 결과 저장
    summary_df.to_csv(RESULT_DIR / "pca_model_results.csv")
    
    print("PCA 결과 테이블 저장 완료")

# 주성분 변환 데이터 생성
def transform_data(pca, data, n_components=None):
    """데이터를 주성분 공간으로 변환"""
    if n_components:
        pca.n_components = n_components
    
    # 분석할 변수 선택 (SWAPPER, WALLET_STATUS 제외)
    features = ['EXPECTED_ROI', 'ROI_STANDARD_DEVIATION', 'SHARPE_RATIO', 
                'WIN_LOSS_RATIO', 'MAX_TRADE_PROPORTION']
    X = data[features]
    
    # PCA 변환
    transformed_data = pca.transform(X)
    
    # DataFrame 생성
    pc_cols = [f'PC{i+1}' for i in range(transformed_data.shape[1])]
    transformed_df = pd.DataFrame(transformed_data, columns=pc_cols)
    
    # 원래 데이터에서 지갑 주소와 지갑 상태 추가
    transformed_df['SWAPPER'] = data['SWAPPER'].values
    transformed_df['WALLET_STATUS'] = data['WALLET_STATUS'].values
    
    # 결과 저장
    transformed_df.to_csv(RESULT_DIR / "transformed_pca_data.csv", index=False)
    
    print(f"데이터 변환 완료: {transformed_df.shape[0]} 행, {transformed_df.shape[1]} 열")
    
    return transformed_df

# 바이플롯 생성
def create_biplot(pca, data, results, components=(1, 2)):
    """PCA 바이플롯 생성"""
    features = results['features']
    
    # 특성 데이터 선택
    X = data[features]
    
    # PCA 변환
    X_pca = pca.transform(X)
    
    # 2D 플롯 준비
    pc1_idx, pc2_idx = components[0] - 1, components[1] - 1
    
    # 플롯 생성
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # 데이터 포인트 그리기 (상태에 따라 다른 색상)
    status_colors = {'active': 'blue', 'churned': 'red'}
    
    for status, color in status_colors.items():
        mask = data['WALLET_STATUS'] == status
        ax.scatter(X_pca[mask, pc1_idx], X_pca[mask, pc2_idx], 
                   color=color, alpha=0.5, label=status)
    
    # 주성분 벡터 그리기
    loadings = pca.components_
    for i, feature in enumerate(features):
        ax.arrow(0, 0, loadings[pc1_idx, i] * 5, loadings[pc2_idx, i] * 5,
                 head_width=0.2, head_length=0.2, fc='green', ec='green')
        ax.text(loadings[pc1_idx, i] * 5.2, loadings[pc2_idx, i] * 5.2, feature, 
                color='green', ha='center', va='center', fontsize=12)
    
    # 축 레이블
    var_ratio1 = pca.explained_variance_ratio_[pc1_idx] * 100
    var_ratio2 = pca.explained_variance_ratio_[pc2_idx] * 100
    ax.set_xlabel(f'PC{components[0]} ({var_ratio1:.1f}%)', fontsize=14)
    ax.set_ylabel(f'PC{components[1]} ({var_ratio2:.1f}%)', fontsize=14)
    
    # 축 눈금 0에 선 추가
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.3)
    ax.axvline(x=0, color='gray', linestyle='--', alpha=0.3)
    
    # 타이틀 설정
    ax.set_title(f'PCA 바이플롯: PC{components[0]} vs PC{components[1]}', fontsize=16)
    
    # 범례 추가
    ax.legend(title="지갑 상태")
    
    # 저장
    plt.tight_layout()
    plt.savefig(RESULT_DIR / f"pca_biplot_pc{components[0]}_pc{components[1]}.png", dpi=300)
    plt.close()
    
    print(f"바이플롯 생성 완료: PC{components[0]} vs PC{components[1]}")

# 메인 함수
def main():
    """PCA 분석 실행"""
    # 데이터 로드
    data = load_data()
    
    # PCA 실행
    pca, results = run_pca(data)
    
    # 결과 분석 및 시각화
    save_pca_results(results)
    create_scree_plot(results)
    loadings_df = save_loadings_table(results)
    create_loadings_heatmap(loadings_df)
    
    # 데이터 변환
    transformed_df = transform_data(pca, data)
    
    # 바이플롯 생성
    create_biplot(pca, data, results, components=(1, 2))
    create_biplot(pca, data, results, components=(1, 3))
    create_biplot(pca, data, results, components=(2, 3))
    
    print("PCA 분석 완료")

# 실행
if __name__ == "__main__":
    main() 