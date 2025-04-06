#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
기대수익률과 이탈률 간 상관관계 분석 스크립트
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import logit
import os

# 그래프 설정
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12
sns.set_palette("viridis")

def load_data(filepath):
    """
    CSV 파일에서 데이터 로드
    
    Parameters:
    -----------
    filepath : str
        데이터 파일 경로
        
    Returns:
    --------
    pd.DataFrame
        로드된 데이터 프레임
    """
    print(f"로드 중: {filepath}")
    data = pd.read_csv(filepath)
    print(f"데이터 로드 완료: {data.shape[0]} 행, {data.shape[1]} 열")
    return data

def preprocess_data(data):
    """
    데이터 전처리
    
    Parameters:
    -----------
    data : pd.DataFrame
        원본 데이터
        
    Returns:
    --------
    pd.DataFrame
        전처리된 데이터
    """
    # 중복 제거
    data = data.drop_duplicates(subset=['wallet_address'])
    
    # 이상치 처리 (극단적인 기대수익률, 예: ±1000%)
    q1 = data['expected_roi'].quantile(0.01)
    q3 = data['expected_roi'].quantile(0.99)
    data_filtered = data[(data['expected_roi'] >= q1) & (data['expected_roi'] <= q3)]
    
    print(f"전처리 후 데이터: {data_filtered.shape[0]} 행")
    print(f"제거된 이상치: {data.shape[0] - data_filtered.shape[0]} 행")
    
    return data_filtered

def analyze_correlation(data):
    """
    기대수익률과 이탈 간 상관관계 분석
    
    Parameters:
    -----------
    data : pd.DataFrame
        분석할 데이터
        
    Returns:
    --------
    dict
        상관관계 분석 결과
    """
    # 피어슨 상관계수
    pearson_corr, pearson_p = stats.pearsonr(data['expected_roi'], data['has_exited'])
    
    # 스피어만 순위 상관계수
    spearman_corr, spearman_p = stats.spearmanr(data['expected_roi'], data['has_exited'])
    
    # 포인트 바이시리얼 상관계수
    point_biserial_corr, point_biserial_p = stats.pointbiserialr(data['has_exited'], data['expected_roi'])
    
    # 로지스틱 회귀
    X = sm.add_constant(data['expected_roi'])
    model = sm.Logit(data['has_exited'], X)
    result = model.fit(disp=0)
    
    # 각 기대수익률 구간별 평균 이탈률
    roi_bins = [-np.inf, -0.5, -0.2, 0, 0.2, 0.5, 1.0, np.inf]
    roi_labels = ['Very Negative (<-50%)', 'Negative (-50% to -20%)', 
                  'Slightly Negative (-20% to 0%)', 'Slightly Positive (0% to 20%)', 
                  'Positive (20% to 50%)', 'Very Positive (50% to 100%)', 
                  'Extremely Positive (>100%)']
    
    data['roi_category'] = pd.cut(data['expected_roi'], bins=roi_bins, labels=roi_labels)
    exit_rates_by_category = data.groupby('roi_category')['has_exited'].mean()
    
    # 결과 정리
    results = {
        'pearson_correlation': pearson_corr,
        'pearson_p_value': pearson_p,
        'spearman_correlation': spearman_corr,
        'spearman_p_value': spearman_p,
        'point_biserial_correlation': point_biserial_corr,
        'point_biserial_p_value': point_biserial_p,
        'logit_result': result,
        'exit_rates_by_category': exit_rates_by_category
    }
    
    # 결과 출력
    print("\n===== 상관관계 분석 결과 =====")
    print(f"피어슨 상관계수: {pearson_corr:.4f} (p-value: {pearson_p:.4f})")
    print(f"스피어만 순위 상관계수: {spearman_corr:.4f} (p-value: {spearman_p:.4f})")
    print(f"포인트 바이시리얼 상관계수: {point_biserial_corr:.4f} (p-value: {point_biserial_p:.4f})")
    print("\n로지스틱 회귀 결과:")
    print(result.summary().tables[1])
    
    print("\n각 기대수익률 구간별 평균 이탈률:")
    for category, exit_rate in exit_rates_by_category.items():
        print(f"{category}: {exit_rate:.4f}")
    
    return results

def plot_correlation(data, output_dir):
    """
    상관관계 시각화
    
    Parameters:
    -----------
    data : pd.DataFrame
        분석할 데이터
    output_dir : str
        결과 저장 디렉토리
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. 기대수익률과 이탈 여부의 산점도
    plt.figure(figsize=(12, 8))
    sns.scatterplot(data=data, x='expected_roi', y='has_exited', alpha=0.5)
    
    # 회귀선
    X = data['expected_roi']
    y = data['has_exited']
    X2 = sm.add_constant(X)
    est = sm.OLS(y, X2)
    est2 = est.fit()
    plt.plot(X, est2.predict(X2), color='red')
    
    plt.title('기대수익률과 이탈 여부의 관계')
    plt.xlabel('기대수익률 (Expected ROI)')
    plt.ylabel('이탈 여부 (0=활성, 1=이탈)')
    plt.axhline(y=0.5, color='green', linestyle='--')
    plt.grid(True)
    plt.savefig(f"{output_dir}/roi_exit_scatter.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. 기대수익률 범주별 이탈률 막대 그래프
    plt.figure(figsize=(14, 8))
    
    roi_bins = [-np.inf, -0.5, -0.2, 0, 0.2, 0.5, 1.0, np.inf]
    roi_labels = ['Very Negative\n(<-50%)', 'Negative\n(-50% to -20%)', 
                 'Slightly Negative\n(-20% to 0%)', 'Slightly Positive\n(0% to 20%)', 
                 'Positive\n(20% to 50%)', 'Very Positive\n(50% to 100%)', 
                 'Extremely Positive\n(>100%)']
    
    data['roi_category'] = pd.cut(data['expected_roi'], bins=roi_bins, labels=roi_labels)
    exit_rates = data.groupby('roi_category')['has_exited'].mean()
    counts = data.groupby('roi_category').size()
    
    fig, ax1 = plt.subplots(figsize=(14, 8))
    
    bars = sns.barplot(x=exit_rates.index, y=exit_rates.values, palette='viridis', ax=ax1)
    ax1.set_ylabel('이탈률 (Exit Rate)', fontsize=14)
    ax1.set_ylim(0, 1)
    
    # 각 막대 위에 이탈률 표시
    for i, bar in enumerate(bars.patches):
        bars.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{exit_rates.values[i]:.2%}",
            ha='center', va='bottom',
            fontsize=12
        )
    
    # 두 번째 y축에 지갑 수 표시
    ax2 = ax1.twinx()
    ax2.plot(range(len(counts)), counts.values, 'r-', marker='o', linewidth=2)
    ax2.set_ylabel('지갑 수 (Wallet Count)', color='red', fontsize=14)
    ax2.tick_params(axis='y', labelcolor='red')
    
    plt.title('기대수익률 범주별 이탈률', fontsize=16)
    plt.grid(False)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/exit_rate_by_roi_category.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. ROI 데실(10분위)별 이탈률
    plt.figure(figsize=(14, 8))
    
    data['roi_decile'] = pd.qcut(data['expected_roi'], 10, labels=False)
    decile_stats = data.groupby('roi_decile').agg({
        'expected_roi': ['mean', 'min', 'max'],
        'has_exited': 'mean',
        'wallet_address': 'count'
    })
    
    decile_stats.columns = ['mean_roi', 'min_roi', 'max_roi', 'exit_rate', 'count']
    
    fig, ax1 = plt.subplots(figsize=(14, 8))
    
    bars = sns.barplot(x=decile_stats.index, y=decile_stats['exit_rate'], palette='viridis', ax=ax1)
    ax1.set_ylabel('이탈률 (Exit Rate)', fontsize=14)
    ax1.set_ylim(0, 1)
    ax1.set_xlabel('기대수익률 10분위 (낮음 → 높음)', fontsize=14)
    
    # 각 막대 위에 이탈률 표시
    for i, bar in enumerate(bars.patches):
        bars.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{decile_stats['exit_rate'].values[i]:.2%}",
            ha='center', va='bottom',
            fontsize=12
        )
    
    # 각 분위 아래에 평균 ROI 표시
    for i, roi in enumerate(decile_stats['mean_roi']):
        ax1.text(
            i,
            -0.05,
            f"{roi:.1%}",
            ha='center', va='top',
            fontsize=10
        )
    
    # 두 번째 y축에 지갑 수 표시
    ax2 = ax1.twinx()
    ax2.plot(decile_stats.index, decile_stats['count'], 'r-', marker='o', linewidth=2)
    ax2.set_ylabel('지갑 수 (Wallet Count)', color='red', fontsize=14)
    ax2.tick_params(axis='y', labelcolor='red')
    
    plt.title('기대수익률 10분위별 이탈률', fontsize=16)
    plt.grid(False)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/exit_rate_by_roi_decile.png", dpi=300, bbox_inches='tight')
    plt.close()

def analyze_exit_probability(data, output_dir):
    """
    로지스틱 회귀를 사용한 이탈 확률 분석
    
    Parameters:
    -----------
    data : pd.DataFrame
        분석할 데이터
    output_dir : str
        결과 저장 디렉토리
    """
    # 기본 로지스틱 회귀 (기대수익률만 사용)
    X = sm.add_constant(data['expected_roi'])
    model = sm.Logit(data['has_exited'], X)
    result = model.fit(disp=0)
    
    print("\n===== 로지스틱 회귀 분석 결과 =====")
    print(result.summary())
    
    # 예측 확률 계산을 위한 ROI 범위
    roi_range = np.linspace(data['expected_roi'].min(), data['expected_roi'].max(), 100)
    X_pred = sm.add_constant(roi_range)
    
    # 예측 확률
    pred_probs = result.predict(X_pred)
    
    # 시각화
    plt.figure(figsize=(12, 8))
    plt.plot(roi_range, pred_probs, 'b-', linewidth=2)
    
    # 실제 데이터 포인트 추가
    plt.scatter(data['expected_roi'], data['has_exited'], alpha=0.3, color='green')
    
    plt.title('기대수익률에 따른 이탈 확률', fontsize=16)
    plt.xlabel('기대수익률 (Expected ROI)', fontsize=14)
    plt.ylabel('이탈 확률 (Exit Probability)', fontsize=14)
    plt.grid(True)
    plt.axhline(y=0.5, color='red', linestyle='--')
    
    # ROI=0 지점 표시
    plt.axvline(x=0, color='gray', linestyle='--')
    
    # ROI=0에서의 이탈 확률 계산
    zero_exit_prob = result.predict(sm.add_constant([0]))[0]
    plt.scatter([0], [zero_exit_prob], color='red', s=100, zorder=5)
    plt.annotate(f'ROI=0에서의 이탈 확률: {zero_exit_prob:.2%}', 
                xy=(0, zero_exit_prob), 
                xytext=(0.1, zero_exit_prob + 0.1),
                arrowprops=dict(facecolor='black', shrink=0.05, width=1.5),
                fontsize=12)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/exit_probability_curve.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 이탈 확률 50%가 되는 기대수익률 임계값 찾기
    def find_threshold(coef, intercept, target_prob=0.5):
        # logit(p) = intercept + coef * roi
        # roi = (logit(p) - intercept) / coef
        # logit(p) = log(p / (1-p))
        logit_p = np.log(target_prob / (1 - target_prob))
        threshold = (logit_p - intercept) / coef
        return threshold
    
    exit_threshold = find_threshold(result.params['expected_roi'], result.params['const'])
    print(f"\n이탈 확률 50%가 되는 기대수익률 임계값: {exit_threshold:.2%}")
    
    return result

def write_report(data, correlation_results, logit_results, output_dir):
    """
    분석 결과 보고서 작성
    
    Parameters:
    -----------
    data : pd.DataFrame
        분석한 데이터
    correlation_results : dict
        상관관계 분석 결과
    logit_results : statsmodels.discrete.discrete_model.BinaryResultsWrapper
        로지스틱 회귀 분석 결과
    output_dir : str
        결과 저장 디렉토리
    """
    report_path = f"{output_dir}/correlation_analysis_report.md"
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# 기대수익률과 이탈률 상관관계 분석 보고서\n\n")
        
        f.write("## 1. 데이터 개요\n\n")
        f.write(f"- 분석 대상 지갑 수: {data.shape[0]:,}개\n")
        f.write(f"- 이탈한 지갑 수: {data['has_exited'].sum():,}개 ({data['has_exited'].mean():.2%})\n")
        f.write(f"- 활성 지갑 수: {(data.shape[0] - data['has_exited'].sum()):,}개 ({1 - data['has_exited'].mean():.2%})\n")
        f.write(f"- 기대수익률 범위: {data['expected_roi'].min():.2%} ~ {data['expected_roi'].max():.2%}\n")
        f.write(f"- 평균 기대수익률: {data['expected_roi'].mean():.2%}\n")
        f.write(f"- 중앙값 기대수익률: {data['expected_roi'].median():.2%}\n\n")
        
        f.write("## 2. 상관관계 분석 결과\n\n")
        f.write("### 기본 상관계수\n\n")
        f.write("| 상관계수 유형 | 값 | p-value | 해석 |\n")
        f.write("|--------------|-----|---------|------|\n")
        f.write(f"| 피어슨 상관계수 | {correlation_results['pearson_correlation']:.4f} | {correlation_results['pearson_p_value']:.4f} | {'통계적으로 유의함' if correlation_results['pearson_p_value'] < 0.05 else '통계적으로 유의하지 않음'} |\n")
        f.write(f"| 스피어만 순위 상관계수 | {correlation_results['spearman_correlation']:.4f} | {correlation_results['spearman_p_value']:.4f} | {'통계적으로 유의함' if correlation_results['spearman_p_value'] < 0.05 else '통계적으로 유의하지 않음'} |\n")
        f.write(f"| 포인트 바이시리얼 상관계수 | {correlation_results['point_biserial_correlation']:.4f} | {correlation_results['point_biserial_p_value']:.4f} | {'통계적으로 유의함' if correlation_results['point_biserial_p_value'] < 0.05 else '통계적으로 유의하지 않음'} |\n\n")
        
        f.write("### 기대수익률 구간별 이탈률\n\n")
        f.write("| 기대수익률 구간 | 이탈률 |\n")
        f.write("|-----------------|-------|\n")
        for category, exit_rate in correlation_results['exit_rates_by_category'].items():
            f.write(f"| {category} | {exit_rate:.2%} |\n")
        f.write("\n")
        
        f.write("## 3. 로지스틱 회귀 분석 결과\n\n")
        f.write("### 모델 계수\n\n")
        f.write("| 변수 | 계수 | 표준오차 | z-값 | p-값 |\n")
        f.write("|------|------|----------|------|------|\n")
        f.write(f"| 상수항 | {logit_results.params['const']:.4f} | {logit_results.bse['const']:.4f} | {logit_results.tvalues['const']:.4f} | {logit_results.pvalues['const']:.4f} |\n")
        f.write(f"| 기대수익률 | {logit_results.params['expected_roi']:.4f} | {logit_results.bse['expected_roi']:.4f} | {logit_results.tvalues['expected_roi']:.4f} | {logit_results.pvalues['expected_roi']:.4f} |\n\n")
        
        # 이탈 확률 50%가 되는 기대수익률 임계값
        def find_threshold(coef, intercept, target_prob=0.5):
            logit_p = np.log(target_prob / (1 - target_prob))
            threshold = (logit_p - intercept) / coef
            return threshold
        
        exit_threshold = find_threshold(logit_results.params['expected_roi'], logit_results.params['const'])
        
        f.write(f"### 이탈 확률 50%가 되는 기대수익률 임계값: {exit_threshold:.2%}\n\n")
        
        f.write("### 모델 성능 지표\n\n")
        f.write(f"- Log-Likelihood: {logit_results.llf:.4f}\n")
        f.write(f"- AIC: {logit_results.aic:.4f}\n")
        f.write(f"- BIC: {logit_results.bic:.4f}\n")
        f.write(f"- 유사 R-squared (McFadden): {logit_results.prsquared:.4f}\n\n")
        
        f.write("## 4. 결론 및 해석\n\n")
        
        if correlation_results['pearson_correlation'] < -0.2:
            f.write("분석 결과, 기대수익률과 이탈 여부 간에 **유의미한 음의 상관관계**가 발견되었습니다. 즉, 기대수익률이 높을수록 이탈 확률이 낮아지는 경향이 있습니다.\n\n")
        elif correlation_results['pearson_correlation'] > 0.2:
            f.write("분석 결과, 기대수익률과 이탈 여부 간에 **유의미한 양의 상관관계**가 발견되었습니다. 이는 일반적인 직관과 달리, 기대수익률이 높은 사용자일수록 이탈 확률이 높아지는 경향을 보입니다.\n\n")
        else:
            f.write("분석 결과, 기대수익률과 이탈 여부 간에 뚜렷한 선형 상관관계가 발견되지 않았습니다. 이는 단순한 선형 관계보다 더 복잡한 패턴이 존재할 가능성을 시사합니다.\n\n")
            
        if logit_results.pvalues['expected_roi'] < 0.05:
            if logit_results.params['expected_roi'] < 0:
                f.write("로지스틱 회귀 분석 결과, 기대수익률은 이탈 여부에 **유의미한 영향**을 미치는 것으로 나타났습니다. 기대수익률이 1%p 증가할 때마다 이탈 확률의 로그 오즈가 약 ")
                f.write(f"{logit_results.params['expected_roi']:.4f} 감소합니다.\n\n")
            else:
                f.write("로지스틱 회귀 분석 결과, 기대수익률은 이탈 여부에 **유의미한 영향**을 미치는 것으로 나타났습니다. 그러나 일반적인 예상과 달리, 기대수익률이 1%p 증가할 때마다 이탈 확률의 로그 오즈가 약 ")
                f.write(f"{logit_results.params['expected_roi']:.4f} 증가하는 것으로 나타났습니다.\n\n")
        else:
            f.write("로지스틱 회귀 분석 결과, 기대수익률은 이탈 여부에 통계적으로 유의미한 영향을 미치지 않는 것으로 나타났습니다.\n\n")
            
        f.write("### 시사점\n\n")
        
        if correlation_results['pearson_correlation'] < -0.1 and logit_results.params['expected_roi'] < 0 and logit_results.pvalues['expected_roi'] < 0.05:
            f.write("이 분석 결과는 가설을 지지합니다: 사용자들이 경험한 기대수익률이 그들의 이탈 결정에 영향을 미치는 것으로 보입니다. 특히, 기대수익률이 낮은 사용자일수록 시장을 이탈할 가능성이 높아집니다.\n\n")
            f.write(f"이탈 확률이 50%가 되는 임계 기대수익률은 약 {exit_threshold:.2%}로 추정됩니다. 즉, 기대수익률이 이 값보다 높으면 사용자가 활성 상태를 유지할 가능성이 더 높고, 이 값보다 낮으면 이탈할 가능성이 더 높습니다.\n\n")
        elif correlation_results['pearson_correlation'] > 0.1 and logit_results.params['expected_roi'] > 0 and logit_results.pvalues['expected_roi'] < 0.05:
            f.write("흥미롭게도, 이 분석 결과는 일반적인 직관과 상반됩니다: 기대수익률이 높은 사용자일수록 오히려 이탈 확률이 높아지는 경향이 있습니다. 이는 다음과 같은 가능성을 시사합니다:\n\n")
            f.write("1. 높은 기대수익률을 경험한 사용자들은 '목표 달성' 후 이탈하는 경향이 있을 수 있습니다.\n")
            f.write("2. 기대수익률 계산 방식이 실제 투자자 경험을 완전히 반영하지 못할 수 있습니다.\n")
            f.write("3. 다른 중요한 요인들(시장 변동성, 사용자 특성 등)이 통제되지 않았을 수 있습니다.\n\n")
        else:
            f.write("이 분석 결과는 기대수익률과 이탈 간의 관계가 단순한 선형 또는 단일 로지스틱 함수로 설명하기 어려운 복잡한 패턴을 가질 수 있음을 시사합니다. 추가적인 분석에서는 다음을 고려해볼 수 있습니다:\n\n")
            f.write("1. 비선형 관계 탐색\n")
            f.write("2. 다양한 사용자 세그먼트별 분석\n")
            f.write("3. 기대수익률 외에 이탈에 영향을 미칠 수 있는 다른 변수들을 포함한 다변량 분석\n\n")
            
        f.write("### 향후 연구 방향\n\n")
        f.write("1. 사용자 세그먼트별 기대수익률과 이탈 간의 관계 분석\n")
        f.write("2. 기대수익률의 변동성과 이탈 간의 관계 탐색\n")
        f.write("3. 시간에 따른 기대수익률 변화와 이탈 패턴 간의 관계 분석\n")
        f.write("4. 사용자 행동 패턴과 기대수익률, 이탈 간의 관계 탐색\n")
    
    print(f"\n분석 보고서가 작성되었습니다: {report_path}")

def main():
    """
    메인 함수
    """
    # 출력 디렉토리 설정
    output_dir = "meme_expected_roi_churn/results"
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. 샘플 데이터 생성 (실제 분석에서는 SQL 결과 데이터 사용)
    # 여기서는 임의의 데이터를 생성하여 예시로 사용
    np.random.seed(42)
    sample_size = 5000
    
    # 기대수익률 분포 (약간 왜곡된 정규분포)
    expected_roi = np.random.normal(0.1, 0.5, sample_size)
    
    # 이탈 여부 시뮬레이션 (기대수익률이 낮을수록 이탈 확률 높음)
    # 로지스틱 함수 사용: P(exit) = 1 / (1 + exp(2 * expected_roi))
    exit_probs = 1 / (1 + np.exp(2 * expected_roi))
    has_exited = np.random.binomial(1, exit_probs)
    
    # 데이터프레임 생성
    data = pd.DataFrame({
        'wallet_address': [f'wallet_{i}' for i in range(sample_size)],
        'expected_roi': expected_roi,
        'has_exited': has_exited,
        'exit_status': ['EXITED' if x == 1 else 'ACTIVE' for x in has_exited],
        'total_tokens': np.random.randint(1, 30, sample_size),
        'total_holding': np.random.randint(0, 15, sample_size),
        'roi_std_dev': np.random.uniform(0.1, 0.8, sample_size),
        'prob_positive_roi': np.random.uniform(0.3, 0.7, sample_size),
        'prob_negative_roi': np.random.uniform(0.3, 0.7, sample_size),
    })
    
    # 확률 합이 1이 되도록 조정
    data['prob_negative_roi'] = 1 - data['prob_positive_roi']
    
    # 2. 데이터 전처리
    data_clean = preprocess_data(data)
    
    # 3. 상관관계 분석
    correlation_results = analyze_correlation(data_clean)
    
    # 4. 상관관계 시각화
    plot_correlation(data_clean, output_dir)
    
    # 5. 이탈 확률 분석
    logit_results = analyze_exit_probability(data_clean, output_dir)
    
    # 6. 결과 보고서 작성
    write_report(data_clean, correlation_results, logit_results, output_dir)
    
    print("\n분석이 완료되었습니다.")

if __name__ == "__main__":
    main() 