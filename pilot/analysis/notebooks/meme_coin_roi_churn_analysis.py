#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
밈코인 기대수익률과 시장 이탈의 관계 분석
분석 가설: "기대 수익률이 낮은 사람일수록 밈코인 시장에서 더 많이 이탈했을 것이다"
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.formula.api import ols
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import StandardScaler

# 시각화 설정
sns.set(style="whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12

def load_data():
    """실제 쿼리 결과 데이터를 로드합니다."""
    file_path = "/Users/johnny/Desktop/flipside_experiment/expected_roi_churn_final/meme_coin_roi_churn_1.csv"
    df = pd.read_csv(file_path)
    
    # 이탈 여부 컬럼 추가 (ACTIVE=0, CHURNED=1)
    df['IS_CHURNED'] = np.where(df['WALLET_STATUS'] == 'CHURNED', 1, 0)
    
    # ROI 카테고리 생성
    df['ROI_CATEGORY'] = pd.cut(
        df['EXPECTED_ROI'], 
        bins=[-np.inf, -0.5, -0.1, 0, 0.1, 0.5, 1.0, np.inf], 
        labels=['손실 50%+', '손실 10-50%', '손실 0-10%', '이익 0-10%', '이익 10-50%', '이익 50-100%', '이익 100%+']
    )
    
    return df

def analyze_roi_churn_correlation(df):
    """기대수익률과 이탈 간의 상관관계 분석"""
    print("=============================================")
    print("1. 기대수익률과 이탈 간의 상관관계 분석")
    print("=============================================")
    
    # NaN 값 제거
    df_clean = df.dropna(subset=['EXPECTED_ROI', 'AVERAGE_ROI', 'IS_CHURNED'])
    
    # 포인트 바이시리얼 상관계수 계산 (이진 변수 vs 연속 변수)
    correlation = stats.pointbiserialr(df_clean['IS_CHURNED'], df_clean['EXPECTED_ROI'])
    correlation_avg = stats.pointbiserialr(df_clean['IS_CHURNED'], df_clean['AVERAGE_ROI'])
    
    print(f"기대수익률과 이탈 여부의 포인트 바이시리얼 상관계수: {correlation[0]:.4f}, p-value: {correlation[1]:.4f}")
    print(f"평균 ROI와 이탈 여부의 포인트 바이시리얼 상관계수: {correlation_avg[0]:.4f}, p-value: {correlation_avg[1]:.4f}")
    
    # 상관관계 시각화
    plt.figure(figsize=(16, 6))
    
    plt.subplot(1, 2, 1)
    sns.boxplot(x='WALLET_STATUS', y='EXPECTED_ROI', data=df_clean)
    plt.title('Wallet Status vs Expected ROI')
    plt.xlabel('Wallet Status')
    plt.ylabel('Expected ROI')
    
    plt.subplot(1, 2, 2)
    sns.boxplot(x='WALLET_STATUS', y='AVERAGE_ROI', data=df_clean)
    plt.title('Wallet Status vs Average ROI')
    plt.xlabel('Wallet Status')
    plt.ylabel('Average ROI')
    
    plt.tight_layout()
    plt.savefig('roi_churn_correlation.png')
    print("상관관계 분석 그래프가 roi_churn_correlation.png로 저장되었습니다.")
    
    # t-test로 두 그룹 간 ROI 차이 검정
    active_roi = df_clean[df_clean['WALLET_STATUS'] == 'ACTIVE']['EXPECTED_ROI']
    churned_roi = df_clean[df_clean['WALLET_STATUS'] == 'CHURNED']['EXPECTED_ROI']
    t_stat, p_value = stats.ttest_ind(active_roi, churned_roi, equal_var=False)
    
    print(f"\nt-test 결과: t-statistic = {t_stat:.4f}, p-value = {p_value:.4f}")
    if p_value < 0.05:
        print("결론: 활성 사용자와 이탈 사용자의 기대수익률에는 통계적으로 유의미한 차이가 있습니다.")
    else:
        print("결론: 활성 사용자와 이탈 사용자의 기대수익률 차이는 통계적으로 유의미하지 않습니다.")
    print()

def analyze_roi_category_churn_rates(df):
    """ROI 카테고리별 이탈률 분석"""
    print("=============================================")
    print("2. ROI 카테고리별 이탈률 분석")
    print("=============================================")
    
    # NaN 값 제거
    df_clean = df.dropna(subset=['ROI_CATEGORY', 'IS_CHURNED'])
    
    # ROI 카테고리별 이탈률 계산
    roi_churn_rates = df_clean.groupby('ROI_CATEGORY').agg({
        'SWAPPER': 'count',
        'IS_CHURNED': 'mean',
        'EXPECTED_ROI': 'mean'
    }).reset_index()
    
    roi_churn_rates.columns = ['ROI Category', 'Wallet Count', 'Churn Rate', 'Avg Expected ROI']
    print(roi_churn_rates)
    
    # ROI 카테고리별 이탈률 시각화
    plt.figure(figsize=(14, 6))
    
    sns.barplot(x='ROI Category', y='Churn Rate', data=roi_churn_rates)
    plt.title('ROI Category vs Churn Rate')
    plt.ylim(0, 1)
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig('roi_category_churn_rates.png')
    print("ROI 카테고리별 이탈률 그래프가 roi_category_churn_rates.png로 저장되었습니다.")
    
    # 카이제곱 검정 - ROI 카테고리와 이탈 여부의 관련성
    contingency_table = pd.crosstab(df_clean['ROI_CATEGORY'], df_clean['WALLET_STATUS'])
    chi2, p, dof, expected = stats.chi2_contingency(contingency_table)
    
    print(f"\n카이제곱 검정 결과: chi2 = {chi2:.4f}, p-value = {p:.4f}, 자유도 = {dof}")
    if p < 0.05:
        print("결론: ROI 카테고리와 이탈 여부 사이에 통계적으로 유의미한 관련성이 있습니다.")
    else:
        print("결론: ROI 카테고리와 이탈 여부 사이에 통계적으로 유의미한 관련성이 없습니다.")
    print()

def analyze_tokens_traded_effect(df):
    """거래한 토큰 수가 이탈에 미치는 영향 분석"""
    print("=============================================")
    print("3. 거래한 토큰 수가 이탈에 미치는 영향 분석")
    print("=============================================")
    
    # NaN 값 제거
    df_clean = df.dropna(subset=['UNIQUE_TOKENS_TRADED', 'IS_CHURNED', 'EXPECTED_ROI'])
    
    # 거래한 토큰 수에 대한 그룹 생성
    df_clean['TOKEN_COUNT_GROUP'] = pd.cut(
        df_clean['UNIQUE_TOKENS_TRADED'], 
        bins=[0, 5, 10, 20, 50, 100, np.inf], 
        labels=['1-5', '6-10', '11-20', '21-50', '51-100', '100+']
    )
    
    # 토큰 수 그룹별 이탈률 계산
    token_count_churn = df_clean.groupby('TOKEN_COUNT_GROUP').agg({
        'SWAPPER': 'count',
        'IS_CHURNED': 'mean',
        'EXPECTED_ROI': 'mean'
    }).reset_index()
    
    token_count_churn.columns = ['Token Count', 'Wallet Count', 'Churn Rate', 'Avg Expected ROI']
    print(token_count_churn)
    
    # 토큰 수와 이탈률/ROI 시각화
    plt.figure(figsize=(16, 6))
    
    plt.subplot(1, 2, 1)
    sns.barplot(x='Token Count', y='Churn Rate', data=token_count_churn)
    plt.title('Token Count vs Churn Rate')
    plt.ylim(0, 1)
    
    plt.subplot(1, 2, 2)
    sns.barplot(x='Token Count', y='Avg Expected ROI', data=token_count_churn)
    plt.title('Token Count vs Avg Expected ROI')
    
    plt.tight_layout()
    plt.savefig('token_count_effect.png')
    print("거래 토큰 수 영향 분석 그래프가 token_count_effect.png로 저장되었습니다.")
    
    # 토큰 수와 기대수익률의 상관관계
    corr_token_roi = stats.pearsonr(df_clean['UNIQUE_TOKENS_TRADED'], df_clean['EXPECTED_ROI'])
    print(f"\n거래 토큰 수와 기대수익률의 상관계수: {corr_token_roi[0]:.4f}, p-value: {corr_token_roi[1]:.4f}")
    
    # 로지스틱 회귀 분석: 토큰 수와 ROI가 이탈에 미치는 영향
    X = df_clean[['EXPECTED_ROI', 'UNIQUE_TOKENS_TRADED']]
    X = sm.add_constant(X)
    y = df_clean['IS_CHURNED']
    
    model = sm.Logit(y, X)
    result = model.fit(disp=0)
    print("\n로지스틱 회귀 분석 결과:")
    print(result.summary().tables[1])
    print()

def predict_churn_with_roi(df):
    """기대수익률을 활용한 이탈 예측 모델"""
    print("=============================================")
    print("4. 기대수익률을 활용한 이탈 예측 모델")
    print("=============================================")
    
    # NaN 값 제거
    df_clean = df.dropna(subset=['EXPECTED_ROI', 'UNIQUE_TOKENS_TRADED', 'AVERAGE_ROI', 'IS_CHURNED'])
    
    # 데이터 준비
    X = df_clean[['EXPECTED_ROI', 'UNIQUE_TOKENS_TRADED', 'AVERAGE_ROI']]
    y = df_clean['IS_CHURNED']
    
    # 데이터 분할
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # 모델 학습
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # 성능 평가
    y_pred = model.predict(X_test)
    print("분류 보고서:")
    print(classification_report(y_test, y_pred))
    
    # 특성 중요도
    importance = model.feature_importances_
    feature_names = X.columns
    
    plt.figure(figsize=(10, 6))
    plt.barh(feature_names, importance)
    plt.xlabel('Feature Importance')
    plt.title('Feature Importance in Churn Prediction Model')
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    print("특성 중요도 그래프가 feature_importance.png로 저장되었습니다.")
    
    # ROC 곡선
    y_prob = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(10, 6))
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc='lower right')
    plt.savefig('roc_curve.png')
    print(f"ROC 곡선이 roc_curve.png로 저장되었습니다. AUC: {roc_auc:.3f}")
    print()

def create_summary_visualization(df):
    """종합 시각화 - 기대수익률과 이탈 관계 요약"""
    print("=============================================")
    print("5. 기대수익률과 이탈 관계 종합 시각화")
    print("=============================================")
    
    # NaN 값 제거
    df_clean = df.dropna(subset=['EXPECTED_ROI', 'UNIQUE_TOKENS_TRADED', 'AVERAGE_ROI', 'IS_CHURNED'])
    
    plt.figure(figsize=(16, 12))
    
    # 1. 기대수익률 구간별 이탈률
    plt.subplot(2, 2, 1)
    roi_bins = [-np.inf, -0.5, -0.2, -0.1, 0, 0.1, 0.2, 0.5, 1.0, np.inf]
    roi_labels = ['-inf~-50%', '-50~-20%', '-20~-10%', '-10~0%', '0~10%', '10~20%', '20~50%', '50~100%', '100%+']
    df_clean['ROI_BIN'] = pd.cut(df_clean['EXPECTED_ROI'], bins=roi_bins, labels=roi_labels)
    
    roi_bin_churn = df_clean.groupby('ROI_BIN').agg({
        'SWAPPER': 'count',
        'IS_CHURNED': 'mean'
    }).reset_index()
    
    sns.barplot(x='ROI_BIN', y='IS_CHURNED', data=roi_bin_churn)
    plt.title('Expected ROI Bins vs Churn Rate')
    plt.xlabel('Expected ROI Bins')
    plt.ylabel('Churn Rate')
    plt.ylim(0, 1)
    plt.xticks(rotation=45)
    
    # 2. 토큰 수와 이탈률의 관계 (ROI로 색상 표시)
    plt.subplot(2, 2, 2)
    token_bins = [0, 10, 20, 50, 100, 200, 500, np.inf]
    token_labels = ['1-10', '11-20', '21-50', '51-100', '101-200', '201-500', '500+']
    df_clean['TOKEN_BIN'] = pd.cut(df_clean['UNIQUE_TOKENS_TRADED'], bins=token_bins, labels=token_labels)
    
    roi_token_churn = df_clean.groupby(['TOKEN_BIN', 'ROI_CATEGORY']).agg({
        'SWAPPER': 'count',
        'IS_CHURNED': 'mean'
    }).reset_index()
    
    pivot_table = roi_token_churn.pivot(index='TOKEN_BIN', columns='ROI_CATEGORY', values='IS_CHURNED')
    sns.heatmap(pivot_table, annot=True, cmap='coolwarm', vmin=0, vmax=1, fmt='.2f')
    plt.title('Token Count and ROI Category vs Churn Rate')
    plt.tight_layout()
    
    # 3. 기대수익률 vs 평균 ROI와 이탈 관계
    plt.subplot(2, 2, 3)
    sns.scatterplot(x='EXPECTED_ROI', y='AVERAGE_ROI', hue='WALLET_STATUS', data=df_clean, alpha=0.5)
    plt.title('Expected ROI vs Average ROI by Wallet Status')
    plt.xlabel('Expected ROI')
    plt.ylabel('Average ROI')
    
    # 4. 이탈률 예측 모델의 정확도
    plt.subplot(2, 2, 4)
    
    # 간단한 로지스틱 회귀 모델
    X = df_clean[['EXPECTED_ROI', 'UNIQUE_TOKENS_TRADED']]
    X = StandardScaler().fit_transform(X)
    y = df_clean['IS_CHURNED']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    model_names = ['RF', 'Logit']
    test_accuracies = []
    
    # 랜덤 포레스트
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    test_accuracies.append(rf.score(X_test, y_test))
    
    # 로지스틱 회귀
    from sklearn.linear_model import LogisticRegression
    logit = LogisticRegression(random_state=42)
    logit.fit(X_train, y_train)
    test_accuracies.append(logit.score(X_test, y_test))
    
    # 막대 그래프로 모델 정확도 표시
    sns.barplot(x=model_names, y=test_accuracies)
    plt.title('Churn Prediction Model Accuracy')
    plt.xlabel('Model')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig('summary_visualization.png')
    print("종합 시각화 그래프가 summary_visualization.png로 저장되었습니다.")
    print()

def main():
    # 1. 데이터 로드
    print("데이터 로드 중...")
    df = load_data()
    print(f"로드된 데이터: {df.shape[0]} 행, {df.shape[1]} 열")
    print("\n데이터 샘플:")
    print(df.head())
    print("\n데이터 기본 통계:")
    print(df.describe())
    print("\n활성/이탈 지갑 비율:")
    print(df['WALLET_STATUS'].value_counts(normalize=True))
    
    # 2. 분석 실행
    analyze_roi_churn_correlation(df)
    analyze_roi_category_churn_rates(df)
    analyze_tokens_traded_effect(df)
    predict_churn_with_roi(df)
    create_summary_visualization(df)
    
    # 3. 결론
    print("=============================================")
    print("결론: 기대수익률과 이탈의 관계")
    print("=============================================")
    
    # 여기서는 실제 분석 결과를 바탕으로 결론을 작성할 예정입니다.
    # 실제 데이터 분석 후 수정 예정

if __name__ == "__main__":
    main() 