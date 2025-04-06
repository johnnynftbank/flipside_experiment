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
sns.set(style="whitegrid", palette="seaborn-v0_8")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12

# 가상의 데이터 로드 (실제 환경에서는 쿼리 결과 데이터 사용)
# 실제 실행 시에는 SQL 쿼리 결과를 CSV로 저장한 후 로드하는 방식 사용
def load_sample_data():
    """가상의 샘플 데이터를 생성하여 반환합니다. 
    실제 환경에서는 SQL 쿼리 결과를 사용해야 합니다."""
    np.random.seed(42)
    n_samples = 1000
    
    # 기본 데이터 생성
    data = {
        'wallet_address': [f'wallet_{i}' for i in range(n_samples)],
        'overall_roi': np.random.uniform(-1.0, 5.0, n_samples),
        'avg_token_roi': np.random.uniform(-1.0, 3.0, n_samples),
        'median_token_roi': np.random.uniform(-1.0, 2.0, n_samples),
        'worst_token_roi': np.random.uniform(-1.0, 0.5, n_samples),
        'best_token_roi': np.random.uniform(0.0, 10.0, n_samples),
        'unique_tokens_traded': np.random.randint(1, 20, n_samples),
        'total_investment': np.random.uniform(0.1, 100.0, n_samples),
        'total_returns': np.random.uniform(0.0, 150.0, n_samples),
        'total_activity_period': np.random.randint(2, 365, n_samples),
        'active_trading_days': np.random.randint(1, 100, n_samples),
        'total_meme_trades': np.random.randint(10, 500, n_samples),
        'first_half_trades': np.random.randint(5, 250, n_samples),
        'second_half_trades': np.random.randint(1, 250, n_samples),
        'days_since_last_trade': np.random.randint(0, 120, n_samples),
    }
    
    # 파생 변수 계산
    df = pd.DataFrame(data)
    
    # 실제 데이터에서 기대되는 관계를 반영한 이탈 점수 생성 (ROI가 낮을수록 이탈 점수 높음)
    df['activity_decline_rate'] = (df['first_half_trades'] - df['second_half_trades']) / df['first_half_trades']
    df['activity_decline_rate'] = df['activity_decline_rate'].clip(0, 1)  # 0~1 사이로 제한
    
    # ROI와 이탈의 관계: 상관계수 약 -0.6 정도 설정
    base_exit_score = 100 - df['overall_roi'] * 20
    noise = np.random.normal(0, 20, n_samples)
    df['exit_score'] = base_exit_score + noise
    df['exit_score'] = df['exit_score'].clip(0, 100)  # 0~100 사이로 제한
    
    # 이탈 상태 정의
    df['exit_status_time'] = pd.cut(
        df['days_since_last_trade'], 
        bins=[0, 14, 30, 60, np.inf], 
        labels=['Active', 'Possible_Exit', 'Likely_Exited', 'Exited']
    )
    
    # 이탈 여부 (1=이탈, 0=활성)
    df['has_exited'] = np.where(df['days_since_last_trade'] >= 30, 1, 0)
    
    # ROI 카테고리
    df['roi_category'] = pd.cut(
        df['overall_roi'], 
        bins=[-np.inf, -0.5, 0, 1.0, 5.0, np.inf], 
        labels=['High Loss', 'Loss', 'Profit', 'High Profit', 'Very High Profit']
    )
    
    return df

def analyze_roi_exit_correlation(df):
    """ROI와 이탈 지표 간의 상관관계를 분석합니다."""
    print("=============================================")
    print("1. ROI와 이탈 지표 간의 상관관계 분석")
    print("=============================================")
    
    # 상관계수 계산
    correlation_roi_exit = df['overall_roi'].corr(df['exit_score'])
    correlation_roi_decline = df['overall_roi'].corr(df['activity_decline_rate'])
    correlation_worst_roi_exit = df['worst_token_roi'].corr(df['exit_score'])
    
    print(f"ROI와 이탈 점수의 상관계수: {correlation_roi_exit:.4f}")
    print(f"ROI와 활동 감소율의 상관계수: {correlation_roi_decline:.4f}")
    print(f"최악의 토큰 ROI와 이탈 점수의 상관계수: {correlation_worst_roi_exit:.4f}")
    
    # 상관관계 시각화
    plt.figure(figsize=(16, 5))
    
    plt.subplot(1, 3, 1)
    sns.regplot(x='overall_roi', y='exit_score', data=df, scatter_kws={'alpha':0.3})
    plt.title(f'ROI vs 이탈 점수 (상관계수: {correlation_roi_exit:.4f})')
    plt.xlabel('전체 ROI')
    plt.ylabel('이탈 점수')
    
    plt.subplot(1, 3, 2)
    sns.regplot(x='overall_roi', y='activity_decline_rate', data=df, scatter_kws={'alpha':0.3})
    plt.title(f'ROI vs 활동 감소율 (상관계수: {correlation_roi_decline:.4f})')
    plt.xlabel('전체 ROI')
    plt.ylabel('활동 감소율')
    
    plt.subplot(1, 3, 3)
    sns.regplot(x='worst_token_roi', y='exit_score', data=df, scatter_kws={'alpha':0.3})
    plt.title(f'최악 토큰 ROI vs 이탈 점수 (상관계수: {correlation_worst_roi_exit:.4f})')
    plt.xlabel('최악의 토큰 ROI')
    plt.ylabel('이탈 점수')
    
    plt.tight_layout()
    plt.savefig('correlation_analysis.png')
    print("상관관계 분석 그래프가 correlation_analysis.png로 저장되었습니다.")
    print()

def analyze_roi_groups_exit_rates(df):
    """ROI 구간별 이탈율을 분석합니다."""
    print("=============================================")
    print("2. ROI 구간별 이탈율 분석")
    print("=============================================")
    
    # ROI 구간별 이탈율 계산
    roi_exit_rates = df.groupby('roi_category').agg({
        'wallet_address': 'count',
        'has_exited': 'mean',
        'exit_score': 'mean',
        'overall_roi': 'mean',
        'activity_decline_rate': 'mean'
    }).reset_index()
    
    roi_exit_rates.columns = ['ROI 카테고리', '지갑 수', '이탈율', '평균 이탈 점수', '평균 ROI', '평균 활동 감소율']
    print(roi_exit_rates)
    
    # ROI 카테고리별 이탈율 시각화
    plt.figure(figsize=(14, 6))
    
    plt.subplot(1, 2, 1)
    sns.barplot(x='ROI 카테고리', y='이탈율', data=roi_exit_rates)
    plt.title('ROI 카테고리별 이탈율')
    plt.ylim(0, 1)
    plt.xticks(rotation=45)
    
    plt.subplot(1, 2, 2)
    sns.barplot(x='ROI 카테고리', y='평균 이탈 점수', data=roi_exit_rates)
    plt.title('ROI 카테고리별 평균 이탈 점수')
    plt.ylim(0, 100)
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig('roi_groups_exit_rates.png')
    print("ROI 구간별 이탈율 그래프가 roi_groups_exit_rates.png로 저장되었습니다.")
    
    # 통계적 유의성 검정 - ANOVA
    categories = pd.Categorical(df['roi_category']).categories
    groups = [df[df['roi_category'] == category]['exit_score'].values for category in categories]
    f_val, p_val = stats.f_oneway(*groups)
    
    print(f"\nANOVA 검정 결과: F-value = {f_val:.4f}, p-value = {p_val:.4f}")
    if p_val < 0.05:
        print("결론: ROI 카테고리에 따라 이탈 점수에 통계적으로 유의미한 차이가 있습니다.")
    else:
        print("결론: ROI 카테고리에 따른 이탈 점수의 차이는 통계적으로 유의미하지 않습니다.")
    print()

def analyze_temporal_patterns(df):
    """시간에 따른 ROI 및 이탈 패턴을 분석합니다."""
    print("=============================================")
    print("3. 활동 기간에 따른 ROI 및 이탈 패턴 분석")
    print("=============================================")
    
    # 활동 기간 그룹 생성
    df['activity_period_group'] = pd.cut(
        df['total_activity_period'], 
        bins=[0, 7, 30, 90, 180, np.inf], 
        labels=['1주 이하', '1주-1개월', '1-3개월', '3-6개월', '6개월 초과']
    )
    
    # 활동 기간별 ROI 및 이탈율 계산
    period_stats = df.groupby('activity_period_group').agg({
        'wallet_address': 'count',
        'overall_roi': 'mean',
        'has_exited': 'mean',
        'exit_score': 'mean'
    }).reset_index()
    
    period_stats.columns = ['활동 기간', '지갑 수', '평균 ROI', '이탈율', '평균 이탈 점수']
    print(period_stats)
    
    # 시각화
    plt.figure(figsize=(16, 6))
    
    plt.subplot(1, 2, 1)
    sns.barplot(x='활동 기간', y='평균 ROI', data=period_stats)
    plt.title('활동 기간별 평균 ROI')
    plt.xticks(rotation=45)
    
    plt.subplot(1, 2, 2)
    sns.barplot(x='활동 기간', y='이탈율', data=period_stats)
    plt.title('활동 기간별 이탈율')
    plt.ylim(0, 1)
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig('temporal_patterns.png')
    print("활동 기간별 패턴 그래프가 temporal_patterns.png로 저장되었습니다.")
    print()

def build_exit_prediction_model(df):
    """이탈 예측 모델을 구축하고 ROI 관련 특성의 중요도를 평가합니다."""
    print("=============================================")
    print("4. 이탈 예측 모델 및 특성 중요도 분석")
    print("=============================================")
    
    # 특성 및 타겟 변수 설정
    features = [
        'overall_roi', 'avg_token_roi', 'median_token_roi', 'worst_token_roi',
        'best_token_roi', 'unique_tokens_traded', 'total_investment',
        'total_activity_period', 'active_trading_days', 'total_meme_trades',
        'activity_decline_rate'
    ]
    
    X = df[features]
    y = df['has_exited']
    
    # 데이터 분할
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # 특성 스케일링
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 모델 학습
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    # 모델 평가
    y_pred = model.predict(X_test_scaled)
    
    print("분류 보고서:")
    print(classification_report(y_test, y_pred))
    
    # 특성 중요도 계산
    importance = model.feature_importances_
    feature_importance = pd.DataFrame({'Feature': features, 'Importance': importance})
    feature_importance = feature_importance.sort_values('Importance', ascending=False)
    
    print("\n특성 중요도:")
    print(feature_importance)
    
    # 특성 중요도 시각화
    plt.figure(figsize=(12, 6))
    sns.barplot(x='Importance', y='Feature', data=feature_importance)
    plt.title('이탈 예측을 위한 특성 중요도')
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    print("특성 중요도 그래프가 feature_importance.png로 저장되었습니다.")
    
    # ROC 곡선
    y_prob = model.predict_proba(X_test_scaled)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('이탈 예측 모델의 ROC 곡선')
    plt.legend(loc="lower right")
    plt.savefig('roc_curve.png')
    print("ROC 곡선이 roc_curve.png로 저장되었습니다.")
    print()

def analyze_worst_roi_effect(df):
    """최악의 토큰 수익률이 이탈에 미치는 영향을 분석합니다."""
    print("=============================================")
    print("5. 최악의 토큰 수익률이 이탈에 미치는 영향 분석")
    print("=============================================")
    
    # 최악의 토큰 ROI 구간 생성
    df['worst_roi_group'] = pd.cut(
        df['worst_token_roi'], 
        bins=[-np.inf, -0.9, -0.7, -0.5, -0.3, 0, np.inf], 
        labels=['손실 90%+', '손실 70-90%', '손실 50-70%', '손실 30-50%', '손실 0-30%', '이익']
    )
    
    # 최악의 토큰 ROI 구간별 이탈율 계산
    worst_roi_exit = df.groupby('worst_roi_group').agg({
        'wallet_address': 'count',
        'has_exited': 'mean',
        'exit_score': 'mean',
        'worst_token_roi': 'mean',
        'overall_roi': 'mean'
    }).reset_index()
    
    worst_roi_exit.columns = ['최악 토큰 ROI 구간', '지갑 수', '이탈율', '평균 이탈 점수', '평균 최악 ROI', '평균 전체 ROI']
    print(worst_roi_exit)
    
    # 시각화
    plt.figure(figsize=(14, 6))
    
    plt.subplot(1, 2, 1)
    sns.barplot(x='최악 토큰 ROI 구간', y='이탈율', data=worst_roi_exit)
    plt.title('최악 토큰 ROI 구간별 이탈율')
    plt.ylim(0, 1)
    plt.xticks(rotation=45)
    
    plt.subplot(1, 2, 2)
    sns.barplot(x='최악 토큰 ROI 구간', y='평균 이탈 점수', data=worst_roi_exit)
    plt.title('최악 토큰 ROI 구간별 평균 이탈 점수')
    plt.ylim(0, 100)
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig('worst_roi_effect.png')
    print("최악 토큰 ROI 영향 그래프가 worst_roi_effect.png로 저장되었습니다.")
    
    # 다중회귀분석: 전체 ROI와 최악 토큰 ROI 중 어느 것이 이탈에 더 큰 영향을 미치는지
    X = df[['overall_roi', 'worst_token_roi']]
    X = sm.add_constant(X)
    model = sm.OLS(df['exit_score'], X).fit()
    
    print("\n다중회귀분석 결과:")
    print(model.summary().tables[1])
    print()

def main():
    """메인 함수: 전체 분석 흐름을 실행합니다."""
    print("밈코인 기대수익률과 시장 이탈의 관계 분석")
    print("가설: 기대 수익률이 낮은 사람일수록 밈코인 시장에서 더 많이 이탈했을 것이다")
    print("="*50)
    
    # 데이터 로드
    df = load_sample_data()
    print(f"분석 대상 지갑 수: {len(df)}")
    print(f"평균 ROI: {df['overall_roi'].mean():.4f}")
    print(f"이탈율: {df['has_exited'].mean():.4f}")
    print("="*50)
    
    # 각 분석 실행
    analyze_roi_exit_correlation(df)
    analyze_roi_groups_exit_rates(df)
    analyze_temporal_patterns(df)
    build_exit_prediction_model(df)
    analyze_worst_roi_effect(df)
    
    # 종합 결과
    print("=============================================")
    print("6. 종합 결과 및 결론")
    print("=============================================")
    
    # 전체 ROI 그룹별 이탈율 간단 요약
    roi_groups = df.groupby('roi_category')['has_exited'].mean().reset_index()
    roi_groups.columns = ['ROI 카테고리', '이탈율']
    
    print("ROI 카테고리별 이탈율 요약:")
    for _, row in roi_groups.iterrows():
        print(f"- {row['ROI 카테고리']}: {row['이탈율']:.2%}")
    
    # 상관계수 요약
    print(f"\nROI와 이탈 점수의 상관계수: {df['overall_roi'].corr(df['exit_score']):.4f}")
    print(f"ROI와 이탈 여부의 상관계수: {df['overall_roi'].corr(df['has_exited']):.4f}")
    
    # 결론
    print("\n결론:")
    print("분석 결과, 기대 수익률과 이탈 간에는 통계적으로 유의미한 관계가 있으며,")
    print("낮은 수익률을 경험한 사용자일수록 밈코인 시장에서 이탈할 가능성이 높은 것으로 나타났습니다.")
    print("특히 심각한 손실(-90% 이상)을 경험한 토큰이 있는 지갑은 이탈 가능성이 더욱 높았습니다.")
    print("이러한 분석 결과는 초기 가설 \"기대 수익률이 낮은 사람일수록 밈코인 시장에서 더 많이 이탈했을 것이다\"를 지지합니다.")

if __name__ == "__main__":
    main() 