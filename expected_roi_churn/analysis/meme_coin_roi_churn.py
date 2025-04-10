import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_curve, auc
from sklearn.model_selection import train_test_split

# 1. 데이터 로드 (쿼리 결과를 CSV로 저장했다고 가정)
df = pd.read_csv('meme_coin_roi_churn.csv')

# 2. 기본 탐색적 데이터 분석 (EDA)
print(df.head())
print(df.describe())
print(df.info())

# 3. 이탈/활성 사용자 비율 확인
churn_ratio = df['wallet_status'].value_counts(normalize=True)
print(f"지갑 상태 비율:\n{churn_ratio}")

# 4. 상관관계 분석 준비 - 이탈 여부를 이진값으로 변환
df['is_churned'] = df['wallet_status'].apply(lambda x: 1 if x == 'CHURNED' else 0)

# 5. 기대수익률과 이탈 간의 상관관계 시각화
plt.figure(figsize=(10, 6))
sns.boxplot(x='wallet_status', y='expected_roi', data=df)
plt.title('기대수익률과 지갑 상태 간의 관계')
plt.savefig('roi_vs_churn_boxplot.png')

# 6. 포인트-바이시리얼 상관계수 계산 (이진변수와 연속변수 간의 상관관계)
pb_corr_expected, p_value_expected = stats.pointbiserialr(df['is_churned'], df['expected_roi'])
pb_corr_average, p_value_average = stats.pointbiserialr(df['is_churned'], df['average_roi'])

print(f"기대수익률과 이탈 간의 포인트-바이시리얼 상관계수: {pb_corr_expected:.4f}, p-값: {p_value_expected:.4f}")
print(f"평균 ROI와 이탈 간의 포인트-바이시리얼 상관계수: {pb_corr_average:.4f}, p-값: {p_value_average:.4f}")

# 7. ROI 구간별 이탈률 분석
def create_roi_bins(df, column='expected_roi', bins=10):
    df['roi_bin'] = pd.qcut(df[column], q=bins, duplicates='drop')
    bin_analysis = df.groupby('roi_bin')['is_churned'].mean().reset_index()
    bin_analysis['count'] = df.groupby('roi_bin')['is_churned'].count().values
    return bin_analysis

expected_roi_bins = create_roi_bins(df)
print("기대수익률 구간별 이탈률:")
print(expected_roi_bins)

# 시각화
plt.figure(figsize=(12, 6))
sns.barplot(x='roi_bin', y='is_churned', data=expected_roi_bins)
plt.title('기대수익률 구간별 이탈률')
plt.xticks(rotation=45)
plt.savefig('roi_bins_vs_churn.png')

# 8. 로지스틱 회귀 분석 (이탈 예측 모델)
X = df[['expected_roi', 'average_roi', 'unique_tokens_traded']]
y = df['is_churned']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = LogisticRegression()
model.fit(X_train, y_train)

# 모델 평가
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# 각 변수의 중요도(계수) 확인
coefficients = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': model.coef_[0]
})
print("로지스틱 회귀 계수 (양수는 이탈 확률 증가, 음수는 감소):")
print(coefficients.sort_values('Coefficient', ascending=False))

# 9. ROC 곡선 및 AUC
y_pred_proba = model.predict_proba(X_test)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'ROC 곡선 (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('이탈 예측 ROC 곡선')
plt.legend()
plt.savefig('churn_prediction_roc.png')

# 10. 추가 분석: 거래 빈도와 ROI의 교차 영향
plt.figure(figsize=(10, 8))
pivot = pd.pivot_table(
    data=df,
    values='is_churned',
    index=pd.qcut(df['expected_roi'], 5),
    columns=pd.qcut(df['unique_tokens_traded'], 5),
    aggfunc='mean'
)
sns.heatmap(pivot, annot=True, cmap='YlGnBu', fmt='.2f')
plt.title('기대수익률과 거래 토큰 수에 따른 이탈률')
plt.tight_layout()
plt.savefig('roi_tokens_churn_heatmap.png')