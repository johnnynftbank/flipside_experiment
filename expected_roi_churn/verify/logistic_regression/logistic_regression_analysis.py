import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, roc_auc_score
from sklearn.preprocessing import StandardScaler
import os
import warnings
warnings.filterwarnings('ignore')

# 출력 디렉토리 설정
output_dir = "verify/logistic_regression"
os.makedirs(output_dir, exist_ok=True)

# 데이터 로드
def load_data(filepath):
    """데이터 로드 및 전처리"""
    print(f"데이터를 로드합니다: {filepath}")
    df = pd.read_csv(filepath)
    
    # 지갑 상태를 이진값으로 변환
    df['IS_CHURNED'] = (df['WALLET_STATUS'] == 'CHURNED').astype(int)
    
    print(f"{len(df)}개의 지갑 데이터를 로드했습니다")
    return df

def clean_data(df):
    """데이터 정제 및 전처리"""
    # 이상치 제거
    Q1 = df['EXPECTED_ROI'].quantile(0.25)
    Q3 = df['EXPECTED_ROI'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 3 * IQR
    upper_bound = Q3 + 3 * IQR
    
    # 이상치가 제거된 데이터프레임 생성
    df_clean = df[(df['EXPECTED_ROI'] >= lower_bound) & 
                 (df['EXPECTED_ROI'] <= upper_bound) &
                 (~df['EXPECTED_ROI'].isna()) &
                 (~np.isinf(df['EXPECTED_ROI']))]
    
    print(f"원본 데이터: {len(df)}개 레코드")
    print(f"정제된 데이터: {len(df_clean)}개 레코드 (이상치 제거)")
    
    return df_clean

def perform_logistic_regression(df_clean):
    """로지스틱 회귀 분석 수행"""
    print("\n=== 로지스틱 회귀 분석 ===")
    
    # 특성(X)과 타겟(y) 정의
    X = df_clean[['EXPECTED_ROI']]
    y = df_clean['IS_CHURNED']
    
    # 데이터 분할: 훈련 세트(70%)와 테스트 세트(30%)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # 특성 스케일링
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 로지스틱 회귀 모델 훈련
    model = LogisticRegression(random_state=42)
    model.fit(X_train_scaled, y_train)
    
    # 회귀 계수 및 절편 추출
    coef = model.coef_[0][0]
    intercept = model.intercept_[0]
    
    print(f"로지스틱 회귀 계수(beta): {coef:.6f}")
    print(f"절편(intercept): {intercept:.6f}")
    print(f"오즈비(Odds Ratio): {np.exp(coef):.6f}")
    
    # 테스트 세트에서 예측 수행
    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)[:, 1]
    
    # 분류 보고서
    report = classification_report(y_test, y_pred, output_dict=True)
    accuracy = report['accuracy']
    precision = report['1']['precision']
    recall = report['1']['recall']
    f1 = report['1']['f1-score']
    
    print(f"정확도: {accuracy:.4f}")
    print(f"정밀도(이탈 예측 정확성): {precision:.4f}")
    print(f"재현율(실제 이탈 감지율): {recall:.4f}")
    print(f"F1 점수: {f1:.4f}")
    
    # ROC 곡선 및 AUC 계산
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    auc = roc_auc_score(y_test, y_prob)
    print(f"AUC(Area Under Curve): {auc:.4f}")
    
    # 결과 시각화
    create_visualizations(df_clean, model, scaler, X_test_scaled, y_test, y_prob, fpr, tpr, auc)
    
    # 결과 요약 저장
    save_results(df_clean, coef, intercept, accuracy, precision, recall, f1, auc)
    
    return {
        'coef': coef,
        'intercept': intercept,
        'odds_ratio': np.exp(coef),
        'accuracy': accuracy,
        'precision': precision, 
        'recall': recall,
        'f1': f1,
        'auc': auc
    }

def create_visualizations(df_clean, model, scaler, X_test_scaled, y_test, y_prob, fpr, tpr, auc):
    """분석 결과 시각화"""
    
    # 1. ROI에 따른 이탈 확률 시각화
    plt.figure(figsize=(12, 8))
    
    # 예측 확률 곡선 생성을 위한 데이터
    roi_range = np.linspace(df_clean['EXPECTED_ROI'].min(), df_clean['EXPECTED_ROI'].max(), 100)
    roi_range_scaled = scaler.transform(roi_range.reshape(-1, 1))
    churn_probs = model.predict_proba(roi_range_scaled)[:, 1]
    
    # 실제 데이터 포인트 시각화 (산점도)
    plt.scatter(df_clean['EXPECTED_ROI'], df_clean['IS_CHURNED'], 
                alpha=0.3, s=10, c=df_clean['IS_CHURNED'], 
                cmap='coolwarm', label='실제 데이터')
    
    # 예측 확률 곡선
    plt.plot(roi_range, churn_probs, color='blue', linewidth=3, 
             label='이탈 확률 예측 곡선')
    
    # 의사결정 경계선 (확률 0.5)
    decision_boundary = (-model.intercept_[0] / model.coef_[0][0]) * scaler.scale_[0] + scaler.mean_[0]
    plt.axvline(x=decision_boundary, color='red', linestyle='--', 
                label=f'의사결정 경계 (ROI = {decision_boundary:.4f})')
    
    plt.title('기대 수익률(ROI)에 따른 이탈 확률', fontsize=16)
    plt.xlabel('기대 수익률(Expected ROI)', fontsize=14)
    plt.ylabel('이탈 확률', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 그래프 저장
    plt.tight_layout()
    plt.savefig(f"{output_dir}/churn_probability_vs_roi.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. ROC 곡선
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC 곡선 (AUC = {auc:.4f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (1 - 특이도)', fontsize=14)
    plt.ylabel('True Positive Rate (재현율)', fontsize=14)
    plt.title('ROC 곡선', fontsize=16)
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/roc_curve.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. 오즈비 해석 시각화
    odds_ratio = np.exp(model.coef_[0][0])
    plt.figure(figsize=(10, 6))
    
    # 오즈비 시각화
    plt.barh(['기대 수익률(ROI)'], [odds_ratio], color='skyblue')
    plt.axvline(x=1, color='red', linestyle='--')
    
    # 값 레이블 추가
    plt.text(odds_ratio, 0, f'  {odds_ratio:.4f}', va='center', fontsize=12)
    
    plt.title('기대 수익률(ROI)의 오즈비', fontsize=16)
    plt.xlabel('오즈비(Odds Ratio)', fontsize=14)
    plt.xlim(0, max(odds_ratio * 1.2, 1.2))
    plt.grid(True, alpha=0.3, axis='x')
    
    # 해석 텍스트 추가
    if odds_ratio < 1:
        interpretation = f"ROI가 1단위 증가할 때마다\n이탈 확률이 {(1 - odds_ratio) * 100:.1f}% 감소"
    else:
        interpretation = f"ROI가 1단위 증가할 때마다\n이탈 확률이 {(odds_ratio - 1) * 100:.1f}% 증가"
    
    plt.annotate(interpretation, xy=(odds_ratio / 2, 0), xytext=(0, -40), 
                textcoords='offset points', ha='center', va='top',
                fontsize=12, bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.3))
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/odds_ratio.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. 혼동 행렬(Confusion Matrix) 시각화
    plt.figure(figsize=(10, 8))
    conf_matrix = confusion_matrix(y_test, model.predict(X_test_scaled))
    
    # 비율로 변환
    conf_matrix_normalized = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]
    
    sns.heatmap(conf_matrix_normalized, annot=conf_matrix, fmt='d', cmap='Blues',
                xticklabels=['활성', '이탈'], yticklabels=['활성', '이탈'])
    
    plt.title('혼동 행렬', fontsize=16)
    plt.ylabel('실제 레이블', fontsize=14)
    plt.xlabel('예측 레이블', fontsize=14)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/confusion_matrix.png", dpi=300, bbox_inches='tight')
    plt.close()

def save_results(df_clean, coef, intercept, accuracy, precision, recall, f1, auc):
    """분석 결과를 텍스트 파일로 저장"""
    
    # 활성 및 이탈 지갑의 기대 수익률 통계
    active_roi = df_clean[df_clean['IS_CHURNED'] == 0]['EXPECTED_ROI']
    churned_roi = df_clean[df_clean['IS_CHURNED'] == 1]['EXPECTED_ROI']
    
    result_text = f"""
로지스틱 회귀 분석 결과
==========================

기본 정보:
- 총 지갑 수: {len(df_clean)}
- 활성 지갑: {len(active_roi)}개 ({len(active_roi) / len(df_clean) * 100:.2f}%)
- 이탈 지갑: {len(churned_roi)}개 ({len(churned_roi) / len(df_clean) * 100:.2f}%)

기술 통계:
- 활성 지갑 ROI - 평균: {active_roi.mean():.6f}, 중앙값: {active_roi.median():.6f}, 표준편차: {active_roi.std():.6f}
- 이탈 지갑 ROI - 평균: {churned_roi.mean():.6f}, 중앙값: {churned_roi.median():.6f}, 표준편차: {churned_roi.std():.6f}
- 평균 차이 (활성 - 이탈): {active_roi.mean() - churned_roi.mean():.6f}

로지스틱 회귀 모델 결과:
- 회귀 계수(beta): {coef:.6f}
- 절편(intercept): {intercept:.6f}
- 오즈비(Odds Ratio): {np.exp(coef):.6f}
- 해석: 기대 수익률(ROI)이 1단위 증가할 때마다 이탈 오즈가 {np.exp(coef):.6f}배 변화

모델 성능:
- 정확도: {accuracy:.4f}
- 정밀도(이탈 예측 정확성): {precision:.4f}
- 재현율(실제 이탈 감지율): {recall:.4f}
- F1 점수: {f1:.4f}
- AUC(Area Under Curve): {auc:.4f}

결론:
이 로지스틱 회귀 분석은 기대 수익률(ROI)과 지갑 이탈 간의 관계를 모델링했습니다.
오즈비가 {np.exp(coef):.4f}{"보다 작아" if np.exp(coef) < 1 else "보다 커"} 기대 수익률이 증가할수록 이탈 확률이 {"감소함" if np.exp(coef) < 1 else "증가함"}을 나타냅니다.
구체적으로, 기대 수익률이 1단위 증가할 때마다 이탈 오즈가 {abs(1 - np.exp(coef)) * 100:.1f}% {"감소합니다" if np.exp(coef) < 1 else "증가합니다"}.
모델의 AUC는 {auc:.4f}로, 이는 기대 수익률이 지갑 이탈을 예측하는 데 {"매우 좋은" if auc > 0.8 else "중간 수준의" if auc > 0.7 else "약한"} 예측력을 가짐을 시사합니다.
"""
    
    with open(f"{output_dir}/logistic_regression_results.txt", "w") as f:
        f.write(result_text)
    
    print(f"결과가 저장되었습니다: {output_dir}/logistic_regression_results.txt")

def main():
    # 데이터 파일 경로
    data_path = "expected_roi_churn_final/query_result/meme_coin_roi_churn_3.csv"
    
    # 데이터 로드
    df = load_data(data_path)
    
    # 데이터 정제
    df_clean = clean_data(df)
    
    # 로지스틱 회귀 분석 수행
    results = perform_logistic_regression(df_clean)
    
    # 요약 출력
    print("\n=== 요약 ===")
    print(f"로지스틱 회귀 계수: {results['coef']:.6f}")
    print(f"오즈비: {results['odds_ratio']:.6f}")
    print(f"정확도: {results['accuracy']:.4f}")
    print(f"AUC: {results['auc']:.4f}")
    print(f"해석: 기대 수익률(ROI)이 1단위 증가할 때마다 이탈 오즈가 {results['odds_ratio']:.4f}배 변화")
    print(f"      즉, 기대 수익률이 1단위 증가할 때마다 이탈 확률이 {abs(1 - results['odds_ratio']) * 100:.1f}% {'감소' if results['odds_ratio'] < 1 else '증가'}")

if __name__ == "__main__":
    main() 