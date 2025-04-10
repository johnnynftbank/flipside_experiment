import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os

# 출력 디렉토리 설정
output_dir = "verify/mann_whitney_test"
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

# 기본 통계 분석
def basic_statistics(df):
    """기본 통계 분석 수행"""
    print("\n=== 기본 통계 ===")
    
    # 상태별 카운트 및 비율
    status_counts = df['WALLET_STATUS'].value_counts()
    status_pcts = df['WALLET_STATUS'].value_counts(normalize=True) * 100
    
    print(f"활성 지갑: {status_counts.get('ACTIVE', 0)} ({status_pcts.get('ACTIVE', 0):.2f}%)")
    print(f"이탈 지갑: {status_counts.get('CHURNED', 0)} ({status_pcts.get('CHURNED', 0):.2f}%)")
    
    # 지갑 상태별 통계 그룹화
    stats = df.groupby('WALLET_STATUS')['EXPECTED_ROI'].describe()
    print("\n지갑 상태별 예상 ROI 통계:")
    print(stats)
    
    return stats

# Mann-Whitney U 검정 수행
def perform_mann_whitney_test(df):
    """활성 지갑과 이탈 지갑 간의 Mann-Whitney U 검정 수행"""
    print("\n=== Mann-Whitney U 검정 ===")
    
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
    
    # 활성 및 이탈 지갑의 예상 ROI 추출
    active_roi = df_clean[df_clean['WALLET_STATUS'] == 'ACTIVE']['EXPECTED_ROI'].values
    churned_roi = df_clean[df_clean['WALLET_STATUS'] == 'CHURNED']['EXPECTED_ROI'].values
    
    # Mann-Whitney U 검정 수행
    u_stat, p_value = stats.mannwhitneyu(active_roi, churned_roi)
    
    print(f"Mann-Whitney U 통계량: {u_stat:.4f}")
    print(f"p-value: {p_value:.10f}")
    
    # 해석
    alpha = 0.05
    if p_value < alpha:
        interpretation = "두 그룹의 분포가 통계적으로 유의미하게 다릅니다 (귀무가설 기각)"
    else:
        interpretation = "두 그룹의 분포 차이가 통계적으로 유의미하지 않습니다 (귀무가설 채택)"
    
    print(f"해석: {interpretation}")
    
    # 각 그룹의 ROI 분포 시각화
    plt.figure(figsize=(12, 8))
    
    # 밀도 플롯
    sns.kdeplot(
        data=df_clean, 
        x='EXPECTED_ROI', 
        hue='WALLET_STATUS',
        fill=True,
        alpha=0.5,
        palette=['green', 'red'],
        common_norm=False,
        linewidth=2
    )
    
    # 각 그룹의 평균값 표시
    active_mean = np.mean(active_roi)
    churned_mean = np.mean(churned_roi)
    plt.axvline(x=active_mean, color='green', linestyle='--', 
                label=f'ACTIVE 평균: {active_mean:.4f}')
    plt.axvline(x=churned_mean, color='red', linestyle='--', 
                label=f'CHURNED 평균: {churned_mean:.4f}')
    
    plt.title('지갑 상태별 예상 ROI 분포 (이상치 제거)', fontsize=14)
    plt.xlabel('예상 ROI', fontsize=12)
    plt.ylabel('밀도', fontsize=12)
    plt.legend(title='지갑 상태')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # 시각화 저장
    plt.savefig(f"{output_dir}/roi_distribution_by_status.png", dpi=300, bbox_inches='tight')
    print(f"시각화가 저장되었습니다: {output_dir}/roi_distribution_by_status.png")
    
    # 결과를 텍스트 파일로 저장
    result_text = f"""
Mann-Whitney U 검정 결과
============================

기본 정보:
- 총 지갑 수: {len(df)} (정제 후: {len(df_clean)})
- 활성 지갑: {len(active_roi)}개
- 이탈 지갑: {len(churned_roi)}개
- 이탈률: {len(churned_roi) / len(df_clean) * 100:.2f}%

기술 통계:
- 활성 지갑 ROI - 평균: {np.mean(active_roi):.6f}, 중앙값: {np.median(active_roi):.6f}, 표준편차: {np.std(active_roi):.6f}
- 이탈 지갑 ROI - 평균: {np.mean(churned_roi):.6f}, 중앙값: {np.median(churned_roi):.6f}, 표준편차: {np.std(churned_roi):.6f}
- 평균 차이 (ACTIVE - CHURNED): {np.mean(active_roi) - np.mean(churned_roi):.6f}

Mann-Whitney U 검정 결과:
- U 통계량: {u_stat:.4f}
- p-value: {p_value:.10f}
- 해석: {interpretation}

결론:
이 분석은 활성 지갑과 이탈 지갑 간의 예상 ROI 분포가 통계적으로 {"유의미한" if p_value < alpha else "유의미하지 않은"} 차이를 보임을 확인했습니다.
Mann-Whitney U 검정은 비모수적 방법으로, 두 독립 표본 간의 분포 차이를 확인하는 데 사용됩니다.
이는 데이터가 정규 분포를 따르지 않거나 이상치가 있는 경우에 특히 적합합니다.
"""
    
    with open(f"{output_dir}/mann_whitney_test_results.txt", "w") as f:
        f.write(result_text)
    
    print(f"결과가 저장되었습니다: {output_dir}/mann_whitney_test_results.txt")
    
    return {
        'u_stat': u_stat,
        'p_value': p_value,
        'active_roi_mean': np.mean(active_roi),
        'churned_roi_mean': np.mean(churned_roi),
        'active_roi_median': np.median(active_roi),
        'churned_roi_median': np.median(churned_roi),
        'active_roi_std': np.std(active_roi),
        'churned_roi_std': np.std(churned_roi),
        'interpretation': interpretation
    }

def main():
    # 데이터 파일 경로
    data_path = "expected_roi_churn_final/query_result/meme_coin_roi_churn_3.csv"
    
    # 데이터 로드
    df = load_data(data_path)
    
    # 기본 통계 계산
    basic_statistics(df)
    
    # Mann-Whitney 검정 수행
    results = perform_mann_whitney_test(df)
    
    # 요약 출력
    print("\n=== 요약 ===")
    print(f"Mann-Whitney U 통계량: {results['u_stat']:.4f}")
    print(f"p-value: {results['p_value']:.10f}")
    print(f"활성 지갑 ROI 평균: {results['active_roi_mean']:.6f}")
    print(f"이탈 지갑 ROI 평균: {results['churned_roi_mean']:.6f}")
    print(f"평균 차이: {results['active_roi_mean'] - results['churned_roi_mean']:.6f}")
    print(f"해석: {results['interpretation']}")

if __name__ == "__main__":
    main() 