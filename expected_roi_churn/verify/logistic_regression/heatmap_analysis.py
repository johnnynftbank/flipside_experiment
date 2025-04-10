import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# 출력 디렉토리 설정
output_dir = "verify/logistic_regression"
os.makedirs(output_dir, exist_ok=True)

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

def create_heatmap(df_clean):
    """ROI와 거래일수에 따른 이탈률 히트맵 생성"""
    print("\n=== ROI와 거래일수에 따른 이탈률 히트맵 ===")
    
    try:
        # ROI와 거래일수를 5분위로 그룹화
        df_clean['ROI_GROUP'] = pd.qcut(df_clean['EXPECTED_ROI'], 5, labels=False)
        df_clean['DAYS_GROUP'] = pd.qcut(df_clean['TRADED_DAYS'], 5, labels=False)
        
        # ROI와 거래일수 그룹별 이탈률 계산
        heatmap_data = df_clean.groupby(['ROI_GROUP', 'DAYS_GROUP']).agg({
            'IS_CHURNED': 'mean',
            'WALLET_STATUS': 'count'
        }).reset_index()
        
        # 피벗 테이블 생성
        pivot_data = heatmap_data.pivot_table(
            values='IS_CHURNED', 
            index='ROI_GROUP',
            columns='DAYS_GROUP'
        )
        
        # 데이터 확인
        print("이탈률 피벗 테이블:")
        print(pivot_data)
        
        # 히트맵 시각화
        plt.figure(figsize=(12, 9))
        sns.heatmap(pivot_data, annot=True, cmap='YlGnBu_r', fmt=".2f", vmin=0, vmax=1)
        plt.title('기대 수익률(ROI)과 거래일수에 따른 이탈률', fontsize=16)
        plt.xlabel('거래일수 그룹 (높을수록 더 많은 거래일수)', fontsize=14)
        plt.ylabel('ROI 그룹 (높을수록 더 높은 ROI)', fontsize=14)
        
        # 축 레이블 한글화
        plt.xticks(ticks=np.arange(0.5, 5.5, 1), labels=['매우 적음', '적음', '보통', '많음', '매우 많음'], fontsize=12)
        plt.yticks(ticks=np.arange(0.5, 5.5, 1), labels=['매우 낮음', '낮음', '보통', '높음', '매우 높음'], fontsize=12)
        
        # 컬러바 레이블 추가
        cbar = plt.gca().collections[0].colorbar
        cbar.set_label('이탈률 (높을수록 더 많은 이탈)', fontsize=12)
        
        # 그래프 저장
        plt.tight_layout()
        plt.savefig(f"{output_dir}/roi_days_churn_heatmap.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print("ROI와 거래일수 히트맵이 생성되었습니다.")
        
        # 추가 분석: 그룹별 평균 통계 출력
        roi_group_stats = df_clean.groupby('ROI_GROUP').agg({
            'EXPECTED_ROI': ['mean', 'median'],
            'IS_CHURNED': 'mean',
            'WALLET_STATUS': 'count'
        })
        
        days_group_stats = df_clean.groupby('DAYS_GROUP').agg({
            'TRADED_DAYS': ['mean', 'median'],
            'IS_CHURNED': 'mean',
            'WALLET_STATUS': 'count'
        })
        
        print("\nROI 그룹별 통계:")
        print(roi_group_stats)
        
        print("\n거래일수 그룹별 통계:")
        print(days_group_stats)
        
        # ROI와 거래일수의 상관관계
        correlation = df_clean[['EXPECTED_ROI', 'TRADED_DAYS']].corr().iloc[0, 1]
        print(f"\nROI와 거래일수 간의 상관계수: {correlation:.4f}")
        
        # 거래일수를 통제한 후의 ROI와 이탈 간의 관계 분석
        controlled_effects = []
        for days_group in range(5):
            subset = df_clean[df_clean['DAYS_GROUP'] == days_group]
            active_roi = subset[subset['IS_CHURNED'] == 0]['EXPECTED_ROI'].mean()
            churned_roi = subset[subset['IS_CHURNED'] == 1]['EXPECTED_ROI'].mean()
            effect = active_roi - churned_roi
            controlled_effects.append({
                'DAYS_GROUP': days_group,
                'ACTIVE_ROI': active_roi,
                'CHURNED_ROI': churned_roi,
                'DIFFERENCE': effect
            })
        
        controlled_df = pd.DataFrame(controlled_effects)
        print("\n거래일수 그룹별 ROI 차이 (활성 - 이탈):")
        print(controlled_df)
        
        return pivot_data, controlled_df
        
    except Exception as e:
        print(f"히트맵 생성 중 오류 발생: {e}")
        return None, None

def main():
    # 데이터 파일 경로
    data_path = "expected_roi_churn_final/query_result/meme_coin_roi_churn_3.csv"
    
    # 데이터 로드
    df = load_data(data_path)
    
    # 데이터 정제
    df_clean = clean_data(df)
    
    # 히트맵 생성
    pivot_data, controlled_df = create_heatmap(df_clean)
    
    print("\n분석이 완료되었습니다.")

if __name__ == "__main__":
    main() 