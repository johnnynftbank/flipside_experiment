import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime

# 한글 폰트 설정
plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False
sns.set(style="whitegrid")

# 결과 저장 폴더 경로
RESULT_PATH = "../report/"
os.makedirs(RESULT_PATH, exist_ok=True)

# 데이터 로드
def load_data(file_path):
    print(f"데이터 파일 로드 중: {file_path}")
    data = pd.read_csv(file_path)
    
    # 열 이름을 소문자로 변환
    data.columns = [col.lower() for col in data.columns]
    
    print(f"로드 완료: {len(data)} 행, {list(data.columns)}")
    return data

# 기존 군집 정의에 따라 지갑 분류
def classify_wallets(data):
    """
    기존 K-means 결과를 바탕으로 각 지갑을 군집으로 분류
    
    군집 0: 안정적 투자형
    군집 1: 잭팟 추구형
    군집 2: 모험적 투자형
    군집 3: 일반 투자형
    군집 -1: 미분류
    """
    # 결측치 제거
    data = data.dropna(subset=['expected_roi', 'roi_standard_deviation', 'sharpe_ratio', 
                               'win_loss_ratio', 'max_trade_proportion'])
    
    # 군집 할당 초기화
    data['cluster'] = -1  # 기본값은 미분류(-1)
    
    # 군집 0: 안정적 투자형 (높은 ROI, 낮은 표준편차, 안정적 투자형)
    data.loc[(data['expected_roi'] > -0.02) & 
             (data['roi_standard_deviation'] < 0.1) &
             (data['win_loss_ratio'] > 0.6), 'cluster'] = 0
    
    # 군집 1: 잭팟 추구형 (극도로 낮은 ROI, 낮은 승률, 잭팟 추구형)
    data.loc[(data['expected_roi'] < -0.5) & 
             (data['sharpe_ratio'] < -5.0) &
             (data['win_loss_ratio'] < 0.1) &
             (data['max_trade_proportion'] > 0.35), 'cluster'] = 1
    
    # 군집 2: 모험적 투자형 (중간 ROI, 높은 표준편차, 모험적 투자형)
    data.loc[(data['expected_roi'] > -0.1) & (data['expected_roi'] < 0) &
             (data['roi_standard_deviation'] > 0.5) &
             (data['win_loss_ratio'] > 0.6), 'cluster'] = 2
    
    # 군집 3: 일반 투자형 (낮은 ROI, 중간 표준편차, 일반 투자형)
    data.loc[(data['expected_roi'] > -0.3) & (data['expected_roi'] < -0.1) &
             (data['roi_standard_deviation'] > 0.2) & (data['roi_standard_deviation'] < 0.5) &
             (data['win_loss_ratio'] > 0.4) & (data['win_loss_ratio'] < 0.6), 'cluster'] = 3
    
    # 분류 결과 요약
    print("\n군집 분류 결과:")
    cluster_counts = data['cluster'].value_counts().sort_index()
    total_wallets = len(data)
    
    for cluster_id, count in cluster_counts.items():
        cluster_name = "미분류" if cluster_id == -1 else f"군집 {cluster_id}"
        percentage = count / total_wallets * 100
        print(f"{cluster_name}: {count}개 지갑 ({percentage:.2f}%)")
    
    return data

# 군집별 통계 계산
def calculate_cluster_stats(data):
    # 각 군집별 주요 지표의 평균과 표준편차 계산
    cluster_stats = data.groupby('cluster').agg({
        'expected_roi': ['mean', 'std'],
        'roi_standard_deviation': ['mean', 'std'],
        'sharpe_ratio': ['mean', 'std'],
        'win_loss_ratio': ['mean', 'std'],
        'max_trade_proportion': ['mean', 'std'],
        'unique_tokens_traded': ['mean', 'std'],
        'total_trades': ['mean', 'std'],
        'swapper': 'count'
    })
    
    cluster_stats.columns = ['_'.join(col).strip() for col in cluster_stats.columns.values]
    cluster_stats = cluster_stats.rename(columns={'swapper_count': 'wallet_count'})
    
    # 비율 계산 추가
    total_wallets = len(data)
    cluster_stats['percentage'] = cluster_stats['wallet_count'] / total_wallets * 100
    
    return cluster_stats

# 시각화 생성
def create_visualizations(data, stats):
    # 1. 군집 분포 파이 차트
    plt.figure(figsize=(10, 7))
    cluster_names = {
        -1: '미분류',
        0: '안정적 투자형',
        1: '잭팟 추구형',
        2: '모험적 투자형',
        3: '일반 투자형'
    }
    
    # 군집 분포 계산
    cluster_counts = data['cluster'].value_counts().sort_index()
    cluster_labels = [cluster_names[i] for i in cluster_counts.index]
    
    # 파이 차트 생성
    plt.pie(cluster_counts, labels=cluster_labels, autopct='%1.1f%%', startangle=90, 
           textprops={'fontsize': 12}, colors=sns.color_palette('viridis', len(cluster_counts)))
    plt.title('2025년 3월 밈코인 투자자 군집 분포', fontsize=16)
    plt.tight_layout()
    plt.savefig(f"{RESULT_PATH}march_2025_cluster_distribution_pie.png", dpi=300)
    plt.close()
    
    # 2. 주요 지표별 군집 분포 (박스 플롯)
    key_metrics = ['expected_roi', 'roi_standard_deviation', 'sharpe_ratio', 
                  'win_loss_ratio', 'max_trade_proportion']
    
    for metric in key_metrics:
        plt.figure(figsize=(10, 6))
        ax = sns.boxplot(x='cluster', y=metric, data=data, palette='viridis')
        
        plt.title(f'군집별 {metric} 분포 (2025년 3월)', fontsize=14)
        plt.xlabel('군집', fontsize=12)
        plt.ylabel(metric, fontsize=12)
        
        # x축 레이블 변경
        ax.set_xticklabels([cluster_names[i] for i in sorted(data['cluster'].unique())])
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig(f"{RESULT_PATH}march_2025_cluster_{metric}_boxplot.png", dpi=300)
        plt.close()
    
    # 3. 2차원 산점도 (ROI vs 최대 거래 비중)
    plt.figure(figsize=(12, 8))
    
    # 군집별 색상 맵핑
    cluster_colors = {
        -1: 'gray',    # 미분류
        0: 'green',    # 안정적 투자형
        1: 'red',      # 잭팟 추구형
        2: 'purple',   # 모험적 투자형
        3: 'blue'      # 일반 투자형
    }
    
    # 군집별로 산점도 그리기
    for cluster_id in sorted(data['cluster'].unique()):
        cluster_data = data[data['cluster'] == cluster_id]
        plt.scatter(
            cluster_data['expected_roi'], 
            cluster_data['max_trade_proportion'],
            c=cluster_colors[cluster_id],
            label=cluster_names[cluster_id],
            alpha=0.7,
            edgecolors='w',
            linewidth=0.5
        )
    
    plt.title('기대 수익률 vs 최대 거래 비중 (2025년 3월)', fontsize=16)
    plt.xlabel('기대 수익률 (Expected ROI)', fontsize=14)
    plt.ylabel('최대 거래 비중 (Max Trade Proportion)', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{RESULT_PATH}march_2025_roi_vs_max_trade_scatter.png", dpi=300)
    plt.close()
    
    # 4. 샤프 비율 vs 승패 비율 산점도
    plt.figure(figsize=(12, 8))
    
    for cluster_id in sorted(data['cluster'].unique()):
        cluster_data = data[data['cluster'] == cluster_id]
        plt.scatter(
            cluster_data['sharpe_ratio'], 
            cluster_data['win_loss_ratio'],
            c=cluster_colors[cluster_id],
            label=cluster_names[cluster_id],
            alpha=0.7,
            edgecolors='w',
            linewidth=0.5
        )
    
    plt.title('샤프 비율 vs 승패 비율 (2025년 3월)', fontsize=16)
    plt.xlabel('샤프 비율 (Sharpe Ratio)', fontsize=14)
    plt.ylabel('승패 비율 (Win-Loss Ratio)', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{RESULT_PATH}march_2025_sharpe_vs_winloss_scatter.png", dpi=300)
    plt.close()

# 보고서 생성
def create_report(data, stats):
    # 현재 날짜
    today = datetime.now().strftime("%Y-%m-%d")
    
    report_content = f"""# 2025년 3월 밈코인 투자자 군집 분석 보고서
*생성일: {today}*

## 1. 분석 개요

본 보고서는 2025년 3월 솔라나 밈코인 거래 데이터를 기존에 정의된 4개 군집(안정적 투자형, 잭팟 추구형, 모험적 투자형, 일반 투자형)에 따라 분류하고 분석한 결과입니다.

### 데이터셋 정보
- **분석 기간**: 2025년 3월 1일 ~ 3월 31일
- **대상 지갑 수**: {len(data)}개 지갑
- **사용 쿼리**: monthly_cluster_analysis.sql

## 2. 군집 분포 결과

![군집 분포](march_2025_cluster_distribution_pie.png)

### 군집별 지갑 수 및 비율
"""
    
    # 군집 테이블 생성
    cluster_names = {
        -1: '미분류',
        0: '안정적 투자형',
        1: '잭팟 추구형',
        2: '모험적 투자형',
        3: '일반 투자형'
    }
    
    for cluster_id in sorted(stats.index):
        cluster_info = stats.loc[cluster_id]
        report_content += f"""
| **{cluster_names[cluster_id]}** (군집 {cluster_id}) |
|:-------------------------------------------|
| 지갑 수: {int(cluster_info['wallet_count'])}개 ({cluster_info['percentage']:.2f}%) |
| 기대 수익률: {cluster_info['expected_roi_mean']:.4f} (±{cluster_info['expected_roi_std']:.4f}) |
| ROI 표준편차: {cluster_info['roi_standard_deviation_mean']:.4f} (±{cluster_info['roi_standard_deviation_std']:.4f}) |
| 샤프 비율: {cluster_info['sharpe_ratio_mean']:.4f} (±{cluster_info['sharpe_ratio_std']:.4f}) |
| 승패 비율: {cluster_info['win_loss_ratio_mean']:.4f} (±{cluster_info['win_loss_ratio_std']:.4f}) |
| 최대 거래 비중: {cluster_info['max_trade_proportion_mean']:.4f} (±{cluster_info['max_trade_proportion_std']:.4f}) |
| 평균 거래 토큰 수: {cluster_info['unique_tokens_traded_mean']:.2f} |
| 평균 거래 횟수: {cluster_info['total_trades_mean']:.2f} |
"""
    
    report_content += """
## 3. 군집 특성 분석

### 주요 지표별 군집 분포

![기대 수익률 분포](march_2025_cluster_expected_roi_boxplot.png)

![ROI 표준편차 분포](march_2025_cluster_roi_standard_deviation_boxplot.png)

![샤프 비율 분포](march_2025_cluster_sharpe_ratio_boxplot.png)

![승패 비율 분포](march_2025_cluster_win_loss_ratio_boxplot.png)

![최대 거래 비중 분포](march_2025_cluster_max_trade_proportion_boxplot.png)

### 2차원 분포 분석

**기대 수익률 vs 최대 거래 비중**
![ROI vs 최대 거래 비중](march_2025_roi_vs_max_trade_scatter.png)

이 그래프에서 잭팟 추구형 투자자(군집 1)는 낮은 기대 수익률과 높은 최대 거래 비중을 보이며 뚜렷하게 구분됩니다.

**샤프 비율 vs 승패 비율**
![샤프 비율 vs 승패 비율](march_2025_sharpe_vs_winloss_scatter.png)

이 그래프에서 각 군집의 위험 대비 수익 특성과 승패 패턴의 관계를 확인할 수 있습니다.

## 4. 결론 및 인사이트

### 주요 발견사항
1. **군집 분포**: 2025년 3월 데이터에서 각 군집의 분포는 위와 같이 나타났으며, 이는 기존 분석과 비교하여 [유사점/차이점]을 보입니다.
2. **잭팟 추구형 비율**: 잭팟 추구형 투자자(군집 1)는 전체의 [비율]%를 차지합니다.
3. **미분류 비율**: 기존 정의된 군집에 맞지 않는 투자자는 [미분류 비율]%로 나타났습니다.

### 향후 분석 방향
- 월별 군집 분포 변화 추적을 통한 시장 트렌드 분석
- 군집별 행동 패턴의 시간에 따른 변화 연구
"""
    
    # 보고서 저장
    with open(f"{RESULT_PATH}march_2025_cluster_report.md", "w", encoding="utf-8") as f:
        f.write(report_content)
    
    print(f"\n보고서가 저장되었습니다: {RESULT_PATH}march_2025_cluster_report.md")

# 메인 함수
def main():
    print("25년 3월 밈코인 투자자 군집 분석 시작")
    
    # 1. 데이터 로드
    data_path = "/Users/johnny/Desktop/flipside_experiment/cluster_time_series/query/query_result/monthly_cluster_analysis_25_03.csv"
    data = load_data(data_path)
    
    # 2. 군집 분류
    classified_data = classify_wallets(data)
    
    # 3. 군집별 통계 계산
    cluster_stats = calculate_cluster_stats(classified_data)
    print("\n군집별 통계:")
    print(cluster_stats)
    
    # 4. 결과 저장
    classified_data.to_csv(f"{RESULT_PATH}march_2025_classified_data.csv", index=False)
    cluster_stats.to_csv(f"{RESULT_PATH}march_2025_cluster_stats.csv")
    
    # 5. 시각화 생성
    create_visualizations(classified_data, cluster_stats)
    
    # 6. 보고서 생성
    create_report(classified_data, cluster_stats)
    
    print("\n25년 3월 밈코인 투자자 군집 분석 완료")

if __name__ == "__main__":
    main() 