import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

# 데이터 로드
df = pd.read_csv('../data/results/chain_meme_volume.csv')

# 결과 디렉토리 생성
output_dir = Path('../data/analysis/images')
output_dir.mkdir(parents=True, exist_ok=True)

# 스타일 설정
colors = ['#FF9999', '#66B2FF', '#99FF99', '#FFCC99', '#FF99CC', '#99FFCC', '#FFB366', '#FF99FF']

# 1. 체인별 총 거래량 파이 차트
def create_volume_pie_chart():
    volume_by_chain = df.groupby('blockchain')['volumeUSD'].sum().sort_values(ascending=False)
    total_volume = volume_by_chain.sum()
    
    plt.figure(figsize=(12, 8))
    plt.pie(volume_by_chain, labels=[f'{chain}\n({value/total_volume:.1%})' for chain, value in volume_by_chain.items()],
            autopct='%1.1f%%', colors=colors)
    plt.title('체인별 거래량 분포')
    plt.savefig(output_dir / 'volume_distribution.png', bbox_inches='tight', dpi=300)
    plt.close()

# 2. 일일 거래량 추이
def create_daily_volume_trend():
    # 상위 4개 체인 선택
    top_chains = df.groupby('blockchain')['volumeUSD'].sum().nlargest(4).index
    daily_volume = df[df['blockchain'].isin(top_chains)].pivot_table(
        index='block_date', columns='blockchain', values='volumeUSD'
    )
    
    plt.figure(figsize=(15, 8))
    for chain in top_chains:
        plt.plot(daily_volume.index, daily_volume[chain], label=chain, marker='o', markersize=4)
    
    plt.title('체인별 일일 거래량 추이')
    plt.xlabel('날짜')
    plt.ylabel('거래량 (USD)')
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid(True)
    plt.ticklabel_format(style='plain', axis='y')
    plt.savefig(output_dir / 'daily_volume_trend.png', bbox_inches='tight', dpi=300)
    plt.close()

# 3. 시계열 특성 시각화
def create_time_series_analysis():
    # 상위 3개 체인 선택
    top_chains = ['Solana', 'Base', 'BSC']
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))
    
    # 거래량 추이
    for chain in top_chains:
        chain_data = df[df['blockchain'] == chain]
        ax1.plot(chain_data['block_date'], chain_data['volumeUSD'], 
                label=chain, marker='o', markersize=4)
    
    ax1.set_title('거래량 추이')
    ax1.set_xlabel('날짜')
    ax1.set_ylabel('거래량 (USD)')
    ax1.legend()
    ax1.grid(True)
    ax1.tick_params(axis='x', rotation=45)
    
    # 신규 사용자 추이
    for chain in top_chains:
        chain_data = df[df['blockchain'] == chain]
        ax2.plot(chain_data['block_date'], chain_data['numberOfNewUsers'],
                label=chain, marker='o', markersize=4)
    
    ax2.set_title('신규 사용자 추이')
    ax2.set_xlabel('날짜')
    ax2.set_ylabel('신규 사용자 수')
    ax2.legend()
    ax2.grid(True)
    ax2.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'time_series_analysis.png', bbox_inches='tight', dpi=300)
    plt.close()

if __name__ == "__main__":
    create_volume_pie_chart()
    create_daily_volume_trend()
    create_time_series_analysis() 