import pandas as pd
import plotly.graph_objects as go

# 데이터 로드
loadings = pd.read_csv('../report/pca_loadings.csv', index_col=0)
pca_data = pd.read_csv('../report/transformed_pca_data_for_kmeans.csv')

# 주성분 설명된 분산 비율 (보고서에서 가져옴)
explained_var = {'PC1': 55.46, 'PC2': 18.37, 'PC3': 14.28}

# 3D 그래프 생성
fig = go.Figure()

# 투자자 상태별 데이터 포인트 추가
colors = {'active': 'blue', 'churned': 'red'}
for status, color in colors.items():
    df_status = pca_data[pca_data['WALLET_STATUS'] == status]
    fig.add_trace(go.Scatter3d(
        x=df_status['PC1'],
        y=df_status['PC2'],
        z=df_status['PC3'],
        mode='markers',
        marker=dict(size=3, color=color, opacity=0.5),
        name=status
    ))

# 변수 로딩 벡터 추가 (스케일 조정)
scale_factor = 5
for i, var in enumerate(loadings.index):
    x, y, z = loadings.iloc[i, 0:3].values
    fig.add_trace(go.Scatter3d(
        x=[0, x*scale_factor],
        y=[0, y*scale_factor],
        z=[0, z*scale_factor],
        mode='lines+markers+text',
        line=dict(color='black', width=5),
        marker=dict(size=5, color='black'),
        name=var,
        text=['', var],
        textposition='top center'
    ))

# 레이아웃 설정
fig.update_layout(
    title='3D PCA Biplot',
    scene=dict(
        xaxis_title=f'PC1 ({explained_var["PC1"]}%)',
        yaxis_title=f'PC2 ({explained_var["PC2"]}%)',
        zaxis_title=f'PC3 ({explained_var["PC3"]}%)'
    ),
    width=1000,
    height=800
)

# HTML 파일로 저장
fig.write_html('../report/pca_3d_biplot.html')
print('3D PCA Biplot을 HTML 파일로 저장했습니다: ../report/pca_3d_biplot.html') 