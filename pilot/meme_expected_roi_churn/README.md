# 밈코인 기대 ROI와 이탈률 분석 프로젝트

이 프로젝트는 밈코인 트레이더의 기대수익률(Expected ROI)과 이탈률 간의 관계를 분석하기 위한 SQL 쿼리 및 Python 분석 코드를 포함합니다.

## 프로젝트 목적

이 프로젝트의 주요 목적은 다음과 같습니다:

1. 트레이더의 기대수익률과 이탈 행동 간의 상관관계 탐색
2. 특정 수익률 임계값 이상/이하에서 이탈률 변화 패턴 분석
3. 트레이더 이탈 예측을 위한 통계적 모델 개발

## 디렉토리 구조

```
meme_expected_roi_churn/
├── 01_active_wallets.sql             # 활성 지갑 샘플 추출
├── 02_historical_transactions.sql     # 샘플 지갑의 거래내역 수집
├── 03_filter_disposable.sql          # 일회성 지갑 필터링
├── 04_user_activity_status.sql       # 활성/이탈 지갑 분류
├── 05_empirical_expected_returns.sql  # 실증적 기대수익률 계산
├── 06_correlation_analysis.sql       # 기대수익률과 이탈률 상관관계 분석
├── python/                           # Python 분석 스크립트
│   └── analyze_correlations.py       # 상관관계 분석 및 시각화
├── results/                          # 분석 결과 저장 디렉토리
└── README.md                         # 프로젝트 설명
```

## SQL 쿼리 실행 순서

SQL 쿼리는 다음 순서로 실행해야 합니다:

1. **01_active_wallets.sql**: 2025년 2월 거래 데이터에서 10,000개의 중간 규모 거래 지갑 추출
2. **02_historical_transactions.sql**: 선택된 지갑의 모든 거래 내역 수집 및 밈코인 거래 분류
3. **03_filter_disposable.sql**: 2일 이상 거래 활동이 있는 지갑만 필터링
4. **04_user_activity_status.sql**: 2025년 3월 1일 이후 거래가 없는 지갑을 '이탈'로 분류
5. **05_empirical_expected_returns.sql**: 각 토큰의 수익률 및 실제 평균 수익률 기반 기대수익률 계산
6. **06_correlation_analysis.sql**: 기대수익률과 이탈률 간의 상관관계 분석

## Python 분석 실행 방법

Python 분석을 실행하려면 다음 단계를 따르세요:
```bash
pip install pandas numpy matplotlib seaborn scipy statsmodels
```

2. SQL 쿼리 결과를 CSV 파일로 내보내고 다음 경로에 저장:
```
meme_expected_roi_churn/results/correlation_data.csv
```

3. 분석 스크립트 실행:
```bash
cd meme_expected_roi_churn
python python/analyze_correlations.py
```

4. 결과 확인:
분석 결과는 `results` 디렉토리에 저장됩니다:
- 상관관계 보고서: `correlation_analysis_report.md`
- 시각화 그래프: `.png` 파일들

## 결과 해석

분석 결과 보고서는 다음 섹션을 포함합니다:
1. 데이터 개요
2. 상관관계 분석 결과
3. 로지스틱 회귀 분석 결과
4. 결론 및 해석

특히, 이탈 확률이 50%가 되는 기대수익률 임계값을 찾는 것이 중요합니다. 이 임계값은 트레이더가 시장에 계속 참여할지 이탈할지 결정하는 중요한 지표일 수 있습니다.

## 주의사항

- 이 분석은 가상의 데이터 모델을 기반으로 합니다. 실제 분석에서는 적절한 데이터 소스를 사용해야 합니다.
- 이탈 행동에는 기대수익률 외에도 다양한 요소가 영향을 미칠 수 있습니다.
- 분석 결과는 해당 데이터셋 및 시간 범위에 한정됩니다. 