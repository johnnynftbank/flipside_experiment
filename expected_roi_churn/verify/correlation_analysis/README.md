# 기대 수익률과 이탈 지갑 간의 상관 관계 분석

## 개요
이 분석은 `meme_coin_roi_churn_3.csv` 데이터를 기반으로 이탈 지갑과 기대 수익률 간의 상관 관계를 파악하는 것을 목적으로 합니다.

## 분석 방법
다음과 같은 통계적 방법을 사용하여 분석을 진행했습니다:

1. **포인트 바이시리얼 상관계수(Point-Biserial Correlation)** - 이진 변수(이탈 여부)와 연속 변수(기대 수익률) 간의 상관 관계를 측정
2. **독립표본 t-검정(Independent t-test)** - 활성 지갑과 이탈 지갑 간의 기대 수익률에 유의미한 차이가 있는지 검정

## 데이터 설명
- 데이터 위치: `expected_roi_churn_final/query_result/meme_coin_roi_churn_3.csv`
- 주요 변수:
  - `SWAPPER`: 지갑 주소
  - `EXPECTED_ROI`: 기대 수익률
  - `WALLET_STATUS`: 지갑 상태 (ACTIVE/CHURNED)
  - 기타 보조 변수: `UNIQUE_TOKENS_TRADED`, `MEME_TRADE_COUNT`, `LAST_MEME_DATE`, `TRADED_DAYS`

## 파일 구조
- `roi_churn_correlation.py`: 상관 분석 수행 스크립트
- `results/`: 분석 결과 폴더
  - `correlation_analysis_summary.txt`: 분석 결과 요약
  - `roi_wallet_status_boxplot.png`: 지갑 상태별 기대 수익률 분포 박스 플롯
  - `roi_wallet_status_histogram.png`: 지갑 상태별 기대 수익률 분포 히스토그램

## 실행 방법
다음 명령어로 분석을 실행할 수 있습니다:
```bash
cd /path/to/flipside_experiment
python3 verify/correlation_analysis/roi_churn_correlation.py
```

## 결과 해석
분석 결과는 `results/correlation_analysis_summary.txt` 파일에 요약되어 있습니다. 주요 결과는 다음과 같습니다:

1. 포인트 바이시리얼 상관계수: 기대 수익률과 이탈 지갑 간의 상관관계를 측정
2. t-검정 결과: 활성 지갑과 이탈 지갑 사이의 기대 수익률 차이에 대한 통계적 유의성 확인
3. 평균 기대 수익률 비교: 활성 지갑과 이탈 지갑의 평균 기대 수익률 비교 