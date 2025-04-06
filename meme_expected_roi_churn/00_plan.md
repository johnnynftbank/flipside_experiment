# 밈코인 기대수익률과 이탈 관계 분석 계획

## 연구 개요

### 분석 목적
밈코인 거래자의 경험적 기대수익률과 시장 이탈 간의 상관관계를 분석하여, 수익률이 낮은 사용자가 밈코인 시장에서 더 많이 이탈하는지 검증한다.

### 연구 가설
"기대 수익률이 낮은 사람일수록 밈코인 시장에서 더 많이 이탈했을 것이다."

### 이탈 정의
- **명확한 이탈 기준**: 2025년 3월 1일부터 밈코인 거래 내역이 없는 지갑을 이탈 지갑으로 정의

## 분석 단계

### 1단계: 활성 지갑 샘플 추출 (01_active_wallets.sql)
- **목표**: 2025년 2월 중 중간 규모 거래(10-1000건)가 있는 지갑 10,000개 무작위 추출
- **작업 내용**:
  - pump.fun에서 2025년 2월 거래 데이터 필터링 
  - 거래 건수 10-1000건 사이인 지갑만 포함
  - RANDOM() 함수를 사용해 10,000개 지갑 무작위 추출
  - 결과 저장: 지갑 주소 및 2월 거래 수

### 2단계: 전체 거래 내역 수집 (02_historical_transactions.sql)
- **목표**: 선정된 10,000개 지갑의 전체 기간 거래 내역 추출
- **작업 내용**:
  - 선정된 지갑들의 모든 시점 pump.fun 거래 데이터 추출
  - 구매(SOL→밈코인) 및 판매(밈코인→SOL) 거래 구분
  - 지갑별 거래 기간, 거래 토큰 수, 거래 패턴 등 요약 정보 계산
  - 결과 저장: 거래 상세 내역 및 지갑별 요약 통계

### 3단계: 일회성 지갑 필터링 (03_filter_disposable.sql)
- **목표**: 밈코인 거래 기간이 충분한 지갑만 선별
- **작업 내용**:
  - 밈코인 거래 기간(첫 거래일~마지막 거래일)이 2일 이상인 지갑만 선택
  - 일회성 또는 단기 투기 목적 지갑 제외
  - 유의미한 행동 패턴 분석이 가능한 지갑만 유지
  - 결과 저장: 필터링된 지갑 목록 및 거래 특성

### 4단계: 이탈/활성 지갑 분류 (04_user_activity_status.sql)
- **목표**: 명확한 이탈 기준(2025년 3월 1일 이후 거래 없음)에 따라 활성/이탈 구분
- **작업 내용**:
  - 2025년 3월 1일 이후 pump.fun 거래 유무 확인
  - 활성 지갑과 이탈 지갑으로 명확히 구분 (binary classification)
  - 각 그룹별 지갑 수 및 비율 계산
  - 결과 저장: 지갑 ID, 활성/이탈 상태, 마지막 거래일, 활동 통계

### 5단계: 경험적 기대수익률 계산 (05_calculate_expected_roi.sql)
- **목표**: 각 지갑의 경험적 기대수익률 계산
- **작업 내용**:
  1. 지갑별, 토큰별 투자 및 회수 금액 계산
     ```sql
     token_total_sol_spent = SUM(CASE WHEN trade_type = 'MEME_BUY' THEN sol_amount ELSE 0 END)
     token_total_sol_received = SUM(CASE WHEN trade_type = 'MEME_SELL' THEN sol_amount ELSE 0 END)
     ```
  
  2. 토큰별 ROI 계산
     ```sql
     token_roi = (token_total_sol_received - token_total_sol_spent) / token_total_sol_spent
     ```
  
  3. ROI 값 구간화 및 확률 분포 계산
     ```sql
     -- ROI 구간 정의
     CASE
       WHEN token_roi <= -0.9 THEN '-90%~-100%'
       WHEN token_roi <= -0.8 THEN '-80%~-89%'
       ...
       WHEN token_roi <= 1.0 THEN '90%~100%'
       WHEN token_roi <= 2.0 THEN '100%~200%'
       WHEN token_roi <= 5.0 THEN '200%~500%'
       ELSE 'Over 500%'
     END as roi_bucket
     
     -- 지갑별 구간 확률 계산
     tokens_in_bucket / total_tokens_traded as probability
     ```
  
  4. 각 구간의 실제 평균 수익률과 확률을 곱한 가중 평균으로 기대수익률 계산
     ```sql
     WITH token_roi_buckets AS (
       SELECT
         wallet_address,
         CASE
           WHEN token_roi <= -0.9 THEN '-90%~-100%'
           WHEN token_roi <= -0.8 THEN '-80%~-89%'
           ...
           WHEN token_roi <= 1.0 THEN '90%~100%'
           WHEN token_roi <= 2.0 THEN '100%~200%'
           WHEN token_roi <= 5.0 THEN '200%~500%'
           ELSE 'Over 500%'
         END as roi_bucket,
         token_roi
       FROM wallet_token_roi
     ),
     bucket_stats AS (
       SELECT
         wallet_address,
         roi_bucket,
         AVG(token_roi) as avg_bucket_roi, -- 구간별 실제 평균 수익률
         COUNT(*) / SUM(COUNT(*)) OVER (PARTITION BY wallet_address) as bucket_probability
       FROM token_roi_buckets
       GROUP BY wallet_address, roi_bucket
     )
     SELECT
       wallet_address,
       SUM(avg_bucket_roi * bucket_probability) as expected_roi -- 기대수익률 계산
     FROM bucket_stats
     GROUP BY wallet_address
     ```
  
  5. 지갑별 실현 ROI와 기대 ROI 모두 저장
     ```sql
     -- 실현 ROI
     (total_sol_received - total_sol_spent) / total_sol_spent as realized_roi
     
     -- 경험적 기대수익률
     expected_roi
     ```

### 6단계: 수익률-이탈 상관관계 분석 (06_roi_exit_correlation.sql)
- **목표**: 경험적 기대수익률과 이탈 여부 간의 관계 분석
- **작업 내용**:
  - 활성/이탈 그룹별 기대수익률 통계(평균, 중앙값, 분포) 비교
  - 기대수익률 구간별 이탈율 계산
  - 기대수익률과 이탈 여부의 상관계수(피어슨, 스피어만) 계산
  - 통계적 유의성 검정(t-test, 카이제곱 등)
  - 결과 저장: 상관관계 분석 결과 및 통계치

### 7단계: 심층 분석 및 시각화 (07_python_analysis.py)
- **목표**: 경험적 기대수익률과 이탈의 관계에 대한 심층 분석 및 시각화
- **작업 내용**:
  - SQL 결과를 pandas DataFrame으로 로드
  - 기대수익률 구간별 이탈율 시각화(막대 그래프, 박스 플롯)
  - 다양한 요인을 통제한 다변량 분석(회귀 분석)
  - 토큰 다양성, 거래 규모 등 추가 변수의 영향 분석
  - 시각화 결과 저장: 그래프 및 분석 보고서

### 8단계: 종합 결론 도출 및 보고서 작성 (08_final_report.md)
- **목표**: 분석 결과 정리 및 인사이트 도출
- **작업 내용**:
  - 각 단계별 분석 결과 종합
  - 가설 "기대 수익률이 낮은 사람일수록 이탈했을 것이다" 검증
  - 통계적 유의성 및 효과 크기 평가
  - 최종 보고서 작성(Markdown 문서)
  - 추가 연구 방향 및 제한점 제시

## 경험적 기대수익률 계산 설명

### 기본 개념
경험적 기대수익률은 각 지갑이 거래한 고유 토큰들의 수익률 분포를 기반으로 계산된 확률적 기대값입니다. 
이는 단순 실현 수익률과 달리, 투자자의 투자 패턴에서 도출된 확률 분포를 반영합니다.

### 계산 예시
예를 들어, 한 지갑이 10개 토큰을 거래했고, 수익률이 다음과 같다면:

1. 구간별 실제 평균 수익률과 확률:
   - 2개 토큰: -92%, -95% → -90%~-100% 구간
     - 구간 평균 ROI = (-0.92 + (-0.95)) / 2 = -0.935 (-93.5%)
     - 구간 확률 = 2/10 = 0.2 (20%)
   - 3개 토큰: -35%, -42%, -46% → -30%~-50% 구간
     - 구간 평균 ROI = (-0.35 + (-0.42) + (-0.46)) / 3 = -0.41 (-41%)
     - 구간 확률 = 3/10 = 0.3 (30%)
   - 4개 토큰: +25%, +28%, +31%, +34% → +20%~+40% 구간
     - 구간 평균 ROI = (0.25 + 0.28 + 0.31 + 0.34) / 4 = 0.295 (29.5%)
     - 구간 확률 = 4/10 = 0.4 (40%)
   - 1개 토큰: +180% → +100%~+200% 구간
     - 구간 평균 ROI = 1.80 (180%)
     - 구간 확률 = 1/10 = 0.1 (10%)

2. 기대수익률 계산:
   - (-0.935 × 0.2) + (-0.41 × 0.3) + (0.295 × 0.4) + (1.80 × 0.1)
   - = -0.187 + (-0.123) + 0.118 + 0.18
   - = -0.012 = -1.2%

이 지갑의 경험적 기대수익률은 -1.2%로, 각 구간의 실제 평균 수익률과 해당 구간 확률을 사용하여 계산됩니다.

### 장점
1. 투자자의 토큰 선택 패턴에서 도출된 확률 분포 활용
2. 각 구간의 실제 평균 수익률을 사용하여 더 정확한 기대값 계산
3. 극단적 수익/손실의 빈도와 실제 크기를 고려한 현실적인 기대값
4. 이탈과의 상관관계 분석에 더 유용한 예측 지표

## 기술적 고려사항
1. **SQL 최적화**: 대용량 데이터 처리를 위한 쿼리 최적화
2. **중간 결과 저장**: 각 단계별 결과는 임시 테이블 또는 파일로 저장
3. **버전 관리**: 모든 쿼리와 분석 코드는 experiment/churn-and-expected-returns 브랜치에서 관리
4. **테스트**: 중요 단계마다 소규모 샘플로 테스트 후 전체 데이터에 적용