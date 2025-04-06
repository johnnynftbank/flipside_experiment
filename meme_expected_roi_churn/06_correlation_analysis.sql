-- 목적: 기대수익률과 이탈 여부 간의 상관관계 분석
-- 입력: 05_empirical_expected_returns.sql의 결과
-- 출력: 기대수익률과 이탈률 간의 관계 분석

-- 1. 이전 단계 결과 기반 데이터 준비 (기대수익률 및 이탈 상태)
WITH expected_returns_data AS (
  -- 실제 환경에서는 05_empirical_expected_returns.sql의 결과를 사용
  SELECT 
    wallet_address,
    expected_roi,
    expected_roi_category,
    expected_roi_decile,
    prob_positive_roi,
    prob_negative_roi,
    expected_positive_roi,
    expected_negative_roi,
    roi_std_dev,
    sharpe_ratio,
    total_tokens,
    total_holding,
    has_exited,
    exit_status
  FROM (
    -- 샘플 데이터 (실제 실행 시 05_empirical_expected_returns.sql 결과로 대체)
    -- 이 예시에서는 간단한 모의 데이터 생성
    WITH sample_wallets AS (
      SELECT 
        SWAPPER as wallet_address,
        RANDOM() as random_value
      FROM 
        solana.defi.fact_swaps
      WHERE 
        SUCCEEDED = true
        AND BLOCK_TIMESTAMP >= '2025-02-01'
        AND BLOCK_TIMESTAMP < '2025-03-01'
        AND swap_program = 'pump.fun'
      GROUP BY 
        SWAPPER
      HAVING 
        COUNT(*) >= 10 
        AND COUNT(*) < 1000
      ORDER BY 
        RANDOM()
      LIMIT 10000
    ),
    sample_expected_returns AS (
      SELECT 
        wallet_address,
        -- 예시로 임의의 기대수익률 생성 (-0.8 ~ 2.0 범위)
        (random_value * 2.8) - 0.8 as expected_roi,
        -- 임의의 이탈 여부 생성
        CASE WHEN RANDOM() < 0.4 THEN 1 ELSE 0 END as has_exited,
        CASE WHEN has_exited = 1 THEN 'EXITED' ELSE 'ACTIVE' END as exit_status,
        -- 추가 지표 생성
        random_value * 0.8 as prob_positive_roi,
        1 - (random_value * 0.8) as prob_negative_roi,
        (random_value * 2.8) - 0.8 as expected_positive_roi,
        ((1 - random_value) * 0.5) - 0.5 as expected_negative_roi,
        RANDOM() * 0.5 as roi_std_dev,
        CEILING(RANDOM() * 20) as total_tokens,
        CEILING(RANDOM() * 10) as total_holding
      FROM 
        sample_wallets
    )
    SELECT 
      wallet_address,
      expected_roi,
      CASE
        WHEN expected_roi <= -0.5 THEN 'Very Negative (< -50%)'
        WHEN expected_roi <= -0.2 THEN 'Negative (-50% to -20%)'
        WHEN expected_roi < 0 THEN 'Slightly Negative (-20% to 0%)'
        WHEN expected_roi = 0 THEN 'Zero (0%)'
        WHEN expected_roi < 0.2 THEN 'Slightly Positive (0% to 20%)'
        WHEN expected_roi < 0.5 THEN 'Positive (20% to 50%)'
        WHEN expected_roi < 1 THEN 'Very Positive (50% to 100%)'
        ELSE 'Extremely Positive (> 100%)'
      END as expected_roi_category,
      NTILE(10) OVER (ORDER BY expected_roi) as expected_roi_decile,
      prob_positive_roi,
      prob_negative_roi,
      expected_positive_roi,
      expected_negative_roi,
      roi_std_dev,
      CASE 
        WHEN roi_std_dev > 0 THEN expected_roi / roi_std_dev 
        ELSE NULL 
      END as sharpe_ratio,
      total_tokens,
      total_holding,
      has_exited,
      exit_status
    FROM 
      sample_expected_returns
  )
),

-- 2. 기대수익률 구간별 이탈률 계산
exit_rates_by_expected_roi_category AS (
  SELECT 
    expected_roi_category,
    COUNT(*) as wallet_count,
    SUM(has_exited) as exited_count,
    (SUM(has_exited)::FLOAT / COUNT(*)) as exit_rate,
    AVG(expected_roi) as avg_expected_roi,
    AVG(CASE WHEN has_exited = 1 THEN expected_roi ELSE NULL END) as avg_expected_roi_exited,
    AVG(CASE WHEN has_exited = 0 THEN expected_roi ELSE NULL END) as avg_expected_roi_active,
    -- 위험 조정 수익률 기준
    AVG(sharpe_ratio) as avg_sharpe_ratio,
    AVG(CASE WHEN has_exited = 1 THEN sharpe_ratio ELSE NULL END) as avg_sharpe_ratio_exited,
    AVG(CASE WHEN has_exited = 0 THEN sharpe_ratio ELSE NULL END) as avg_sharpe_ratio_active,
    -- 긍정적 수익 확률
    AVG(prob_positive_roi) as avg_prob_positive_roi,
    AVG(CASE WHEN has_exited = 1 THEN prob_positive_roi ELSE NULL END) as avg_prob_positive_roi_exited,
    AVG(CASE WHEN has_exited = 0 THEN prob_positive_roi ELSE NULL END) as avg_prob_positive_roi_active
  FROM 
    expected_returns_data
  GROUP BY 
    expected_roi_category
  ORDER BY 
    avg_expected_roi
),

-- 3. 기대수익률 데실(십분위수)별 이탈률 계산
exit_rates_by_expected_roi_decile AS (
  SELECT 
    expected_roi_decile,
    COUNT(*) as wallet_count,
    SUM(has_exited) as exited_count,
    (SUM(has_exited)::FLOAT / COUNT(*)) as exit_rate,
    MIN(expected_roi) as min_expected_roi,
    MAX(expected_roi) as max_expected_roi,
    AVG(expected_roi) as avg_expected_roi,
    MEDIAN(expected_roi) as median_expected_roi,
    AVG(CASE WHEN has_exited = 1 THEN expected_roi ELSE NULL END) as avg_expected_roi_exited,
    AVG(CASE WHEN has_exited = 0 THEN expected_roi ELSE NULL END) as avg_expected_roi_active
  FROM 
    expected_returns_data
  GROUP BY 
    expected_roi_decile
  ORDER BY 
    expected_roi_decile
),

-- 4. 상관관계 지표 계산
correlation_metrics AS (
  SELECT 
    -- 피어슨 상관계수 (기대수익률과 이탈 간)
    CORR(expected_roi, has_exited) as roi_exit_correlation,
    
    -- 로지스틱 회귀 유사 계산 (경사도 추정)
    REGR_SLOPE(has_exited, expected_roi) as exit_roi_slope,
    REGR_INTERCEPT(has_exited, expected_roi) as exit_roi_intercept,
    
    -- 위험 조정 수익률과 이탈 간 상관관계
    CORR(sharpe_ratio, has_exited) as sharpe_exit_correlation,
    
    -- 긍정적 수익 확률과 이탈 간 상관관계
    CORR(prob_positive_roi, has_exited) as prob_positive_exit_correlation,
    
    -- 전체 통계
    AVG(expected_roi) as avg_expected_roi,
    AVG(CASE WHEN has_exited = 1 THEN expected_roi ELSE NULL END) as avg_expected_roi_exited,
    AVG(CASE WHEN has_exited = 0 THEN expected_roi ELSE NULL END) as avg_expected_roi_active,
    
    -- 이탈률
    SUM(has_exited) as total_exited,
    COUNT(*) as total_wallets,
    (SUM(has_exited)::FLOAT / COUNT(*)) as overall_exit_rate
  FROM 
    expected_returns_data
),

-- 5. 기대수익률과 이탈률 간의 비선형 관계 탐색
nonlinear_patterns AS (
  SELECT 
    ROUND(expected_roi, 1) as roi_rounded,
    COUNT(*) as wallet_count,
    SUM(has_exited) as exited_count,
    (SUM(has_exited)::FLOAT / COUNT(*)) as exit_rate
  FROM 
    expected_returns_data
  GROUP BY 
    ROUND(expected_roi, 1)
  HAVING 
    COUNT(*) >= 10  -- 샘플 수가 충분한 경우만 고려
  ORDER BY 
    roi_rounded
),

-- 6. 기대수익률의 구성요소별 이탈 분석
component_analysis AS (
  SELECT 
    -- 기대 양수 수익률 그룹화
    CASE
      WHEN expected_positive_roi <= 0 THEN '0% or less'
      WHEN expected_positive_roi <= 0.1 THEN '0% to 10%'
      WHEN expected_positive_roi <= 0.3 THEN '10% to 30%'
      WHEN expected_positive_roi <= 0.5 THEN '30% to 50%'
      WHEN expected_positive_roi <= 1.0 THEN '50% to 100%'
      ELSE 'Over 100%'
    END as positive_roi_group,
    
    -- 기대 음수 수익률 그룹화
    CASE
      WHEN expected_negative_roi >= 0 THEN '0% or more'
      WHEN expected_negative_roi >= -0.1 THEN '-10% to 0%'
      WHEN expected_negative_roi >= -0.3 THEN '-30% to -10%'
      WHEN expected_negative_roi >= -0.5 THEN '-50% to -30%'
      ELSE 'Under -50%'
    END as negative_roi_group,
    
    COUNT(*) as wallet_count,
    SUM(has_exited) as exited_count,
    (SUM(has_exited)::FLOAT / COUNT(*)) as exit_rate,
    AVG(expected_roi) as avg_expected_roi
  FROM 
    expected_returns_data
  GROUP BY 
    positive_roi_group,
    negative_roi_group
  ORDER BY 
    avg_expected_roi
),

-- 7. 이탈 여부에 영향을 미치는 다중 요인 분석
-- (기대수익률, 변동성, 거래 토큰 수 등 종합적 고려)
multi_factor_analysis AS (
  SELECT 
    -- 기대수익률 구간
    CASE
      WHEN expected_roi < 0 THEN 'Negative'
      WHEN expected_roi <= 0.5 THEN 'Moderate'
      ELSE 'High'
    END as roi_group,
    
    -- 변동성 구간
    CASE
      WHEN roi_std_dev <= 0.3 THEN 'Low'
      WHEN roi_std_dev <= 0.6 THEN 'Medium'
      ELSE 'High'
    END as volatility_group,
    
    -- 거래 토큰 수 구간
    CASE
      WHEN total_tokens <= 5 THEN 'Few'
      WHEN total_tokens <= 15 THEN 'Moderate'
      ELSE 'Many'
    END as token_count_group,
    
    COUNT(*) as wallet_count,
    SUM(has_exited) as exited_count,
    (SUM(has_exited)::FLOAT / COUNT(*)) as exit_rate
  FROM 
    expected_returns_data
  GROUP BY 
    roi_group,
    volatility_group,
    token_count_group
  ORDER BY 
    roi_group,
    volatility_group,
    token_count_group
)

-- 최종 결과 출력 (모든 분석 테이블)
-- 1. 상관관계 기본 지표
SELECT 
  'Correlation Metrics' as analysis_type,
  *
FROM 
  correlation_metrics

UNION ALL

-- 2. 기대수익률 카테고리별 이탈률
SELECT 
  'Exit Rates by ROI Category' as analysis_type,
  expected_roi_category as category,
  wallet_count,
  exited_count,
  exit_rate,
  avg_expected_roi,
  avg_expected_roi_exited,
  avg_expected_roi_active,
  avg_sharpe_ratio,
  avg_prob_positive_roi,
  NULL, NULL, NULL, NULL  -- 남은 열 채우기
FROM 
  exit_rates_by_expected_roi_category

UNION ALL

-- 3. 기대수익률 데실별 이탈률
SELECT 
  'Exit Rates by ROI Decile' as analysis_type,
  expected_roi_decile::VARCHAR as category,
  wallet_count,
  exited_count,
  exit_rate,
  avg_expected_roi,
  avg_expected_roi_exited,
  avg_expected_roi_active,
  NULL, NULL, 
  min_expected_roi,
  max_expected_roi,
  median_expected_roi,
  NULL
FROM 
  exit_rates_by_expected_roi_decile

UNION ALL

-- 4. 비선형 패턴 분석
SELECT 
  'Nonlinear Patterns' as analysis_type,
  roi_rounded::VARCHAR as category,
  wallet_count,
  exited_count,
  exit_rate,
  roi_rounded as avg_expected_roi,
  NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL
FROM 
  nonlinear_patterns

ORDER BY 
  analysis_type,
  category; 