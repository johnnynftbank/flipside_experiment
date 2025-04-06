-- 시간에 따른 수익률과 이탈 패턴 분석
WITH roi_exit_data AS (
  -- 06_user_roi_and_exit.sql의 결과를 사용한다고 가정
  -- 실제 환경에서는 임시 테이블 또는 이전 쿼리 결과를 활용
  SELECT 
    wallet_address,
    overall_roi,
    avg_token_roi,
    median_token_roi,
    exit_score,
    exit_status_time,
    activity_decline_rate,
    days_since_last_trade,
    total_investment,
    unique_tokens_traded,
    first_meme_trade,
    last_meme_trade,
    total_activity_period,
    active_trading_days
  FROM 
    (
      -- 06_user_roi_and_exit.sql 쿼리의 일부분으로 가정
      -- 실제 실행 시 이 부분은 06_user_roi_and_exit.sql의 전체 쿼리로 대체
      -- 또는 임시 테이블에서 데이터를 가져옴
      SELECT 
        r.*,
        e.first_meme_trade,
        e.last_meme_trade,
        e.total_activity_period,
        e.active_trading_days
      FROM 
        wallet_overall_roi r
        JOIN wallet_activity_pattern e ON r.wallet_address = e.wallet_address
    )
),

-- 월별 첫 거래 시작 지갑들의 수익률 및 이탈율
monthly_cohort_analysis AS (
  SELECT 
    DATE_TRUNC('month', first_meme_trade) as cohort_month,
    COUNT(*) as wallets_in_cohort,
    AVG(overall_roi) as avg_cohort_roi,
    MEDIAN(overall_roi) as median_cohort_roi,
    COUNT(CASE WHEN exit_status_time IN ('Exited', 'Likely_Exited') THEN 1 END)::FLOAT / COUNT(*) as cohort_exit_rate,
    AVG(exit_score) as avg_exit_score,
    AVG(activity_decline_rate) as avg_activity_decline,
    AVG(total_activity_period) as avg_activity_period,
    AVG(active_trading_days) as avg_active_days
  FROM 
    roi_exit_data
  GROUP BY 
    cohort_month
  HAVING 
    wallets_in_cohort >= 10  -- 최소 10개 이상의 지갑이 있는 코호트만 포함
  ORDER BY 
    cohort_month
),

-- 코호트별 수익률 분포
cohort_roi_distribution AS (
  SELECT 
    DATE_TRUNC('month', first_meme_trade) as cohort_month,
    CASE 
      WHEN overall_roi < 0 THEN 'Negative ROI'
      WHEN overall_roi = 0 THEN 'Zero ROI'
      WHEN overall_roi <= 1.0 THEN 'Positive ROI (0-100%)'
      WHEN overall_roi <= 5.0 THEN 'High ROI (100-500%)'
      ELSE 'Very High ROI (500%+)'
    END as roi_group,
    COUNT(*) as wallets_count,
    COUNT(*) / SUM(COUNT(*)) OVER (PARTITION BY DATE_TRUNC('month', first_meme_trade)) as roi_group_pct
  FROM 
    roi_exit_data
  GROUP BY 
    cohort_month,
    roi_group
  ORDER BY 
    cohort_month,
    CASE 
      WHEN roi_group = 'Negative ROI' THEN 1
      WHEN roi_group = 'Zero ROI' THEN 2
      WHEN roi_group = 'Positive ROI (0-100%)' THEN 3
      WHEN roi_group = 'High ROI (100-500%)' THEN 4
      ELSE 5
    END
),

-- 활동 기간에 따른 수익률 분석
activity_period_roi AS (
  SELECT 
    CASE 
      WHEN total_activity_period <= 7 THEN '1 Week or Less'
      WHEN total_activity_period <= 30 THEN '1 Week - 1 Month'
      WHEN total_activity_period <= 90 THEN '1-3 Months'
      WHEN total_activity_period <= 180 THEN '3-6 Months'
      ELSE 'Over 6 Months'
    END as activity_period_group,
    COUNT(*) as wallets_count,
    AVG(overall_roi) as avg_roi,
    MEDIAN(overall_roi) as median_roi,
    COUNT(CASE WHEN overall_roi >= 0 THEN 1 END)::FLOAT / COUNT(*) as profit_ratio,
    AVG(exit_score) as avg_exit_score,
    COUNT(CASE WHEN exit_status_time IN ('Exited', 'Likely_Exited') THEN 1 END)::FLOAT / COUNT(*) as exit_rate
  FROM 
    roi_exit_data
  GROUP BY 
    activity_period_group
  ORDER BY 
    MIN(total_activity_period)
),

-- 최근 활동 상태별 수익률 분석
recent_activity_roi AS (
  SELECT 
    exit_status_time,
    COUNT(*) as wallets_count,
    AVG(overall_roi) as avg_roi,
    MEDIAN(overall_roi) as median_roi,
    COUNT(CASE WHEN overall_roi >= 0 THEN 1 END)::FLOAT / COUNT(*) as profit_ratio,
    AVG(activity_decline_rate) as avg_activity_decline
  FROM 
    roi_exit_data
  GROUP BY 
    exit_status_time
  ORDER BY 
    CASE 
      WHEN exit_status_time = 'Active' THEN 1
      WHEN exit_status_time = 'Possible_Exit' THEN 2
      WHEN exit_status_time = 'Likely_Exited' THEN 3
      ELSE 4
    END
)

-- 최종 결과 출력
SELECT 
  'Monthly Cohort Analysis' as analysis_type,
  cohort_month::VARCHAR as dimension,
  wallets_in_cohort as count,
  avg_cohort_roi as value_1,
  median_cohort_roi as value_2,
  cohort_exit_rate as value_3,
  avg_exit_score as value_4,
  avg_activity_decline as value_5
FROM 
  monthly_cohort_analysis

UNION ALL

SELECT 
  'Cohort ROI Distribution' as analysis_type,
  cohort_month::VARCHAR || ' - ' || roi_group as dimension,
  wallets_count as count,
  roi_group_pct as value_1,
  NULL as value_2,
  NULL as value_3,
  NULL as value_4,
  NULL as value_5
FROM 
  cohort_roi_distribution

UNION ALL

SELECT 
  'Activity Period ROI Analysis' as analysis_type,
  activity_period_group as dimension,
  wallets_count as count,
  avg_roi as value_1,
  median_roi as value_2,
  profit_ratio as value_3,
  avg_exit_score as value_4,
  exit_rate as value_5
FROM 
  activity_period_roi

UNION ALL

SELECT 
  'Recent Activity Status ROI' as analysis_type,
  exit_status_time as dimension,
  wallets_count as count,
  avg_roi as value_1,
  median_roi as value_2,
  profit_ratio as value_3,
  avg_activity_decline as value_4,
  NULL as value_5
FROM 
  recent_activity_roi

ORDER BY 
  analysis_type,
  dimension; 