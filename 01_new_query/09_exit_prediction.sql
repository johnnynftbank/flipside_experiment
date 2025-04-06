-- 이탈 예측 모델을 위한 데이터 준비
WITH roi_exit_data AS (
  -- 06_user_roi_and_exit.sql의 결과를 사용한다고 가정
  -- 실제 환경에서는 임시 테이블 또는 이전 쿼리 결과를 활용
  SELECT 
    wallet_address,
    overall_roi,
    avg_token_roi,
    median_token_roi,
    worst_token_roi,
    best_token_roi,
    unique_tokens_traded,
    total_investment,
    total_returns,
    
    exit_score,
    exit_status_time,
    activity_decline_rate,
    days_since_last_trade,
    
    first_meme_trade,
    last_meme_trade,
    total_activity_period,
    active_trading_days,
    total_meme_trades,
    first_half_trades,
    second_half_trades
  FROM 
    (
      -- 06_user_roi_and_exit.sql 쿼리 결과를 가정
      -- 실제 실행 시 이 부분은 06_user_roi_and_exit.sql의 결과를 사용
      SELECT * FROM table(result_scan(last_query_id()))
    )
),

-- 이탈 여부 정의 (이진 분류를 위한 타겟 변수)
-- 이탈: 마지막 거래로부터 30일 이상 경과 또는 활동 감소율 50% 이상
exit_labels AS (
  SELECT 
    wallet_address,
    CASE
      WHEN days_since_last_trade >= 30 OR 
           (activity_decline_rate IS NOT NULL AND activity_decline_rate >= 0.5)
      THEN 1 -- 이탈
      ELSE 0 -- 활성
    END as has_exited,
    
    exit_score,
    exit_status_time,
    days_since_last_trade,
    activity_decline_rate
  FROM 
    roi_exit_data
),

-- 특성(Feature) 엔지니어링
prediction_features AS (
  SELECT 
    r.wallet_address,
    r.overall_roi,
    r.avg_token_roi,
    r.median_token_roi,
    r.worst_token_roi,
    r.best_token_roi,
    
    -- 수익률 관련 파생 특성
    CASE WHEN r.overall_roi < 0 THEN 1 ELSE 0 END as is_losing_wallet,
    CASE WHEN r.worst_token_roi < -0.9 THEN 1 ELSE 0 END as has_total_loss,
    CASE 
      WHEN r.overall_roi <= -0.5 THEN 'High Loss'
      WHEN r.overall_roi < 0 THEN 'Loss'
      WHEN r.overall_roi = 0 THEN 'Breakeven'
      WHEN r.overall_roi <= 1 THEN 'Profit'
      ELSE 'High Profit'
    END as roi_category,
    
    -- 거래 활동 관련 특성
    r.unique_tokens_traded,
    r.total_investment,
    r.total_returns,
    r.total_activity_period,
    r.active_trading_days,
    r.total_meme_trades,
    
    -- 활동 패턴 특성
    r.active_trading_days::FLOAT / NULLIF(r.total_activity_period, 0) as activity_density,
    r.total_meme_trades::FLOAT / NULLIF(r.active_trading_days, 0) as trades_per_active_day,
    r.first_half_trades,
    r.second_half_trades,
    r.activity_decline_rate,
    
    -- 시간 관련 특성
    DATEDIFF('day', r.first_meme_trade, CURRENT_DATE()) as days_since_first_trade,
    r.days_since_last_trade,
    DATE_PART('month', r.first_meme_trade) as start_month,
    DATE_PART('dayofweek', r.first_meme_trade) as start_day_of_week,
    
    -- 타겟 변수
    e.has_exited
  FROM 
    roi_exit_data r
    JOIN exit_labels e ON r.wallet_address = e.wallet_address
),

-- 주요 특성별 이탈율 분석
feature_exit_rates AS (
  -- ROI 카테고리별 이탈율
  SELECT 
    'ROI_Category' as feature_type,
    roi_category as feature_value,
    COUNT(*) as wallet_count,
    SUM(has_exited) as exit_count,
    SUM(has_exited)::FLOAT / COUNT(*) as exit_rate,
    AVG(overall_roi) as avg_roi,
    AVG(days_since_last_trade) as avg_days_inactive
  FROM 
    prediction_features
  GROUP BY 
    roi_category
  
  UNION ALL
  
  -- 거래한 토큰 수에 따른 이탈율
  SELECT 
    'Unique_Tokens' as feature_type,
    CASE 
      WHEN unique_tokens_traded = 1 THEN '1 Token'
      WHEN unique_tokens_traded = 2 THEN '2 Tokens'
      WHEN unique_tokens_traded <= 5 THEN '3-5 Tokens'
      WHEN unique_tokens_traded <= 10 THEN '6-10 Tokens'
      ELSE '10+ Tokens'
    END as feature_value,
    COUNT(*) as wallet_count,
    SUM(has_exited) as exit_count,
    SUM(has_exited)::FLOAT / COUNT(*) as exit_rate,
    AVG(overall_roi) as avg_roi,
    AVG(days_since_last_trade) as avg_days_inactive
  FROM 
    prediction_features
  GROUP BY 
    feature_value
  
  UNION ALL
  
  -- 활동 기간에 따른 이탈율
  SELECT 
    'Activity_Period' as feature_type,
    CASE 
      WHEN total_activity_period <= 7 THEN '1 Week or Less'
      WHEN total_activity_period <= 30 THEN '1 Week - 1 Month'
      WHEN total_activity_period <= 90 THEN '1-3 Months'
      WHEN total_activity_period <= 180 THEN '3-6 Months'
      ELSE 'Over 6 Months'
    END as feature_value,
    COUNT(*) as wallet_count,
    SUM(has_exited) as exit_count,
    SUM(has_exited)::FLOAT / COUNT(*) as exit_rate,
    AVG(overall_roi) as avg_roi,
    AVG(days_since_last_trade) as avg_days_inactive
  FROM 
    prediction_features
  GROUP BY 
    feature_value
),

-- 전체 데이터셋 통계
dataset_stats AS (
  SELECT 
    'Dataset Statistics' as stat_type,
    COUNT(*) as total_wallets,
    SUM(has_exited) as total_exits,
    SUM(has_exited)::FLOAT / COUNT(*) as overall_exit_rate,
    AVG(overall_roi) as avg_overall_roi,
    CORR(overall_roi, has_exited::FLOAT) as roi_exit_correlation,
    CORR(worst_token_roi, has_exited::FLOAT) as worst_roi_exit_correlation
  FROM 
    prediction_features
)

-- 최종 결과: 예측 모델 학습용 데이터 및 분석 결과
SELECT 
  'Prediction Dataset' as result_type,
  wallet_address as id,
  overall_roi,
  avg_token_roi,
  median_token_roi,
  worst_token_roi,
  best_token_roi,
  is_losing_wallet,
  has_total_loss,
  roi_category,
  unique_tokens_traded,
  total_investment,
  total_returns,
  total_activity_period,
  active_trading_days,
  total_meme_trades,
  activity_density,
  trades_per_active_day,
  activity_decline_rate,
  days_since_first_trade,
  days_since_last_trade,
  start_month,
  start_day_of_week,
  has_exited
FROM 
  prediction_features

UNION ALL

-- 특성별 이탈율 요약
SELECT 
  'Feature Exit Rates' as result_type,
  feature_type || ': ' || feature_value as id,
  exit_rate as overall_roi,
  wallet_count as avg_token_roi,
  exit_count as median_token_roi,
  avg_roi as worst_token_roi,
  avg_days_inactive as best_token_roi,
  NULL as is_losing_wallet,
  NULL as has_total_loss,
  NULL as roi_category,
  NULL as unique_tokens_traded,
  NULL as total_investment,
  NULL as total_returns,
  NULL as total_activity_period,
  NULL as active_trading_days,
  NULL as total_meme_trades,
  NULL as activity_density,
  NULL as trades_per_active_day,
  NULL as activity_decline_rate,
  NULL as days_since_first_trade,
  NULL as days_since_last_trade,
  NULL as start_month,
  NULL as start_day_of_week,
  NULL as has_exited
FROM 
  feature_exit_rates

UNION ALL

-- 데이터셋 통계 요약
SELECT 
  'Dataset Statistics' as result_type,
  'Dataset Overall Stats' as id,
  avg_overall_roi as overall_roi,
  roi_exit_correlation as avg_token_roi,
  worst_roi_exit_correlation as median_token_roi,
  total_wallets as worst_token_roi,
  total_exits as best_token_roi,
  overall_exit_rate as is_losing_wallet,
  NULL as has_total_loss,
  NULL as roi_category,
  NULL as unique_tokens_traded,
  NULL as total_investment,
  NULL as total_returns,
  NULL as total_activity_period,
  NULL as active_trading_days,
  NULL as total_meme_trades,
  NULL as activity_density,
  NULL as trades_per_active_day,
  NULL as activity_decline_rate,
  NULL as days_since_first_trade,
  NULL as days_since_last_trade,
  NULL as start_month,
  NULL as start_day_of_week,
  NULL as has_exited
FROM 
  dataset_stats; 