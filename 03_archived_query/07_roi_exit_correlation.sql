-- 수익률과 이탈의 관계 분석
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
    unique_tokens_traded
  FROM 
    (
      -- 06_user_roi_and_exit.sql 쿼리 결과
      -- 실제 실행 시 이 부분은 06_user_roi_and_exit.sql의 전체 쿼리로 대체
      -- 또는 임시 테이블에서 데이터를 가져옴
      SELECT * FROM table(result_scan(last_query_id()))
    )
),

-- ROI 구간별 그룹화
roi_buckets AS (
  SELECT 
    CASE
      WHEN overall_roi <= -0.9 THEN '-90%~-100%'
      WHEN overall_roi <= -0.8 THEN '-80%~-89%'
      WHEN overall_roi <= -0.7 THEN '-70%~-79%'
      WHEN overall_roi <= -0.6 THEN '-60%~-69%'
      WHEN overall_roi <= -0.5 THEN '-50%~-59%'
      WHEN overall_roi <= -0.4 THEN '-40%~-49%'
      WHEN overall_roi <= -0.3 THEN '-30%~-39%'
      WHEN overall_roi <= -0.2 THEN '-20%~-29%'
      WHEN overall_roi <= -0.1 THEN '-10%~-19%'
      WHEN overall_roi < 0 THEN '0%~-9%'
      WHEN overall_roi = 0 THEN '0%'
      WHEN overall_roi <= 0.1 THEN '0%~10%'
      WHEN overall_roi <= 0.2 THEN '10%~20%'
      WHEN overall_roi <= 0.3 THEN '20%~30%'
      WHEN overall_roi <= 0.4 THEN '30%~40%'
      WHEN overall_roi <= 0.5 THEN '40%~50%'
      WHEN overall_roi <= 0.6 THEN '50%~60%'
      WHEN overall_roi <= 0.7 THEN '60%~70%'
      WHEN overall_roi <= 0.8 THEN '70%~80%'
      WHEN overall_roi <= 0.9 THEN '80%~90%'
      WHEN overall_roi <= 1.0 THEN '90%~100%'
      WHEN overall_roi <= 2.0 THEN '100%~200%'
      WHEN overall_roi <= 3.0 THEN '200%~300%'
      WHEN overall_roi <= 5.0 THEN '300%~500%'
      WHEN overall_roi <= 10.0 THEN '500%~1000%'
      ELSE 'Over 1000%'
    END as roi_bucket,
    wallet_address,
    overall_roi,
    exit_score,
    exit_status_time,
    activity_decline_rate,
    days_since_last_trade
  FROM 
    roi_exit_data
),

-- 각 ROI 구간별 집계
roi_bucket_stats AS (
  SELECT 
    roi_bucket,
    COUNT(*) as wallets_in_bucket,
    
    -- 이탈 관련 통계
    AVG(exit_score) as avg_exit_score,
    MEDIAN(exit_score) as median_exit_score,
    
    -- 이탈 상태별 분포
    COUNT(CASE WHEN exit_status_time = 'Exited' THEN 1 END) as exited_count,
    COUNT(CASE WHEN exit_status_time = 'Likely_Exited' THEN 1 END) as likely_exited_count,
    COUNT(CASE WHEN exit_status_time = 'Possible_Exit' THEN 1 END) as possible_exit_count,
    COUNT(CASE WHEN exit_status_time = 'Active' THEN 1 END) as active_count,
    
    -- 이탈율 계산
    (exited_count + likely_exited_count)::FLOAT / COUNT(*) as exit_rate,
    
    -- 평균 활동 감소율
    AVG(activity_decline_rate) as avg_activity_decline,
    
    -- 마지막 거래 이후 평균 경과일
    AVG(days_since_last_trade) as avg_days_since_last_trade
  FROM 
    roi_buckets
  GROUP BY 
    roi_bucket
),

-- 상관관계 분석을 위한 데이터
correlation_data AS (
  SELECT 
    CORR(overall_roi, exit_score) as roi_exit_correlation,
    CORR(overall_roi, activity_decline_rate) as roi_decline_correlation,
    CORR(overall_roi, days_since_last_trade) as roi_inactive_days_correlation
  FROM 
    roi_exit_data
  WHERE 
    exit_score IS NOT NULL AND
    activity_decline_rate IS NOT NULL
),

-- ROI 그룹별 이탈율
roi_grouped_exit AS (
  SELECT 
    CASE 
      WHEN overall_roi < 0 THEN 'Negative ROI'
      WHEN overall_roi = 0 THEN 'Zero ROI'
      WHEN overall_roi <= 1.0 THEN 'Positive ROI (0-100%)'
      WHEN overall_roi <= 5.0 THEN 'High ROI (100-500%)'
      ELSE 'Very High ROI (500%+)'
    END as roi_group,
    COUNT(*) as wallets_count,
    AVG(exit_score) as avg_exit_score,
    COUNT(CASE WHEN exit_status_time IN ('Exited', 'Likely_Exited') THEN 1 END)::FLOAT / COUNT(*) as exit_rate,
    AVG(activity_decline_rate) as avg_activity_decline
  FROM 
    roi_exit_data
  GROUP BY 
    roi_group
  ORDER BY 
    MIN(overall_roi)
)

-- 상관관계 및 ROI 구간별 이탈율 출력
SELECT 
  'Correlation Analysis' as analysis_type,
  NULL as bucket_name,
  NULL as wallets_count,
  r.roi_exit_correlation as value_1,
  r.roi_decline_correlation as value_2,
  r.roi_inactive_days_correlation as value_3,
  NULL as value_4,
  NULL as value_5
FROM 
  correlation_data r

UNION ALL

SELECT 
  'ROI Group Exit Rates' as analysis_type,
  roi_group as bucket_name,
  wallets_count,
  avg_exit_score as value_1,
  exit_rate as value_2,
  avg_activity_decline as value_3,
  NULL as value_4,
  NULL as value_5
FROM 
  roi_grouped_exit

UNION ALL

SELECT 
  'ROI Bucket Details' as analysis_type,
  roi_bucket as bucket_name,
  wallets_in_bucket as wallets_count,
  avg_exit_score as value_1,
  exit_rate as value_2,
  avg_activity_decline as value_3,
  avg_days_since_last_trade as value_4,
  active_count::FLOAT / wallets_in_bucket as value_5
FROM 
  roi_bucket_stats
ORDER BY 
  analysis_type,
  bucket_name; 