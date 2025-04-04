WITH target_wallets AS (
  -- 2025년 2월 이후에 거래한 지갑 중 이탈한 지갑 식별
  SELECT DISTINCT
    swapper AS wallet_address
  FROM solana.defi.fact_swaps
  WHERE block_timestamp >= '2025-02-01'
    AND swap_program = 'pump.fun'
  GROUP BY swapper
  HAVING COUNT(*) BETWEEN 10 AND 1000
    AND DATEDIFF('day', MAX(block_timestamp), CURRENT_DATE()) > 30
),

user_activity AS (
  -- 식별된 지갑의 전체 거래 내역 분석 (2월 이전 포함)
  SELECT 
    s.swapper AS wallet_address,
    MIN(s.block_timestamp) AS first_trade_time,
    MAX(s.block_timestamp) AS last_trade_time,
    COUNT(DISTINCT CASE
      WHEN s.swap_from_mint = 'So11111111111111111111111111111111111111112' THEN s.swap_to_mint
      WHEN s.swap_to_mint = 'So11111111111111111111111111111111111111112' THEN s.swap_from_mint
    END) AS unique_tokens_traded,
    COUNT(*) AS total_trades
  FROM solana.defi.fact_swaps s
  JOIN target_wallets w ON s.swapper = w.wallet_address
  WHERE swap_program = 'pump.fun'
  GROUP BY s.swapper
),

duration_stats AS (
  -- 활동 기간을 다양한 단위로 계산
  SELECT
    wallet_address,
    unique_tokens_traded,
    total_trades,
    first_trade_time,
    last_trade_time,
    DATEDIFF('day', first_trade_time, last_trade_time) AS days_active,
    DATEDIFF('hour', first_trade_time, last_trade_time) AS hours_active,
    DATEDIFF('minute', first_trade_time, last_trade_time) AS minutes_active
  FROM user_activity
),

duration_distribution AS (
  -- 활동 기간 분포 계산
  SELECT
    CASE
      WHEN days_active < 1 THEN '1일 미만'
      WHEN days_active < 7 THEN '1-7일'
      WHEN days_active < 30 THEN '7-30일'
      WHEN days_active < 90 THEN '30-90일'
      ELSE '90일 이상'
    END AS duration_category,
    COUNT(*) AS wallet_count,
    AVG(unique_tokens_traded) AS avg_tokens_traded,
    MEDIAN(unique_tokens_traded) AS median_tokens_traded,
    AVG(total_trades) AS avg_total_trades,
    MEDIAN(total_trades) AS median_total_trades,
    MIN(days_active) AS min_days,
    MAX(days_active) AS max_days,
    AVG(days_active) AS avg_days
  FROM duration_stats
  GROUP BY 1
),

token_distribution AS (
  -- 거래 토큰 수 분포 계산
  SELECT
    CASE
      WHEN unique_tokens_traded <= 5 THEN '1-5개'
      WHEN unique_tokens_traded <= 10 THEN '6-10개'
      WHEN unique_tokens_traded <= 20 THEN '11-20개'
      WHEN unique_tokens_traded <= 50 THEN '21-50개'
      ELSE '50개 이상'
    END AS token_count_category,
    COUNT(*) AS wallet_count,
    AVG(days_active) AS avg_days_active,
    MEDIAN(days_active) AS median_days_active,
    AVG(total_trades) AS avg_trades,
    MEDIAN(total_trades) AS median_trades
  FROM duration_stats
  GROUP BY 1
)

-- 최종 결과 출력
SELECT 'Duration Distribution' AS analysis_type, 
       duration_category AS category,
       wallet_count,
       avg_tokens_traded,
       median_tokens_traded,
       avg_total_trades,
       median_total_trades,
       min_days,
       max_days,
       avg_days
FROM duration_distribution
UNION ALL
SELECT 'Token Count Distribution' AS analysis_type,
       token_count_category AS category,
       wallet_count,
       NULL AS avg_tokens_traded,
       NULL AS median_tokens_traded,
       avg_trades AS avg_total_trades,
       median_trades AS median_total_trades,
       avg_days_active AS min_days,
       median_days_active AS max_days,
       NULL AS avg_days
FROM token_distribution
ORDER BY analysis_type, category; 