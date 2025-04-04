-- Fee distribution analysis for Solana transactions
-- Shows the distribution of transaction fees over the past 7 days
-- Filters out noise (transactions with count <= 10)
-- Visualizes the distribution using ASCII art

WITH fee_stats AS (
  SELECT 
    fee,
    COUNT(*) as tx_count,
    COUNT(*) * 100.0 / SUM(COUNT(*)) OVER () as percentage
  FROM solana.core.fact_transactions
  WHERE block_timestamp >= DATEADD(day, -7, CURRENT_TIMESTAMP())  -- 최근 7일 데이터만 조회
  GROUP BY fee
  HAVING tx_count > 10  -- 노이즈 제거
  ORDER BY tx_count DESC
  LIMIT 20  -- 상위 20개 fee 값만 조회
)
SELECT 
  fee,
  tx_count,
  ROUND(percentage, 2) as percentage,
  RPAD('█', FLOOR(percentage/2)::INT, '█') as distribution_viz
FROM fee_stats; 