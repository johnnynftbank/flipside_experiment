WITH fee_ranges AS (
  SELECT 
    CASE 
      WHEN fee = 5000 THEN '1. 5000 (기본 수수료)'
      WHEN fee BETWEEN 5001 AND 5999 THEN '2. 5001-5999 (기본 수수료 근접)'
      WHEN fee BETWEEN 6000 AND 9999 THEN '3. 6000-9999 (낮은 수수료)'
      WHEN fee BETWEEN 10000 AND 49999 THEN '4. 10000-49999 (중간 수수료)'
      WHEN fee BETWEEN 50000 AND 99999 THEN '5. 50000-99999 (중고 수수료)'
      WHEN fee BETWEEN 100000 AND 999999 THEN '6. 100000-999999 (높은 수수료)'
      WHEN fee >= 1000000 THEN '7. 1000000+ (매우 높은 수수료)'
      ELSE '0. 기타'
    END as fee_range,
    fee,
    COUNT(*) as tx_count,
    COUNT(*) * 100.0 / SUM(COUNT(*)) OVER () as percentage
  FROM solana.core.fact_transactions
  WHERE block_timestamp >= DATEADD(day, -7, CURRENT_TIMESTAMP())
  GROUP BY 1, 2
),
range_summary AS (
  SELECT 
    fee_range,
    SUM(tx_count) as total_tx_count,
    SUM(tx_count) * 100.0 / SUM(SUM(tx_count)) OVER () as range_percentage,
    MIN(fee) as min_fee,
    MAX(fee) as max_fee,
    AVG(fee) as avg_fee,
    COUNT(DISTINCT fee) as unique_fee_values
  FROM fee_ranges
  GROUP BY 1
)
SELECT 
  fee_range,
  total_tx_count,
  ROUND(range_percentage, 2) as percentage,
  min_fee as min_fee_lamports,
  max_fee as max_fee_lamports,
  ROUND(avg_fee, 2) as avg_fee_lamports,
  unique_fee_values,
  ROUND(min_fee/1e9, 9) as min_fee_sol,
  ROUND(max_fee/1e9, 9) as max_fee_sol,
  ROUND(avg_fee/1e9, 9) as avg_fee_sol,
  RPAD('█', FLOOR(range_percentage/2)::INT, '█') as distribution_viz
FROM range_summary
WHERE fee_range != '0. 기타'  -- 정의된 범위 외의 값 제외
ORDER BY fee_range; 