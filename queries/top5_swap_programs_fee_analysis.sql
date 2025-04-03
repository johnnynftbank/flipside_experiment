WITH top_5_programs AS (
  SELECT DISTINCT swap_program
  FROM (
    VALUES 
      ('Raydium Liquidity Pool V4'),
      ('raydium concentrated liquidity'),
      ('pump.fun'),
      ('raydium constant product market maker'),
      ('meteora dlmm pools program')
  ) AS t(swap_program)
),
fee_stats AS (
  SELECT 
    s.swap_program,
    -- Fee 구간 분류 (더 세분화된 구간)
    CASE 
      WHEN s.fee_amount = 0 THEN '0 (No Fee)'
      WHEN s.fee_amount < 0.000005 THEN '< 0.000005 SOL'
      WHEN s.fee_amount = 0.000005 THEN '0.000005 SOL (Base Fee)'
      WHEN s.fee_amount <= 0.00001 THEN '0.000005-0.00001 SOL'
      WHEN s.fee_amount <= 0.00005 THEN '0.00001-0.00005 SOL'
      WHEN s.fee_amount <= 0.0001 THEN '0.00005-0.0001 SOL'
      WHEN s.fee_amount <= 0.0005 THEN '0.0001-0.0005 SOL'
      WHEN s.fee_amount <= 0.001 THEN '0.0005-0.001 SOL'
      WHEN s.fee_amount <= 0.005 THEN '0.001-0.005 SOL'
      WHEN s.fee_amount <= 0.01 THEN '0.005-0.01 SOL'
      WHEN s.fee_amount <= 0.05 THEN '0.01-0.05 SOL'
      ELSE '> 0.05 SOL'
    END as fee_range,
    COUNT(*) as swap_count,
    COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (PARTITION BY s.swap_program) as percentage,
    AVG(s.fee_amount) as avg_fee,
    MIN(s.fee_amount) as min_fee,
    MAX(s.fee_amount) as max_fee
  FROM solana.defi.fact_swaps s
  JOIN top_5_programs t ON s.swap_program = t.swap_program
  WHERE block_timestamp >= DATEADD(day, -30, CURRENT_TIMESTAMP())
  GROUP BY 1, 2
)
SELECT 
  swap_program,
  fee_range,
  swap_count,
  ROUND(percentage, 2) as percentage_in_program,
  ROUND(avg_fee, 9) as avg_fee_sol,  -- 더 많은 소수점 자리 표시
  ROUND(min_fee, 9) as min_fee_sol,
  ROUND(max_fee, 9) as max_fee_sol,
  RPAD('█', FLOOR(percentage/2)::INT, '█') as distribution_viz
FROM fee_stats
ORDER BY 
  swap_program,
  CASE fee_range
    WHEN '0 (No Fee)' THEN 1
    WHEN '< 0.000005 SOL' THEN 2
    WHEN '0.000005 SOL (Base Fee)' THEN 3
    WHEN '0.000005-0.00001 SOL' THEN 4
    WHEN '0.00001-0.00005 SOL' THEN 5
    WHEN '0.00005-0.0001 SOL' THEN 6
    WHEN '0.0001-0.0005 SOL' THEN 7
    WHEN '0.0005-0.001 SOL' THEN 8
    WHEN '0.001-0.005 SOL' THEN 9
    WHEN '0.005-0.01 SOL' THEN 10
    WHEN '0.01-0.05 SOL' THEN 11
    ELSE 12
  END; 