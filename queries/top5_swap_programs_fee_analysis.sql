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
    -- Fee 구간 분류
    CASE 
      WHEN s.fee_amount = 0 THEN '0 (No Fee)'
      WHEN s.fee_amount < 0.001 THEN '< 0.001 SOL'
      WHEN s.fee_amount < 0.01 THEN '0.001 - 0.01 SOL'
      WHEN s.fee_amount < 0.1 THEN '0.01 - 0.1 SOL'
      WHEN s.fee_amount < 1 THEN '0.1 - 1 SOL'
      ELSE '> 1 SOL'
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
  ROUND(avg_fee, 6) as avg_fee_sol,
  ROUND(min_fee, 6) as min_fee_sol,
  ROUND(max_fee, 6) as max_fee_sol,
  RPAD('█', FLOOR(percentage/2)::INT, '█') as distribution_viz
FROM fee_stats
ORDER BY 
  swap_program,
  CASE fee_range
    WHEN '0 (No Fee)' THEN 1
    WHEN '< 0.001 SOL' THEN 2
    WHEN '0.001 - 0.01 SOL' THEN 3
    WHEN '0.01 - 0.1 SOL' THEN 4
    WHEN '0.1 - 1 SOL' THEN 5
    ELSE 6
  END; 