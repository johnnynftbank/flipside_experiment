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
    -- Fee 구간 분류 (SOL 단위로 표현)
    CASE 
      WHEN t.fee = 0.000005 THEN '0.000005 SOL (기본 수수료)'
      WHEN t.fee > 0.000005 AND t.fee <= 0.000006 THEN '0.000005-0.000006 SOL (기본 수수료 +20% 이내)'
      WHEN t.fee > 0.000006 AND t.fee <= 0.00001 THEN '0.000006-0.00001 SOL (기본 수수료 2배 이내)'
      WHEN t.fee > 0.00001 AND t.fee <= 0.00005 THEN '0.00001-0.00005 SOL (기본 수수료 2-10배)'
      WHEN t.fee > 0.00005 AND t.fee <= 0.0001 THEN '0.00005-0.0001 SOL (기본 수수료 10-20배)'
      WHEN t.fee > 0.0001 AND t.fee <= 0.001 THEN '0.0001-0.001 SOL (기본 수수료 20-200배)'
      WHEN t.fee > 0.001 THEN '> 0.001 SOL (기본 수수료 200배 초과)'
      ELSE '< 0.000005 SOL (기본 수수료 미만)'
    END as fee_range,
    COUNT(*) as swap_count,
    COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (PARTITION BY s.swap_program) as percentage,
    AVG(t.fee) as avg_fee,
    MIN(t.fee) as min_fee,
    MAX(t.fee) as max_fee
  FROM solana.defi.fact_swaps s
  JOIN solana.core.fact_transactions t ON s.tx_id = t.tx_id
  JOIN top_5_programs p ON s.swap_program = p.swap_program
  WHERE s.block_timestamp >= '2024-03-01' 
    AND s.block_timestamp < '2024-04-01'
  GROUP BY 1, 2
)
SELECT 
  swap_program,
  fee_range,
  swap_count,
  ROUND(percentage, 2) as percentage_in_program,
  ROUND(avg_fee, 9) as avg_fee_sol,
  ROUND(min_fee, 9) as min_fee_sol,
  ROUND(max_fee, 9) as max_fee_sol,
  RPAD('█', FLOOR(percentage/2)::INT, '█') as distribution_viz
FROM fee_stats
ORDER BY 
  swap_program,
  CASE fee_range
    WHEN '< 0.000005 SOL (기본 수수료 미만)' THEN 1
    WHEN '0.000005 SOL (기본 수수료)' THEN 2
    WHEN '0.000005-0.000006 SOL (기본 수수료 +20% 이내)' THEN 3
    WHEN '0.000006-0.00001 SOL (기본 수수료 2배 이내)' THEN 4
    WHEN '0.00001-0.00005 SOL (기본 수수료 2-10배)' THEN 5
    WHEN '0.00005-0.0001 SOL (기본 수수료 10-20배)' THEN 6
    WHEN '0.0001-0.001 SOL (기본 수수료 20-200배)' THEN 7
    WHEN '> 0.001 SOL (기본 수수료 200배 초과)' THEN 8
  END; 