-- Analysis of distinct swap programs in Solana DEX
-- Shows the distribution and usage statistics of different swap programs

WITH swap_program_stats AS (
  SELECT 
    swap_program,
    COUNT(*) as swap_count,
    COUNT(DISTINCT swapper) as unique_swappers,
    COUNT(*) * 100.0 / SUM(COUNT(*)) OVER () as percentage_of_total_swaps,
    MIN(block_timestamp) as first_swap_seen,
    MAX(block_timestamp) as last_swap_seen
  FROM solana.defi.fact_swaps
  WHERE block_timestamp >= DATEADD(day, -30, CURRENT_TIMESTAMP())  -- 최근 30일 데이터
  GROUP BY swap_program
  ORDER BY swap_count DESC
)
SELECT 
  swap_program,
  swap_count,
  unique_swappers,
  ROUND(percentage_of_total_swaps, 2) as percentage_of_total_swaps,
  first_swap_seen,
  last_swap_seen,
  RPAD('█', FLOOR(percentage_of_total_swaps/2)::INT, '█') as distribution_viz
FROM swap_program_stats; 