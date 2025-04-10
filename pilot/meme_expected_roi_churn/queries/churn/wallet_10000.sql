SELECT swapper, COUNT(*) AS trade_count, MAX(block_timestamp) AS last_trade
FROM solana.defi.fact_swaps
WHERE block_timestamp >= '2025-02-01'
AND swap_program = 'pump.fun'
GROUP BY swapper
HAVING trade_count BETWEEN 10 AND 1000
ORDER BY RANDOM()
LIMIT 10000
