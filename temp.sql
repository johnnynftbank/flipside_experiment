SELECT DISTINCT swap_program, COUNT(*) as tx_count FROM solana.defi.fact_swaps WHERE BLOCK_TIMESTAMP >= '2024-02-01' AND BLOCK_TIMESTAMP < '2024-03-01' GROUP BY swap_program ORDER BY tx_count DESC LIMIT 10;
