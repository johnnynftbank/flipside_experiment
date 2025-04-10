WITH feb_active_wallets AS (
  SELECT 
    swapper AS wallet_address,
    COUNT(*) AS feb_trade_count,
    MAX(block_timestamp) AS last_feb_trade
  FROM solana.defi.fact_swaps
  WHERE block_timestamp >= '2025-02-01'
    AND block_timestamp < '2025-03-01'
    AND swap_program = 'pump.fun'
  GROUP BY swapper
  HAVING feb_trade_count BETWEEN 10 AND 1000
  ORDER BY RANDOM()
  LIMIT 10000
),

-- 기존 CTE들은 동일하게 유지...
trade_data AS (
  SELECT
    s.wallet_address,
    fs.tx_id,
    fs.block_timestamp,
    fs.swap_from_mint AS send_token,
    fs.swap_to_mint AS receive_token,
    fs.swap_from_amount AS send_amount,
    fs.swap_to_amount AS receive_amount,
    CASE
      WHEN fs.swap_from_mint = 'So11111111111111111111111111111111111111112' THEN 'BUY'
      WHEN fs.swap_to_mint = 'So11111111111111111111111111111111111111112' THEN 'SELL'
      ELSE 'OTHER'
    END AS trade_type
  FROM solana.defi.fact_swaps fs
  JOIN feb_active_wallets s ON fs.swapper = s.wallet_address
  WHERE fs.swap_program = 'pump.fun'
),

token_summary AS (
  SELECT
    wallet_address,
    CASE
      WHEN trade_type = 'BUY' THEN receive_token
      WHEN trade_type = 'SELL' THEN send_token
      ELSE NULL
    END AS traded_token,
    SUM(CASE WHEN trade_type = 'BUY' THEN send_amount ELSE 0 END) AS total_buy_sol,
    SUM(CASE WHEN trade_type = 'BUY' THEN receive_amount ELSE 0 END) AS total_buy_tokens,
    SUM(CASE WHEN trade_type = 'SELL' THEN receive_amount ELSE 0 END) AS total_sell_sol,
    SUM(CASE WHEN trade_type = 'SELL' THEN send_amount ELSE 0 END) AS total_sell_tokens,
    SUM(CASE WHEN trade_type = 'SELL' THEN receive_amount ELSE 0 END) - 
    SUM(CASE WHEN trade_type = 'BUY' THEN send_amount ELSE 0 END) AS profit_loss_sol
  FROM trade_data
  WHERE trade_type IN ('BUY', 'SELL')
  GROUP BY wallet_address, 
    CASE
      WHEN trade_type = 'BUY' THEN receive_token
      WHEN trade_type = 'SELL' THEN send_token
      ELSE NULL
    END
),

wallet_summary AS (
  SELECT
    ts.wallet_address,
    COUNT(DISTINCT ts.traded_token) AS unique_tokens,
    SUM(ts.profit_loss_sol) AS total_pnl,
    COUNT(*) AS total_trades,
    CASE
      WHEN NOT EXISTS (
        SELECT 1 
        FROM solana.defi.fact_swaps fs 
        WHERE fs.swapper = ts.wallet_address 
          AND fs.swap_program = 'pump.fun'
          AND fs.block_timestamp >= '2025-03-01'
      )
      THEN TRUE
      ELSE FALSE
    END AS is_churned
  FROM token_summary ts
  GROUP BY ts.wallet_address
)

-- 통계 분석 결과
SELECT 
  -- PnL 통계
  MIN(total_pnl) as min_pnl,
  MAX(total_pnl) as max_pnl,
  AVG(total_pnl) as avg_pnl,
  MEDIAN(total_pnl) as median_pnl,
  STDDEV(total_pnl) as stddev_pnl,
  PERCENTILE_CONT(0.10) WITHIN GROUP (ORDER BY total_pnl) as pnl_p10,
  PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY total_pnl) as pnl_p25,
  PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY total_pnl) as pnl_p75,
  PERCENTILE_CONT(0.90) WITHIN GROUP (ORDER BY total_pnl) as pnl_p90,
  
  -- 토큰 수 통계
  MIN(unique_tokens) as min_tokens,
  MAX(unique_tokens) as max_tokens,
  AVG(unique_tokens) as avg_tokens,
  MEDIAN(unique_tokens) as median_tokens,
  STDDEV(unique_tokens) as stddev_tokens,
  PERCENTILE_CONT(0.10) WITHIN GROUP (ORDER BY unique_tokens) as tokens_p10,
  PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY unique_tokens) as tokens_p25,
  PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY unique_tokens) as tokens_p75,
  PERCENTILE_CONT(0.90) WITHIN GROUP (ORDER BY unique_tokens) as tokens_p90
FROM wallet_summary
WHERE is_churned = TRUE;