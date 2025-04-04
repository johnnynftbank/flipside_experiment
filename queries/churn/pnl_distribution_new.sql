-- 기존 CTE들 재사용
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
    SUM(CASE WHEN trade_type = 'SELL' THEN receive_amount ELSE 0 END) AS total_sell_sol,
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
    wallet_address,
    SUM(profit_loss_sol) AS total_pnl,
    CASE 
      WHEN NOT EXISTS (
        SELECT 1 
        FROM solana.defi.fact_swaps fs 
        WHERE fs.swapper = wallet_address 
          AND fs.swap_program = 'pump.fun'
          AND fs.block_timestamp >= '2025-03-01'
      )
      THEN TRUE
      ELSE FALSE
    END AS is_churned
  FROM token_summary
  GROUP BY wallet_address
)

-- 실제 데이터 분포 기반 PnL 구간별 분포
SELECT
  is_churned,
  CASE 
    WHEN total_pnl < -92.107 THEN 'EXTREME LOSS (Bottom 1%)'           -- P1 미만
    WHEN total_pnl < -14.9209 THEN 'SEVERE LOSS (P1-P5)'              -- P1-P5
    WHEN total_pnl < -5.8275 THEN 'LARGE LOSS (P5-P10)'               -- P5-P10
    WHEN total_pnl < -1.2885 THEN 'MEDIUM LOSS (P10-P25)'             -- P10-P25
    WHEN total_pnl < 0 THEN 'SMALL LOSS (P25-P50)'                    -- P25-P50
    WHEN total_pnl = 0 THEN 'BREAK EVEN'
    WHEN total_pnl <= 1.0296 THEN 'SMALL PROFIT (P50-P90)'            -- P50-P90
    WHEN total_pnl <= 3.3078 THEN 'MEDIUM PROFIT (P90-P95)'           -- P90-P95
    WHEN total_pnl <= 32.1304 THEN 'LARGE PROFIT (P95-P99)'           -- P95-P99
    ELSE 'EXTREME PROFIT (Top 1%)'                                     -- P99 초과
  END AS pnl_category,
  COUNT(*) AS wallet_count,
  ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (PARTITION BY is_churned), 2) AS percentage,
  ROUND(AVG(total_pnl), 4) AS avg_pnl_in_group,
  ROUND(MEDIAN(total_pnl), 4) AS median_pnl_in_group
FROM wallet_summary
GROUP BY is_churned, pnl_category
ORDER BY 
  is_churned,
  CASE pnl_category
    WHEN 'EXTREME LOSS (Bottom 1%)' THEN 1
    WHEN 'SEVERE LOSS (P1-P5)' THEN 2
    WHEN 'LARGE LOSS (P5-P10)' THEN 3
    WHEN 'MEDIUM LOSS (P10-P25)' THEN 4
    WHEN 'SMALL LOSS (P25-P50)' THEN 5
    WHEN 'BREAK EVEN' THEN 6
    WHEN 'SMALL PROFIT (P50-P90)' THEN 7
    WHEN 'MEDIUM PROFIT (P90-P95)' THEN 8
    WHEN 'LARGE PROFIT (P95-P99)' THEN 9
    WHEN 'EXTREME PROFIT (Top 1%)' THEN 10
  END; 