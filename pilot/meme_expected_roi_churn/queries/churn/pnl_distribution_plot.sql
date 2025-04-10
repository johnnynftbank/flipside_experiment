-- 기본 CTE들은 동일하게 유지
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

-- 1. 전체 지갑의 PnL 분포 데이터
SELECT
  'all' as user_group,
  total_pnl,
  COUNT(*) as wallet_count
FROM wallet_summary
WHERE total_pnl BETWEEN -100 AND 100  -- 극단치 제외
GROUP BY total_pnl
UNION ALL
-- 2. 이탈 사용자의 PnL 분포 데이터
SELECT
  'churned' as user_group,
  total_pnl,
  COUNT(*) as wallet_count
FROM wallet_summary
WHERE is_churned = TRUE
  AND total_pnl BETWEEN -100 AND 100  -- 극단치 제외
GROUP BY total_pnl
UNION ALL
-- 3. 활성 사용자의 PnL 분포 데이터
SELECT
  'active' as user_group,
  total_pnl,
  COUNT(*) as wallet_count
FROM wallet_summary
WHERE is_churned = FALSE
  AND total_pnl BETWEEN -100 AND 100  -- 극단치 제외
GROUP BY total_pnl
ORDER BY user_group, total_pnl; 