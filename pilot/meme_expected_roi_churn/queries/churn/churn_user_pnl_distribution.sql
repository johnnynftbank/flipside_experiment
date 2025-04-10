-- 이탈 사용자의 손익 분포 분석 쿼리
WITH sampled_wallets AS (
  SELECT 
    swapper AS wallet_address,
    COUNT(*) AS trade_count,
    MAX(block_timestamp) AS last_trade,
    CASE 
      WHEN COUNT(*) BETWEEN 10 AND 30 THEN 'low'
      WHEN COUNT(*) BETWEEN 31 AND 100 THEN 'medium'
      ELSE 'high'
    END AS activity_level
  FROM solana.defi.fact_swaps
  WHERE block_timestamp >= '2025-02-01'
  AND swap_program = 'pump.fun'
  GROUP BY swapper
  HAVING trade_count BETWEEN 10 AND 1000
  ORDER BY RANDOM()
  LIMIT 10000
),

-- 거래 데이터 추출
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
  JOIN sampled_wallets s ON fs.swapper = s.wallet_address
  WHERE fs.swap_program = 'pump.fun'
  AND fs.block_timestamp >= '2025-02-01'
),

-- 토큰별 손익 계산
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
    SUM(CASE WHEN trade_type = 'BUY' THEN send_amount ELSE 0 END) AS profit_loss_sol,
    MIN(CASE WHEN trade_type = 'BUY' THEN block_timestamp ELSE NULL END) AS first_buy,
    MAX(CASE WHEN trade_type = 'SELL' THEN block_timestamp ELSE NULL END) AS last_sell
  FROM trade_data
  WHERE trade_type IN ('BUY', 'SELL')
  GROUP BY wallet_address, 
    CASE
      WHEN trade_type = 'BUY' THEN receive_token
      WHEN trade_type = 'SELL' THEN send_token
      ELSE NULL
    END
),

-- 지갑별 총 손익 및 이탈 여부 계산
wallet_summary AS (
  SELECT
    ts.wallet_address,
    SUM(ts.profit_loss_sol) AS total_pnl,
    -- 이탈 여부: 마지막 거래로부터 30일 이상 지났으면 이탈
    CASE
      WHEN DATEDIFF('day', MAX(COALESCE(ts.last_sell, ts.first_buy)), CURRENT_DATE()) > 30 
      THEN TRUE
      ELSE FALSE
    END AS is_churned
  FROM token_summary ts
  GROUP BY ts.wallet_address
),

-- 이탈 사용자의 손익 구간별 분류
churned_user_pnl_distribution AS (
  SELECT
    CASE
      WHEN total_pnl < -3.35 THEN '< -3.35 SOL (하위 10%)'
      WHEN total_pnl < -0.95 THEN '-3.35 ~ -0.95 SOL (하위 10-25%)'
      WHEN total_pnl < -0.19 THEN '-0.95 ~ -0.19 SOL (하위 25-50%)'
      WHEN total_pnl < 0 THEN '-0.19 ~ 0 SOL (하위 50-75%)'
      WHEN total_pnl = 0 THEN '0 SOL (손익 없음)'
      WHEN total_pnl <= 1.11 THEN '0 ~ 1.11 SOL (상위 75-90%)'
      ELSE '> 1.11 SOL (상위 10%)'
    END AS pnl_category,
    COUNT(*) AS wallet_count
  FROM wallet_summary
  WHERE is_churned = TRUE  -- 이탈 사용자만 선택
  GROUP BY 
    CASE
      WHEN total_pnl < -3.35 THEN '< -3.35 SOL (하위 10%)'
      WHEN total_pnl < -0.95 THEN '-3.35 ~ -0.95 SOL (하위 10-25%)'
      WHEN total_pnl < -0.19 THEN '-0.95 ~ -0.19 SOL (하위 25-50%)'
      WHEN total_pnl < 0 THEN '-0.19 ~ 0 SOL (하위 50-75%)'
      WHEN total_pnl = 0 THEN '0 SOL (손익 없음)'
      WHEN total_pnl <= 1.11 THEN '0 ~ 1.11 SOL (상위 75-90%)'
      ELSE '> 1.11 SOL (상위 10%)'
    END
)

-- 최종 결과: 이탈 사용자의 손익 분포 및 비율 계산
SELECT
  pnl_category,
  wallet_count,
  wallet_count / SUM(wallet_count) OVER () AS percentage,
  SUM(wallet_count) OVER (ORDER BY 
    CASE 
      WHEN pnl_category = '< -3.35 SOL (하위 10%)' THEN 1
      WHEN pnl_category = '-3.35 ~ -0.95 SOL (하위 10-25%)' THEN 2
      WHEN pnl_category = '-0.95 ~ -0.19 SOL (하위 25-50%)' THEN 3
      WHEN pnl_category = '-0.19 ~ 0 SOL (하위 50-75%)' THEN 4
      WHEN pnl_category = '0 SOL (손익 없음)' THEN 5
      WHEN pnl_category = '0 ~ 1.11 SOL (상위 75-90%)' THEN 6
      ELSE 7
    END
  ) / SUM(wallet_count) OVER () AS cumulative_percentage
FROM churned_user_pnl_distribution
ORDER BY 
  CASE 
    WHEN pnl_category = '< -3.35 SOL (하위 10%)' THEN 1
    WHEN pnl_category = '-3.35 ~ -0.95 SOL (하위 10-25%)' THEN 2
    WHEN pnl_category = '-0.95 ~ -0.19 SOL (하위 25-50%)' THEN 3
    WHEN pnl_category = '-0.19 ~ 0 SOL (하위 50-75%)' THEN 4
    WHEN pnl_category = '0 SOL (손익 없음)' THEN 5
    WHEN pnl_category = '0 ~ 1.11 SOL (상위 75-90%)' THEN 6
    ELSE 7
  END