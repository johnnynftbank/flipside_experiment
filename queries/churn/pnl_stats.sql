-- 1단계: 2025년 2월에 거래한 지갑 샘플링 (10000개로 제한)
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

-- 2단계: 식별된 지갑의 전체 거래 데이터 가져오기
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

-- 3단계: 토큰별 거래 요약 및 손익 계산
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

-- 4단계: 지갑별 총 PnL 계산
wallet_summary AS (
  SELECT
    wallet_address,
    SUM(profit_loss_sol) AS total_pnl
  FROM token_summary
  GROUP BY wallet_address
)

-- 최종 결과: PnL 분포 통계
SELECT
  -- 기본 통계량
  COUNT(*) as total_wallets,
  ROUND(AVG(total_pnl), 4) as mean_pnl,
  ROUND(MEDIAN(total_pnl), 4) as median_pnl,
  ROUND(STDDEV(total_pnl), 4) as stddev_pnl,
  ROUND(MIN(total_pnl), 4) as min_pnl,
  ROUND(MAX(total_pnl), 4) as max_pnl,
  
  -- 주요 percentile 값들
  ROUND(PERCENTILE_CONT(0.01) WITHIN GROUP (ORDER BY total_pnl), 4) as p1,
  ROUND(PERCENTILE_CONT(0.05) WITHIN GROUP (ORDER BY total_pnl), 4) as p5,
  ROUND(PERCENTILE_CONT(0.10) WITHIN GROUP (ORDER BY total_pnl), 4) as p10,
  ROUND(PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY total_pnl), 4) as p25,
  ROUND(PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY total_pnl), 4) as p75,
  ROUND(PERCENTILE_CONT(0.90) WITHIN GROUP (ORDER BY total_pnl), 4) as p90,
  ROUND(PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY total_pnl), 4) as p95,
  ROUND(PERCENTILE_CONT(0.99) WITHIN GROUP (ORDER BY total_pnl), 4) as p99,

  -- 구간별 지갑 수 분포
  COUNT(CASE WHEN total_pnl < 0 THEN 1 END) as loss_wallets,
  COUNT(CASE WHEN total_pnl = 0 THEN 1 END) as breakeven_wallets,
  COUNT(CASE WHEN total_pnl > 0 THEN 1 END) as profit_wallets
FROM wallet_summary; 