-- 목적: active_days가 10일 이하인 지갑들의 분포 분석
-- 입력: 02_historical_transactions.sql의 결과
-- 출력: active_days별 지갑 수 집계

WITH target_wallets AS (
  -- 첫 번째 쿼리의 결과를 재현
  SELECT 
    SWAPPER as wallet_address,
    COUNT(*) as feb_trade_count
  FROM 
    solana.defi.fact_swaps
  WHERE 
    SUCCEEDED = true
    AND BLOCK_TIMESTAMP >= '2025-02-01'
    AND BLOCK_TIMESTAMP < '2025-03-01'
    AND swap_program = 'pump.fun'  -- pump.fun 거래만 포함
  GROUP BY 
    SWAPPER
  HAVING 
    feb_trade_count >= 10 
    AND feb_trade_count < 1000
  ORDER BY 
    RANDOM()
  LIMIT 10000
),

wallet_all_trades AS (
  -- 선정된 지갑들의 전체 거래 내역
  SELECT 
    s.*,
    w.feb_trade_count,
    CASE
      WHEN s.swap_program = 'pump.fun' AND s.swap_from_mint = 'So11111111111111111111111111111111111111112' THEN 'MEME_BUY'
      WHEN s.swap_program = 'pump.fun' AND s.swap_to_mint = 'So11111111111111111111111111111111111111112' THEN 'MEME_SELL'
      ELSE 'OTHER'
    END AS trade_type
  FROM 
    solana.defi.fact_swaps s
    INNER JOIN target_wallets w ON s.SWAPPER = w.wallet_address
  WHERE 
    s.SUCCEEDED = true
),

wallet_summary AS (
  -- 지갑별 거래 기간 및 거래 패턴 요약
  SELECT 
    SWAPPER as wallet_address,
    COUNT(*) as total_trade_count,
    COUNT(CASE WHEN trade_type IN ('MEME_BUY', 'MEME_SELL') THEN 1 END) as meme_trade_count,
    COUNT(DISTINCT DATE_TRUNC('day', BLOCK_TIMESTAMP)) as active_days,
    COUNT(DISTINCT CASE WHEN trade_type IN ('MEME_BUY', 'MEME_SELL') THEN DATE_TRUNC('day', BLOCK_TIMESTAMP) END) as meme_active_days
  FROM 
    wallet_all_trades
  GROUP BY 
    SWAPPER
)

-- 밈코인 거래 활동일 수 기준 분포
SELECT 
  meme_active_days as active_days,
  COUNT(*) as wallet_count,
  ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 2) as percentage
FROM 
  wallet_summary
WHERE 
  meme_active_days <= 10
GROUP BY 
  meme_active_days
ORDER BY 
  meme_active_days; 