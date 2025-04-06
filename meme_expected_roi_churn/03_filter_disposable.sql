-- 목적: 일회성 지갑 필터링
-- 입력: 02_historical_transactions.sql의 결과
-- 출력: 일회성이 아닌 지갑들의 거래 내역 및 필터링 통계
-- 필터링 기준:
--   1. 밈 코인 거래 기간이 2일 이상인 지갑만 선택
--   2. 첫 밈 코인 거래와 마지막 밈 코인 거래 사이의 간격으로 계산

WITH target_wallets AS (
  -- 2025년 2월 활성 지갑 10,000개 추출
  SELECT 
    SWAPPER as wallet_address,
    COUNT(*) as feb_trade_count
  FROM 
    solana.defi.fact_swaps
  WHERE 
    SUCCEEDED = true
    AND BLOCK_TIMESTAMP >= '2025-02-01'
    AND BLOCK_TIMESTAMP < '2025-03-01'
    AND swap_program = 'pump.fun'
  GROUP BY 
    SWAPPER
  HAVING 
    feb_trade_count >= 10 
    AND feb_trade_count < 1000
  ORDER BY 
    RANDOM()
  LIMIT 10000
),

all_trades AS (
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

meme_trade_periods AS (
  -- 각 지갑의 밈 코인 거래 기간 계산
  SELECT 
    SWAPPER as wallet_address,
    MIN(CASE WHEN trade_type IN ('MEME_BUY', 'MEME_SELL') THEN BLOCK_TIMESTAMP END) as first_meme_trade,
    MAX(CASE WHEN trade_type IN ('MEME_BUY', 'MEME_SELL') THEN BLOCK_TIMESTAMP END) as last_meme_trade,
    DATEDIFF('day', first_meme_trade, last_meme_trade) + 1 as meme_trading_days,
    COUNT(DISTINCT CASE WHEN trade_type IN ('MEME_BUY', 'MEME_SELL') THEN DATE_TRUNC('day', BLOCK_TIMESTAMP) END) as unique_meme_trading_days
  FROM 
    all_trades
  GROUP BY 
    SWAPPER
),

wallet_summary AS (
  -- 지갑별 거래 패턴 요약
  SELECT 
    t.SWAPPER as wallet_address,
    COUNT(*) as total_trade_count,
    COUNT(CASE WHEN trade_type IN ('MEME_BUY', 'MEME_SELL') THEN 1 END) as meme_trade_count,
    COUNT(CASE WHEN trade_type = 'OTHER' THEN 1 END) as other_trade_count,
    p.first_meme_trade,
    p.last_meme_trade,
    p.meme_trading_days,
    p.unique_meme_trading_days,
    COUNT(CASE WHEN trade_type = 'MEME_BUY' THEN 1 END) as meme_buy_count,
    COUNT(CASE WHEN trade_type = 'MEME_SELL' THEN 1 END) as meme_sell_count
  FROM 
    all_trades t
    INNER JOIN meme_trade_periods p ON t.SWAPPER = p.wallet_address
  WHERE 
    p.meme_trading_days >= 2  -- 2일 이상 거래 기간
  GROUP BY 
    t.SWAPPER,
    p.first_meme_trade,
    p.last_meme_trade,
    p.meme_trading_days,
    p.unique_meme_trading_days
)

-- 최종 결과: 필터링된 지갑들의 거래 패턴
SELECT 
  *,
  unique_meme_trading_days::FLOAT / meme_trading_days as meme_activity_density,
  meme_trade_count::FLOAT / unique_meme_trading_days as meme_trades_per_active_day,
  meme_sell_count::FLOAT / NULLIF(meme_buy_count, 0) as meme_sell_buy_ratio
FROM 
  wallet_summary
ORDER BY 
  meme_trading_days DESC; 