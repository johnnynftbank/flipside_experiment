-- 목적: active_days가 1-10일인 지갑들의 경험적 기대수익률 분포 분석
-- 입력: solana.defi.fact_swaps 테이블 (pump.fun 거래만)
-- 출력: active_days별 지갑들의 경험적 기대수익률

WITH target_wallets AS (
  -- 2월 거래 기준 대상 지갑 추출
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

wallet_all_trades AS (
  -- 선정된 지갑들의 전체 거래 내역과 수익률 구간 분류
  SELECT 
    s.SWAPPER,
    s.BLOCK_TIMESTAMP,
    s.swap_from_amount,
    s.swap_to_amount,
    s.swap_from_mint,
    s.swap_to_mint,
    s.swap_program,
    CASE
      WHEN s.swap_program = 'pump.fun' AND s.swap_from_mint = 'So11111111111111111111111111111111111111112' THEN 'MEME_BUY'
      WHEN s.swap_program = 'pump.fun' AND s.swap_to_mint = 'So11111111111111111111111111111111111111112' THEN 'MEME_SELL'
      ELSE 'OTHER'
    END AS trade_type,
    CASE 
      WHEN s.swap_from_amount <= 0 THEN NULL
      ELSE (s.swap_to_amount::decimal / NULLIF(s.swap_from_amount::decimal, 0)) - 1
    END as trade_return,
    CASE
      WHEN s.swap_from_amount <= 0 THEN NULL
      WHEN (s.swap_to_amount::decimal / NULLIF(s.swap_from_amount::decimal, 0)) - 1 <= -1.0 THEN -1.0
      WHEN (s.swap_to_amount::decimal / NULLIF(s.swap_from_amount::decimal, 0)) - 1 <= -0.9 THEN -0.9
      WHEN (s.swap_to_amount::decimal / NULLIF(s.swap_from_amount::decimal, 0)) - 1 <= -0.8 THEN -0.8
      WHEN (s.swap_to_amount::decimal / NULLIF(s.swap_from_amount::decimal, 0)) - 1 <= -0.7 THEN -0.7
      WHEN (s.swap_to_amount::decimal / NULLIF(s.swap_from_amount::decimal, 0)) - 1 <= -0.6 THEN -0.6
      WHEN (s.swap_to_amount::decimal / NULLIF(s.swap_from_amount::decimal, 0)) - 1 <= -0.5 THEN -0.5
      WHEN (s.swap_to_amount::decimal / NULLIF(s.swap_from_amount::decimal, 0)) - 1 <= -0.4 THEN -0.4
      WHEN (s.swap_to_amount::decimal / NULLIF(s.swap_from_amount::decimal, 0)) - 1 <= -0.3 THEN -0.3
      WHEN (s.swap_to_amount::decimal / NULLIF(s.swap_from_amount::decimal, 0)) - 1 <= -0.2 THEN -0.2
      WHEN (s.swap_to_amount::decimal / NULLIF(s.swap_from_amount::decimal, 0)) - 1 <= -0.1 THEN -0.1
      WHEN (s.swap_to_amount::decimal / NULLIF(s.swap_from_amount::decimal, 0)) - 1 <= 0.0 THEN 0.0
      WHEN (s.swap_to_amount::decimal / NULLIF(s.swap_from_amount::decimal, 0)) - 1 <= 0.1 THEN 0.1
      WHEN (s.swap_to_amount::decimal / NULLIF(s.swap_from_amount::decimal, 0)) - 1 <= 0.25 THEN 0.25
      WHEN (s.swap_to_amount::decimal / NULLIF(s.swap_from_amount::decimal, 0)) - 1 <= 0.5 THEN 0.5
      WHEN (s.swap_to_amount::decimal / NULLIF(s.swap_from_amount::decimal, 0)) - 1 <= 1.0 THEN 1.0
      WHEN (s.swap_to_amount::decimal / NULLIF(s.swap_from_amount::decimal, 0)) - 1 <= 2.0 THEN 2.0
      WHEN (s.swap_to_amount::decimal / NULLIF(s.swap_from_amount::decimal, 0)) - 1 <= 5.0 THEN 5.0
      ELSE 10.0
    END as return_bucket
  FROM 
    solana.defi.fact_swaps s
    INNER JOIN target_wallets w ON s.SWAPPER = w.wallet_address
  WHERE 
    s.SUCCEEDED = true
),

wallet_stats AS (
  -- 각 지갑의 거래 통계
  SELECT 
    SWAPPER,
    COUNT(*) as total_trades,
    COUNT(DISTINCT DATE_TRUNC('day', BLOCK_TIMESTAMP)) as total_days,
    COUNT(CASE WHEN swap_from_amount <= 0 THEN 1 END) as invalid_amount_trades,
    COUNT(CASE WHEN trade_type = 'MEME_BUY' THEN 1 END) as meme_buy_count,
    COUNT(CASE WHEN trade_type = 'MEME_SELL' THEN 1 END) as meme_sell_count,
    COUNT(DISTINCT CASE WHEN trade_type IN ('MEME_BUY', 'MEME_SELL') THEN DATE_TRUNC('day', BLOCK_TIMESTAMP) END) as meme_active_days,
    MIN(BLOCK_TIMESTAMP) as first_trade,
    MAX(BLOCK_TIMESTAMP) as last_trade
  FROM 
    wallet_all_trades
  GROUP BY 
    SWAPPER
),

wallet_summary AS (
  -- 지갑별 거래 기간 및 수익률 분포
  SELECT 
    SWAPPER as wallet_address,
    COUNT(DISTINCT CASE WHEN trade_type IN ('MEME_BUY', 'MEME_SELL') THEN DATE_TRUNC('day', BLOCK_TIMESTAMP) END) as meme_active_days,
    AVG(CASE WHEN trade_type = 'MEME_SELL' AND swap_program = 'pump.fun' THEN return_bucket END) as expected_return  -- pump.fun에서의 밈코인 판매 거래의 경험적 기대수익률
  FROM 
    wallet_all_trades
  GROUP BY 
    SWAPPER
  HAVING 
    meme_active_days <= 10
)

-- 단계별 지갑 수 확인
SELECT 
  '1. Initial Sample' as step,
  COUNT(*) as wallet_count
FROM 
  target_wallets

UNION ALL

SELECT 
  '2. Wallet Stats' as step,
  COUNT(*) as wallet_count
FROM 
  wallet_stats

UNION ALL

SELECT 
  '3. Final Summary' as step,
  COUNT(*) as wallet_count
FROM 
  wallet_summary

UNION ALL

-- 최종 결과: 밈코인 활동일수별 수익률 분포
SELECT 
  '4. By Active Days: ' || meme_active_days::varchar as step,
  wallet_count
FROM (
  SELECT 
    meme_active_days,
    COUNT(*) as wallet_count,
    ROUND(AVG(expected_return) * 100, 2) as empirical_expected_return_pct
  FROM 
    wallet_summary
  GROUP BY 
    meme_active_days
  ORDER BY 
    meme_active_days
); 