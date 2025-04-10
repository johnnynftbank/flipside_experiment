-- 목적: 선정된 10,000개 지갑의 전체 거래 내역 추출
-- 입력: 
--   1. solana.defi.fact_swaps (pump.fun 거래만)
--   2. 01_active_wallets.sql의 결과 (target_wallets CTE로 재현)
-- 출력: 선정된 지갑들의 전체 기간 거래 내역 및 지갑별 요약 정보
-- 필터링: pump.fun 거래소의 거래만 포함
-- 참고: pump.fun에서 SOL과의 거래를 밈코인 거래로 간주

WITH target_wallets AS (
  -- 첫 번째 쿼리의 결과를 재현
  -- 실제 운영시에는 이 부분을 첫 번째 쿼리 결과로 대체하거나 임시 테이블로 저장하여 사용
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

wallet_tokens AS (
  -- 각 지갑이 거래한 고유 밈코인 토큰 수 계산 (pump.fun에서의 거래만)
  SELECT
    SWAPPER,
    COUNT(DISTINCT CASE 
      WHEN swap_program = 'pump.fun' AND swap_from_mint = 'So11111111111111111111111111111111111111112' THEN swap_to_mint  -- MEME_BUY: 받은 토큰
      WHEN swap_program = 'pump.fun' AND swap_to_mint = 'So11111111111111111111111111111111111111112' THEN swap_from_mint  -- MEME_SELL: 보낸 토큰
    END) as unique_meme_tokens,
    COUNT(DISTINCT CASE 
      WHEN swap_program != 'pump.fun' AND (
        swap_from_mint = 'So11111111111111111111111111111111111111112' OR 
        swap_to_mint = 'So11111111111111111111111111111111111111112'
      ) THEN 
        CASE 
          WHEN swap_from_mint = 'So11111111111111111111111111111111111111112' THEN swap_to_mint
          ELSE swap_from_mint
        END
    END) as unique_other_tokens
  FROM 
    wallet_all_trades
  GROUP BY 
    SWAPPER
),

wallet_summary AS (
  -- 지갑별 거래 기간 및 거래 패턴 요약
  SELECT 
    t.SWAPPER as wallet_address,
    COUNT(*) as total_trade_count,
    COUNT(CASE WHEN trade_type IN ('MEME_BUY', 'MEME_SELL') THEN 1 END) as meme_trade_count,
    COUNT(CASE WHEN trade_type = 'OTHER' THEN 1 END) as other_trade_count,
    MIN(BLOCK_TIMESTAMP) as first_trade_date,
    MAX(BLOCK_TIMESTAMP) as last_trade_date,
    DATEDIFF('day', first_trade_date, last_trade_date) + 1 as wallet_age_days,
    COUNT(DISTINCT DATE_TRUNC('day', BLOCK_TIMESTAMP)) as active_days,
    COUNT(DISTINCT CASE WHEN trade_type IN ('MEME_BUY', 'MEME_SELL') THEN DATE_TRUNC('day', BLOCK_TIMESTAMP) END) as meme_active_days,
    COUNT(CASE WHEN trade_type = 'MEME_BUY' THEN 1 END) as meme_buy_count,
    COUNT(CASE WHEN trade_type = 'MEME_SELL' THEN 1 END) as meme_sell_count,
    tok.unique_meme_tokens,
    tok.unique_other_tokens,
    feb_trade_count
  FROM 
    wallet_all_trades t
    LEFT JOIN wallet_tokens tok ON t.SWAPPER = tok.SWAPPER
  GROUP BY 
    t.SWAPPER,
    tok.unique_meme_tokens,
    tok.unique_other_tokens,
    feb_trade_count
)

-- 최종 출력: 지갑별 요약 정보
SELECT 
  *,
  meme_active_days::FLOAT / wallet_age_days as meme_activity_density,
  meme_trade_count::FLOAT / NULLIF(meme_active_days, 0) as meme_trades_per_active_day,
  meme_sell_count::FLOAT / NULLIF(meme_buy_count, 0) as meme_sell_buy_ratio
FROM 
  wallet_summary
ORDER BY 
  wallet_age_days DESC; 