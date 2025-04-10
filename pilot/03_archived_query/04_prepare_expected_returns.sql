WITH target_wallets AS (
  -- 1단계: 2025년 2월 활성 지갑 10,000개 추출
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
  -- 2단계: 선정된 지갑들의 전체 거래 내역
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

filtered_wallets AS (
  -- 3단계: 일회성 지갑 필터링
  SELECT 
    SWAPPER as wallet_address,
    MIN(CASE WHEN trade_type IN ('MEME_BUY', 'MEME_SELL') THEN BLOCK_TIMESTAMP END) as first_meme_trade,
    MAX(CASE WHEN trade_type IN ('MEME_BUY', 'MEME_SELL') THEN BLOCK_TIMESTAMP END) as last_meme_trade,
    DATEDIFF('day', MIN(CASE WHEN trade_type IN ('MEME_BUY', 'MEME_SELL') THEN BLOCK_TIMESTAMP END),
             MAX(CASE WHEN trade_type IN ('MEME_BUY', 'MEME_SELL') THEN BLOCK_TIMESTAMP END)) + 1 as meme_trading_days
  FROM 
    all_trades
  GROUP BY 
    SWAPPER
  HAVING 
    meme_trading_days >= 2
),

meme_trades_detail AS (
  -- 4단계: 필터링된 지갑들의 밈코인 거래 상세 정보
  SELECT 
    t.SWAPPER as wallet_address,
    t.BLOCK_TIMESTAMP,
    t.trade_type,
    CASE 
      WHEN trade_type = 'MEME_BUY' THEN t.swap_from_amount  -- SOL 지출량
      WHEN trade_type = 'MEME_SELL' THEN t.swap_to_amount   -- SOL 수취량
    END as sol_amount,
    CASE 
      WHEN trade_type = 'MEME_BUY' THEN t.swap_to_amount    -- 밈코인 수취량
      WHEN trade_type = 'MEME_SELL' THEN t.swap_from_amount -- 밈코인 지출량
    END as token_amount,
    CASE 
      WHEN trade_type = 'MEME_BUY' THEN t.swap_to_mint      -- 구매한 밈코인 주소
      WHEN trade_type = 'MEME_SELL' THEN t.swap_from_mint   -- 판매한 밈코인 주소
    END as token_mint,
    ROW_NUMBER() OVER (PARTITION BY t.SWAPPER, CASE 
      WHEN trade_type = 'MEME_BUY' THEN t.swap_to_mint
      WHEN trade_type = 'MEME_SELL' THEN t.swap_from_mint
    END ORDER BY t.BLOCK_TIMESTAMP) as token_trade_sequence
  FROM 
    all_trades t
    INNER JOIN filtered_wallets w ON t.SWAPPER = w.wallet_address
  WHERE 
    trade_type IN ('MEME_BUY', 'MEME_SELL')
)

-- 최종 출력: 기대수익률 계산을 위한 정제된 데이터
SELECT 
  wallet_address,
  token_mint,
  token_trade_sequence,
  BLOCK_TIMESTAMP,
  trade_type,
  sol_amount,
  token_amount,
  SUM(CASE WHEN trade_type = 'MEME_BUY' THEN sol_amount ELSE 0 END) 
    OVER (PARTITION BY wallet_address, token_mint ORDER BY BLOCK_TIMESTAMP) as token_total_sol_spent,
  SUM(CASE WHEN trade_type = 'MEME_BUY' THEN token_amount ELSE 0 END) 
    OVER (PARTITION BY wallet_address, token_mint ORDER BY BLOCK_TIMESTAMP) as token_total_bought,
  SUM(CASE WHEN trade_type = 'MEME_SELL' THEN sol_amount ELSE 0 END) 
    OVER (PARTITION BY wallet_address, token_mint ORDER BY BLOCK_TIMESTAMP) as token_total_sol_received,
  SUM(CASE WHEN trade_type = 'MEME_SELL' THEN token_amount ELSE 0 END) 
    OVER (PARTITION BY wallet_address, token_mint ORDER BY BLOCK_TIMESTAMP) as token_total_sold
FROM 
  meme_trades_detail
ORDER BY 
  wallet_address,
  token_mint,
  BLOCK_TIMESTAMP; 