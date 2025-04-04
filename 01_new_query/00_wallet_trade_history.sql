WITH target_wallets AS (
  -- 2025년 2월 기준 대상 지갑 추출 (기존과 동일)
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
  LIMIT 10
),

trade_types AS (
  -- 거래 유형별 분류
  SELECT 
    s.BLOCK_TIMESTAMP,
    s.SWAPPER,
    s.swap_from_mint,
    s.swap_to_mint,
    s.swap_from_amount,
    s.swap_to_amount,
    CASE
      WHEN s.swap_from_mint = 'So11111111111111111111111111111111111111112' THEN 'BUY'
      WHEN s.swap_to_mint = 'So11111111111111111111111111111111111111112' THEN 'SELL'
      ELSE 'OTHER'
    END AS trade_type
  FROM 
    solana.defi.fact_swaps s
    INNER JOIN target_wallets w ON s.SWAPPER = w.wallet_address
  WHERE 
    s.SUCCEEDED = true
    AND s.swap_program = 'pump.fun'
)

-- 각 거래 유형별로 동일한 수의 샘플 추출
SELECT * FROM (
  SELECT *, ROW_NUMBER() OVER (PARTITION BY trade_type ORDER BY BLOCK_TIMESTAMP DESC) as rn
  FROM trade_types
)
WHERE rn <= 30  -- 각 유형별 30개씩 표시
ORDER BY 
  trade_type,   -- 거래 유형으로 먼저 정렬
  BLOCK_TIMESTAMP DESC;