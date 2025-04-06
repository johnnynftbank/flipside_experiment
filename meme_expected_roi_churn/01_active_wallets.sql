-- 목적: 2025년 2월 한 달 동안 중간 규모의 거래(10-1000건)가 있는 지갑 10,000개를 무작위 추출
-- 입력: solana.defi.fact_swaps (pump.fun 거래만)
-- 출력: 2월 중간 규모 거래 지갑 중 무작위 10,000개 지갑 목록
-- 필터링: 
--   1. 극단적인 거래 패턴(10건 미만 또는 1000건 이상)은 제외
--   2. pump.fun 거래소의 거래만 포함

WITH february_trades AS (
  SELECT 
    SWAPPER,
    COUNT(*) as trade_count
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
    trade_count >= 10 
    AND trade_count < 1000
)
SELECT 
  SWAPPER as wallet_address,
  trade_count
FROM 
  february_trades
ORDER BY 
  RANDOM() -- Snowflake의 RANDOM() 함수를 사용하여 무작위 정렬
LIMIT 10000; 