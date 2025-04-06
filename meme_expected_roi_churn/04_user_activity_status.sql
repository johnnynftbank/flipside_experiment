-- 목적: 이탈/활성 지갑 분류
-- 입력: 03_filter_disposable.sql의 결과 (2일 이상 활동한 지갑들)
-- 출력: 지갑별 이탈/활성 상태 및 관련 지표
-- 이탈 정의: 2025년 3월 1일 이후 밈코인 거래 내역이 없는 지갑

WITH filtered_wallets AS (
  -- 3단계 필터링된 지갑 목록 (2일 이상 활동 지갑)
  SELECT 
    wallet_address,
    first_meme_trade,
    last_meme_trade,
    meme_trading_days,
    unique_meme_trading_days,
    meme_trade_count,
    meme_buy_count,
    meme_sell_count
  FROM (
    -- 실제 환경에서는 03_filter_disposable.sql 쿼리 결과를 사용
    -- 여기서는 필터링 조건을 직접 구현
    WITH target_wallets AS (
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
    SELECT * FROM wallet_summary
  )
),

-- 3월 1일 이후 밈코인 거래 여부 확인
march_activity AS (
  SELECT 
    fw.wallet_address,
    MAX(CASE WHEN s.BLOCK_TIMESTAMP >= '2025-03-01' AND 
              (
                (s.swap_program = 'pump.fun' AND s.swap_from_mint = 'So11111111111111111111111111111111111111112') OR
                (s.swap_program = 'pump.fun' AND s.swap_to_mint = 'So11111111111111111111111111111111111111112')
              )
              THEN s.BLOCK_TIMESTAMP ELSE NULL END) as last_march_trade_date,
    COUNT(CASE WHEN s.BLOCK_TIMESTAMP >= '2025-03-01' AND 
              (
                (s.swap_program = 'pump.fun' AND s.swap_from_mint = 'So11111111111111111111111111111111111111112') OR
                (s.swap_program = 'pump.fun' AND s.swap_to_mint = 'So11111111111111111111111111111111111111112')
              )
              THEN 1 ELSE NULL END) as march_trade_count
  FROM 
    filtered_wallets fw
    LEFT JOIN solana.defi.fact_swaps s ON fw.wallet_address = s.SWAPPER
  WHERE 
    s.SUCCEEDED = true
  GROUP BY 
    fw.wallet_address
),

-- 최종 이탈/활성 상태 및 지표 계산
wallet_status AS (
  SELECT 
    fw.*,
    ma.last_march_trade_date,
    ma.march_trade_count,
    CASE WHEN ma.march_trade_count > 0 THEN 0 ELSE 1 END as has_exited,
    CASE WHEN ma.march_trade_count > 0 THEN 'ACTIVE' ELSE 'EXITED' END as exit_status
  FROM 
    filtered_wallets fw
    LEFT JOIN march_activity ma ON fw.wallet_address = ma.wallet_address
)

-- 최종 결과: 지갑별 이탈/활성 상태 및 통계
SELECT 
  *,
  -- 추가 지표 계산
  DATEDIFF('day', last_meme_trade, CURRENT_DATE()) as days_since_last_trade,
  
  -- 2월 이전 vs 2월 거래 비교 (활동 추세 분석)
  CASE 
    WHEN first_meme_trade < '2025-02-01' THEN 
      DATEDIFF('day', first_meme_trade, '2025-02-01') 
    ELSE 0 
  END as pre_feb_days,
  
  CASE 
    WHEN first_meme_trade >= '2025-02-01' THEN 0
    ELSE (
      SELECT COUNT(*) 
      FROM solana.defi.fact_swaps 
      WHERE SWAPPER = wallet_status.wallet_address 
      AND SUCCEEDED = true
      AND BLOCK_TIMESTAMP < '2025-02-01'
      AND (
        (swap_program = 'pump.fun' AND swap_from_mint = 'So11111111111111111111111111111111111111112') OR
        (swap_program = 'pump.fun' AND swap_to_mint = 'So11111111111111111111111111111111111111112')
      )
    )
  END as pre_feb_trade_count,
  
  CASE 
    WHEN pre_feb_days > 0 AND pre_feb_trade_count > 0 
    THEN pre_feb_trade_count::FLOAT / pre_feb_days 
    ELSE NULL 
  END as pre_feb_trades_per_day,
  
  CASE 
    WHEN pre_feb_trades_per_day > 0 AND meme_trade_count > pre_feb_trade_count
    THEN ((meme_trade_count - pre_feb_trade_count)::FLOAT / 28) / pre_feb_trades_per_day
    ELSE NULL
  END as activity_growth_rate
FROM 
  wallet_status
ORDER BY 
  exit_status, last_meme_trade DESC; 