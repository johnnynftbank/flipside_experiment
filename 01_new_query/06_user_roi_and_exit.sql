WITH target_wallets AS (
  -- 2025년 2월 활성 지갑 추출 (이전 쿼리와 동일)
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

filtered_wallets AS (
  -- 일회성 지갑 필터링 (2일 이상 거래)
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

wallet_token_roi AS (
  -- 지갑별, 토큰별 ROI 계산
  SELECT 
    t.SWAPPER as wallet_address,
    CASE 
      WHEN trade_type = 'MEME_BUY' THEN t.swap_to_mint
      WHEN trade_type = 'MEME_SELL' THEN t.swap_from_mint
    END as token_mint,
    SUM(CASE WHEN trade_type = 'MEME_BUY' THEN t.swap_from_amount ELSE 0 END) as total_sol_invested,
    SUM(CASE WHEN trade_type = 'MEME_SELL' THEN t.swap_to_amount ELSE 0 END) as total_sol_returned,
    CASE 
      WHEN SUM(CASE WHEN trade_type = 'MEME_BUY' THEN t.swap_from_amount ELSE 0 END) > 0 
      THEN (SUM(CASE WHEN trade_type = 'MEME_SELL' THEN t.swap_to_amount ELSE 0 END) - 
            SUM(CASE WHEN trade_type = 'MEME_BUY' THEN t.swap_from_amount ELSE 0 END)) / 
           SUM(CASE WHEN trade_type = 'MEME_BUY' THEN t.swap_from_amount ELSE 0 END)
      ELSE NULL 
    END as token_roi
  FROM 
    all_trades t
    INNER JOIN filtered_wallets w ON t.SWAPPER = w.wallet_address
  WHERE 
    trade_type IN ('MEME_BUY', 'MEME_SELL')
  GROUP BY 
    t.SWAPPER,
    token_mint
  HAVING 
    total_sol_invested > 0
),

wallet_overall_roi AS (
  -- 지갑별 종합 ROI 계산
  SELECT 
    wallet_address,
    COUNT(DISTINCT token_mint) as unique_tokens_traded,
    SUM(total_sol_invested) as total_investment,
    SUM(total_sol_returned) as total_returns,
    CASE 
      WHEN SUM(total_sol_invested) > 0 
      THEN (SUM(total_sol_returned) - SUM(total_sol_invested)) / SUM(total_sol_invested)
      ELSE NULL 
    END as overall_roi,
    AVG(token_roi) as avg_token_roi,
    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY token_roi) as median_token_roi,
    MIN(token_roi) as worst_token_roi,
    MAX(token_roi) as best_token_roi
  FROM 
    wallet_token_roi
  GROUP BY 
    wallet_address
),

wallet_activity_pattern AS (
  -- 지갑별 활동 패턴 분석
  SELECT 
    t.SWAPPER as wallet_address,
    MIN(CASE WHEN trade_type IN ('MEME_BUY', 'MEME_SELL') THEN BLOCK_TIMESTAMP END) as first_meme_trade,
    MAX(CASE WHEN trade_type IN ('MEME_BUY', 'MEME_SELL') THEN BLOCK_TIMESTAMP END) as last_meme_trade,
    DATEDIFF('day', MIN(CASE WHEN trade_type IN ('MEME_BUY', 'MEME_SELL') THEN BLOCK_TIMESTAMP END),
             MAX(CASE WHEN trade_type IN ('MEME_BUY', 'MEME_SELL') THEN BLOCK_TIMESTAMP END)) + 1 as total_activity_period,
    COUNT(CASE WHEN trade_type IN ('MEME_BUY', 'MEME_SELL') THEN 1 END) as total_meme_trades,
    COUNT(DISTINCT CASE WHEN trade_type IN ('MEME_BUY', 'MEME_SELL') THEN DATE_TRUNC('day', BLOCK_TIMESTAMP) END) as active_trading_days,
    
    -- 시간에 따른 활동 변화 측정 (마지막 거래 이후 경과 시간)
    DATEDIFF('day', MAX(CASE WHEN trade_type IN ('MEME_BUY', 'MEME_SELL') THEN BLOCK_TIMESTAMP END), CURRENT_DATE()) as days_since_last_trade,
    
    -- 거래 빈도 변화 측정 (첫 절반 기간의 거래와 나중 절반 기간의 거래 비교)
    COUNT(CASE WHEN trade_type IN ('MEME_BUY', 'MEME_SELL') AND 
               DATEDIFF('day', MIN(CASE WHEN trade_type IN ('MEME_BUY', 'MEME_SELL') THEN BLOCK_TIMESTAMP END) OVER (PARTITION BY t.SWAPPER), 
                          BLOCK_TIMESTAMP) <= 
               DATEDIFF('day', MIN(CASE WHEN trade_type IN ('MEME_BUY', 'MEME_SELL') THEN BLOCK_TIMESTAMP END) OVER (PARTITION BY t.SWAPPER),
                          MAX(CASE WHEN trade_type IN ('MEME_BUY', 'MEME_SELL') THEN BLOCK_TIMESTAMP END) OVER (PARTITION BY t.SWAPPER)) / 2
               THEN 1 END) OVER (PARTITION BY t.SWAPPER) as first_half_trades,
               
    COUNT(CASE WHEN trade_type IN ('MEME_BUY', 'MEME_SELL') AND 
               DATEDIFF('day', MIN(CASE WHEN trade_type IN ('MEME_BUY', 'MEME_SELL') THEN BLOCK_TIMESTAMP END) OVER (PARTITION BY t.SWAPPER), 
                          BLOCK_TIMESTAMP) > 
               DATEDIFF('day', MIN(CASE WHEN trade_type IN ('MEME_BUY', 'MEME_SELL') THEN BLOCK_TIMESTAMP END) OVER (PARTITION BY t.SWAPPER),
                          MAX(CASE WHEN trade_type IN ('MEME_BUY', 'MEME_SELL') THEN BLOCK_TIMESTAMP END) OVER (PARTITION BY t.SWAPPER)) / 2
               THEN 1 END) OVER (PARTITION BY t.SWAPPER) as second_half_trades
  FROM 
    all_trades t
    INNER JOIN filtered_wallets w ON t.SWAPPER = w.wallet_address
  GROUP BY 
    t.SWAPPER
),

exit_metrics AS (
  -- 이탈 지표 계산
  SELECT 
    wallet_address,
    total_activity_period,
    total_meme_trades,
    active_trading_days,
    days_since_last_trade,
    first_half_trades,
    second_half_trades,
    -- 이탈 지표 1: 마지막 거래 이후 시간 (높을수록 이탈 가능성 높음)
    CASE 
      WHEN days_since_last_trade >= 60 THEN 'Exited'
      WHEN days_since_last_trade >= 30 THEN 'Likely_Exited'
      WHEN days_since_last_trade >= 14 THEN 'Possible_Exit'
      ELSE 'Active'
    END as exit_status_time,
    
    -- 이탈 지표 2: 거래 빈도 감소율 (높을수록 이탈 가능성 높음)
    CASE
      WHEN first_half_trades = 0 THEN NULL
      ELSE (first_half_trades - second_half_trades)::FLOAT / first_half_trades
    END as activity_decline_rate,
    
    -- 이탈 지표 3: 종합 이탈 점수 (0-100, 높을수록 이탈 가능성 높음)
    CASE
      WHEN days_since_last_trade >= 60 THEN 100
      ELSE LEAST(100, (days_since_last_trade / 60.0 * 50) + 
                      (CASE WHEN first_half_trades = 0 THEN 0 
                            ELSE ((first_half_trades - second_half_trades)::FLOAT / first_half_trades) * 50 
                       END))
    END as exit_score
  FROM 
    wallet_activity_pattern
)

-- 최종 결과: ROI와 이탈 지표의 연관성
SELECT 
  r.*,
  e.days_since_last_trade,
  e.total_activity_period,
  e.active_trading_days,
  e.first_half_trades,
  e.second_half_trades,
  e.activity_decline_rate,
  e.exit_status_time,
  e.exit_score
FROM 
  wallet_overall_roi r
  JOIN exit_metrics e ON r.wallet_address = e.wallet_address
ORDER BY 
  r.overall_roi; 