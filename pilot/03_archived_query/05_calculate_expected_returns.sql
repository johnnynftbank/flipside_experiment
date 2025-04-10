WITH target_wallets AS (
  -- 2025년 2월 활성 지갑 100개 추출
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
  LIMIT 100  -- 테스트를 위해 100개로 제한
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
  -- 일회성 지갑 필터링
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

trade_history AS (
  -- 필터링된 지갑들의 밈코인 거래 상세 정보와 누적값
  SELECT 
    t.SWAPPER as wallet_address,
    t.BLOCK_TIMESTAMP,
    t.trade_type,
    CASE 
      WHEN trade_type = 'MEME_BUY' THEN t.swap_from_amount
      WHEN trade_type = 'MEME_SELL' THEN t.swap_to_amount
    END as sol_amount,
    CASE 
      WHEN trade_type = 'MEME_BUY' THEN t.swap_to_amount
      WHEN trade_type = 'MEME_SELL' THEN t.swap_from_amount
    END as token_amount,
    CASE 
      WHEN trade_type = 'MEME_BUY' THEN t.swap_to_mint
      WHEN trade_type = 'MEME_SELL' THEN t.swap_from_mint
    END as token_mint,
    SUM(CASE WHEN trade_type = 'MEME_BUY' THEN t.swap_from_amount ELSE 0 END) 
      OVER (PARTITION BY t.SWAPPER, CASE 
        WHEN trade_type = 'MEME_BUY' THEN t.swap_to_mint
        WHEN trade_type = 'MEME_SELL' THEN t.swap_from_mint
      END ORDER BY t.BLOCK_TIMESTAMP) as token_total_sol_spent,
    SUM(CASE WHEN trade_type = 'MEME_SELL' THEN t.swap_to_amount ELSE 0 END) 
      OVER (PARTITION BY t.SWAPPER, CASE 
        WHEN trade_type = 'MEME_BUY' THEN t.swap_to_mint
        WHEN trade_type = 'MEME_SELL' THEN t.swap_from_mint
      END ORDER BY t.BLOCK_TIMESTAMP) as token_total_sol_received
  FROM 
    all_trades t
    INNER JOIN filtered_wallets w ON t.SWAPPER = w.wallet_address
  WHERE 
    trade_type IN ('MEME_BUY', 'MEME_SELL')
),

token_roi_calculation AS (
  -- 각 (지갑, 토큰) 쌍의 ROI 계산
  SELECT 
    wallet_address,
    token_mint,
    MAX(token_total_sol_spent) as investment,
    MAX(token_total_sol_received) as revenue,
    CASE 
      WHEN MAX(token_total_sol_spent) > 0 
      THEN (MAX(token_total_sol_received) - MAX(token_total_sol_spent)) / MAX(token_total_sol_spent)
      ELSE NULL 
    END as roi,
    DATEDIFF('day', MIN(BLOCK_TIMESTAMP), MAX(BLOCK_TIMESTAMP)) as holding_period_days
  FROM 
    trade_history
  GROUP BY 
    wallet_address,
    token_mint
  HAVING 
    investment > 0
),

roi_distribution AS (
  -- ROI 구간 설정 및 분포 계산
  SELECT 
    CASE
      WHEN roi = -1.0 THEN '-100%'
      WHEN roi <= -0.9 THEN '-90%~-99%'
      WHEN roi <= -0.8 THEN '-80%~-89%'
      WHEN roi <= -0.7 THEN '-70%~-79%'
      WHEN roi <= -0.6 THEN '-60%~-69%'
      WHEN roi <= -0.5 THEN '-50%~-59%'
      WHEN roi <= -0.4 THEN '-40%~-49%'
      WHEN roi <= -0.3 THEN '-30%~-39%'
      WHEN roi <= -0.2 THEN '-20%~-29%'
      WHEN roi <= -0.1 THEN '-10%~-19%'
      WHEN roi < 0 THEN '0%~-9%'
      WHEN roi = 0 THEN '0%'
      WHEN roi <= 0.1 THEN '0%~10%'
      WHEN roi <= 0.2 THEN '10%~20%'
      WHEN roi <= 0.3 THEN '20%~30%'
      WHEN roi <= 0.4 THEN '30%~40%'
      WHEN roi <= 0.5 THEN '40%~50%'
      WHEN roi <= 0.6 THEN '50%~60%'
      WHEN roi <= 0.7 THEN '60%~70%'
      WHEN roi <= 0.8 THEN '70%~80%'
      WHEN roi <= 0.9 THEN '80%~90%'
      WHEN roi <= 1.0 THEN '90%~100%'
      WHEN roi <= 2.0 THEN '100%~200%'
      WHEN roi <= 3.0 THEN '200%~300%'
      WHEN roi <= 5.0 THEN '300%~500%'
      WHEN roi <= 7.0 THEN '500%~700%'
      WHEN roi <= 10.0 THEN '700%~1000%'
      ELSE 'Over 1000%'
    END as roi_bucket,
    COUNT(*) as trades_in_bucket,
    COUNT(*) / SUM(COUNT(*)) OVER () as probability,
    AVG(roi) as avg_roi,
    AVG(holding_period_days) as avg_holding_days,
    MIN(roi) as min_roi,
    MAX(roi) as max_roi,
    SUM(investment) as total_investment,
    SUM(revenue) as total_revenue
  FROM 
    token_roi_calculation
  WHERE 
    roi IS NOT NULL
  GROUP BY 
    roi_bucket
)

-- 최종 결과
SELECT 
  roi_bucket,
  trades_in_bucket,
  ROUND(probability, 4) as probability,
  ROUND(avg_roi * 100, 2) || '%' as avg_roi_percent,
  ROUND(avg_holding_days, 1) as avg_holding_days,
  ROUND(min_roi * 100, 2) || '%' as min_roi_percent,
  ROUND(max_roi * 100, 2) || '%' as max_roi_percent,
  ROUND(total_investment, 2) as total_investment_sol,
  ROUND(total_revenue, 2) as total_revenue_sol,
  ROUND(probability * avg_roi * 100, 4) || '%' as contribution_to_expected_roi
FROM 
  roi_distribution
ORDER BY 
  min_roi; 