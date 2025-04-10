-- 손실 비율 구간에 따른 이탈률 분석 (3000개 지갑 대상)
-- 1단계: 샘플 지갑 추출 
WITH sampled_wallets AS (
  SELECT 
    swapper AS wallet_address,
    COUNT(*) AS trade_count,
    MAX(block_timestamp) AS last_trade,
    CASE 
      WHEN COUNT(*) BETWEEN 10 AND 30 THEN 'low'
      WHEN COUNT(*) BETWEEN 31 AND 100 THEN 'medium'
      ELSE 'high'
    END AS activity_level
  FROM solana.defi.fact_swaps
  WHERE block_timestamp >= '2025-02-01'
  AND swap_program = 'pump.fun'
  GROUP BY swapper
  HAVING trade_count BETWEEN 10 AND 1000
  ORDER BY RANDOM()
  LIMIT 3000
),

-- 2단계: 거래 데이터 추출
trade_data AS (
  SELECT
    s.wallet_address,
    fs.tx_id,
    fs.block_timestamp,
    fs.swap_from_mint AS send_token,
    fs.swap_to_mint AS receive_token,
    fs.swap_from_amount AS send_amount,
    fs.swap_to_amount AS receive_amount,
    CASE
      WHEN fs.swap_from_mint = 'So11111111111111111111111111111111111111112' THEN 'BUY'
      WHEN fs.swap_to_mint = 'So11111111111111111111111111111111111111112' THEN 'SELL'
      ELSE 'OTHER'
    END AS trade_type
  FROM solana.defi.fact_swaps fs
  JOIN sampled_wallets s ON fs.swapper = s.wallet_address
  WHERE fs.swap_program = 'pump.fun'
  AND fs.block_timestamp >= '2025-02-01'
),

-- 3단계: 토큰별 손익 계산
token_summary AS (
  SELECT
    wallet_address,
    CASE
      WHEN trade_type = 'BUY' THEN receive_token
      WHEN trade_type = 'SELL' THEN send_token
      ELSE NULL
    END AS traded_token,
    SUM(CASE WHEN trade_type = 'BUY' THEN send_amount ELSE 0 END) AS total_buy_sol,
    SUM(CASE WHEN trade_type = 'BUY' THEN receive_amount ELSE 0 END) AS total_buy_tokens,
    SUM(CASE WHEN trade_type = 'SELL' THEN receive_amount ELSE 0 END) AS total_sell_sol,
    SUM(CASE WHEN trade_type = 'SELL' THEN send_amount ELSE 0 END) AS total_sell_tokens,
    SUM(CASE WHEN trade_type = 'SELL' THEN receive_amount ELSE 0 END) - 
    SUM(CASE WHEN trade_type = 'BUY' THEN send_amount ELSE 0 END) AS profit_loss_sol,
    MIN(CASE WHEN trade_type = 'BUY' THEN block_timestamp ELSE NULL END) AS first_buy,
    MAX(CASE WHEN trade_type = 'SELL' THEN block_timestamp ELSE NULL END) AS last_sell,
    -- 거래가 이익인지 손실인지
    CASE 
      WHEN SUM(CASE WHEN trade_type = 'SELL' THEN receive_amount ELSE 0 END) - 
           SUM(CASE WHEN trade_type = 'BUY' THEN send_amount ELSE 0 END) >= 0 
      THEN 0 ELSE 1 
    END AS is_loss
  FROM trade_data
  WHERE trade_type IN ('BUY', 'SELL')
  GROUP BY wallet_address, 
    CASE
      WHEN trade_type = 'BUY' THEN receive_token
      WHEN trade_type = 'SELL' THEN send_token
      ELSE NULL
    END
),

-- 4단계: 지갑별 거래 요약 및 이탈 여부 판단
wallet_summary AS (
  SELECT
    ts.wallet_address,
    COUNT(DISTINCT ts.traded_token) AS unique_tokens,
    SUM(ts.profit_loss_sol) AS total_pnl,
    SUM(CASE WHEN ts.profit_loss_sol > 0 THEN 1 ELSE 0 END) AS profitable_trades,
    SUM(CASE WHEN ts.profit_loss_sol < 0 THEN 1 ELSE 0 END) AS loss_trades,
    SUM(ts.is_loss) AS total_losses,
    COUNT(*) AS total_trades,
    -- 손실 비율
    SUM(ts.is_loss) / COUNT(*)::FLOAT AS loss_ratio,
    -- 이탈 여부: 마지막 거래로부터 30일 이상 지났으면 이탈
    CASE
      WHEN DATEDIFF('day', MAX(COALESCE(ts.last_sell, ts.first_buy)), CURRENT_DATE()) > 30 
      THEN TRUE
      ELSE FALSE
    END AS is_churned
  FROM token_summary ts
  GROUP BY ts.wallet_address
),

-- 5단계: 손실 비율 구간화 및 이탈률 계산
loss_ratio_analysis AS (
  SELECT
    -- 손실 비율을 5개 구간으로 나누기
    CASE
      WHEN loss_ratio < 0.2 THEN '0-20% (매우 낮음)'
      WHEN loss_ratio < 0.4 THEN '20-40% (낮음)'
      WHEN loss_ratio < 0.6 THEN '40-60% (중간)'
      WHEN loss_ratio < 0.8 THEN '60-80% (높음)'
      ELSE '80-100% (매우 높음)'
    END AS loss_ratio_category,
    COUNT(*) AS wallet_count,
    SUM(CASE WHEN is_churned THEN 1 ELSE 0 END) AS churned_count,
    SUM(CASE WHEN is_churned THEN 1 ELSE 0 END) / COUNT(*)::FLOAT AS churn_rate,
    AVG(total_pnl) AS avg_pnl,
    AVG(unique_tokens) AS avg_tokens_traded
  FROM wallet_summary
  GROUP BY 
    CASE
      WHEN loss_ratio < 0.2 THEN '0-20% (매우 낮음)'
      WHEN loss_ratio < 0.4 THEN '20-40% (낮음)'
      WHEN loss_ratio < 0.6 THEN '40-60% (중간)'
      WHEN loss_ratio < 0.8 THEN '60-80% (높음)'
      ELSE '80-100% (매우 높음)'
    END
)

-- 6단계: 결과 출력
SELECT
  loss_ratio_category,
  wallet_count,
  churned_count,
  churn_rate,
  avg_pnl,
  avg_tokens_traded
FROM loss_ratio_analysis
ORDER BY 
  CASE 
    WHEN loss_ratio_category = '0-20% (매우 낮음)' THEN 1
    WHEN loss_ratio_category = '20-40% (낮음)' THEN 2
    WHEN loss_ratio_category = '40-60% (중간)' THEN 3
    WHEN loss_ratio_category = '60-80% (높음)' THEN 4
    ELSE 5
  END