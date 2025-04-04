-- 1단계: 2025년 2월에 거래한 지갑 샘플링 (10000개로 제한)
WITH feb_active_wallets AS (
  SELECT 
    swapper AS wallet_address,
    COUNT(*) AS feb_trade_count,
    MAX(block_timestamp) AS last_feb_trade
  FROM solana.defi.fact_swaps
  WHERE block_timestamp >= '2025-02-01'
    AND block_timestamp < '2025-03-01'  -- 2월 한 달만 체크
    AND swap_program = 'pump.fun'
  GROUP BY swapper
  HAVING feb_trade_count BETWEEN 10 AND 1000
  ORDER BY RANDOM()  -- 무작위 샘플링
  LIMIT 10000
),

-- 2단계: 식별된 지갑의 전체 거래 데이터 가져오기 (전체 기간)
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
  JOIN feb_active_wallets s ON fs.swapper = s.wallet_address
  WHERE fs.swap_program = 'pump.fun'
  -- 여기서는 날짜 필터 제거 (전체 생애 주기 분석)
),

-- 3단계: 토큰별 거래 요약 및 손익 계산
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
    -- 이탈 여부: 3월 1일 이후 거래가 없으면 이탈
    CASE
      WHEN NOT EXISTS (
        SELECT 1 
        FROM solana.defi.fact_swaps fs 
        WHERE fs.swapper = ts.wallet_address 
          AND fs.swap_program = 'pump.fun'
          AND fs.block_timestamp >= '2025-03-01'
      )
      THEN TRUE
      ELSE FALSE
    END AS is_churned
  FROM token_summary ts
  GROUP BY ts.wallet_address
)

-- 최종 결과: 이탈 여부와 손실 비율의 관계
SELECT
  is_churned,
  COUNT(*) AS wallet_count,
  AVG(total_losses / total_trades) AS avg_loss_ratio,
  MEDIAN(total_losses / total_trades) AS median_loss_ratio,
  AVG(profitable_trades / (profitable_trades + loss_trades)) AS avg_win_rate,
  MEDIAN(profitable_trades / (profitable_trades + loss_trades)) AS median_win_rate,
  AVG(total_pnl) AS avg_total_pnl,
  MEDIAN(total_pnl) AS median_total_pnl,
  AVG(unique_tokens) AS avg_unique_tokens,
  MEDIAN(unique_tokens) AS median_unique_tokens
FROM wallet_summary
GROUP BY is_churned
ORDER BY is_churned;