-- 목적: 지갑별 경험적 기대수익률 계산
-- 입력: 이전 단계에서 추출한 지갑들의 토큰별 거래 및 수익률
-- 출력: 지갑별 경험적 기대수익률 및 관련 지표
-- 방법: 실제 평균 수익률 기반 기대수익률 계산 (개선된 방식)

-- 1. 타겟 지갑 목록 (04_user_activity_status.sql 결과)
WITH target_wallets AS (
  -- 실제 환경에서는 04_user_activity_status.sql 결과를 사용
  SELECT 
    wallet_address,
    exit_status,
    has_exited
  FROM (
    -- 샘플 데이터 (실제 실행 시 04_user_activity_status.sql 결과로 대체)
    WITH filtered_wallets AS (
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
        INNER JOIN filtered_wallets w ON s.SWAPPER = w.wallet_address
      WHERE 
        s.SUCCEEDED = true
    ),
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
    )
    SELECT 
      fw.wallet_address,
      CASE WHEN ma.march_trade_count > 0 THEN 'ACTIVE' ELSE 'EXITED' END as exit_status,
      CASE WHEN ma.march_trade_count > 0 THEN 0 ELSE 1 END as has_exited
    FROM 
      filtered_wallets fw
      LEFT JOIN march_activity ma ON fw.wallet_address = ma.wallet_address
  )
),

-- 2. 지갑별 토큰 거래 내역 및 ROI 계산
wallet_token_trades AS (
  SELECT 
    t.SWAPPER as wallet_address,
    t.swap_from_mint as token_address,
    t.BLOCK_TIMESTAMP,
    
    -- 거래 타입 분류
    CASE
      WHEN t.swap_program = 'pump.fun' AND t.swap_from_mint = 'So11111111111111111111111111111111111111112' THEN 'BUY'
      WHEN t.swap_program = 'pump.fun' AND t.swap_to_mint = 'So11111111111111111111111111111111111111112' THEN 'SELL'
      ELSE 'OTHER'
    END AS trade_direction,
    
    -- 거래 금액 추출
    CASE
      WHEN trade_direction = 'BUY' THEN t.swap_from_amount
      WHEN trade_direction = 'SELL' THEN t.swap_to_amount
      ELSE 0
    END AS sol_amount,
    
    -- 토큰 수량 추출
    CASE
      WHEN trade_direction = 'BUY' THEN t.swap_to_amount
      WHEN trade_direction = 'SELL' THEN t.swap_from_amount
      ELSE 0
    END AS token_amount,
    
    -- 토큰 가격 계산 (SOL 기준)
    CASE
      WHEN trade_direction = 'BUY' AND t.swap_to_amount > 0 THEN t.swap_from_amount / t.swap_to_amount
      WHEN trade_direction = 'SELL' AND t.swap_from_amount > 0 THEN t.swap_to_amount / t.swap_from_amount
      ELSE 0
    END AS token_price_in_sol
  FROM 
    solana.defi.fact_swaps t
    INNER JOIN target_wallets w ON t.SWAPPER = w.wallet_address
  WHERE 
    t.SUCCEEDED = true
    AND t.BLOCK_TIMESTAMP >= '2025-01-01'  -- 2025년 1월 이후 거래만 고려
    AND t.BLOCK_TIMESTAMP < '2025-03-01'   -- 3월 1일 이전까지만 (이탈 기준일)
    AND (
      (t.swap_program = 'pump.fun' AND t.swap_from_mint = 'So11111111111111111111111111111111111111112') OR
      (t.swap_program = 'pump.fun' AND t.swap_to_mint = 'So11111111111111111111111111111111111111112')
    )
),

-- 3. 지갑별 토큰별 최초/최종 거래 가격 추출
token_trade_summary AS (
  SELECT 
    wallet_address,
    token_address,
    COUNT(*) as trade_count,
    MIN(CASE WHEN trade_direction = 'BUY' THEN BLOCK_TIMESTAMP END) as first_buy_date,
    MAX(CASE WHEN trade_direction = 'BUY' THEN BLOCK_TIMESTAMP END) as last_buy_date,
    MIN(CASE WHEN trade_direction = 'SELL' THEN BLOCK_TIMESTAMP END) as first_sell_date,
    MAX(CASE WHEN trade_direction = 'SELL' THEN BLOCK_TIMESTAMP END) as last_sell_date,
    COUNT(CASE WHEN trade_direction = 'BUY' THEN 1 END) as buy_count,
    COUNT(CASE WHEN trade_direction = 'SELL' THEN 1 END) as sell_count,
    
    -- 최초 매수 가격 (시간순 첫 매수)
    FIRST_VALUE(token_price_in_sol) OVER (
      PARTITION BY wallet_address, token_address 
      ORDER BY BLOCK_TIMESTAMP 
      ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING
    ) as first_buy_price,
    
    -- 최종 매도 가격 (시간순 마지막 매도)
    -- 매도가 없는 경우, 2월 마지막 주 평균 가격 사용 (대체 방법 필요)
    LAST_VALUE(CASE WHEN trade_direction = 'SELL' THEN token_price_in_sol END IGNORE NULLS) OVER (
      PARTITION BY wallet_address, token_address 
      ORDER BY BLOCK_TIMESTAMP 
      ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING
    ) as last_sell_price,
    
    -- 토큰별 순 매수/매도 금액 계산
    SUM(CASE WHEN trade_direction = 'BUY' THEN sol_amount ELSE 0 END) as total_buy_sol,
    SUM(CASE WHEN trade_direction = 'SELL' THEN sol_amount ELSE 0 END) as total_sell_sol,
    
    -- 토큰별 순 매수/매도 수량 계산
    SUM(CASE WHEN trade_direction = 'BUY' THEN token_amount ELSE 0 END) as total_buy_tokens,
    SUM(CASE WHEN trade_direction = 'SELL' THEN token_amount ELSE 0 END) as total_sell_tokens
  FROM 
    wallet_token_trades
  GROUP BY 
    wallet_address, 
    token_address
),

-- 4. 토큰별 마지막 주 가격 정보 (매도 없는 토큰 평가용)
token_last_week_prices AS (
  SELECT 
    token_address,
    AVG(token_price_in_sol) as avg_last_week_price
  FROM 
    wallet_token_trades
  WHERE 
    BLOCK_TIMESTAMP >= '2025-02-21'  -- 2월 마지막 주 기준
    AND BLOCK_TIMESTAMP < '2025-03-01'
  GROUP BY 
    token_address
),

-- 5. 토큰별 ROI 계산
token_roi AS (
  SELECT 
    s.*,
    p.avg_last_week_price,
    
    -- 매도 없는 경우 마지막 주 평균 가격 사용
    COALESCE(last_sell_price, avg_last_week_price) as effective_sell_price,
    
    -- ROI 계산: (sell_price - buy_price) / buy_price
    CASE 
      WHEN first_buy_price > 0 THEN 
        (COALESCE(last_sell_price, avg_last_week_price) - first_buy_price) / first_buy_price
      ELSE NULL
    END as token_roi,
    
    -- 보유 여부 판단
    CASE WHEN total_buy_tokens > total_sell_tokens THEN 1 ELSE 0 END as is_holding,
    
    -- 남은 토큰 수량
    (total_buy_tokens - total_sell_tokens) as remaining_tokens,
    
    -- 실현 손익 (SOL)
    (total_sell_sol - total_buy_sol) as realized_pnl_sol
  FROM 
    token_trade_summary s
    LEFT JOIN token_last_week_prices p ON s.token_address = p.token_address
),

-- 6. ROI 구간 정의
roi_buckets AS (
  SELECT
    CASE
      WHEN token_roi <= -0.9 THEN '-90% to -100%'
      WHEN token_roi <= -0.7 THEN '-70% to -90%'
      WHEN token_roi <= -0.5 THEN '-50% to -70%'
      WHEN token_roi <= -0.3 THEN '-30% to -50%'
      WHEN token_roi <= -0.1 THEN '-10% to -30%'
      WHEN token_roi < 0 THEN '0% to -10%'
      WHEN token_roi = 0 THEN '0%'
      WHEN token_roi <= 0.2 THEN '0% to 20%'
      WHEN token_roi <= 0.4 THEN '20% to 40%'
      WHEN token_roi <= 0.6 THEN '40% to 60%'
      WHEN token_roi <= 0.8 THEN '60% to 80%'
      WHEN token_roi <= 1.0 THEN '80% to 100%'
      WHEN token_roi <= 2.0 THEN '100% to 200%'
      WHEN token_roi <= 5.0 THEN '200% to 500%'
      WHEN token_roi <= 10.0 THEN '500% to 1000%'
      ELSE '1000%+'
    END AS roi_bucket,
    CASE
      WHEN token_roi <= -0.9 THEN 1
      WHEN token_roi <= -0.7 THEN 2
      WHEN token_roi <= -0.5 THEN 3
      WHEN token_roi <= -0.3 THEN 4
      WHEN token_roi <= -0.1 THEN 5
      WHEN token_roi < 0 THEN 6
      WHEN token_roi = 0 THEN 7
      WHEN token_roi <= 0.2 THEN 8
      WHEN token_roi <= 0.4 THEN 9
      WHEN token_roi <= 0.6 THEN 10
      WHEN token_roi <= 0.8 THEN 11
      WHEN token_roi <= 1.0 THEN 12
      WHEN token_roi <= 2.0 THEN 13
      WHEN token_roi <= 5.0 THEN 14
      WHEN token_roi <= 10.0 THEN 15
      ELSE 16
    END AS bucket_order,
    wallet_address,
    token_address,
    token_roi,
    trade_count,
    buy_count,
    sell_count,
    is_holding,
    remaining_tokens,
    realized_pnl_sol
  FROM
    token_roi
  WHERE
    token_roi IS NOT NULL
    AND first_buy_price > 0
),

-- 7. 지갑별 ROI 구간 통계
wallet_roi_bucket_stats AS (
  SELECT
    wallet_address,
    roi_bucket,
    bucket_order,
    COUNT(*) as token_count,
    AVG(token_roi) as avg_roi_in_bucket,
    SUM(CASE WHEN is_holding = 1 THEN 1 ELSE 0 END) as holding_count,
    SUM(remaining_tokens) as total_remaining_tokens,
    SUM(realized_pnl_sol) as total_realized_pnl
  FROM
    roi_buckets
  GROUP BY
    wallet_address,
    roi_bucket,
    bucket_order
),

-- 8. 지갑별 전체 토큰 통계
wallet_token_stats AS (
  SELECT
    wallet_address,
    COUNT(*) as total_tokens,
    SUM(CASE WHEN is_holding = 1 THEN 1 ELSE 0 END) as total_holding_tokens
  FROM
    roi_buckets
  GROUP BY
    wallet_address
),

-- 9. 지갑별 구간별 확률 계산
wallet_bucket_probabilities AS (
  SELECT
    w.wallet_address,
    b.roi_bucket,
    b.bucket_order,
    b.token_count,
    b.avg_roi_in_bucket,
    b.holding_count,
    b.total_remaining_tokens,
    b.total_realized_pnl,
    w.total_tokens,
    (b.token_count::FLOAT / w.total_tokens) as bucket_probability
  FROM
    wallet_roi_bucket_stats b
    JOIN wallet_token_stats w ON b.wallet_address = w.wallet_address
),

-- 10. 지갑별 기대수익률 계산
wallet_expected_returns AS (
  SELECT
    wallet_address,
    SUM(avg_roi_in_bucket * bucket_probability) as expected_roi,
    SUM(CASE WHEN avg_roi_in_bucket >= 0 THEN bucket_probability ELSE 0 END) as prob_positive_roi,
    SUM(CASE WHEN avg_roi_in_bucket < 0 THEN bucket_probability ELSE 0 END) as prob_negative_roi,
    
    -- 양수/음수 ROI의 기대값
    SUM(CASE WHEN avg_roi_in_bucket >= 0 THEN avg_roi_in_bucket * bucket_probability ELSE 0 END) as expected_positive_roi,
    SUM(CASE WHEN avg_roi_in_bucket < 0 THEN avg_roi_in_bucket * bucket_probability ELSE 0 END) as expected_negative_roi,
    
    -- 분산 및 표준편차 추정
    SUM(POWER(avg_roi_in_bucket - SUM(avg_roi_in_bucket * bucket_probability) OVER (PARTITION BY wallet_address), 2) * bucket_probability) as roi_variance,
    
    -- 구간별 통계
    COUNT(*) as bucket_count,
    SUM(token_count) as total_tokens,
    SUM(holding_count) as total_holding,
    SUM(total_realized_pnl) as total_realized_pnl
  FROM
    wallet_bucket_probabilities
  GROUP BY
    wallet_address
),

-- 11. 최종 결과: 지갑별 기대수익률 및 이탈 상태
final_results AS (
  SELECT
    e.*,
    SQRT(e.roi_variance) as roi_std_dev,
    t.exit_status,
    t.has_exited,
    ROW_NUMBER() OVER (ORDER BY expected_roi DESC) as rank_by_expected_roi,
    NTILE(10) OVER (ORDER BY expected_roi) as expected_roi_decile
  FROM
    wallet_expected_returns e
    JOIN target_wallets t ON e.wallet_address = t.wallet_address
)

-- 최종 출력
SELECT 
  *,
  -- 추가 지표
  CASE 
    WHEN roi_std_dev > 0 THEN expected_roi / roi_std_dev 
    ELSE NULL 
  END as sharpe_ratio,
  
  -- 기대수익률 구간화
  CASE
    WHEN expected_roi <= -0.5 THEN 'Very Negative (< -50%)'
    WHEN expected_roi <= -0.2 THEN 'Negative (-50% to -20%)'
    WHEN expected_roi < 0 THEN 'Slightly Negative (-20% to 0%)'
    WHEN expected_roi = 0 THEN 'Zero (0%)'
    WHEN expected_roi < 0.2 THEN 'Slightly Positive (0% to 20%)'
    WHEN expected_roi < 0.5 THEN 'Positive (20% to 50%)'
    WHEN expected_roi < 1 THEN 'Very Positive (50% to 100%)'
    ELSE 'Extremely Positive (> 100%)'
  END as expected_roi_category
FROM 
  final_results
ORDER BY 
  expected_roi DESC; 