-- 목적: 기대수익률과 이탈 여부 간의 상관관계 분석
-- 입력: 05_empirical_expected_returns.sql의 결과
-- 출력: 기대수익률과 이탈률 간의 관계 분석
-- 참고: 2025년 1월 1일 이후 거래만 고려함 (데이터 볼륨 제한)

-- 1. 이전 단계의 실제 데이터를 사용하여 기대수익률 및 이탈 상태 분석
WITH 
-- 첫 단계: 밈 코인 거래만 식별하여 별도 CTE로 관리
meme_trades AS (
  SELECT 
    SWAPPER as wallet_address,
    BLOCK_TIMESTAMP,
    TX_ID
  FROM 
    solana.defi.fact_swaps
  WHERE 
    SUCCEEDED = true
    AND BLOCK_TIMESTAMP >= '2025-02-01'
    AND BLOCK_TIMESTAMP < '2025-03-01'
    AND swap_program = 'pump.fun'
    AND (swap_from_mint = 'So11111111111111111111111111111111111111112' OR 
         swap_to_mint = 'So11111111111111111111111111111111111111112')
),

-- 두 번째 단계: 지갑별 거래 통계 및 밈 코인 거래일 수 계산
wallet_stats AS (
  SELECT 
    t.wallet_address,
    COUNT(*) as meme_trade_count,
    COUNT(DISTINCT DATE(t.BLOCK_TIMESTAMP)) as meme_trading_days
  FROM 
    meme_trades t
  GROUP BY 
    t.wallet_address
),

-- 세 번째 단계: 총 거래 수도 계산 (pump.fun의 모든 거래)
all_trades_count AS (
  SELECT
    SWAPPER as wallet_address,
    COUNT(*) as total_trades
  FROM
    solana.defi.fact_swaps
  WHERE
    SUCCEEDED = true
    AND BLOCK_TIMESTAMP >= '2025-02-01'
    AND BLOCK_TIMESTAMP < '2025-03-01'
    AND swap_program = 'pump.fun'
  GROUP BY
    SWAPPER
),

-- 네 번째 단계: 필터링 조건 적용
filtered_wallets AS (
  SELECT
    w.wallet_address,
    w.meme_trading_days,
    w.meme_trade_count,
    COALESCE(a.total_trades, 0) as feb_trade_count
  FROM
    wallet_stats w
    LEFT JOIN all_trades_count a ON w.wallet_address = a.wallet_address
  WHERE
    COALESCE(a.total_trades, 0) >= 10
    AND COALESCE(a.total_trades, 0) < 1000
    AND w.meme_trading_days >= 2  -- 실제 밈 코인 거래일 2일 이상
  ORDER BY
    RANDOM()
  LIMIT 10000
),

-- 다섯 번째 단계: 3월 이후 밈 코인 거래 여부로 이탈 상태 계산
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

-- 여섯 번째 단계: 최종 이탈 상태 결정
user_activity_status AS (
  SELECT 
    fw.wallet_address,
    CASE WHEN ma.march_trade_count > 0 THEN 'ACTIVE' ELSE 'EXITED' END as exit_status,
    CASE WHEN ma.march_trade_count > 0 THEN 0 ELSE 1 END as has_exited,
    fw.meme_trading_days,
    fw.meme_trade_count,
    fw.feb_trade_count
  FROM 
    filtered_wallets fw
    LEFT JOIN march_activity ma ON fw.wallet_address = ma.wallet_address
),

-- 일곱 번째 단계: 전체 거래 내역 (all_trades)
all_trades AS (
  SELECT 
    s.*,
    CASE
      WHEN s.swap_program = 'pump.fun' AND s.swap_from_mint = 'So11111111111111111111111111111111111111112' THEN 'MEME_BUY'
      WHEN s.swap_program = 'pump.fun' AND s.swap_to_mint = 'So11111111111111111111111111111111111111112' THEN 'MEME_SELL'
      ELSE 'OTHER'
    END AS trade_type
  FROM 
    solana.defi.fact_swaps s
    INNER JOIN user_activity_status w ON s.SWAPPER = w.wallet_address
  WHERE 
    s.SUCCEEDED = true
    AND s.BLOCK_TIMESTAMP >= '2025-01-01'
),

-- 토큰 거래 내역 및 ROI 계산을 위한 CTEs
wallet_token_trades AS (
  SELECT 
    t.SWAPPER as wallet_address,
    CASE
      WHEN t.swap_program = 'pump.fun' AND t.swap_from_mint = 'So11111111111111111111111111111111111111112' THEN t.swap_to_mint
      WHEN t.swap_program = 'pump.fun' AND t.swap_to_mint = 'So11111111111111111111111111111111111111112' THEN t.swap_from_mint
    END AS token_address,
    t.BLOCK_TIMESTAMP,
    
    -- 토큰 심볼 생성 (민트 주소 마지막 4자리 사용)
    CASE
      WHEN t.swap_program = 'pump.fun' AND t.swap_from_mint = 'So11111111111111111111111111111111111111112' 
        THEN CONCAT('TOKEN_', RIGHT(t.swap_to_mint, 4))
      WHEN t.swap_program = 'pump.fun' AND t.swap_to_mint = 'So11111111111111111111111111111111111111112' 
        THEN CONCAT('TOKEN_', RIGHT(t.swap_from_mint, 4))
    END AS token_symbol,
    
    -- 거래 타입 분류
    CASE
      WHEN t.swap_program = 'pump.fun' AND t.swap_from_mint = 'So11111111111111111111111111111111111111112' THEN 'BUY'
      WHEN t.swap_program = 'pump.fun' AND t.swap_to_mint = 'So11111111111111111111111111111111111111112' THEN 'SELL'
      ELSE 'OTHER'
    END AS trade_direction,
    
    -- 거래 금액 추출
    CASE
      WHEN t.swap_program = 'pump.fun' AND t.swap_from_mint = 'So11111111111111111111111111111111111111112' THEN t.swap_from_amount
      WHEN t.swap_program = 'pump.fun' AND t.swap_to_mint = 'So11111111111111111111111111111111111111112' THEN t.swap_to_amount
      ELSE 0
    END AS sol_amount,
    
    -- 토큰 수량 추출
    CASE
      WHEN t.swap_program = 'pump.fun' AND t.swap_from_mint = 'So11111111111111111111111111111111111111112' THEN t.swap_to_amount
      WHEN t.swap_program = 'pump.fun' AND t.swap_to_mint = 'So11111111111111111111111111111111111111112' THEN t.swap_from_amount
      ELSE 0
    END AS token_amount,
    
    -- 토큰 가격 계산 (SOL 기준)
    CASE
      WHEN t.swap_program = 'pump.fun' AND t.swap_from_mint = 'So11111111111111111111111111111111111111112' AND t.swap_to_amount > 0 
        THEN t.swap_from_amount / t.swap_to_amount
      WHEN t.swap_program = 'pump.fun' AND t.swap_to_mint = 'So11111111111111111111111111111111111111112' AND t.swap_from_amount > 0 
        THEN t.swap_to_amount / t.swap_from_amount
      ELSE 0
    END AS token_price_in_sol
  FROM 
    solana.defi.fact_swaps t
    INNER JOIN user_activity_status w ON t.SWAPPER = w.wallet_address
  WHERE 
    t.SUCCEEDED = true
    AND t.BLOCK_TIMESTAMP >= '2025-01-01'
    AND t.BLOCK_TIMESTAMP < '2025-03-01'
    AND (
      (t.swap_program = 'pump.fun' AND t.swap_from_mint = 'So11111111111111111111111111111111111111112') OR
      (t.swap_program = 'pump.fun' AND t.swap_to_mint = 'So11111111111111111111111111111111111111112')
    )
),

token_trade_summary AS (
  SELECT 
    wallet_address,
    token_address,
    token_symbol,
    COUNT(*) as trade_count,
    MIN(CASE WHEN trade_direction = 'BUY' THEN BLOCK_TIMESTAMP END) as first_buy_date,
    MAX(CASE WHEN trade_direction = 'BUY' THEN BLOCK_TIMESTAMP END) as last_buy_date,
    MIN(CASE WHEN trade_direction = 'SELL' THEN BLOCK_TIMESTAMP END) as first_sell_date,
    MAX(CASE WHEN trade_direction = 'SELL' THEN BLOCK_TIMESTAMP END) as last_sell_date,
    COUNT(CASE WHEN trade_direction = 'BUY' THEN 1 END) as buy_count,
    COUNT(CASE WHEN trade_direction = 'SELL' THEN 1 END) as sell_count,
    
    -- 토큰별 최초 구매 가격 (가장 중요한 기준가)
    MIN(CASE WHEN trade_direction = 'BUY' THEN token_price_in_sol END) as first_buy_price,
    
    -- 토큰별 최종 판매 가격
    MAX(CASE WHEN trade_direction = 'SELL' THEN token_price_in_sol END) as last_sell_price,
    
    -- 토큰별 평균 구매 가격
    SUM(CASE WHEN trade_direction = 'BUY' THEN sol_amount ELSE 0 END) / 
      NULLIF(SUM(CASE WHEN trade_direction = 'BUY' THEN token_amount ELSE 0 END), 0) as avg_buy_price,
      
    -- 토큰별 평균 판매 가격
    SUM(CASE WHEN trade_direction = 'SELL' THEN sol_amount ELSE 0 END) / 
      NULLIF(SUM(CASE WHEN trade_direction = 'SELL' THEN token_amount ELSE 0 END), 0) as avg_sell_price,
    
    -- 토큰별 순 매수/매도 금액 계산
    SUM(CASE WHEN trade_direction = 'BUY' THEN sol_amount ELSE 0 END) as total_buy_sol,
    SUM(CASE WHEN trade_direction = 'SELL' THEN sol_amount ELSE 0 END) as total_sell_sol,
    
    -- 토큰별 순 매수/매도 수량 계산
    SUM(CASE WHEN trade_direction = 'BUY' THEN token_amount ELSE 0 END) as total_buy_tokens,
    SUM(CASE WHEN trade_direction = 'SELL' THEN token_amount ELSE 0 END) as total_sell_tokens,
    
    -- 남은 토큰 수량
    SUM(CASE WHEN trade_direction = 'BUY' THEN token_amount ELSE -token_amount END) as remaining_tokens,
    
    -- 실현 손익 (SOL)
    SUM(CASE WHEN trade_direction = 'SELL' THEN sol_amount ELSE -sol_amount END) as realized_pnl_sol
  FROM 
    wallet_token_trades
  GROUP BY 
    wallet_address, 
    token_address,
    token_symbol
),

token_last_week_prices AS (
  SELECT 
    token_address,
    AVG(token_price_in_sol) as avg_last_week_price,
    COUNT(*) as price_data_count
  FROM 
    wallet_token_trades
  WHERE 
    BLOCK_TIMESTAMP >= '2025-02-21'
    AND BLOCK_TIMESTAMP < '2025-03-01'
  GROUP BY 
    token_address
),

token_roi AS (
  SELECT 
    s.*,
    p.avg_last_week_price,
    p.price_data_count,
    
    -- 2025년 1월 1일 이전 구매 가능성 확인
    CASE
      -- 첫 구매일이 1월 1일인 경우 (실제로는 그 이전일 수 있음)
      WHEN first_buy_date = '2025-01-01' THEN true
      -- 구매 기록은 없지만 판매 기록이 있는 경우 (1월 이전 구매)
      WHEN first_buy_date IS NULL AND first_sell_date IS NOT NULL THEN true
      -- 정상적인 경우
      ELSE false
    END as potential_pre_2025_purchase,
    
    -- 토큰 상태 분류
    CASE 
      WHEN total_sell_tokens >= total_buy_tokens * 0.99 THEN 'FULLY_SOLD'
      WHEN total_sell_tokens > 0 THEN 'PARTIALLY_SOLD'
      ELSE 'HOLDING'
    END as token_status,
    
    -- 실효 판매 가격 계산
    CASE
      WHEN total_sell_tokens >= total_buy_tokens * 0.99 THEN avg_sell_price
      WHEN total_sell_tokens > 0 THEN
        (total_sell_sol + (remaining_tokens * COALESCE(avg_last_week_price, avg_buy_price))) / total_buy_tokens
      ELSE COALESCE(avg_last_week_price, avg_buy_price)
    END as effective_sell_price,
    
    -- ROI 계산: (effective_sell_price - first_buy_price) / first_buy_price
    CASE 
      -- 첫 구매 기록이 없는 경우 (2025년 1월 이전 구매)
      WHEN first_buy_price IS NULL OR first_buy_price = 0 THEN NULL
      -- 실효 판매 가격이 계산되지 않는 경우
      WHEN (
        token_status = 'HOLDING' AND 
        avg_last_week_price IS NULL AND 
        total_buy_tokens > 0
      ) THEN NULL
      -- 정상적인 ROI 계산
      ELSE 
        CASE
          WHEN token_status = 'FULLY_SOLD' THEN
            (avg_sell_price / first_buy_price) - 1
          WHEN token_status = 'PARTIALLY_SOLD' THEN
            ((total_sell_sol + (remaining_tokens * COALESCE(avg_last_week_price, avg_buy_price))) / 
            total_buy_sol) - 1
          ELSE
            (COALESCE(avg_last_week_price, avg_buy_price) / first_buy_price) - 1
        END
    END as token_roi
  FROM 
    token_trade_summary s
    LEFT JOIN token_last_week_prices p ON s.token_address = p.token_address
),

roi_buckets AS (
  SELECT
    wallet_address,
    token_address,
    token_symbol,
    token_status,
    potential_pre_2025_purchase,
    token_roi,
    -- ROI 범위 분류 (세분화된 버전)
    CASE
      WHEN token_roi IS NULL THEN 'UNKNOWN_ROI'
      -- 음수 ROI (10% 간격)
      WHEN token_roi <= -0.9 THEN 'LOSS_90_TO_100'
      WHEN token_roi <= -0.8 THEN 'LOSS_80_TO_90'
      WHEN token_roi <= -0.7 THEN 'LOSS_70_TO_80'
      WHEN token_roi <= -0.6 THEN 'LOSS_60_TO_70'
      WHEN token_roi <= -0.5 THEN 'LOSS_50_TO_60'
      WHEN token_roi <= -0.4 THEN 'LOSS_40_TO_50'
      WHEN token_roi <= -0.3 THEN 'LOSS_30_TO_40'
      WHEN token_roi <= -0.2 THEN 'LOSS_20_TO_30'
      WHEN token_roi <= -0.1 THEN 'LOSS_10_TO_20'
      WHEN token_roi < 0 THEN 'LOSS_0_TO_10'
      -- 0% 정확히 (변동 없음)
      WHEN token_roi = 0 THEN 'BREAKEVEN'
      -- 양수 ROI (탄력적 간격)
      WHEN token_roi <= 0.1 THEN 'GAIN_0_TO_10'
      WHEN token_roi <= 0.25 THEN 'GAIN_10_TO_25'
      WHEN token_roi <= 0.5 THEN 'GAIN_25_TO_50'
      WHEN token_roi <= 1.0 THEN 'GAIN_50_TO_100'
      WHEN token_roi <= 2.0 THEN 'GAIN_100_TO_200'
      WHEN token_roi <= 5.0 THEN 'GAIN_200_TO_500'
      WHEN token_roi <= 10.0 THEN 'GAIN_500_TO_1000'
      WHEN token_roi <= 20.0 THEN 'GAIN_1000_TO_2000'
      ELSE 'GAIN_OVER_2000'
    END as roi_bucket,
    -- ROI 구간 순서 (분석 및 시각화용)
    CASE
      WHEN token_roi IS NULL THEN 0
      WHEN token_roi <= -0.9 THEN 1
      WHEN token_roi <= -0.8 THEN 2
      WHEN token_roi <= -0.7 THEN 3
      WHEN token_roi <= -0.6 THEN 4
      WHEN token_roi <= -0.5 THEN 5
      WHEN token_roi <= -0.4 THEN 6
      WHEN token_roi <= -0.3 THEN 7
      WHEN token_roi <= -0.2 THEN 8
      WHEN token_roi <= -0.1 THEN 9
      WHEN token_roi < 0 THEN 10
      WHEN token_roi = 0 THEN 11
      WHEN token_roi <= 0.1 THEN 12
      WHEN token_roi <= 0.25 THEN 13
      WHEN token_roi <= 0.5 THEN 14
      WHEN token_roi <= 1.0 THEN 15
      WHEN token_roi <= 2.0 THEN 16
      WHEN token_roi <= 5.0 THEN 17
      WHEN token_roi <= 10.0 THEN 18
      WHEN token_roi <= 20.0 THEN 19
      ELSE 20
    END as roi_bucket_order
  FROM 
    token_roi
),

wallet_roi_bucket_stats AS (
  SELECT
    wallet_address,
    roi_bucket,
    COUNT(*) as token_count,
    AVG(token_roi) as avg_roi_in_bucket,
    MIN(token_roi) as min_roi_in_bucket,
    MAX(token_roi) as max_roi_in_bucket,
    MIN(roi_bucket_order) as bucket_order
  FROM 
    roi_buckets
  WHERE 
    roi_bucket != 'UNKNOWN_ROI'
    AND potential_pre_2025_purchase = false
  GROUP BY 
    wallet_address, roi_bucket
),

wallet_token_stats AS (
  SELECT
    wallet_address,
    COUNT(*) as total_token_count,
    -- 유효한 ROI 계산 가능 토큰 수
    SUM(CASE WHEN token_roi IS NOT NULL AND potential_pre_2025_purchase = false THEN 1 ELSE 0 END) as valid_roi_token_count,
    -- 2025년 1월 이전 구매 가능성 있는 토큰 수
    SUM(CASE WHEN potential_pre_2025_purchase = true THEN 1 ELSE 0 END) as pre_2025_token_count,
    -- 토큰 상태별 카운트
    SUM(CASE WHEN token_status = 'HOLDING' THEN 1 ELSE 0 END) as holding_token_count,
    SUM(CASE WHEN token_status = 'FULLY_SOLD' THEN 1 ELSE 0 END) as fully_sold_token_count,
    SUM(CASE WHEN token_status = 'PARTIALLY_SOLD' THEN 1 ELSE 0 END) as partially_sold_token_count,
    -- ROI 양수/음수 토큰 카운트
    SUM(CASE WHEN token_roi > 0 THEN 1 ELSE 0 END) as positive_roi_token_count,
    SUM(CASE WHEN token_roi < 0 THEN 1 ELSE 0 END) as negative_roi_token_count,
    -- 유효한 토큰에 대한 평균 ROI
    AVG(CASE WHEN token_roi IS NOT NULL AND potential_pre_2025_purchase = false THEN token_roi ELSE NULL END) as avg_token_roi,
    -- 데이터 품질 지표
    CASE 
      WHEN COUNT(*) > 0 THEN 
        SUM(CASE WHEN potential_pre_2025_purchase = true THEN 1 ELSE 0 END)::FLOAT / COUNT(*)
      ELSE 0
    END as pre_2025_token_ratio,
    -- ROI 계산 가능 토큰 비율
    CASE 
      WHEN COUNT(*) > 0 THEN 
        SUM(CASE WHEN token_roi IS NOT NULL AND potential_pre_2025_purchase = false THEN 1 ELSE 0 END)::FLOAT / COUNT(*)
      ELSE 0
    END as valid_roi_token_ratio
  FROM 
    roi_buckets
  GROUP BY 
    wallet_address
),

wallet_roi_distribution AS (
  SELECT
    r.wallet_address,
    r.roi_bucket,
    r.token_count,
    s.valid_roi_token_count,
    -- 확률 계산: 해당 ROI 구간의 토큰 수 / 유효한 ROI 토큰 수
    CASE 
      WHEN s.valid_roi_token_count > 0 THEN 
        r.token_count::FLOAT / s.valid_roi_token_count
      ELSE 0
    END as probability,
    r.avg_roi_in_bucket,
    r.bucket_order
  FROM 
    wallet_roi_bucket_stats r
    JOIN wallet_token_stats s ON r.wallet_address = s.wallet_address
),

wallet_expected_roi_base AS (
  SELECT
    w.wallet_address,
    -- 기대 수익률 = 각 ROI 구간의 (확률 × 평균 ROI)의 합
    SUM(w.probability * w.avg_roi_in_bucket) as expected_roi,
    -- 양수/음수 ROI 확률
    SUM(CASE WHEN w.avg_roi_in_bucket > 0 THEN w.probability ELSE 0 END) as prob_positive_roi,
    SUM(CASE WHEN w.avg_roi_in_bucket < 0 THEN w.probability ELSE 0 END) as prob_negative_roi,
    -- 토큰 정보
    MAX(s.total_token_count) as total_token_count,
    MAX(s.valid_roi_token_count) as valid_roi_token_count,
    MAX(s.pre_2025_token_count) as pre_2025_token_count,
    MAX(s.pre_2025_token_ratio) as pre_2025_token_ratio,
    MAX(s.valid_roi_token_ratio) as valid_roi_token_ratio,
    MAX(s.holding_token_count) as holding_token_count,
    MAX(s.fully_sold_token_count) as fully_sold_token_count,
    MAX(s.partially_sold_token_count) as partially_sold_token_count,
    MAX(s.avg_token_roi) as avg_token_roi
  FROM 
    wallet_roi_distribution w
    JOIN wallet_token_stats s ON w.wallet_address = s.wallet_address
  GROUP BY 
    w.wallet_address
),

wallet_expected_returns AS (
  SELECT 
    b.wallet_address,
    b.expected_roi,
    b.prob_positive_roi,
    b.prob_negative_roi,
    -- 분산 및 표준편차 계산
    SUM(w.probability * POWER(w.avg_roi_in_bucket - b.expected_roi, 2)) as variance,
    SQRT(SUM(w.probability * POWER(w.avg_roi_in_bucket - b.expected_roi, 2))) as std_dev,
    -- 위험조정 기대수익률 (Sharpe ratio 아이디어 차용)
    CASE 
      WHEN SQRT(SUM(w.probability * POWER(w.avg_roi_in_bucket - b.expected_roi, 2))) > 0 
      THEN b.expected_roi / SQRT(SUM(w.probability * POWER(w.avg_roi_in_bucket - b.expected_roi, 2)))
      ELSE NULL
    END as risk_adjusted_roi,
    -- 기타 토큰 정보는 기본 CTE에서 가져옴
    b.total_token_count,
    b.valid_roi_token_count,
    b.pre_2025_token_count,
    b.pre_2025_token_ratio,
    b.valid_roi_token_ratio,
    b.holding_token_count,
    b.fully_sold_token_count,
    b.partially_sold_token_count,
    b.avg_token_roi
  FROM 
    wallet_expected_roi_base b
    JOIN wallet_roi_distribution w ON b.wallet_address = w.wallet_address
  GROUP BY 
    b.wallet_address,
    b.expected_roi,
    b.prob_positive_roi,
    b.prob_negative_roi,
    b.total_token_count,
    b.valid_roi_token_count,
    b.pre_2025_token_count,
    b.pre_2025_token_ratio,
    b.valid_roi_token_ratio,
    b.holding_token_count,
    b.fully_sold_token_count,
    b.partially_sold_token_count,
    b.avg_token_roi
),

expected_returns_data AS (
  SELECT
    e.wallet_address,
    e.expected_roi,
    CASE
      WHEN e.expected_roi IS NULL THEN 'UNKNOWN_ROI'
      -- 음수 ROI (10% 간격)
      WHEN e.expected_roi <= -0.9 THEN 'LOSS_90_TO_100'
      WHEN e.expected_roi <= -0.8 THEN 'LOSS_80_TO_90'
      WHEN e.expected_roi <= -0.7 THEN 'LOSS_70_TO_80'
      WHEN e.expected_roi <= -0.6 THEN 'LOSS_60_TO_70'
      WHEN e.expected_roi <= -0.5 THEN 'LOSS_50_TO_60'
      WHEN e.expected_roi <= -0.4 THEN 'LOSS_40_TO_50'
      WHEN e.expected_roi <= -0.3 THEN 'LOSS_30_TO_40'
      WHEN e.expected_roi <= -0.2 THEN 'LOSS_20_TO_30'
      WHEN e.expected_roi <= -0.1 THEN 'LOSS_10_TO_20'
      WHEN e.expected_roi < 0 THEN 'LOSS_0_TO_10'
      -- 0% 정확히 (변동 없음)
      WHEN e.expected_roi = 0 THEN 'BREAKEVEN'
      -- 양수 ROI (탄력적 간격)
      WHEN e.expected_roi <= 0.1 THEN 'GAIN_0_TO_10'
      WHEN e.expected_roi <= 0.25 THEN 'GAIN_10_TO_25'
      WHEN e.expected_roi <= 0.5 THEN 'GAIN_25_TO_50'
      WHEN e.expected_roi <= 1.0 THEN 'GAIN_50_TO_100'
      WHEN e.expected_roi <= 2.0 THEN 'GAIN_100_TO_200'
      WHEN e.expected_roi <= 5.0 THEN 'GAIN_200_TO_500'
      WHEN e.expected_roi <= 10.0 THEN 'GAIN_500_TO_1000'
      WHEN e.expected_roi <= 20.0 THEN 'GAIN_1000_TO_2000'
      ELSE 'GAIN_OVER_2000'
    END as expected_roi_category,
    e.risk_adjusted_roi,
    e.std_dev as roi_std_dev,
    e.prob_positive_roi,
    e.prob_negative_roi,
    e.total_token_count,
    e.valid_roi_token_count,
    e.pre_2025_token_count,
    e.pre_2025_token_ratio,
    e.valid_roi_token_ratio,
    e.holding_token_count,
    u.has_exited,
    u.exit_status,
    -- 데이터 품질 지표
    CASE 
      WHEN e.pre_2025_token_ratio > 0.3 THEN false -- 30% 이상이 2025년 1월 이전 구매로 추정됨
      WHEN e.valid_roi_token_count < 5 THEN false -- 유효한 ROI 토큰 수가 5개 미만
      WHEN e.valid_roi_token_ratio < 0.7 THEN false -- 70% 미만의 토큰만 ROI 계산 가능
      ELSE true
    END as is_data_reliable
  FROM 
    wallet_expected_returns e
    JOIN user_activity_status u ON e.wallet_address = u.wallet_address
),

-- 2. 기대수익률 구간별 이탈률 계산
exit_rates_by_expected_roi_category AS (
  SELECT 
    expected_roi_category,
    CAST(COUNT(*) AS NUMBER) as wallet_count,
    CAST(SUM(has_exited) AS NUMBER) as exited_count,
    (CAST(SUM(has_exited) AS NUMBER) / NULLIF(CAST(COUNT(*) AS NUMBER), 0)) as exit_rate,
    AVG(expected_roi) as avg_expected_roi,
    AVG(CASE WHEN has_exited = 1 THEN expected_roi ELSE NULL END) as avg_expected_roi_exited,
    AVG(CASE WHEN has_exited = 0 THEN expected_roi ELSE NULL END) as avg_expected_roi_active,
    -- 위험 조정 수익률 기준
    AVG(risk_adjusted_roi) as avg_risk_adjusted_roi,
    AVG(CASE WHEN has_exited = 1 THEN risk_adjusted_roi ELSE NULL END) as avg_risk_adjusted_roi_exited,
    AVG(CASE WHEN has_exited = 0 THEN risk_adjusted_roi ELSE NULL END) as avg_risk_adjusted_roi_active,
    -- 긍정적 수익 확률
    AVG(prob_positive_roi) as avg_prob_positive_roi,
    AVG(CASE WHEN has_exited = 1 THEN prob_positive_roi ELSE NULL END) as avg_prob_positive_roi_exited,
    AVG(CASE WHEN has_exited = 0 THEN prob_positive_roi ELSE NULL END) as avg_prob_positive_roi_active,
    -- 데이터 품질
    CAST(COUNT(CASE WHEN is_data_reliable = true THEN 1 END) AS NUMBER) as reliable_wallet_count,
    CAST(SUM(CASE WHEN is_data_reliable = true THEN has_exited ELSE 0 END) AS NUMBER) as reliable_exited_count,
    CASE 
      WHEN COUNT(CASE WHEN is_data_reliable = true THEN 1 END) > 0 
      THEN CAST(SUM(CASE WHEN is_data_reliable = true THEN has_exited ELSE 0 END) AS NUMBER) / 
           NULLIF(CAST(COUNT(CASE WHEN is_data_reliable = true THEN 1 END) AS NUMBER), 0)
      ELSE NULL
    END as reliable_exit_rate
  FROM 
    expected_returns_data
  GROUP BY 
    expected_roi_category
  ORDER BY 
    avg_expected_roi
),

-- 3. 기대수익률 데실(십분위수)별 이탈률 계산
exit_rates_by_expected_roi_decile AS (
  WITH roi_deciles AS (
    SELECT 
      wallet_address,
      expected_roi,
      has_exited,
      is_data_reliable,
      NTILE(5) OVER (ORDER BY expected_roi) as expected_roi_decile  -- 10에서 5로 감소
    FROM 
      expected_returns_data
  )
  SELECT 
    expected_roi_decile,
    CAST(COUNT(*) AS NUMBER) as wallet_count,
    CAST(SUM(has_exited) AS NUMBER) as exited_count,
    (CAST(SUM(has_exited) AS NUMBER) / NULLIF(CAST(COUNT(*) AS NUMBER), 0)) as exit_rate,
    MIN(expected_roi) as min_expected_roi,
    MAX(expected_roi) as max_expected_roi,
    AVG(expected_roi) as avg_expected_roi,
    MEDIAN(expected_roi) as median_expected_roi,
    AVG(CASE WHEN has_exited = 1 THEN expected_roi ELSE NULL END) as avg_expected_roi_exited,
    AVG(CASE WHEN has_exited = 0 THEN expected_roi ELSE NULL END) as avg_expected_roi_active,
    -- 신뢰할 수 있는 데이터만 사용한 이탈률
    CAST(COUNT(CASE WHEN is_data_reliable = true THEN 1 END) AS NUMBER) as reliable_wallet_count,
    CAST(SUM(CASE WHEN is_data_reliable = true THEN has_exited ELSE 0 END) AS NUMBER) as reliable_exited_count,
    CASE 
      WHEN COUNT(CASE WHEN is_data_reliable = true THEN 1 END) > 0 
      THEN CAST(SUM(CASE WHEN is_data_reliable = true THEN has_exited ELSE 0 END) AS NUMBER) / 
           NULLIF(CAST(COUNT(CASE WHEN is_data_reliable = true THEN 1 END) AS NUMBER), 0)
      ELSE NULL
    END as reliable_exit_rate
  FROM 
    roi_deciles
  GROUP BY 
    expected_roi_decile
  ORDER BY 
    expected_roi_decile
),

-- 4. 상관관계 지표 계산
correlation_metrics AS (
  SELECT 
    -- 피어슨 상관계수 (기대수익률과 이탈 간)
    CORR(expected_roi, has_exited) as roi_exit_correlation,
    
    -- 로지스틱 회귀 유사 계산 (경사도 추정)
    REGR_SLOPE(has_exited, expected_roi) as exit_roi_slope,
    REGR_INTERCEPT(has_exited, expected_roi) as exit_roi_intercept,
    
    -- 위험 조정 수익률과 이탈 간 상관관계
    CORR(risk_adjusted_roi, has_exited) as risk_adjusted_roi_exit_correlation,
    
    -- 긍정적 수익 확률과 이탈 간 상관관계
    CORR(prob_positive_roi, has_exited) as prob_positive_exit_correlation,
    
    -- 전체 통계
    AVG(expected_roi) as avg_expected_roi,
    AVG(CASE WHEN has_exited = 1 THEN expected_roi ELSE NULL END) as avg_expected_roi_exited,
    AVG(CASE WHEN has_exited = 0 THEN expected_roi ELSE NULL END) as avg_expected_roi_active,
    
    -- 이탈률
    CAST(SUM(has_exited) AS NUMBER) as total_exited,
    CAST(COUNT(*) AS NUMBER) as total_wallets,
    (CAST(SUM(has_exited) AS NUMBER) / NULLIF(CAST(COUNT(*) AS NUMBER), 0)) as overall_exit_rate,
    
    -- 신뢰할 수 있는 데이터만 사용한 상관관계
    CORR(CASE WHEN is_data_reliable = true THEN expected_roi ELSE NULL END, 
         CASE WHEN is_data_reliable = true THEN has_exited ELSE NULL END) as reliable_roi_exit_correlation,
         
    -- 신뢰할 수 있는 데이터만 사용한 이탈률
    CAST(COUNT(CASE WHEN is_data_reliable = true THEN 1 END) AS NUMBER) as reliable_wallet_count,
    CAST(SUM(CASE WHEN is_data_reliable = true THEN has_exited ELSE 0 END) AS NUMBER) as reliable_exited_count,
    CASE 
      WHEN COUNT(CASE WHEN is_data_reliable = true THEN 1 END) > 0 
      THEN CAST(SUM(CASE WHEN is_data_reliable = true THEN has_exited ELSE 0 END) AS NUMBER) / 
           NULLIF(CAST(COUNT(CASE WHEN is_data_reliable = true THEN 1 END) AS NUMBER), 0)
      ELSE NULL
    END as reliable_exit_rate
  FROM 
    expected_returns_data
),

-- 5. 기대수익률과 이탈률 간의 비선형 관계 탐색
nonlinear_patterns AS (
  SELECT 
    ROUND(expected_roi, 1) as roi_rounded,
    CAST(COUNT(*) AS NUMBER) as wallet_count,
    CAST(SUM(has_exited) AS NUMBER) as exited_count,
    (CAST(SUM(has_exited) AS NUMBER) / NULLIF(CAST(COUNT(*) AS NUMBER), 0)) as exit_rate,
    -- 신뢰할 수 있는 데이터만 사용한 이탈률
    CAST(COUNT(CASE WHEN is_data_reliable = true THEN 1 END) AS NUMBER) as reliable_wallet_count,
    CAST(SUM(CASE WHEN is_data_reliable = true THEN has_exited ELSE 0 END) AS NUMBER) as reliable_exited_count,
    CASE 
      WHEN COUNT(CASE WHEN is_data_reliable = true THEN 1 END) > 0 
      THEN CAST(SUM(CASE WHEN is_data_reliable = true THEN has_exited ELSE 0 END) AS NUMBER) / 
           NULLIF(CAST(COUNT(CASE WHEN is_data_reliable = true THEN 1 END) AS NUMBER), 0)
      ELSE NULL
    END as reliable_exit_rate
  FROM 
    expected_returns_data
  GROUP BY 
    ROUND(expected_roi, 1)
  HAVING 
    COUNT(*) >= 3  -- 샘플 수 기준 완화
  ORDER BY 
    roi_rounded
),

-- 6. 수익/손실 확률과 이탈률 분석
probability_analysis AS (
  SELECT 
    -- 양수 ROI 확률 그룹화
    CASE
      WHEN prob_positive_roi <= 0.2 THEN 'Very Low (0-20%)'
      WHEN prob_positive_roi <= 0.4 THEN 'Low (20-40%)'
      WHEN prob_positive_roi <= 0.6 THEN 'Medium (40-60%)'
      WHEN prob_positive_roi <= 0.8 THEN 'High (60-80%)'
      ELSE 'Very High (80-100%)'
    END as positive_roi_prob_group,
    
    CAST(COUNT(*) AS NUMBER) as wallet_count,
    CAST(SUM(has_exited) AS NUMBER) as exited_count,
    (CAST(SUM(has_exited) AS NUMBER) / NULLIF(CAST(COUNT(*) AS NUMBER), 0)) as exit_rate,
    AVG(expected_roi) as avg_expected_roi,
    AVG(roi_std_dev) as avg_roi_std_dev,
    -- 신뢰할 수 있는 데이터만 사용한 이탈률
    CAST(COUNT(CASE WHEN is_data_reliable = true THEN 1 END) AS NUMBER) as reliable_wallet_count,
    CAST(SUM(CASE WHEN is_data_reliable = true THEN has_exited ELSE 0 END) AS NUMBER) as reliable_exited_count,
    CASE 
      WHEN COUNT(CASE WHEN is_data_reliable = true THEN 1 END) > 0 
      THEN CAST(SUM(CASE WHEN is_data_reliable = true THEN has_exited ELSE 0 END) AS NUMBER) / 
           NULLIF(CAST(COUNT(CASE WHEN is_data_reliable = true THEN 1 END) AS NUMBER), 0)
      ELSE NULL
    END as reliable_exit_rate
  FROM 
    expected_returns_data
  GROUP BY 
    positive_roi_prob_group
  ORDER BY 
    avg_expected_roi
),

-- 7. 이탈 여부에 영향을 미치는 다중 요인 분석
-- (기대수익률, 변동성, 거래 토큰 수 등 종합적 고려)
multi_factor_analysis AS (
  SELECT 
    -- 기대수익률 구간 - 간소화
    CASE
      WHEN expected_roi < 0 THEN 'Negative'
      ELSE 'Positive'
    END as roi_group,
    
    -- 변동성 구간 - 간소화
    CASE
      WHEN roi_std_dev <= 0.3 THEN 'Low'
      ELSE 'High'
    END as volatility_group,
    
    -- 거래 토큰 수 구간 - 간소화
    CASE
      WHEN total_token_count <= 5 THEN 'Few'
      ELSE 'Many'
    END as token_count_group,
    
    CAST(COUNT(*) AS NUMBER) as wallet_count,
    CAST(SUM(has_exited) AS NUMBER) as exited_count,
    (CAST(SUM(has_exited) AS NUMBER) / NULLIF(CAST(COUNT(*) AS NUMBER), 0)) as exit_rate,
    
    -- 신뢰할 수 있는 데이터만 사용한 이탈률
    CAST(COUNT(CASE WHEN is_data_reliable = true THEN 1 END) AS NUMBER) as reliable_wallet_count,
    CAST(SUM(CASE WHEN is_data_reliable = true THEN has_exited ELSE 0 END) AS NUMBER) as reliable_exited_count,
    CASE 
      WHEN COUNT(CASE WHEN is_data_reliable = true THEN 1 END) > 0 
      THEN CAST(SUM(CASE WHEN is_data_reliable = true THEN has_exited ELSE 0 END) AS NUMBER) / 
           NULLIF(CAST(COUNT(CASE WHEN is_data_reliable = true THEN 1 END) AS NUMBER), 0)
      ELSE NULL
    END as reliable_exit_rate
  FROM 
    expected_returns_data
  GROUP BY 
    roi_group,
    volatility_group,
    token_count_group
  ORDER BY 
    roi_group,
    volatility_group,
    token_count_group
),

-- 8. 데이터 품질에 따른 상관관계 분석
data_quality_impact AS (
  SELECT
    'All Data' as data_segment,
    CAST(COUNT(*) AS NUMBER) as wallet_count,
    CAST(SUM(has_exited) AS NUMBER) as exited_count,
    (CAST(SUM(has_exited) AS NUMBER) / NULLIF(CAST(COUNT(*) AS NUMBER), 0)) as exit_rate,
    CORR(expected_roi, has_exited) as roi_exit_correlation,
    AVG(expected_roi) as avg_expected_roi,
    STDDEV(expected_roi) as std_expected_roi
  FROM
    expected_returns_data
  
  UNION ALL
  
  SELECT
    'Reliable Data Only' as data_segment,
    CAST(COUNT(*) AS NUMBER) as wallet_count,
    CAST(SUM(has_exited) AS NUMBER) as exited_count,
    (CAST(SUM(has_exited) AS NUMBER) / NULLIF(CAST(COUNT(*) AS NUMBER), 0)) as exit_rate,
    CORR(expected_roi, has_exited) as roi_exit_correlation,
    AVG(expected_roi) as avg_expected_roi,
    STDDEV(expected_roi) as std_expected_roi
  FROM
    expected_returns_data
  WHERE
    is_data_reliable = true
    
  UNION ALL
  
  SELECT
    'Unreliable Data Only' as data_segment,
    CAST(COUNT(*) AS NUMBER) as wallet_count,
    CAST(SUM(has_exited) AS NUMBER) as exited_count,
    (CAST(SUM(has_exited) AS NUMBER) / NULLIF(CAST(COUNT(*) AS NUMBER), 0)) as exit_rate,
    CORR(expected_roi, has_exited) as roi_exit_correlation,
    AVG(expected_roi) as avg_expected_roi,
    STDDEV(expected_roi) as std_expected_roi
  FROM
    expected_returns_data
  WHERE
    is_data_reliable = false
)

-- 최종 결과 출력
-- 각 분석 유형에 대해 일관된 열 구조를 유지하기 위해 NULL 값으로 채우거나 명시적 열 선택

-- 공통 열 구조: analysis_type, category, 데이터 열들

-- 1. 상관관계 지표
SELECT 
  'CORRELATION_METRICS' as analysis_type, 
  NULL as category, 
  roi_exit_correlation,
  exit_roi_slope,
  exit_roi_intercept,
  risk_adjusted_roi_exit_correlation,
  prob_positive_exit_correlation,
  avg_expected_roi,
  avg_expected_roi_exited,
  avg_expected_roi_active,
  total_exited,
  total_wallets,
  overall_exit_rate,
  reliable_roi_exit_correlation,
  reliable_wallet_count,
  reliable_exited_count,
  reliable_exit_rate,
  NULL as min_expected_roi,
  NULL as max_expected_roi,
  NULL as median_expected_roi,
  NULL as avg_risk_adjusted_roi,
  NULL as avg_risk_adjusted_roi_exited,
  NULL as avg_risk_adjusted_roi_active,
  NULL as avg_prob_positive_roi,
  NULL as avg_prob_positive_roi_exited,
  NULL as avg_prob_positive_roi_active,
  NULL as avg_roi_std_dev,
  NULL as std_expected_roi
FROM correlation_metrics

UNION ALL

-- 2. 기대수익률 구간별 이탈률
SELECT 
  'EXIT_RATES_BY_CATEGORY' as analysis_type, 
  expected_roi_category as category,
  NULL as roi_exit_correlation,
  NULL as exit_roi_slope,
  NULL as exit_roi_intercept,
  NULL as risk_adjusted_roi_exit_correlation,
  NULL as prob_positive_exit_correlation,
  avg_expected_roi,
  avg_expected_roi_exited,
  avg_expected_roi_active,
  exited_count as total_exited,
  wallet_count as total_wallets,
  exit_rate as overall_exit_rate,
  NULL as reliable_roi_exit_correlation,
  reliable_wallet_count,
  reliable_exited_count,
  reliable_exit_rate,
  NULL as min_expected_roi,
  NULL as max_expected_roi,
  NULL as median_expected_roi,
  avg_risk_adjusted_roi,
  avg_risk_adjusted_roi_exited,
  avg_risk_adjusted_roi_active,
  avg_prob_positive_roi,
  avg_prob_positive_roi_exited,
  avg_prob_positive_roi_active,
  avg_roi_std_dev,
  NULL as std_expected_roi
FROM exit_rates_by_expected_roi_category

UNION ALL

-- 3. 기대수익률 데실(십분위수)별 이탈률
SELECT 
  'EXIT_RATES_BY_DECILE' as analysis_type, 
  expected_roi_decile::VARCHAR as category,
  NULL as roi_exit_correlation,
  NULL as exit_roi_slope,
  NULL as exit_roi_intercept,
  NULL as risk_adjusted_roi_exit_correlation,
  NULL as prob_positive_exit_correlation,
  avg_expected_roi,
  avg_expected_roi_exited,
  avg_expected_roi_active,
  exited_count as total_exited,
  wallet_count as total_wallets,
  exit_rate as overall_exit_rate,
  NULL as reliable_roi_exit_correlation,
  reliable_wallet_count,
  reliable_exited_count,
  reliable_exit_rate,
  min_expected_roi,
  max_expected_roi,
  median_expected_roi,
  NULL as avg_risk_adjusted_roi,
  NULL as avg_risk_adjusted_roi_exited,
  NULL as avg_risk_adjusted_roi_active,
  NULL as avg_prob_positive_roi,
  NULL as avg_prob_positive_roi_exited,
  NULL as avg_prob_positive_roi_active,
  avg_roi_std_dev,
  NULL as std_expected_roi
FROM exit_rates_by_expected_roi_decile

UNION ALL

-- 4. 비선형 패턴 분석
SELECT 
  'NONLINEAR_PATTERNS' as analysis_type, 
  roi_rounded::VARCHAR as category,
  NULL as roi_exit_correlation,
  NULL as exit_roi_slope,
  NULL as exit_roi_intercept,
  NULL as risk_adjusted_roi_exit_correlation,
  NULL as prob_positive_exit_correlation,
  roi_rounded as avg_expected_roi,
  NULL as avg_expected_roi_exited, 
  NULL as avg_expected_roi_active,
  exited_count as total_exited,
  wallet_count as total_wallets,
  exit_rate as overall_exit_rate,
  NULL as reliable_roi_exit_correlation,
  reliable_wallet_count,
  reliable_exited_count,
  reliable_exit_rate,
  NULL as min_expected_roi,
  NULL as max_expected_roi,
  NULL as median_expected_roi,
  NULL as avg_risk_adjusted_roi,
  NULL as avg_risk_adjusted_roi_exited,
  NULL as avg_risk_adjusted_roi_active,
  NULL as avg_prob_positive_roi,
  NULL as avg_prob_positive_roi_exited,
  NULL as avg_prob_positive_roi_active,
  avg_roi_std_dev,
  NULL as std_expected_roi
FROM nonlinear_patterns

UNION ALL

-- 5. 확률 분석
SELECT 
  'PROBABILITY_ANALYSIS' as analysis_type, 
  positive_roi_prob_group as category,
  NULL as roi_exit_correlation,
  NULL as exit_roi_slope,
  NULL as exit_roi_intercept,
  NULL as risk_adjusted_roi_exit_correlation,
  NULL as prob_positive_exit_correlation,
  avg_expected_roi,
  NULL as avg_expected_roi_exited,
  NULL as avg_expected_roi_active,
  exited_count as total_exited,
  wallet_count as total_wallets,
  exit_rate as overall_exit_rate,
  NULL as reliable_roi_exit_correlation,
  reliable_wallet_count,
  reliable_exited_count,
  reliable_exit_rate,
  NULL as min_expected_roi,
  NULL as max_expected_roi,
  NULL as median_expected_roi,
  NULL as avg_risk_adjusted_roi,
  NULL as avg_risk_adjusted_roi_exited,
  NULL as avg_risk_adjusted_roi_active,
  NULL as avg_prob_positive_roi,
  NULL as avg_prob_positive_roi_exited,
  NULL as avg_prob_positive_roi_active,
  avg_roi_std_dev,
  NULL as std_expected_roi
FROM probability_analysis

UNION ALL

-- 6. 다중 요인 분석
SELECT 
  'MULTI_FACTOR_ANALYSIS' as analysis_type, 
  CONCAT(roi_group, ' ROI, ', volatility_group, ' Volatility, ', token_count_group, ' Tokens') as category,
  NULL as roi_exit_correlation,
  NULL as exit_roi_slope,
  NULL as exit_roi_intercept,
  NULL as risk_adjusted_roi_exit_correlation,
  NULL as prob_positive_exit_correlation,
  NULL as avg_expected_roi,
  NULL as avg_expected_roi_exited,
  NULL as avg_expected_roi_active,
  exited_count as total_exited,
  wallet_count as total_wallets,
  exit_rate as overall_exit_rate,
  NULL as reliable_roi_exit_correlation,
  reliable_wallet_count,
  reliable_exited_count,
  reliable_exit_rate,
  NULL as min_expected_roi,
  NULL as max_expected_roi,
  NULL as median_expected_roi,
  NULL as avg_risk_adjusted_roi,
  NULL as avg_risk_adjusted_roi_exited,
  NULL as avg_risk_adjusted_roi_active,
  NULL as avg_prob_positive_roi,
  NULL as avg_prob_positive_roi_exited,
  NULL as avg_prob_positive_roi_active,
  avg_roi_std_dev,
  NULL as std_expected_roi
FROM multi_factor_analysis

UNION ALL

-- 7. 데이터 품질에 따른 상관관계 분석
SELECT 
  'DATA_QUALITY_IMPACT' as analysis_type, 
  data_segment as category,
  roi_exit_correlation,
  NULL as exit_roi_slope,
  NULL as exit_roi_intercept,
  NULL as risk_adjusted_roi_exit_correlation,
  NULL as prob_positive_exit_correlation,
  avg_expected_roi,
  NULL as avg_expected_roi_exited,
  NULL as avg_expected_roi_active,
  exited_count as total_exited,
  wallet_count as total_wallets,
  exit_rate as overall_exit_rate,
  NULL as reliable_roi_exit_correlation,
  NULL as reliable_wallet_count,
  NULL as reliable_exited_count,
  NULL as reliable_exit_rate,
  NULL as min_expected_roi,
  NULL as max_expected_roi,
  NULL as median_expected_roi,
  NULL as avg_risk_adjusted_roi,
  NULL as avg_risk_adjusted_roi_exited,
  NULL as avg_risk_adjusted_roi_active,
  NULL as avg_prob_positive_roi,
  NULL as avg_prob_positive_roi_exited,
  NULL as avg_prob_positive_roi_active,
  avg_roi_std_dev,
  std_expected_roi
FROM data_quality_impact

ORDER BY analysis_type, category
LIMIT 1000; 
