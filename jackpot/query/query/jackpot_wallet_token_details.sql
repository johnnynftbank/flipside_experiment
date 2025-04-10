WITH /* ============================= 1) 특정 지갑의 밈 코인 거래 추출 ============================= */
wallet_meme_swaps AS (
    SELECT
        fs.*,
        CASE 
            WHEN fs.swap_from_mint = 'So11111111111111111111111111111111111111112' THEN 'BUY'
            ELSE 'SELL'
        END AS trade_type,
        CASE 
            WHEN fs.swap_from_mint = 'So11111111111111111111111111111111111111112' THEN fs.swap_from_amount
            ELSE fs.swap_to_amount
        END AS sol_amount,
        CASE 
            WHEN fs.swap_from_mint = 'So11111111111111111111111111111111111111112' THEN fs.swap_to_mint
            ELSE fs.swap_from_mint
        END AS token_mint
    FROM solana.defi.fact_swaps fs
    WHERE fs.swap_program = 'pump.fun'
      AND (
          fs.swap_from_mint = 'So11111111111111111111111111111111111111112'
          OR fs.swap_to_mint = 'So11111111111111111111111111111111111111112'
      )
      AND fs.swapper = :wallet_address  -- 파라미터로 지갑 주소 입력
      AND fs.block_timestamp >= '2025-01-01'  -- 2025년 1월 1일 이후 데이터만
),

/* ============================= 2) 토큰별 거래 통계 계산 ============================= */
token_trade_summary AS (
    SELECT
        swapper AS wallet_address,
        token_mint,
        COUNT(*) AS trade_count,
        SUM(CASE WHEN trade_type = 'BUY' THEN sol_amount ELSE 0 END) AS total_buy_amount,
        SUM(CASE WHEN trade_type = 'SELL' THEN sol_amount ELSE 0 END) AS total_sell_amount,
        CASE 
            WHEN SUM(CASE WHEN trade_type = 'BUY' THEN sol_amount ELSE 0 END) > 0
            THEN (SUM(CASE WHEN trade_type = 'SELL' THEN sol_amount ELSE 0 END) - 
                 SUM(CASE WHEN trade_type = 'BUY' THEN sol_amount ELSE 0 END)) / 
                 SUM(CASE WHEN trade_type = 'BUY' THEN sol_amount ELSE 0 END)
            ELSE NULL
        END AS roi,
        MIN(block_timestamp) AS first_trade_time,
        MAX(block_timestamp) AS last_trade_time,
        DATEDIFF('minute', MIN(block_timestamp), MAX(block_timestamp)) AS holding_period_minutes
    FROM wallet_meme_swaps
    GROUP BY swapper, token_mint
)

/* ============================= 최종 SELECT ============================= */
SELECT
    wallet_address,
    token_mint,
    trade_count,
    total_buy_amount,
    total_sell_amount,
    roi,
    first_trade_time,
    last_trade_time,
    holding_period_minutes
FROM token_trade_summary
ORDER BY trade_count DESC; 