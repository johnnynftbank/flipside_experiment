WITH /* ============================= 1) 기본 밈 코인 거래 샘플링 & 필터링 ============================= */
meme_coin_swaps AS (
    SELECT
        fs.*
    FROM solana.defi.fact_swaps fs
    WHERE fs.swap_program = 'pump.fun'
      AND (
          fs.swap_from_mint = 'So11111111111111111111111111111111111111112'
          OR fs.swap_to_mint   = 'So11111111111111111111111111111111111111112'
      )
),
meme_coin_traders AS (
    -- 2025-02-01 ~ 2025-02-28 사이에 10~1000건
    SELECT
        swapper,
        COUNT(*) AS trade_count
    FROM meme_coin_swaps
    WHERE block_timestamp BETWEEN '2025-02-01' AND '2025-02-28'
    GROUP BY swapper
    HAVING trade_count BETWEEN 10 AND 1000
    ORDER BY RANDOM()
    LIMIT 10000
),
raw_swaps AS (
    -- 추출된 지갑들(1000개)의 25년 1월 1일 이후 밈 코인 거래 내역
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
            WHEN fs.swap_from_mint = 'So11111111111111111111111111111111111111112' THEN fs.swap_to_amount
            ELSE fs.swap_from_amount
        END AS token_amount
    FROM meme_coin_swaps fs
    JOIN meme_coin_traders mt 
        ON fs.swapper = mt.swapper
    WHERE fs.block_timestamp >= '2025-01-01'
),
active_wallets AS (
    -- 일회성 거래 지갑(25년 1월 1일 이후, 거래 일수 2일 미만) 제외
    SELECT 
        swapper
    FROM raw_swaps
    GROUP BY swapper
    HAVING COUNT(DISTINCT DATE(block_timestamp)) >= 2
),
wallet_activity AS (
    -- 이탈 여부(3/1 이후 거래 여부)
    SELECT
        swapper,
        MAX(CASE WHEN block_timestamp >= '2025-03-01' THEN 1 ELSE 0 END) AS is_active_after_mar1
    FROM raw_swaps
    WHERE swapper IN (SELECT swapper FROM active_wallets)
    GROUP BY swapper
),

/* ============================= 2) 간단(총매수 vs 총매도) ROI 계산 ============================= */
token_performance AS (
    SELECT
        rs.swapper,
        CASE 
            WHEN rs.swap_from_mint = 'So11111111111111111111111111111111111111112' THEN rs.swap_to_mint
            ELSE rs.swap_from_mint
        END AS token_mint,
        SUM(CASE WHEN trade_type = 'BUY'  THEN sol_amount ELSE 0 END) AS total_buy,
        SUM(CASE WHEN trade_type = 'SELL' THEN sol_amount ELSE 0 END) AS total_sell
    FROM raw_swaps rs
    WHERE rs.swapper IN (SELECT swapper FROM active_wallets)
    GROUP BY 
        rs.swapper,
        CASE 
            WHEN rs.swap_from_mint = 'So11111111111111111111111111111111111111112' THEN rs.swap_to_mint
            ELSE rs.swap_from_mint
        END
),
token_roi AS (
    SELECT
        tp.swapper,
        tp.token_mint,
        tp.total_buy,
        tp.total_sell,
        CASE 
            WHEN tp.total_buy > 0 AND tp.total_sell > 0
            THEN (tp.total_sell - tp.total_buy) / tp.total_buy
            ELSE NULL
        END AS roi
    FROM token_performance tp
    WHERE tp.total_buy > 0
      AND tp.total_sell > 0
),

/* ============================= 3) ROI 구간화 & 기대수익률 계산 ============================= */
bucketed_roi AS (
    SELECT
        swapper,
        token_mint,
        roi,
        CASE
            WHEN roi < -0.9 THEN '-100%~-90%'
            WHEN roi < -0.8 THEN '-90%~-80%'
            WHEN roi < -0.7 THEN '-80%~-70%'
            WHEN roi < -0.6 THEN '-70%~-60%'
            WHEN roi < -0.5 THEN '-60%~-50%'
            WHEN roi < -0.4 THEN '-50%~-40%'
            WHEN roi < -0.3 THEN '-40%~-30%'
            WHEN roi < -0.2 THEN '-30%~-20%'
            WHEN roi < -0.1 THEN '-20%~-10%'
            WHEN roi < 0    THEN '-10%~0%'
            WHEN roi < 0.1  THEN '0%~10%'
            WHEN roi < 0.25 THEN '10%~25%'
            WHEN roi < 0.5  THEN '25%~50%'
            WHEN roi < 1.0  THEN '50%~100%'
            WHEN roi < 2.0  THEN '100%~200%'
            WHEN roi < 5.0  THEN '200%~500%'
            WHEN roi < 10.0 THEN '500%~1000%'
            ELSE '1000%+'
        END AS roi_bucket
    FROM token_roi
    WHERE roi IS NOT NULL
),
bucket_stats AS (
    SELECT
        swapper,
        roi_bucket,
        AVG(roi) AS bucket_avg_roi,
        COUNT(*) AS token_count_in_bucket
    FROM bucketed_roi
    GROUP BY swapper, roi_bucket
),
bucket_distribution AS (
    SELECT
        bs.*,
        SUM(bs.token_count_in_bucket) OVER (PARTITION BY bs.swapper) AS total_tokens
    FROM bucket_stats bs
),
wallet_expected_roi AS (
    SELECT
        bd.swapper,
        SUM(bd.bucket_avg_roi * (bd.token_count_in_bucket * 1.0 / bd.total_tokens)) AS expected_roi
    FROM bucket_distribution bd
    GROUP BY bd.swapper
),

/* ============================= 4) 최종 ROI 합산 (지갑 단위) ============================= */
final_wallet_roi AS (
    SELECT
        tr.swapper,
        COUNT(DISTINCT tr.token_mint) AS unique_tokens_traded,
        we.expected_roi,
        wa.is_active_after_mar1
    FROM token_roi tr
    LEFT JOIN wallet_expected_roi we 
        ON tr.swapper = we.swapper
    LEFT JOIN wallet_activity wa     
        ON tr.swapper = wa.swapper
    GROUP BY tr.swapper, we.expected_roi, wa.is_active_after_mar1
),

/* ============================= 5) 새롭게 요청하신 지표(거래 횟수, 마지막 거래일, 누적 거래 일수) ============================= */
wallet_summary AS (
    SELECT
        rs.swapper,
        COUNT(*) AS meme_trade_count,                          -- (1) 밈코인 거래 횟수
        MAX(DATE(rs.block_timestamp)) AS last_meme_date,       -- (2) 마지막 거래 날짜
        COUNT(DISTINCT DATE(rs.block_timestamp)) AS traded_days -- (3) 누적 거래 일수
    FROM raw_swaps rs
    GROUP BY rs.swapper
)

/* ============================= 6) 최종 결과 SELECT ============================= */
SELECT
    fwr.swapper,
    fwr.unique_tokens_traded,
    fwr.expected_roi,
    CASE WHEN fwr.is_active_after_mar1 = 0 THEN 'CHURNED' ELSE 'ACTIVE' END AS wallet_status,

    -- 새로 추가된 3개 지표
    ws.meme_trade_count,
    ws.last_meme_date,
    ws.traded_days

FROM final_wallet_roi fwr
LEFT JOIN wallet_summary ws
       ON fwr.swapper = ws.swapper
ORDER BY fwr.expected_roi DESC;
