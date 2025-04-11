WITH /* ============================= 1) 밈 코인 거래 샘플링 & 필터링 ============================= */
meme_coin_swaps AS (
    SELECT
        fs.*
    FROM solana.defi.fact_swaps fs
    WHERE fs.swap_program = 'pump.fun'
      AND (
          fs.swap_from_mint = 'So11111111111111111111111111111111111111112'
          OR fs.swap_to_mint = 'So11111111111111111111111111111111111111112'
      )
),
meme_coin_traders AS (
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
    SELECT 
        swapper
    FROM raw_swaps
    GROUP BY swapper
    HAVING COUNT(DISTINCT DATE(block_timestamp)) >= 10  -- 최소 10일 이상 거래
      AND COUNT(DISTINCT CASE 
          WHEN swap_from_mint = 'So11111111111111111111111111111111111111112' THEN swap_to_mint
          ELSE swap_from_mint
      END) >= 5  -- 최소 5개 이상의 토큰 거래
),

/* ============================= 2) 토큰별 ROI 계산 ============================= */
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

/* ============================= 4) 토큰별 ROI 표준편차 계산 ============================= */
roi_std_dev AS (
    SELECT
        tr.swapper,
        STDDEV(tr.roi) AS roi_standard_deviation,
        COUNT(DISTINCT tr.token_mint) AS unique_tokens_with_roi
    FROM token_roi tr
    GROUP BY tr.swapper
),

/* ============================= 5) 지갑별 Sharpe Ratio 계산 ============================= */
sharpe_ratio AS (
    SELECT
        wr.swapper,
        wr.expected_roi / NULLIF(sd.roi_standard_deviation, 0) AS sharpe_ratio
    FROM wallet_expected_roi wr
    JOIN roi_std_dev sd ON wr.swapper = sd.swapper
),

/* ============================= 6) Win-to-Loss 비율 계산 ============================= */
win_loss_ratio AS (
    SELECT
        swapper,
        SUM(CASE WHEN roi > 0 THEN 1 ELSE 0 END) AS winning_tokens,
        SUM(CASE WHEN roi < 0 THEN 1 ELSE 0 END) AS losing_tokens,
        SUM(CASE WHEN roi > 0 THEN 1 ELSE 0 END) / NULLIF(SUM(CASE WHEN roi < 0 THEN 1 ELSE 0 END), 0) AS win_loss_ratio
    FROM token_roi
    GROUP BY swapper
),

/* ============================= 7) 최대 단일 거래 비중 계산 ============================= */
max_trade_proportion AS (
    SELECT
        rs.swapper,
        MAX(rs.sol_amount) AS max_single_trade_amount,
        SUM(rs.sol_amount) AS total_buy_amount,
        MAX(rs.sol_amount) / NULLIF(SUM(rs.sol_amount), 0) AS max_trade_proportion
    FROM raw_swaps rs
    WHERE rs.trade_type = 'BUY'
      AND rs.swapper IN (SELECT swapper FROM active_wallets)
    GROUP BY rs.swapper
),

/* ============================= 8) 토큰 수익률 극값 계산 ============================= */
wallet_roi_extremes AS (
    SELECT
        swapper,
        MAX(roi) AS max_token_roi,
        MIN(roi) AS min_token_roi,
        MAX(roi) - MIN(roi) AS roi_range
    FROM token_roi
    WHERE roi IS NOT NULL
    GROUP BY swapper
),

/* ============================= 9) 기타 지표 계산 ============================= */
wallet_summary AS (
    SELECT
        rs.swapper,
        COUNT(DISTINCT CASE 
            WHEN rs.swap_from_mint = 'So11111111111111111111111111111111111111112' 
            THEN rs.swap_to_mint
            ELSE rs.swap_from_mint
        END) AS unique_tokens_traded,
        COUNT(*) AS total_trades,
        MIN(rs.block_timestamp) AS first_trade_date,
        MAX(rs.block_timestamp) AS last_trade_date,
        COUNT(DISTINCT DATE(rs.block_timestamp)) AS trading_days
    FROM raw_swaps rs
    WHERE rs.swapper IN (SELECT swapper FROM active_wallets)
    GROUP BY rs.swapper
),

/* ============================= 7) ROI 통계 계산 ============================= */
roi_quartiles AS (
    SELECT 
        swapper,
        PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY roi) as q1_roi,
        PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY roi) as q3_roi,
        PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY roi) - 
        PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY roi) as iqr_roi
    FROM token_roi
    GROUP BY swapper
),

roi_statistics AS (
    SELECT 
        swapper,
        AVG(roi) as expected_roi,
        STDDEV(roi) as roi_standard_deviation,
        COUNT(CASE WHEN roi > 0 THEN 1 END)::FLOAT / NULLIF(COUNT(*), 0) as win_loss_ratio
    FROM token_roi
    GROUP BY swapper
),

/* ============================= 10) 최종 결과 집계 ============================= */
final_results AS (
    SELECT
        ws.swapper,
        sd.expected_roi,
        sd.roi_standard_deviation,
        sr.sharpe_ratio,
        sd.win_loss_ratio,
        mtp.max_trade_proportion,
        wre.max_token_roi,
        wre.min_token_roi,
        wre.roi_range,
        rq.q1_roi,
        rq.q3_roi,
        rq.iqr_roi,
        ws.unique_tokens_traded,
        ws.total_trades,
        ws.first_trade_date,
        ws.last_trade_date,
        ws.trading_days
    FROM wallet_summary ws
    LEFT JOIN roi_statistics sd ON ws.swapper = sd.swapper
    LEFT JOIN sharpe_ratio sr ON ws.swapper = sr.swapper
    LEFT JOIN max_trade_proportion mtp ON ws.swapper = mtp.swapper
    LEFT JOIN wallet_roi_extremes wre ON ws.swapper = wre.swapper
    LEFT JOIN roi_quartiles rq ON ws.swapper = rq.swapper
)

/* ============================= 최종 SELECT ============================= */
SELECT * 
FROM final_results
WHERE expected_roi IS NOT NULL
  AND roi_standard_deviation IS NOT NULL
  AND sharpe_ratio IS NOT NULL
  AND win_loss_ratio IS NOT NULL
ORDER BY expected_roi DESC; 