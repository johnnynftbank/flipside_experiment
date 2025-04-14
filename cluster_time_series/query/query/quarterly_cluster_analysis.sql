/*
밈 코인 투자자 군집 분석을 위한 분기별 데이터 수집 쿼리
- 목적: 미분류 비율이 높은 문제 해결을 위해 분석 기간 확장
- 방식: 특정 월의 거래자 샘플링 후 해당 거래자의 이전 2개월 데이터까지 포함하여 분석 (총 3개월)

사용 방법:
- target_date 값만 원하는 월의 첫 날짜로 변경 (예: '2025-03-01')
- 나머지 날짜 관련 값들은 자동 계산됨
*/

WITH date_params AS (
  -- 여기서 target_date만 변경하면 나머지 날짜 관련 값은 자동 계산됨
  SELECT 
    '2025-03-01'::DATE AS target_date,
    LAST_DAY(target_date) AS target_end_date,
    EXTRACT(YEAR FROM target_date) AS target_year,
    EXTRACT(MONTH FROM target_date) AS target_month,
    DATEADD(month, -2, target_date) AS period_start_date,  -- 2개월 전 시작일 (총 3개월)
    LAST_DAY(target_date) AS period_end_date                -- 대상 월 마지막일
),

/* ============================= 1) 밈 코인 거래 전체 추출 ============================= */
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

/* ============================= 2) 대상 월의 거래자 중 조건 만족하는 지갑 랜덤 추출 ============================= */
sampled_wallets AS (
    SELECT
        swapper,
        COUNT(*) AS trade_count
    FROM meme_coin_swaps
    CROSS JOIN date_params
    WHERE block_timestamp BETWEEN date_params.target_date AND date_params.target_end_date
    GROUP BY swapper
    HAVING trade_count BETWEEN 10 AND 1000  -- 거래 횟수 10-1000회
    ORDER BY RANDOM()
    LIMIT 10000  -- 10,000개 지갑 추출
),

/* ============================= 3) 추출된 지갑의 3개월 거래 정보 추출 ============================= */
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
    JOIN sampled_wallets sw ON fs.swapper = sw.swapper
    CROSS JOIN date_params
    WHERE fs.block_timestamp BETWEEN date_params.period_start_date AND date_params.period_end_date
),

/* ============================= 4) 필터링 조건 적용 ============================= */
active_wallets AS (
    SELECT 
        swapper
    FROM raw_swaps
    GROUP BY swapper
    HAVING COUNT(DISTINCT DATE(block_timestamp)) >= 5  -- 거래 기간 5일 이상
      AND COUNT(DISTINCT CASE 
          WHEN swap_from_mint = 'So11111111111111111111111111111111111111112' THEN swap_to_mint
          ELSE swap_from_mint
      END) >= 5  -- 서로 다른 밈 코인 5개 이상
),

/* ============================= 5) 조건 충족 지갑만 필터링 ============================= */
filtered_swaps AS (
    SELECT rs.*
    FROM raw_swaps rs
    JOIN active_wallets aw ON rs.swapper = aw.swapper
),

/* ============================= 6) 토큰별 ROI 계산 ============================= */
token_performance AS (
    SELECT
        fs.swapper,
        CASE 
            WHEN fs.swap_from_mint = 'So11111111111111111111111111111111111111112' THEN fs.swap_to_mint
            ELSE fs.swap_from_mint
        END AS token_mint,
        SUM(CASE WHEN trade_type = 'BUY'  THEN sol_amount ELSE 0 END) AS total_buy,
        SUM(CASE WHEN trade_type = 'SELL' THEN sol_amount ELSE 0 END) AS total_sell
    FROM filtered_swaps fs
    GROUP BY 
        fs.swapper,
        CASE 
            WHEN fs.swap_from_mint = 'So11111111111111111111111111111111111111112' THEN fs.swap_to_mint
            ELSE fs.swap_from_mint
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

/* ============================= 7) ROI 구간화 & 기대수익률 계산 ============================= */
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

/* ============================= 8) 토큰별 ROI 표준편차 계산 ============================= */
roi_std_dev AS (
    SELECT
        tr.swapper,
        STDDEV(tr.roi) AS roi_standard_deviation,
        COUNT(DISTINCT tr.token_mint) AS unique_tokens_with_roi
    FROM token_roi tr
    GROUP BY tr.swapper
),

/* ============================= 9) 지갑별 Sharpe Ratio 계산 ============================= */
sharpe_ratio AS (
    SELECT
        wr.swapper,
        wr.expected_roi / NULLIF(sd.roi_standard_deviation, 0) AS sharpe_ratio
    FROM wallet_expected_roi wr
    JOIN roi_std_dev sd ON wr.swapper = sd.swapper
),

/* ============================= 10) Win-to-Loss 비율 계산 ============================= */
win_loss_ratio AS (
    SELECT
        swapper,
        SUM(CASE WHEN roi > 0 THEN 1 ELSE 0 END) AS winning_tokens,
        SUM(CASE WHEN roi < 0 THEN 1 ELSE 0 END) AS losing_tokens,
        SUM(CASE WHEN roi > 0 THEN 1 ELSE 0 END) / NULLIF(SUM(CASE WHEN roi < 0 THEN 1 ELSE 0 END), 0) AS win_loss_ratio
    FROM token_roi
    GROUP BY swapper
),

/* ============================= 11) 최대 단일 거래 비중 계산 ============================= */
max_trade_proportion AS (
    SELECT
        fs.swapper,
        MAX(fs.sol_amount) AS max_single_trade_amount,
        SUM(fs.sol_amount) AS total_buy_amount,
        MAX(fs.sol_amount) / NULLIF(SUM(fs.sol_amount), 0) AS max_trade_proportion
    FROM filtered_swaps fs
    WHERE fs.trade_type = 'BUY'
    GROUP BY fs.swapper
),

/* ============================= 12) 기타 지표 계산 ============================= */
wallet_summary AS (
    SELECT
        fs.swapper,
        COUNT(DISTINCT CASE 
            WHEN fs.swap_from_mint = 'So11111111111111111111111111111111111111112' 
            THEN fs.swap_to_mint
            ELSE fs.swap_from_mint
        END) AS unique_tokens_traded,
        COUNT(*) AS total_trades,
        MIN(fs.block_timestamp) AS first_trade_date,
        MAX(fs.block_timestamp) AS last_trade_date,
        COUNT(DISTINCT DATE(fs.block_timestamp)) AS trading_days,
        dp.target_date AS target_month_start,
        dp.target_end_date AS target_month_end,
        dp.period_start_date AS period_start_date,
        dp.period_end_date AS period_end_date,
        dp.target_year,
        dp.target_month
    FROM filtered_swaps fs
    CROSS JOIN date_params dp
    GROUP BY 
        fs.swapper, 
        dp.target_date, 
        dp.target_end_date,
        dp.period_start_date,
        dp.period_end_date,
        dp.target_year, 
        dp.target_month
),

/* ============================= 13) 최종 결과 집계 ============================= */
final_results AS (
    SELECT
        ws.swapper,
        wr.expected_roi,
        sd.roi_standard_deviation,
        sr.sharpe_ratio,
        wlr.win_loss_ratio,
        mtp.max_trade_proportion,
        ws.unique_tokens_traded,
        ws.total_trades,
        ws.first_trade_date,
        ws.last_trade_date,
        ws.trading_days,
        ws.target_year,
        ws.target_month,
        -- 분석 기간 정보 추가
        ws.target_month_start,
        ws.target_month_end,
        ws.period_start_date,
        ws.period_end_date
    FROM wallet_summary ws
    LEFT JOIN wallet_expected_roi wr ON ws.swapper = wr.swapper
    LEFT JOIN roi_std_dev sd ON ws.swapper = sd.swapper
    LEFT JOIN sharpe_ratio sr ON ws.swapper = sr.swapper
    LEFT JOIN win_loss_ratio wlr ON ws.swapper = wlr.swapper
    LEFT JOIN max_trade_proportion mtp ON ws.swapper = mtp.swapper
)

/* ============================= 최종 SELECT ============================= */
SELECT * 
FROM final_results
WHERE expected_roi IS NOT NULL
  AND roi_standard_deviation IS NOT NULL
  AND sharpe_ratio IS NOT NULL
  AND win_loss_ratio IS NOT NULL
ORDER BY expected_roi DESC; 