-- 1단계: allFeePayments CTE
-- GMGN 봇('BB5dnY55FXS1e1NXqZDwCzgdYJdMCj3B92PU6Q5Fb6DT')이 받은 수수료를 추적
WITH
allFeePayments AS (
    SELECT
      tx_id,
      'SOL' AS feeTokenType,
      balance_change / 1e9 AS fee_token_amount,  -- SOL 단위 변환 (lamports -> SOL)
      'So11111111111111111111111111111111111111112' AS fee_token_mint_address 
    FROM
      solana.account_activity
    WHERE
      tx_success
      AND address = 'BB5dnY55FXS1e1NXqZDwCzgdYJdMCj3B92PU6Q5Fb6DT'  -- GMGN 봇 주소
      AND balance_change > 0  -- 양수 잔액 변화만 (수수료 수입)
),

-- 2단계: botTrades CTE
-- DEX 거래 데이터와 수수료 데이터를 결합하고 USD 가치 계산
botTrades AS (
    SELECT
      block_time,
      amount_usd,  -- 거래 금액(USD)
      fee_token_amount * price AS fee_usd  -- 수수료 금액(USD)
    FROM
      dex_solana.trades AS trades
      JOIN allFeePayments AS feePayments ON trades.tx_id = feePayments.tx_id
      LEFT JOIN prices.usd AS feeTokenPrices ON (
        feeTokenPrices.blockchain = 'solana'
        AND fee_token_mint_address = toBase58(feeTokenPrices.contract_address)
        AND date_trunc('minute', block_time) = minute
      )
    WHERE
      trades.trader_id != 'BB5dnY55FXS1e1NXqZDwCzgdYJdMCj3B92PU6Q5Fb6DT'  -- 봇 자신의 거래는 제외
)

-- 3단계: 최종 집계
-- 일별 거래량과 수수료 합계 계산

