WITH raw_swaps AS (
  SELECT
    *,
    swap_from_mint AS send_token,
    swap_to_mint AS receive_token,
    swap_from_amount AS send_amount,
    swap_to_amount AS receive_amount
  FROM solana.defi.fact_swaps
  WHERE swapper = 'ExsQgvYyo5ZwASyY46UwAsY8QuskcTEYuSt4fURvwiK'
    AND swap_program = 'pump.fun'
),

labeled_swaps AS (
  SELECT
    *,
    -- 거래 유형 분류
    CASE
      WHEN send_token = 'So11111111111111111111111111111111111111112' THEN 'BUY'
      WHEN receive_token = 'So11111111111111111111111111111111111111112' THEN 'SELL'
      ELSE 'OTHER'
    END AS trade_type,

    -- 실제 거래한 토큰 식별
    CASE
      WHEN send_token = 'So11111111111111111111111111111111111111112' THEN receive_token
      WHEN receive_token = 'So11111111111111111111111111111111111111112' THEN send_token
      ELSE NULL
    END AS trade_token,

    -- 단가 계산
    CASE
      WHEN send_token = 'So11111111111111111111111111111111111111112' AND receive_amount != 0 THEN send_amount / receive_amount
      WHEN receive_token = 'So11111111111111111111111111111111111111112' AND send_amount != 0 THEN receive_amount / send_amount
      ELSE NULL
    END AS price
  FROM raw_swaps
),

first_buys AS (
  SELECT trade_token, tx_id AS buy_tx_id
  FROM labeled_swaps
  WHERE trade_type = 'BUY'
  QUALIFY ROW_NUMBER() OVER (PARTITION BY trade_token ORDER BY block_timestamp ASC) = 1
),

last_sells AS (
  SELECT trade_token, tx_id AS sell_tx_id
  FROM labeled_swaps
  WHERE trade_type = 'SELL'
  QUALIFY ROW_NUMBER() OVER (PARTITION BY trade_token ORDER BY block_timestamp DESC) = 1
),

buy_sell_ids AS (
  SELECT
    COALESCE(first_buys.trade_token, last_sells.trade_token) AS trade_token,
    first_buys.buy_tx_id,
    last_sells.sell_tx_id
  FROM first_buys
  FULL OUTER JOIN last_sells
    ON first_buys.trade_token = last_sells.trade_token
)

-- 정제된 요약 테이블 생성
SELECT 
  labeled_swaps.trade_token,

  -- 구매 요약
  avg(CASE WHEN trade_type = 'BUY' THEN price END) AS buy_price,
  sum(CASE WHEN trade_type = 'BUY' THEN receive_amount END) AS buy_amount,
  min(CASE WHEN trade_type = 'BUY' THEN block_timestamp END) AS buy_timestamp,

  -- 판매 요약
  avg(CASE WHEN trade_type = 'SELL' THEN price END) AS sell_price,
  sum(CASE WHEN trade_type = 'SELL' THEN send_amount END) AS sell_amount,
  max(CASE WHEN trade_type = 'SELL' THEN block_timestamp END) AS sell_timestamp,

  -- 손익 추정치 (PnL)
  (sum(CASE WHEN trade_type = 'SELL' THEN price * send_amount END) -
   sum(CASE WHEN trade_type = 'BUY' THEN price * receive_amount END)) AS pnl,

  -- 총 거래 금액
  sum(CASE WHEN trade_type = 'BUY' THEN price * receive_amount END) AS buy_value,
  sum(CASE WHEN trade_type = 'SELL' THEN price * send_amount END) AS sell_value,

  -- 포지션 상태
  CASE
    WHEN sum(CASE WHEN trade_type = 'BUY' THEN receive_amount END) = sum(CASE WHEN trade_type = 'SELL' THEN send_amount END) THEN 'closed'
    WHEN sum(CASE WHEN trade_type = 'BUY' THEN receive_amount END) > sum(CASE WHEN trade_type = 'SELL' THEN send_amount END) THEN 'open'
    ELSE 'partial'
  END AS position_status,

  -- 보유 기간
  timestampdiff(hour,
    min(CASE WHEN trade_type = 'BUY' THEN block_timestamp END),
    max(CASE WHEN trade_type = 'SELL' THEN block_timestamp END)
  ) AS holding_duration_hours,

  timestampdiff(minute,
    min(CASE WHEN trade_type = 'BUY' THEN block_timestamp END),
    max(CASE WHEN trade_type = 'SELL' THEN block_timestamp END)
  ) AS holding_duration_minutes,

  timestampdiff(second,
    min(CASE WHEN trade_type = 'BUY' THEN block_timestamp END),
    max(CASE WHEN trade_type = 'SELL' THEN block_timestamp END)
  ) AS holding_duration_seconds,

  -- 트랜잭션 ID
  buy_sell_ids.buy_tx_id,
  buy_sell_ids.sell_tx_id

FROM labeled_swaps
JOIN buy_sell_ids ON labeled_swaps.trade_token = buy_sell_ids.trade_token
WHERE trade_type IN ('BUY', 'SELL')
GROUP BY labeled_swaps.trade_token, buy_sell_ids.buy_tx_id, buy_sell_ids.sell_tx_id
ORDER BY buy_timestamp