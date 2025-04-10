-- 7d by chain
WITH
  tradesPast7d AS (
    SELECT
      block_date,
      blockchain,
      amount_usd,
      fee_usd,
      user,
      tx_hash
    FROM
      dune.bonkbot.result_dex_trading_bot_trades
    WHERE
      isLastTradeInTransaction = true
      AND block_time >= DATE_TRUNC('day', NOW()) - INTERVAL '30' day
  ),
  firstUserOccurrences AS (
    SELECT
      user,
      blockchain,
      MIN(block_date) AS firstTradeDate
    FROM
      tradesPast7d
    GROUP BY
      user,
      blockchain
  ),
  past7dSectorActivityByChainByDay AS (
    SELECT
      block_date,
      tradesPast7d.blockchain,
      SUM(amount_usd) AS volumeUSD,
      SUM(fee_usd) AS botRevenueUSD,
      COUNT(DISTINCT (tradesPast7d.user)) AS numberOfUsers,
      COALESCE(COUNT(DISTINCT (firstUserOccurrences.user)), 0) AS numberOfNewUsers,
      COUNT(DISTINCT (tx_hash)) AS numberOfTrades
    FROM
      tradesPast7d
      LEFT JOIN firstUserOccurrences ON (
        tradesPast7d.user = firstUserOccurrences.user
        AND tradesPast7d.blockchain = firstUserOccurrences.blockchain
        AND tradesPast7d.block_date = firstUserOccurrences.firstTradeDate
      )
    GROUP BY
      block_date,
      tradesPast7d.blockchain
    ORDER BY
      block_date ASC,
      volumeUSD DESC
  )
SELECT
  *,
  volumeUSD / numberOfUsers AS averageVolumePerUserUSD,
  volumeUSD / numberOfTrades AS averageVolumePerTradeUSD
FROM
  past7dSectorActivityByChainByDay
ORDER BY
  block_date ASC,
  volumeUSD DESC