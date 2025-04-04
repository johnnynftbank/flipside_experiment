WITH 
  sol_trading_bot_stats AS (SELECT day, total_volume_usd FROM query_4933839 WHERE day >= CURRENT_DATE - INTERVAL '1' month),
  tradewiz_stats AS (SELECT day, total_volume_usd FROM query_4933845 WHERE day >= CURRENT_DATE - INTERVAL '1' month),
  trojan_stats AS (SELECT day, total_volume_usd FROM query_4933850 WHERE day >= CURRENT_DATE - INTERVAL '1' month),
  vector_stats AS (SELECT day, total_volume_usd FROM query_4933854 WHERE day >= CURRENT_DATE - INTERVAL '1' month),
  moonshot_stats AS (SELECT day, total_volume_usd FROM query_4933863 WHERE day >= CURRENT_DATE - INTERVAL '1' month),
  bonkbot_stats AS (SELECT day, total_volume_usd FROM query_4933958 WHERE day >= CURRENT_DATE - INTERVAL '1' month)

SELECT * FROM (
  SELECT 'Sol Trading Bot' AS bot_name, day, total_volume_usd FROM sol_trading_bot_stats
  UNION ALL
  SELECT 'TradeWizBot' AS bot_name, day, total_volume_usd FROM tradewiz_stats
  UNION ALL
  SELECT 'Trojan' AS bot_name, day, total_volume_usd FROM trojan_stats
  UNION ALL
  SELECT 'Vector' AS bot_name, day, total_volume_usd FROM vector_stats
  UNION ALL
  SELECT 'Moonshot' AS bot_name, day, total_volume_usd FROM moonshot_stats
  UNION ALL
  SELECT 'BonkBot' AS bot_name, day, total_volume_usd FROM bonkbot_stats
) combined
WHERE day < CURRENT_DATE
ORDER BY day DESC;