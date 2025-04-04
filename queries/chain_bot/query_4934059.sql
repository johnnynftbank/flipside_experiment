WITH 
  banana_stats AS (SELECT day, total_volume_usd FROM query_4933529 WHERE day >= TIMESTAMP '2024-10-01' AND day <= TIMESTAMP '2025-03-31'),
  axiom_stats AS (SELECT day, total_volume_usd FROM query_4930804 WHERE day >= TIMESTAMP '2024-10-01' AND day <= TIMESTAMP '2025-03-31'),
  bullx_stats AS (SELECT day, total_volume_usd FROM query_4933975 WHERE day >= TIMESTAMP '2024-10-01' AND day <= TIMESTAMP '2025-03-31'),
  photon_stats AS (SELECT day, total_volume_usd FROM query_4930792 WHERE day >= TIMESTAMP '2024-10-01' AND day <= TIMESTAMP '2025-03-31'),
  bloom_stats AS (SELECT day, total_volume_usd FROM query_4933547 WHERE day >= TIMESTAMP '2024-10-01' AND day <= TIMESTAMP '2025-03-31'),
  dexcelerate_stats AS (SELECT day, total_volume_usd FROM query_4933564 WHERE day >= TIMESTAMP '2024-10-01' AND day <= TIMESTAMP '2025-03-31')

SELECT * FROM (
  SELECT 'Banana Gun' AS bot_name, day, total_volume_usd AS daily_volume FROM banana_stats
  UNION ALL
  SELECT 'Axiom' AS bot_name, day, total_volume_usd FROM axiom_stats
  UNION ALL
  SELECT 'BullX' AS bot_name, day, total_volume_usd FROM bullx_stats
  UNION ALL
  SELECT 'Photon' AS bot_name, day, total_volume_usd FROM photon_stats
  UNION ALL
  SELECT 'Bloom' AS bot_name, day, total_volume_usd FROM bloom_stats
  UNION ALL
  SELECT 'Dexcelerate' AS bot_name, day, total_volume_usd FROM dexcelerate_stats
) combined
WHERE day <= CURRENT_DATE  -- 미래 데이터는 제외
ORDER BY day DESC;