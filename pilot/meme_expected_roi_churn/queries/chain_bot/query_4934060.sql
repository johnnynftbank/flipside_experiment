WITH 
  gmgn_stats AS (SELECT day, total_volume_usd FROM query_4930796 WHERE day >= CURRENT_DATE - INTERVAL '1' month),
  maestro_stats AS (SELECT day, total_volume_usd FROM query_4933568 WHERE day >= CURRENT_DATE - INTERVAL '1' month),
  mevx_stats AS (SELECT day, total_volume_usd FROM query_4933817 WHERE day >= CURRENT_DATE - INTERVAL '1' month),
  nova_stats AS (SELECT day, total_volume_usd FROM query_4546377 WHERE day >= CURRENT_DATE - INTERVAL '1' month),
  shuriken_stats AS (SELECT day, total_volume_usd FROM query_4933828 WHERE day >= CURRENT_DATE - INTERVAL '1' month)

SELECT * FROM (
  SELECT 'GMGN' AS bot_name, day, total_volume_usd FROM gmgn_stats
  UNION ALL
  SELECT 'Maestro' AS bot_name, day, total_volume_usd FROM maestro_stats
  UNION ALL
  SELECT 'MEVX' AS bot_name, day, total_volume_usd FROM mevx_stats
  UNION ALL
  SELECT 'Nova' AS bot_name, day, total_volume_usd FROM nova_stats
  UNION ALL
  SELECT 'Shuriken' AS bot_name, day, total_volume_usd FROM shuriken_stats
) combined
WHERE day < CURRENT_DATE
ORDER BY day DESC;