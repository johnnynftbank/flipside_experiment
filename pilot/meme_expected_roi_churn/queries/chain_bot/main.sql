WITH 
  part1 AS (SELECT * FROM query_4934059),
  part2 AS (SELECT * FROM query_4934060),
  part3 AS (SELECT * FROM query_4934061),
  
  partial_union AS (
    SELECT * FROM part1
    UNION ALL
    SELECT * FROM part2
  )

SELECT * FROM (
  SELECT * FROM partial_union
  UNION ALL
  SELECT * FROM part3
) final
ORDER BY day DESC;
