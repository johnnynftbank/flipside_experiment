-- Experimental SELECT query to get numbers 4, 5, 6
WITH numbers AS (
  SELECT 4 as num
  UNION ALL
  SELECT 5
  UNION ALL
  SELECT 6
)
SELECT num
FROM numbers
ORDER BY num; 