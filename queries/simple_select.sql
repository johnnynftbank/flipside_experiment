-- Simple SELECT query to get numbers 1, 2, 3
WITH numbers AS (
  SELECT 1 as num
  UNION ALL
  SELECT 2
  UNION ALL
  SELECT 3
)
SELECT num
FROM numbers
ORDER BY num; 