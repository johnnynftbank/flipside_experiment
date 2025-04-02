-- Backtesting analysis with dummy trading data
WITH RECURSIVE dates AS (
  SELECT '2024-01-01'::date as date
  UNION ALL
  SELECT date + 1
  FROM dates
  WHERE date < '2024-03-31'
),
trading_data AS (
  SELECT 
    date,
    -- Simulating price with a base of 100 and some random variations
    100 * (1 + 0.1 * sin(extract(epoch from date)/86400.0/7.0) + 
           0.05 * sin(extract(epoch from date)/86400.0/3.0)) as price,
    -- Simulating trading volume with base 1000 and variations
    1000 * (1 + random()) as volume
),
strategy_performance AS (
  SELECT
    date,
    price,
    volume,
    -- Simple moving averages
    AVG(price) OVER (ORDER BY date ROWS BETWEEN 7 PRECEDING AND CURRENT ROW) as sma_7,
    AVG(price) OVER (ORDER BY date ROWS BETWEEN 20 PRECEDING AND CURRENT ROW) as sma_20,
    -- Simulating strategy returns
    CASE 
      WHEN LAG(price) OVER (ORDER BY date) IS NULL THEN 0
      ELSE (price - LAG(price) OVER (ORDER BY date)) / LAG(price) OVER (ORDER BY date)
    END as daily_return
),
cumulative_performance AS (
  SELECT
    date,
    price,
    volume,
    sma_7,
    sma_20,
    daily_return,
    -- Calculating cumulative returns
    (1 + daily_return) OVER (ORDER BY date) as cumulative_return,
    -- Adding some metrics
    AVG(daily_return) OVER (ORDER BY date ROWS BETWEEN 20 PRECEDING AND CURRENT ROW) as avg_20d_return,
    STDDEV(daily_return) OVER (ORDER BY date ROWS BETWEEN 20 PRECEDING AND CURRENT ROW) as volatility_20d
  FROM strategy_performance
)
SELECT
  date,
  ROUND(price::numeric, 2) as price,
  ROUND(volume::numeric, 0) as volume,
  ROUND(sma_7::numeric, 2) as sma_7,
  ROUND(sma_20::numeric, 2) as sma_20,
  ROUND((daily_return * 100)::numeric, 2) as daily_return_pct,
  ROUND(((cumulative_return - 1) * 100)::numeric, 2) as total_return_pct,
  ROUND((avg_20d_return * 100)::numeric, 2) as avg_20d_return_pct,
  ROUND((volatility_20d * 100)::numeric, 2) as volatility_20d_pct
FROM cumulative_performance
ORDER BY date;

-- Visualization hints (for tools like Metabase, Mode, or similar):
/*
Recommended visualizations:

1. Line Chart: Price Movement with Moving Averages
   - X-axis: date
   - Y-axis: price, sma_7, sma_20
   - This shows price trends and potential trading signals

2. Area Chart: Trading Volume
   - X-axis: date
   - Y-axis: volume
   - This shows market activity levels

3. Line Chart: Cumulative Performance
   - X-axis: date
   - Y-axis: total_return_pct
   - This shows overall strategy performance

4. Combo Chart: Returns and Volatility
   - X-axis: date
   - Line: avg_20d_return_pct
   - Bars: volatility_20d_pct
   - This shows risk-return relationship
*/ 