
# Meme Coin ROI and Churn Analysis Report

## 1. Overview
- Total wallets analyzed: 7515
- Active wallets: 3017 (40.15%)
- Churned wallets: 4497 (59.84%)
- Overall churn rate: 59.84%

## 2. Key Findings

### ROI and Churn Relationship
- The average Expected ROI for active wallets is -0.06
- The average Expected ROI for churned wallets is -0.17
- Statistical significance: Not available

### ROI Bucket Analysis
- Highest churn rates are observed in ROI buckets: [0.0, 1.0, 2.0]
- Lowest churn rates are observed in ROI buckets: [7.0, 8.0, 6.0]

### Trading Behavior
- Wallets with higher trade counts tend to have higher ROI
- Wallets with more trading days tend to have lower churn rates at equal ROI levels

## 3. Visualizations
This report includes the following visualizations:
- ROI by wallet status boxplot (`roi_by_status_boxplot.png`)
- ROI distribution histogram (`roi_histogram.png`)
- ROI density distribution (`roi_density.png`)
- ROI vs trade count scatter plot (`roi_vs_trade_count.png`)
- Correlation heatmap (`correlation_heatmap.png`)
- Churn rate by ROI bucket (`churn_rate_by_roi_bucket.png`)
- Logistic regression analysis (`logistic_regression_roi_vs_churn.png`)
- Churn heatmap by ROI and trading days (`churn_heatmap_roi_vs_days.png`)



## 4. Implications
- ROI appears to be a weak predictor of churn
- Traders with unknown ROI are more likely to churn
- The relationship between ROI and churn is moderated by the number of trading days

## 5. Recommendations
- Target retention efforts at wallets with unknown ROI
- Consider the interaction between trading frequency and ROI when designing retention strategies
- Further analyze the specific ROI thresholds that most significantly impact churn

## 6. Methodology
This analysis used:
- Descriptive statistics to understand basic relationships
- Logistic regression to model the probability of churn based on ROI

- Stratified analysis to examine the interaction between ROI and trading behavior
    