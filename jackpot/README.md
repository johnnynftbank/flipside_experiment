# Jackpot Analysis - Sharpe Ratio vs. ROI Standard Deviation

This repository contains a comprehensive analysis of trader behaviors based on risk-return characteristics, focusing specifically on identifying "jackpot-seeking" behavior patterns.

## Analysis Overview

We conducted a 2D decomposition analysis of Expected ROI, ROI Standard Deviation (σ), and Sharpe Ratio across 3,822 cryptocurrency trading wallets to identify different trading behaviors and their outcomes.

### Key Questions Explored

1. How do different trading strategies (high/low ROI, high/low volatility) perform?
2. Can we identify "jackpot-seeking" behavior (high-risk, high-reward strategies)?
3. Which strategies lead to sustainable trading vs. wallet abandonment?
4. What metrics best predict wallet retention/churn?

## Directory Structure

```
jackpot/
├── README.md                   # This file
├── query/
│   ├── query/                  # SQL queries
│   └── query_result/           # Query results
│       └── jackpot_criteria_3822.csv  # Main dataset
├── analysis/
│   ├── source_code/            # Python scripts
│   │   └── sharpe_roi_analysis.py  # Main analysis script
│   └── report/                 # Analysis reports and visualizations
│       ├── analysis_summary.md        # Summary of findings
│       ├── sharpe_roi_analysis_report.md  # Comprehensive report
│       └── *.png               # Visualizations
```

## Key Findings

1. **Four distinct trading profiles** were identified: Jackpot Seekers (high ROI, high σ), Unsuccessful Gamblers (low ROI, high σ), Cautious/Conservative traders (low ROI, low σ), and Skilled Investors (high ROI, low σ).

2. **Jackpot Seekers** (high ROI, high σ) achieved the highest returns but with extreme volatility and moderate sustainability.

3. **Skilled Investors** (high ROI, low σ) demonstrated the best risk-adjusted performance and lowest churn rates.

4. **Positive correlation** (0.74) between ROI and Standard Deviation confirms the risk-return tradeoff in cryptocurrency trading.

5. **Wallet churn is highest** among Unsuccessful Gamblers at 39.5%, indicating that high volatility without corresponding returns leads to abandonment.

## Reports and Visualizations

- [Analysis Summary](analysis/report/analysis_summary.md) - Key findings and insights
- [Comprehensive Report](analysis/report/sharpe_roi_analysis_report.md) - Detailed analysis methodology and results

### Key Visualizations

- [ROI vs. Standard Deviation (Filtered)](analysis/report/roi_vs_stddev_filtered.png) - 2D decomposition with Sharpe overlay
- [K-means Clustering Results](analysis/report/kmeans_clusters.png) - Natural groupings of trading behaviors
- [Wallet Status by Quadrant](analysis/report/wallet_status_by_quadrant.png) - Churn rates across different trading profiles

## Tools and Technologies

- **Python** - Primary analysis language
- **pandas** - Data manipulation and analysis
- **scikit-learn** - Machine learning (K-means clustering)
- **matplotlib/seaborn** - Data visualization

## Dataset Information

- **Source**: jackpot_criteria_3822.csv
- **Size**: 3,822 wallet records
- **Time Period**: Jan-Apr 2025
- **Key Metrics**: Expected ROI, ROI Standard Deviation, Sharpe Ratio, Win-Loss Ratio, etc.

---

*This analysis was conducted as part of the Flipside experiment project.* 