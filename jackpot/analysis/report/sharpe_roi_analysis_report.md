# Sharpe Ratio vs. ROI Standard Deviation: 2D Decomposition Analysis

## Executive Summary

This report presents a comprehensive analysis of trading behaviors by examining the relationship between expected Return on Investment (ROI), ROI Standard Deviation (σ), and Sharpe Ratio across 3,822 wallets. The primary goal is to identify and characterize different trading behaviors, with a particular focus on "jackpot-seeking" behavior (high-risk, high-reward strategies).

**Key Findings:**

1. **Four distinct trading profiles** were identified through both quadrant analysis and K-means clustering: Jackpot Seekers (high ROI, high σ), Unsuccessful Gamblers (low ROI, high σ), Cautious/Conservative traders (low ROI, low σ), and Skilled Investors (high ROI, low σ).

2. **Jackpot Seekers** (high ROI, high σ) displayed the highest average returns but also the highest volatility, with moderate Sharpe ratios. They showed a moderate churn rate of 33.6%, suggesting that while this high-risk strategy can be successful, it's not sustainable for all users.

3. **Skilled Investors** (high ROI, low σ) demonstrated the highest Sharpe ratios (0.37 on average) and win-loss ratios (2.03), with a relatively low churn rate of 31.5%, indicating a more sustainable and skilled trading approach.

4. **Wallet churn is highest** among Unsuccessful Gamblers at 39.5%, confirming that high volatility combined with poor returns leads to user abandonment.

5. **Positive correlation** (0.74) exists between ROI and ROI Standard Deviation, indicating that higher returns generally come with higher risk in this market.

## 1. Introduction

This analysis explores the relationship between risk (measured by ROI Standard Deviation), return (Expected ROI), and risk-adjusted performance (Sharpe Ratio) for 3,822 cryptocurrency trading wallets. By decomposing these metrics in a two-dimensional space, we aim to identify different trading behaviors, particularly "jackpot-seeking" tendencies where traders accept high volatility in pursuit of high returns.

## 2. Dataset Overview

The dataset consists of 3,822 wallet records with the following key metrics:

- `EXPECTED_ROI`: The average return on investment
- `ROI_STANDARD_DEVIATION`: The volatility/standard deviation of returns
- `SHARPE_RATIO`: Risk-adjusted performance metric (ROI/Standard Deviation)
- `WIN_LOSS_RATIO`: Ratio of profitable to unprofitable trades
- `MAX_TRADE_PROPORTION`: Largest proportion allocated to a single trade
- `UNIQUE_TOKENS_TRADED`: Number of different tokens traded
- `TOTAL_TRADES`: Total number of trades executed
- `WALLET_STATUS`: Whether the wallet is "active" or "churned" (abandoned)

### 2.1 Data Quality and Preprocessing

- No missing values were found in the dataset
- Extreme outliers were identified in ROI and Standard Deviation distributions
- For better visualization and analysis, we filtered to the 99th percentile for both metrics

## 3. Distribution Analysis

### 3.1 Expected ROI Distribution

The Expected ROI distribution shows a strong positive skew, with most wallets having modest returns between -0.5 and 0.5, but with a long tail of highly profitable wallets. The top 1% of wallets have an ROI above 4.0.

### 3.2 ROI Standard Deviation Distribution

The distribution of ROI Standard Deviation is also highly skewed, with most wallets showing moderate volatility but a significant number exhibiting extreme volatility. This suggests varying risk appetites among traders.

### 3.3 Sharpe Ratio Distribution

Sharpe Ratios are concentrated in the 0-1 range, indicating that most traders are not achieving exceptional risk-adjusted returns. Very few wallets exceed a Sharpe Ratio of 2.0, suggesting that truly skilled trading is rare in this dataset.

## 4. 2D Analysis: ROI vs. Standard Deviation

The core of our analysis is a two-dimensional decomposition of ROI and Standard Deviation, with Sharpe Ratio represented through color coding.

### 4.1 Quadrant Analysis

We divided the ROI-Standard Deviation plane into four quadrants based on median values:

1. **Q1: High ROI, High σ (Jackpot Seekers)**
   - Average ROI: 0.69
   - Average Standard Deviation: 1.22
   - Average Sharpe Ratio: 0.28
   - Churn Rate: 33.6%

2. **Q2: Low ROI, High σ (Unsuccessful Gamblers)**
   - Average ROI: 0.09
   - Average Standard Deviation: 0.91
   - Average Sharpe Ratio: 0.10
   - Churn Rate: 39.5%

3. **Q3: Low ROI, Low σ (Cautious/Conservative)**
   - Average ROI: 0.09
   - Average Standard Deviation: 0.31
   - Average Sharpe Ratio: 0.30
   - Churn Rate: 33.8%

4. **Q4: High ROI, Low σ (Skilled Investors)**
   - Average ROI: 0.33
   - Average Standard Deviation: 0.33
   - Average Sharpe Ratio: 0.37
   - Churn Rate: 31.5%

This quadrant analysis reveals that the "Jackpot Seekers" (Q1) achieve the highest returns but at the cost of high volatility. The "Skilled Investors" (Q4) have the best risk-adjusted performance (highest Sharpe ratios) and the lowest churn rate, indicating a more sustainable trading approach.

### 4.2 K-means Clustering

We applied K-means clustering with k=4 to identify natural groupings in the data:

- The clustering results largely aligned with our quadrant analysis, confirming the presence of four distinct trading behaviors.
- Cluster centroids show clear separation between the groups, indicating distinct behavioral patterns.
- The cluster statistics closely match the quadrant statistics, lending further support to our categorization.

## 5. Sharpe Ratio Analysis

The Sharpe Ratio represents risk-adjusted returns and provides additional insights when analyzed against its components:

- The highest Sharpe Ratios are concentrated in the region of moderate ROI and low standard deviation
- A negative correlation (-0.23) exists between Sharpe Ratio and Standard Deviation, confirming that lower volatility tends to produce better risk-adjusted returns
- Wallets with the highest Sharpe Ratios tend to have higher win-loss ratios (2.03 on average for "Skilled Investors"), suggesting better trade selection

## 6. Correlation Analysis

Key correlations between metrics reveal important relationships:

- **ROI and Standard Deviation: +0.74** - Higher returns generally come with higher volatility
- **ROI and Sharpe Ratio: +0.22** - Higher returns moderately contribute to better risk-adjusted performance
- **Standard Deviation and Sharpe Ratio: -0.23** - Lower volatility generally leads to better risk-adjusted returns
- **Win-Loss Ratio and Sharpe Ratio: +0.53** - Better trade selection strongly contributes to risk-adjusted performance

## 7. Wallet Status Analysis

The analysis of wallet activity status (active vs. churned) provides insights into the sustainability of different trading strategies:

- **Highest churn rate (39.5%)** observed in "Unsuccessful Gamblers" (low ROI, high σ)
- **Lowest churn rate (31.5%)** found among "Skilled Investors" (high ROI, low σ)
- **Jackpot Seekers** have a moderate churn rate (33.6%), suggesting that while high-risk/high-reward strategies can be successful, they're not sustainable for all users

## 8. Conclusions and Recommendations

### 8.1 Key Takeaways

1. **Four distinct trading profiles exist** in the cryptocurrency market, each with different risk-return characteristics and sustainability.

2. **"Jackpot Seekers"** (high ROI, high σ) represent traders who accept high volatility in pursuit of high returns. While this strategy yields the highest absolute returns for successful traders, it also involves significant risk.

3. **"Skilled Investors"** (high ROI, low σ) demonstrate the most sustainable approach, with the best risk-adjusted returns and lowest churn rates. These traders appear to have superior trade selection skills.

4. **"Unsuccessful Gamblers"** (low ROI, high σ) have the worst outcomes and highest churn rates, indicating that high volatility without corresponding returns is not viable.

5. **Risk and return are positively correlated** in this market, but skilled traders can achieve better risk-adjusted returns through superior trade selection.

### 8.2 Recommendations

1. **For traders:**
   - Consider risk-adjusted performance (Sharpe Ratio) rather than just absolute returns
   - Focus on improving trade selection (win-loss ratio) rather than increasing position sizes
   - Monitor volatility and consider if the risk taken is proportional to expected returns

2. **For platforms:**
   - Develop tools to help users understand their risk profile and trading behavior
   - Provide risk-adjusted performance metrics alongside absolute return figures
   - Consider targeted interventions for users showing "Unsuccessful Gambler" patterns to reduce churn

3. **For further research:**
   - Examine temporal patterns to understand how traders move between different quadrants over time
   - Investigate the specific tokens and trading patterns associated with each trading profile
   - Study the relationship between trading frequency and risk-adjusted performance

## 9. Methodology Notes

- **Outlier handling:** 99th percentile filtering was applied to extreme values for better visualization
- **Quadrant definition:** Based on median values of ROI and Standard Deviation
- **Clustering approach:** K-means with k=4, selected based on silhouette score and elbow method
- **Visualizations:** Created using matplotlib and seaborn with custom color mapping

---

**Dataset Information:**
- Number of wallets analyzed: 3,822
- Time period covered: Jan-Apr 2025
- Data source: jackpot_criteria_3822.csv

---

*This analysis was conducted using Python with pandas, scikit-learn, matplotlib, and seaborn libraries.* 