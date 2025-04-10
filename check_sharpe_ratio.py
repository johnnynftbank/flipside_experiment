import pandas as pd
import numpy as np

# Load the data
df = pd.read_csv('jackpot/query/query_result/jackpot_criteria_3822.csv')

# Analyze Sharpe ratio
sharpe_ratio = df['SHARPE_RATIO']

# Calculate basic statistics
mean = sharpe_ratio.mean()
std = sharpe_ratio.std()
min_value = sharpe_ratio.min()
max_value = sharpe_ratio.max()
median = sharpe_ratio.median()

print(f"Basic Statistics for Sharpe Ratio:")
print(f"Mean: {mean:.4f}")
print(f"Std: {std:.4f}")
print(f"Min: {min_value:.4f}")
print(f"Max: {max_value:.4f}")
print(f"Median: {median:.4f}")
print("\n")

# Calculate Z-score
df['SHARPE_Z_SCORE'] = (df['SHARPE_RATIO'] - mean) / std

# Find the 10 wallets with the lowest Sharpe ratio
print("10 Wallets with Lowest Sharpe Ratio:")
lowest_sharpe = df.sort_values('SHARPE_RATIO').head(10)
for _, row in lowest_sharpe.iterrows():
    print(f"{row['SWAPPER'][:10]}... {row['SHARPE_RATIO']:.4f} (Z-score: {row['SHARPE_Z_SCORE']:.4f})")
print("\n")

# Find wallets with Z-score < -3
print("Wallets with Z-score < -3:")
outliers = df[df['SHARPE_Z_SCORE'] < -3]
for _, row in outliers.iterrows():
    print(f"{row['SWAPPER'][:10]}... {row['SHARPE_RATIO']:.4f} (Z-score: {row['SHARPE_Z_SCORE']:.4f})")
print("\n")

# Count Z-scores
z_below_3 = (df['SHARPE_Z_SCORE'] < -3).sum()
z_above_3 = (df['SHARPE_Z_SCORE'] > 3).sum()
total = len(df)

print("Z-score Summary:")
print(f"Z-score < -3: {z_below_3} ({z_below_3/total*100:.2f}%)")
print(f"Z-score > 3: {z_above_3} ({z_above_3/total*100:.2f}%)")
print(f"Threshold for Z < -3: {mean - 3*std:.4f}")
print(f"Threshold for Z > 3: {mean + 3*std:.4f}") 