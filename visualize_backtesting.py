import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime, timedelta

# Set style
plt.style.use('seaborn-v0_8')  # Updated style name
sns.set_palette("husl")

# Generate dummy data similar to our SQL query
dates = pd.date_range(start='2024-01-01', end='2024-03-31', freq='D')
np.random.seed(42)  # For reproducibility

# Generate price data
t = np.arange(len(dates))
base_price = 100
price = base_price * (1 + 0.1 * np.sin(t/7) + 0.05 * np.sin(t/3))
volume = 1000 * (1 + np.random.random(len(dates)))

# Create DataFrame
df = pd.DataFrame({
    'date': dates,
    'price': price,
    'volume': volume
})

# Calculate indicators
df['sma_7'] = df['price'].rolling(window=7).mean()
df['sma_20'] = df['price'].rolling(window=20).mean()
df['daily_return'] = df['price'].pct_change()
df['cumulative_return'] = (1 + df['daily_return']).cumprod() - 1
df['avg_20d_return'] = df['daily_return'].rolling(window=20).mean()
df['volatility_20d'] = df['daily_return'].rolling(window=20).std()

# Create directory for plots
import os
if not os.path.exists('plots'):
    os.makedirs('plots')

# 1. Price Movement with Moving Averages
plt.figure(figsize=(12, 6))
plt.plot(df['date'], df['price'], label='Price', linewidth=1)
plt.plot(df['date'], df['sma_7'], label='7-day SMA', linewidth=1.5)
plt.plot(df['date'], df['sma_20'], label='20-day SMA', linewidth=1.5)
plt.title('Price Movement with Moving Averages')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('plots/price_movement.png', dpi=300, bbox_inches='tight')
plt.close()

# 2. Trading Volume
plt.figure(figsize=(12, 4))
plt.fill_between(df['date'], df['volume'], alpha=0.3)
plt.plot(df['date'], df['volume'], alpha=0.8)
plt.title('Trading Volume')
plt.xlabel('Date')
plt.ylabel('Volume')
plt.grid(True)
plt.tight_layout()
plt.savefig('plots/trading_volume.png', dpi=300, bbox_inches='tight')
plt.close()

# 3. Cumulative Performance
plt.figure(figsize=(12, 6))
plt.plot(df['date'], df['cumulative_return'] * 100)
plt.title('Cumulative Performance')
plt.xlabel('Date')
plt.ylabel('Total Return (%)')
plt.grid(True)
plt.tight_layout()
plt.savefig('plots/cumulative_performance.png', dpi=300, bbox_inches='tight')
plt.close()

# 4. Returns and Volatility
fig, ax1 = plt.subplots(figsize=(12, 6))

# Plot average returns
color = 'tab:blue'
ax1.set_xlabel('Date')
ax1.set_ylabel('20-day Average Return (%)', color=color)
ax1.plot(df['date'], df['avg_20d_return'] * 100, color=color)
ax1.tick_params(axis='y', labelcolor=color)

# Plot volatility on secondary y-axis
ax2 = ax1.twinx()
color = 'tab:red'
ax2.set_ylabel('20-day Volatility (%)', color=color)
ax2.bar(df['date'], df['volatility_20d'] * 100, alpha=0.3, color=color)
ax2.tick_params(axis='y', labelcolor=color)

plt.title('Returns and Volatility')
plt.grid(True)
plt.tight_layout()
plt.savefig('plots/returns_volatility.png', dpi=300, bbox_inches='tight')
plt.close()

print("Plots have been saved in the 'plots' directory.") 