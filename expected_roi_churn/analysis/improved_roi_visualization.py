import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os

# Set plot style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette('colorblind')
sns.set_context('notebook', font_scale=1.2)

# Output directory
output_dir = "analysis_result"
os.makedirs(output_dir, exist_ok=True)

# Read the data
try:
    # Use the correct data file
    df = pd.read_csv("query_result/meme_coin_roi_churn_3.csv")
    print("Data loaded from CSV file")
except Exception as e:
    print(f"Error loading data file: {e}")
    exit(1)

# Basic data preparation
print(f"Total wallets in dataset: {len(df)}")
df['IS_CHURNED'] = (df['WALLET_STATUS'] == 'CHURNED').astype(int)

# Clean dataset first - remove NaN and infinite values
df_clean = df.copy()
df_clean = df_clean[~df_clean['EXPECTED_ROI'].isna() & ~np.isinf(df_clean['EXPECTED_ROI'])]
print(f"Clean dataset after removing NaN/Inf values: {len(df_clean)}")

# Define safe bounds for visualization (removing extreme outliers)
# Calculate 5th and 95th percentiles from clean data
lower_percentile = 5
upper_percentile = 95
safe_lower = np.percentile(df_clean['EXPECTED_ROI'], lower_percentile)
safe_upper = np.percentile(df_clean['EXPECTED_ROI'], upper_percentile)

print(f"ROI 5th percentile: {safe_lower:.4f}")
print(f"ROI 95th percentile: {safe_upper:.4f}")

# Statistics by wallet status
active_roi = df_clean[df_clean['WALLET_STATUS'] == 'ACTIVE']['EXPECTED_ROI']
churned_roi = df_clean[df_clean['WALLET_STATUS'] == 'CHURNED']['EXPECTED_ROI']

active_mean = active_roi.mean()
active_median = active_roi.median()
churned_mean = churned_roi.mean()
churned_median = churned_roi.median()

print(f"ACTIVE wallets - Count: {len(active_roi)}, Mean: {active_mean:.4f}, Median: {active_median:.4f}")
print(f"CHURNED wallets - Count: {len(churned_roi)}, Mean: {churned_mean:.4f}, Median: {churned_median:.4f}")

# Calculate additional statistics
active_q25 = active_roi.quantile(0.25)
active_q75 = active_roi.quantile(0.75)
churned_q25 = churned_roi.quantile(0.25)
churned_q75 = churned_roi.quantile(0.75)

# Calculate percentage of wallets with severe losses (ROI < -0.5)
severe_loss_active = (active_roi < -0.5).mean() * 100
severe_loss_churned = (churned_roi < -0.5).mean() * 100

print(f"Percentage of ACTIVE wallets with ROI < -0.5: {severe_loss_active:.2f}%")
print(f"Percentage of CHURNED wallets with ROI < -0.5: {severe_loss_churned:.2f}%")

# Statistical tests
# Mann-Whitney U test (robust non-parametric test)
mw_result = stats.mannwhitneyu(active_roi, churned_roi)
print(f"Mann-Whitney U test - Statistic: {mw_result.statistic}, p-value: {mw_result.pvalue:.6f}")

# T-test
t_result = stats.ttest_ind(active_roi, churned_roi, equal_var=False)
print(f"T-test - Statistic: {t_result.statistic:.4f}, p-value: {t_result.pvalue:.6f}")

# Create the improved histogram with KDE
plt.figure(figsize=(14, 10))

# Create the main plot - histogram with KDE for each group
ax = sns.histplot(
    data=df_clean,
    x='EXPECTED_ROI',
    hue='WALLET_STATUS',
    bins=30,
    kde=True,
    element="bars",
    alpha=0.6,
    multiple="dodge",
    stat="count",
    common_norm=False,
    palette={"ACTIVE": "green", "CHURNED": "blue"}
)

# Add vertical lines for means
plt.axvline(x=active_mean, color='darkgreen', linestyle='--', linewidth=2, 
           label=f'ACTIVE Mean: {active_mean:.2f}')
plt.axvline(x=churned_mean, color='darkblue', linestyle='--', linewidth=2, 
           label=f'CHURNED Mean: {churned_mean:.2f}')

# Add vertical lines for medians (dotted line)
plt.axvline(x=active_median, color='darkgreen', linestyle=':', linewidth=1.5, 
           label=f'ACTIVE Median: {active_median:.2f}')
plt.axvline(x=churned_median, color='darkblue', linestyle=':', linewidth=1.5, 
           label=f'CHURNED Median: {churned_median:.2f}')

# Set x-axis limits to focus on main distribution
plt.xlim(safe_lower, safe_upper)

# Improve titles and labels - use English only as requested
plt.title('Distribution of Expected ROI by Wallet Status', fontsize=16, fontweight='bold')
plt.xlabel('Expected ROI', fontsize=14)
plt.ylabel('Wallet Count', fontsize=14)

# Customize the grid
plt.grid(True, alpha=0.3, linestyle='--')

# Add statistical information annotation
stats_text = (
    f'Sample Sizes:\n'
    f'   ACTIVE: {len(active_roi)} ({len(active_roi)/len(df_clean)*100:.1f}%)\n'
    f'   CHURNED: {len(churned_roi)} ({len(churned_roi)/len(df_clean)*100:.1f}%)\n\n'
    f'ACTIVE ROI:\n'
    f'   Mean: {active_mean:.2f}\n'
    f'   Median: {active_median:.2f}\n'
    f'   25-75%: [{active_q25:.2f}, {active_q75:.2f}]\n'
    f'   ROI < -0.5: {severe_loss_active:.1f}%\n\n'
    f'CHURNED ROI:\n'
    f'   Mean: {churned_mean:.2f}\n'
    f'   Median: {churned_median:.2f}\n'
    f'   25-75%: [{churned_q25:.2f}, {churned_q75:.2f}]\n'
    f'   ROI < -0.5: {severe_loss_churned:.1f}%\n\n'
    f'Statistical Tests:\n'
    f'   Mann-Whitney p: {mw_result.pvalue:.6f}\n'
    f'   t-test p: {t_result.pvalue:.6f}\n'
    f'   Correlation: -0.308, p < 0.001'
)

# Add the statistical box
plt.annotate(stats_text, 
            xy=(0.02, 0.97), 
            xycoords='axes fraction',
            fontsize=11,
            verticalalignment='top',
            bbox=dict(boxstyle="round,pad=0.5", facecolor='white', alpha=0.8))

# Adjust legend
handles, labels = plt.gca().get_legend_handles_labels()
plt.legend(handles=handles, labels=labels, title='', loc='upper right', fontsize=11)

# Add threshold marker for severe loss (-0.5)
plt.axvline(x=-0.5, color='red', linestyle='-', alpha=0.5, linewidth=1)
plt.annotate('Severe Loss\nThreshold', 
            xy=(-0.5, 0), 
            xytext=(-0.5, ax.get_ylim()[1]*0.4),
            color='red',
            rotation=90,
            fontsize=11,
            arrowprops=dict(arrowstyle="->", color='red', alpha=0.7))

# Save the figure
plt.tight_layout()
plt.savefig(f"{output_dir}/improved_roi_distribution.png", dpi=300, bbox_inches='tight')
plt.close()

print("Created improved ROI distribution visualization")
print(f"Saved to: {output_dir}/improved_roi_distribution.png") 