import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from sklearn.cluster import KMeans
import os

# Set the style for plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("viridis")

# Set font sizes for better readability
plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 13,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 11,
    'figure.titlesize': 16
})

# Create output folder for visuals
report_path = "../report"
REPORT_DIR = "../visualization"
os.makedirs(report_path, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

# Load data
print("Loading dataset...")
file_path = "../../query/query_result/jackpot_criteria_3822.csv"
df = pd.read_csv(file_path)

# Drop missing values if any
df.dropna(inplace=True)
print(f"After dropping missing values: {df.shape[0]} entries")

# 1. Basic Exploratory Data Analysis
print("\n1. Basic EDA: Summary Statistics")
print(f"Dataset shape: {df.shape}")
print("\nFirst few rows:")
print(df.head())

print("\nData types:")
print(df.dtypes)

print("\nSummary statistics:")
summary_stats = df.describe()
print(summary_stats)

# Check for missing values
print("\nMissing values per column:")
print(df.isnull().sum())

# 2. Analyze distribution of key metrics
print("\n2. Analysis of Key Metrics Distribution")

# Create helper function for plotting distributions
def plot_distribution(column, title, save_path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Histogram with KDE
    sns.histplot(df[column].dropna(), kde=True, ax=ax1)
    ax1.set_title(f"Distribution of {title}")
    ax1.set_xlabel(title)
    ax1.set_ylabel("Frequency")
    
    # Box plot
    sns.boxplot(x=df[column].dropna(), ax=ax2)
    ax2.set_title(f"Box Plot of {title}")
    ax2.set_xlabel(title)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

print("Plotting distributions for key metrics...")
plot_distribution("EXPECTED_ROI", "Expected ROI", f"{report_path}/expected_roi_distribution.png")
plot_distribution("ROI_STANDARD_DEVIATION", "ROI Standard Deviation", f"{report_path}/roi_std_dev_distribution.png")
plot_distribution("SHARPE_RATIO", "Sharpe Ratio", f"{report_path}/sharpe_ratio_distribution.png")

# 3. Handle outliers for visualization
# Create a function to filter extreme outliers for better visualization
def filter_for_visualization(df, column, percentile=0.99):
    threshold = df[column].quantile(percentile)
    return df[df[column] <= threshold]

# Create visualization dataframes with outliers handled
df_viz_roi = filter_for_visualization(df, "EXPECTED_ROI", 0.99)
df_viz_std = filter_for_visualization(df, "ROI_STANDARD_DEVIATION", 0.99)
df_viz_sharpe = filter_for_visualization(df, "SHARPE_RATIO", 0.99)

print(f"\nAfter filtering extreme outliers (99th percentile):")
print(f"ROI visualization dataset: {df_viz_roi.shape[0]} entries")
print(f"Std Dev visualization dataset: {df_viz_std.shape[0]} entries")
print(f"Sharpe visualization dataset: {df_viz_sharpe.shape[0]} entries")

# Re-plot with filtered data
plot_distribution("EXPECTED_ROI", "Expected ROI (99th Percentile)", f"{report_path}/expected_roi_distribution_filtered.png")
plot_distribution("ROI_STANDARD_DEVIATION", "ROI Standard Deviation (99th Percentile)", f"{report_path}/roi_std_dev_distribution_filtered.png")
plot_distribution("SHARPE_RATIO", "Sharpe Ratio (99th Percentile)", f"{report_path}/sharpe_ratio_distribution_filtered.png")

# 4. Create 2D plots and segmentation analysis (ROI vs Standard Deviation)
print("\n4. 2D Analysis: ROI vs Standard Deviation with Sharpe overlay")

# Create a scatter plot of ROI vs. Standard Deviation, colored by Sharpe Ratio
plt.figure(figsize=(12, 10))

# Create a color map from blue (low) to yellow (medium) to red (high)
colors = ["#2D49A0", "#73A0D0", "#FEE090", "#FD8D3C", "#FC4E2A", "#E31A1C", "#BD0026"]
cmap = LinearSegmentedColormap.from_list("custom_cmap", colors)

# Create scatter plot for all data points
scatter = plt.scatter(
    df["ROI_STANDARD_DEVIATION"], 
    df["EXPECTED_ROI"],
    c=df["SHARPE_RATIO"],
    cmap=cmap,
    s=30,
    alpha=0.7
)

# Add color bar
cbar = plt.colorbar(scatter)
cbar.set_label('Sharpe Ratio', rotation=270, labelpad=20)

# Add axis labels and title
plt.xlabel('ROI Standard Deviation (σ)')
plt.ylabel('Expected ROI')
plt.title('2D Analysis: Expected ROI vs. ROI Standard Deviation')

# Add reference line for Sharpe Ratio = 1 (if visible in range)
max_std = df["ROI_STANDARD_DEVIATION"].max()
max_roi = df["EXPECTED_ROI"].max()
plt.plot([0, max_std], [0, max_std], 'k--', alpha=0.5, label='Sharpe Ratio = 1')

plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f"{report_path}/roi_vs_stddev_full.png", dpi=300, bbox_inches='tight')
plt.close()

# Create a filtered version for better visualization
plt.figure(figsize=(12, 10))

# Filter data for visualization (99th percentile for both axes)
percentile = 0.99
roi_threshold = df["EXPECTED_ROI"].quantile(percentile)
std_threshold = df["ROI_STANDARD_DEVIATION"].quantile(percentile)

# Create a proper copy to avoid SettingWithCopyWarning
df_filtered = df[(df["EXPECTED_ROI"] <= roi_threshold) & 
                 (df["ROI_STANDARD_DEVIATION"] <= std_threshold)].copy()

# Add reference lines for median values to divide the plot into quadrants
roi_median = df_filtered["EXPECTED_ROI"].median()
std_median = df_filtered["ROI_STANDARD_DEVIATION"].median()

# Calculate min and max values for Sharpe Ratio to set a better color scale
sharpe_min = df_filtered["SHARPE_RATIO"].quantile(0.01)  # 1st percentile to avoid extreme outliers
sharpe_max = df_filtered["SHARPE_RATIO"].quantile(0.99)  # 99th percentile to avoid extreme outliers
sharpe_mid = 0  # Midpoint for diverging colormap

# Create diverging colormap: blue (negative), white (zero), red (positive)
cmap_diverging = plt.cm.RdBu_r  # Red-Blue reversed colormap (negative is blue, positive is red)

# Draw shaded areas for quadrants
# Q1: High ROI, High σ (top right)
plt.axhspan(roi_median, roi_threshold, xmin=(std_median-df_filtered["ROI_STANDARD_DEVIATION"].min())/(std_threshold-df_filtered["ROI_STANDARD_DEVIATION"].min()), 
            xmax=1, facecolor='lightsalmon', alpha=0.15)
# Q2: Low ROI, High σ (bottom right)
plt.axhspan(df_filtered["EXPECTED_ROI"].min(), roi_median, xmin=(std_median-df_filtered["ROI_STANDARD_DEVIATION"].min())/(std_threshold-df_filtered["ROI_STANDARD_DEVIATION"].min()), 
            xmax=1, facecolor='lightcoral', alpha=0.15)
# Q3: Low ROI, Low σ (bottom left)
plt.axhspan(df_filtered["EXPECTED_ROI"].min(), roi_median, xmin=0, 
            xmax=(std_median-df_filtered["ROI_STANDARD_DEVIATION"].min())/(std_threshold-df_filtered["ROI_STANDARD_DEVIATION"].min()), facecolor='lightsteelblue', alpha=0.15)
# Q4: High ROI, Low σ (top left)
plt.axhspan(roi_median, roi_threshold, xmin=0, 
            xmax=(std_median-df_filtered["ROI_STANDARD_DEVIATION"].min())/(std_threshold-df_filtered["ROI_STANDARD_DEVIATION"].min()), facecolor='lightgreen', alpha=0.15)

# Create scatter plot with filtered data and normalized color scale
scatter = plt.scatter(
    df_filtered["ROI_STANDARD_DEVIATION"], 
    df_filtered["EXPECTED_ROI"],
    c=df_filtered["SHARPE_RATIO"],
    cmap=cmap_diverging,
    vmin=sharpe_min, vmax=sharpe_max,  # Set explicit color limits
    s=40,
    alpha=0.7
)

# Add color bar with better formatting
cbar = plt.colorbar(scatter)
cbar.set_label('Sharpe Ratio', rotation=270, labelpad=20)
cbar.mappable.set_clim(sharpe_min, sharpe_max)  # Ensure colorbar limits match data

# Add axis labels and title
plt.xlabel('ROI Standard Deviation (σ)')
plt.ylabel('Expected ROI')
plt.title('2D Analysis: Expected ROI vs. ROI Standard Deviation (Filtered)')

# Add reference lines for median values to divide the plot into quadrants
plt.axhline(y=roi_median, color='black', linestyle='--', alpha=0.7, linewidth=1.5)
plt.axvline(x=std_median, color='black', linestyle='--', alpha=0.7, linewidth=1.5)

# Show the median values as text
plt.text(std_median + 0.02, df_filtered["EXPECTED_ROI"].min() + 0.02, f'Median σ = {std_median:.3f}', 
         fontweight='bold', bbox=dict(facecolor='white', alpha=0.7))
plt.text(df_filtered["ROI_STANDARD_DEVIATION"].min() + 0.02, roi_median + 0.02, f'Median ROI = {roi_median:.3f}', 
         fontweight='bold', bbox=dict(facecolor='white', alpha=0.7))

# Calculate quadrant counts
q1_count = len(df_filtered[(df_filtered["EXPECTED_ROI"] >= roi_median) & (df_filtered["ROI_STANDARD_DEVIATION"] >= std_median)])
q2_count = len(df_filtered[(df_filtered["EXPECTED_ROI"] < roi_median) & (df_filtered["ROI_STANDARD_DEVIATION"] >= std_median)])
q3_count = len(df_filtered[(df_filtered["EXPECTED_ROI"] < roi_median) & (df_filtered["ROI_STANDARD_DEVIATION"] < std_median)])
q4_count = len(df_filtered[(df_filtered["EXPECTED_ROI"] >= roi_median) & (df_filtered["ROI_STANDARD_DEVIATION"] < std_median)])
total = len(df_filtered)

# Add annotations for each quadrant - position near to the quadrant centers and include counts
plt.text(std_median + (std_threshold - std_median) * 0.5, roi_median + (roi_threshold - roi_median) * 0.5, 
         f"Q1: High ROI, High σ\n(Jackpot Seekers)\n{q1_count} wallets ({q1_count/total*100:.1f}%)", 
         ha='center', va='center', fontweight='bold',
         bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.9))

plt.text(std_median + (std_threshold - std_median) * 0.5, df_filtered["EXPECTED_ROI"].min() + (roi_median - df_filtered["EXPECTED_ROI"].min()) * 0.5, 
         f"Q2: Low ROI, High σ\n(Unsuccessful Gamblers)\n{q2_count} wallets ({q2_count/total*100:.1f}%)", 
         ha='center', va='center', fontweight='bold',
         bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.9))

plt.text(df_filtered["ROI_STANDARD_DEVIATION"].min() + (std_median - df_filtered["ROI_STANDARD_DEVIATION"].min()) * 0.5, df_filtered["EXPECTED_ROI"].min() + (roi_median - df_filtered["EXPECTED_ROI"].min()) * 0.5, 
         f"Q3: Low ROI, Low σ\n(Cautious/Conservative)\n{q3_count} wallets ({q3_count/total*100:.1f}%)", 
         ha='center', va='center', fontweight='bold',
         bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.9))

plt.text(df_filtered["ROI_STANDARD_DEVIATION"].min() + (std_median - df_filtered["ROI_STANDARD_DEVIATION"].min()) * 0.5, roi_median + (roi_threshold - roi_median) * 0.5, 
         f"Q4: High ROI, Low σ\n(Skilled Investors)\n{q4_count} wallets ({q4_count/total*100:.1f}%)", 
         ha='center', va='center', fontweight='bold',
         bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.9))

# Add reference line for Sharpe = 1
max_visible_std = std_threshold
max_visible_roi = roi_threshold
plt.plot([0, max_visible_std], [0, max_visible_std], 'k--', alpha=0.5, label='Sharpe Ratio = 1')

plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f"{report_path}/roi_vs_stddev_filtered.png", dpi=300, bbox_inches='tight')
plt.close()

# 5. Quadrant Analysis
print("\n5. Quadrant Analysis")

# Define quadrants
def assign_quadrant(row):
    if row["EXPECTED_ROI"] >= roi_median and row["ROI_STANDARD_DEVIATION"] >= std_median:
        return "Q1: High ROI, High σ (Jackpot Seekers)"
    elif row["EXPECTED_ROI"] < roi_median and row["ROI_STANDARD_DEVIATION"] >= std_median:
        return "Q2: Low ROI, High σ (Unsuccessful Gamblers)"
    elif row["EXPECTED_ROI"] < roi_median and row["ROI_STANDARD_DEVIATION"] < std_median:
        return "Q3: Low ROI, Low σ (Cautious/Conservative)"
    else:
        return "Q4: High ROI, Low σ (Skilled Investors)"

# Add quadrant information to the dataframe
df_filtered["Quadrant"] = df_filtered.apply(assign_quadrant, axis=1)

# Calculate statistics for each quadrant
quadrant_stats = df_filtered.groupby("Quadrant").agg({
    "EXPECTED_ROI": ["mean", "median"],
    "ROI_STANDARD_DEVIATION": ["mean", "median"],
    "SHARPE_RATIO": ["mean", "median"],
    "WIN_LOSS_RATIO": ["mean", "median"],
    "MAX_TRADE_PROPORTION": ["mean", "median"],
    "UNIQUE_TOKENS_TRADED": ["mean", "median"],
    "TOTAL_TRADES": ["mean", "median"],
    "WALLET_STATUS": lambda x: (x == "churned").mean() * 100  # Churn rate as percentage
})

print("\nQuadrant Statistics:")
print(quadrant_stats)

# Save quadrant statistics to CSV
quadrant_stats.to_csv(f"{report_path}/quadrant_statistics.csv")

# 6. Visualize the quadrant distribution
quadrant_counts = df_filtered["Quadrant"].value_counts()
plt.figure(figsize=(10, 6))
bars = plt.bar(quadrant_counts.index, quadrant_counts.values, color=sns.color_palette("viridis", 4))

# Add count and percentage labels
total = quadrant_counts.sum()
for i, bar in enumerate(bars):
    count = quadrant_counts.values[i]
    percentage = (count / total) * 100
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
             f"{count}\n({percentage:.1f}%)",
             ha='center', va='bottom')

plt.title("Distribution of Wallets Across Quadrants")
plt.ylabel("Number of Wallets")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig(f"{report_path}/quadrant_distribution.png", dpi=300, bbox_inches='tight')
plt.close()

# 7. K-means Clustering
print("\n7. K-means Clustering Analysis")

# Prepare data for clustering
X = df_filtered[["EXPECTED_ROI", "ROI_STANDARD_DEVIATION"]].copy()

# Scale the features for better clustering
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Determine optimal number of clusters using Elbow method
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

inertias = []
silhouette_scores = []
k_range = range(2, 9)

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    inertias.append(kmeans.inertia_)
    
    # Calculate silhouette score
    if k > 1:  # Silhouette score is defined for k > 1
        labels = kmeans.labels_
        silhouette_scores.append(silhouette_score(X_scaled, labels))

# Plot Elbow method and Silhouette scores
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Elbow curve
ax1.plot(k_range, inertias, 'bo-')
ax1.set_title('Elbow Method for Optimal k')
ax1.set_xlabel('Number of Clusters (k)')
ax1.set_ylabel('Inertia')
ax1.grid(True, alpha=0.3)

# Silhouette score - Fix the x range to match the silhouette_scores array
ax2.plot(list(k_range)[0:len(silhouette_scores)], silhouette_scores, 'ro-')
ax2.set_title('Silhouette Score for Optimal k')
ax2.set_xlabel('Number of Clusters (k)')
ax2.set_ylabel('Silhouette Score')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f"{report_path}/kmeans_optimization.png", dpi=300, bbox_inches='tight')
plt.close()

# Choose k=4 for consistency with quadrant analysis
k = 4
kmeans = KMeans(n_clusters=k, random_state=42)
df_filtered['Cluster'] = kmeans.fit_predict(X_scaled)

# Map cluster numbers to meaningful names based on centroids
centroids = scaler.inverse_transform(kmeans.cluster_centers_)
cluster_properties = {}

for i in range(k):
    roi = centroids[i, 0]
    std = centroids[i, 1]
    sharpe = roi / std if std > 0 else 0
    
    # Determine cluster characteristics
    roi_level = "High" if roi > roi_median else "Low"
    std_level = "High" if std > std_median else "Low"
    
    cluster_properties[i] = {
        "ROI": roi,
        "STD": std,
        "Sharpe": sharpe,
        "Description": f"Cluster {i}: {roi_level} ROI, {std_level} σ"
    }

# Map cluster numbers to names
cluster_mapping = {i: prop["Description"] for i, prop in cluster_properties.items()}
df_filtered['Cluster_Name'] = df_filtered['Cluster'].map(cluster_mapping)

# Visualize the clusters
plt.figure(figsize=(12, 10))

# Create a scatter plot colored by cluster
scatter = plt.scatter(
    df_filtered["ROI_STANDARD_DEVIATION"], 
    df_filtered["EXPECTED_ROI"],
    c=df_filtered["Cluster"],
    cmap="viridis",
    s=50,
    alpha=0.7
)

# Plot cluster centroids
plt.scatter(
    centroids[:, 1],  # STD
    centroids[:, 0],  # ROI
    c=range(k),
    cmap="viridis",
    marker='X',
    s=200,
    edgecolors='black',
    label='Centroids'
)

# Add cluster labels near centroids
for i, (std, roi) in enumerate(zip(centroids[:, 1], centroids[:, 0])):
    plt.annotate(
        f"Cluster {i}",
        (std, roi),
        xytext=(10, 10),
        textcoords='offset points',
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.7)
    )

# Add reference lines for median values
plt.axhline(y=roi_median, color='gray', linestyle='--', alpha=0.5)
plt.axvline(x=std_median, color='gray', linestyle='--', alpha=0.5)

# Add a colorbar legend
legend1 = plt.colorbar(scatter)
legend1.set_label("Cluster", rotation=270, labelpad=20)

# Add labels and title
plt.xlabel('ROI Standard Deviation (σ)')
plt.ylabel('Expected ROI')
plt.title('K-means Clustering of Wallets Based on ROI and Standard Deviation')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f"{report_path}/kmeans_clusters.png", dpi=300, bbox_inches='tight')
plt.close()

# 8. Cluster Statistics
print("\n8. Cluster Statistics")

cluster_stats = df_filtered.groupby("Cluster_Name").agg({
    "EXPECTED_ROI": ["mean", "median"],
    "ROI_STANDARD_DEVIATION": ["mean", "median"],
    "SHARPE_RATIO": ["mean", "median"],
    "WIN_LOSS_RATIO": ["mean", "median"],
    "MAX_TRADE_PROPORTION": ["mean", "median"],
    "UNIQUE_TOKENS_TRADED": ["mean", "median"],
    "TOTAL_TRADES": ["mean", "median"],
    "WALLET_STATUS": lambda x: (x == "churned").mean() * 100  # Churn rate as percentage
})

print("\nCluster Statistics:")
print(cluster_stats)

# Save cluster statistics to CSV
cluster_stats.to_csv(f"{report_path}/cluster_statistics.csv")

# 9. Additional Analysis: Sharpe Ratio vs EXPECTED_ROI
print("\n9. Additional Analysis: Sharpe Ratio vs ROI")

plt.figure(figsize=(10, 8))
scatter = plt.scatter(
    df_filtered["EXPECTED_ROI"], 
    df_filtered["SHARPE_RATIO"],
    c=df_filtered["ROI_STANDARD_DEVIATION"],
    cmap="plasma",
    s=40,
    alpha=0.7
)

# Add color bar
cbar = plt.colorbar(scatter)
cbar.set_label('ROI Standard Deviation', rotation=270, labelpad=20)

# Add labels and title
plt.xlabel('Expected ROI')
plt.ylabel('Sharpe Ratio')
plt.title('Sharpe Ratio vs Expected ROI (colored by Standard Deviation)')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f"{report_path}/sharpe_vs_roi.png", dpi=300, bbox_inches='tight')
plt.close()

# 10. Correlation Analysis
print("\n10. Correlation Analysis")

# Calculate correlations between key metrics
correlation_columns = [
    "EXPECTED_ROI", "ROI_STANDARD_DEVIATION", "SHARPE_RATIO", 
    "WIN_LOSS_RATIO", "MAX_TRADE_PROPORTION", "UNIQUE_TOKENS_TRADED", 
    "TOTAL_TRADES"
]

correlation_matrix = df[correlation_columns].corr()
print("\nCorrelation Matrix:")
print(correlation_matrix)

# Plot correlation heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(
    correlation_matrix, 
    annot=True, 
    cmap="coolwarm", 
    center=0,
    linewidths=0.5, 
    fmt=".2f"
)
plt.title("Correlation Matrix of Key Trading Metrics")
plt.tight_layout()
plt.savefig(f"{report_path}/correlation_heatmap.png", dpi=300, bbox_inches='tight')
plt.close()

# 11. Analyze wallet status distribution across clusters
print("\n11. Wallet Status Distribution Across Clusters")

wallet_status_cluster = pd.crosstab(
    df_filtered["Cluster_Name"], 
    df_filtered["WALLET_STATUS"], 
    normalize="index"
) * 100  # Convert to percentages

print("\nWallet Status Distribution by Cluster (%):")
print(wallet_status_cluster)

# Plot wallet status distribution
wallet_status_cluster.plot(
    kind="bar", 
    figsize=(14, 8),
    stacked=True
)
plt.title("Wallet Status Distribution by Cluster")
plt.xlabel("Cluster")
plt.ylabel("Percentage (%)")
plt.legend(title="Wallet Status")
plt.tight_layout()
plt.savefig(f"{report_path}/wallet_status_by_cluster.png", dpi=300, bbox_inches='tight')
plt.close()

# 12. WALLET_STATUS analysis by quadrant
print("\n12. Wallet Status Distribution Across Quadrants")

wallet_status_quadrant = pd.crosstab(
    df_filtered["Quadrant"], 
    df_filtered["WALLET_STATUS"], 
    normalize="index"
) * 100  # Convert to percentages

print("\nWallet Status Distribution by Quadrant (%):")
print(wallet_status_quadrant)

# Plot wallet status distribution by quadrant
wallet_status_quadrant.plot(
    kind="bar", 
    figsize=(14, 8),
    stacked=True
)
plt.title("Wallet Status Distribution by Quadrant")
plt.xlabel("Quadrant")
plt.ylabel("Percentage (%)")
plt.legend(title="Wallet Status")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig(f"{report_path}/wallet_status_by_quadrant.png", dpi=300, bbox_inches='tight')
plt.close()

print("\nAnalysis completed. Results and visualizations saved to the 'report' directory.")

# Update the kmeans clustering function to use the specified cluster names and save to visualization folder
def plot_kmeans_clusters(df, x_col, y_col, k=4, report_dir=REPORT_DIR):
    # Apply K-means clustering
    features = df[[x_col, y_col]].values
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    df['cluster'] = kmeans.fit_predict(features)
    centers = kmeans.cluster_centers_
    
    # Map cluster labels to meaningful names
    cluster_names = {
        0: "Skilled Investors",
        1: "Cautious/Conservative", 
        2: "Unsuccessful Gamblers",
        3: "Jackpot Seekers"
    }
    
    # Set colors and markers for each cluster
    colors = ['#2D68C4', '#65A2D9', '#FF7E79', '#F93822']
    markers = ['o', 's', '^', 'D']
    
    # Create plot
    plt.figure(figsize=(12, 10))
    
    # Plot each cluster with its unique color and marker
    for i in range(k):
        cluster_data = df[df['cluster'] == i]
        plt.scatter(
            cluster_data[x_col], 
            cluster_data[y_col], 
            s=50, 
            c=colors[i], 
            marker=markers[i],
            alpha=0.7, 
            label=f"{cluster_names[i]} (n={len(cluster_data)})"
        )
    
    # Plot cluster centers
    plt.scatter(
        centers[:, 0], 
        centers[:, 1], 
        s=200, 
        c='yellow', 
        marker='*', 
        alpha=1, 
        label='Cluster Centers'
    )
    
    # Calculate statistics for each cluster
    cluster_stats = []
    for i in range(k):
        cluster_data = df[df['cluster'] == i]
        active_rate = (cluster_data['WALLET_STATUS'] == 'active').mean() * 100
        churn_rate = 100 - active_rate
        
        # Add cluster annotation
        plt.annotate(
            f"{cluster_names[i]}\n"
            f"Size: {len(cluster_data)} wallets\n"
            f"Churn Rate: {churn_rate:.1f}%\n"
            f"Mean ROI: {cluster_data[y_col].mean():.3f}\n"
            f"Mean σ: {cluster_data[x_col].mean():.3f}",
            xy=(centers[i, 0], centers[i, 1]),
            xytext=(centers[i, 0] + 0.05, centers[i, 1] + 0.05),
            bbox=dict(boxstyle="round,pad=0.5", fc="white", alpha=0.8),
            fontsize=10
        )
        
        # Store cluster statistics
        cluster_stats.append({
            'cluster': cluster_names[i],
            'size': len(cluster_data),
            'churn_rate': churn_rate,
            'mean_roi': cluster_data[y_col].mean(),
            'mean_std': cluster_data[x_col].mean(),
            'mean_sharpe': cluster_data['SHARPE_RATIO'].mean() if 'SHARPE_RATIO' in cluster_data.columns else None
        })
    
    # Set axis labels, title, etc.
    plt.xlabel(x_col, fontsize=14)
    plt.ylabel(y_col, fontsize=14)
    plt.title(f'K-Means Clustering (k={k}): {y_col} vs {x_col}', fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    
    # Save image and statistics
    plt.tight_layout()
    plt.savefig(os.path.join(report_dir, 'kmeans_clusters.png'), dpi=300)
    
    # Save statistics to CSV
    stats_df = pd.DataFrame(cluster_stats)
    stats_df.to_csv(os.path.join(report_dir, 'cluster_statistics.csv'), index=False)
    
    return df['cluster'], stats_df

# Add the code to run kmeans clustering at the end of the script
print("\n9. K-means Clustering Analysis")
# Run k-means clustering on the filtered dataset
df_clustering = df_filtered.copy()
clusters, cluster_stats = plot_kmeans_clusters(
    df_clustering, 
    x_col="ROI_STANDARD_DEVIATION", 
    y_col="EXPECTED_ROI", 
    k=4,
    report_dir=REPORT_DIR
)

print("K-means clustering completed. Results saved to visualization directory.")
print(cluster_stats) 