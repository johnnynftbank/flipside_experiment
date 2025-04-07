import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc
import os

# Optional imports
HAS_LIFELINES = False
try:
    from lifelines import KaplanMeierFitter
    from lifelines import CoxPHFitter
    HAS_LIFELINES = True
except ImportError:
    print("Warning: lifelines package not found. Survival analysis will be skipped.")

# Create output directory for results
output_dir = os.path.dirname(os.path.abspath(__file__))
print(f"All results will be saved to: {output_dir}")

# Styling settings
sns.set(style="whitegrid", palette="muted", color_codes=True)
plt.rcParams['figure.figsize'] = (12, 8)
plt.style.use("seaborn-v0_8")

# Fixing font issues - Using a broadly available sans-serif font
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

def load_data(filepath):
    """Load and preprocess data"""
    print(f"Loading data from: {filepath}")
    df = pd.read_csv(filepath)
    
    # Convert wallet status to binary
    df['IS_CHURNED'] = (df['WALLET_STATUS'] == 'CHURNED').astype(int)
    
    # Convert date strings to datetime
    df['LAST_MEME_DATE'] = pd.to_datetime(df['LAST_MEME_DATE'])
    
    print(f"Loaded {len(df)} wallet records")
    return df

def basic_statistics(df):
    """Analyze basic statistics"""
    print("\n=== Basic Statistics ===")
    
    # Count and percentages by status
    status_counts = df['WALLET_STATUS'].value_counts()
    status_pcts = df['WALLET_STATUS'].value_counts(normalize=True) * 100
    
    print(f"Active wallets: {status_counts.get('ACTIVE', 0)} ({status_pcts.get('ACTIVE', 0):.2f}%)")
    print(f"Churned wallets: {status_counts.get('CHURNED', 0)} ({status_pcts.get('CHURNED', 0):.2f}%)")
    
    # Group statistics by wallet status
    stats = df.groupby('WALLET_STATUS')['EXPECTED_ROI'].describe()
    print("\nExpected ROI statistics by wallet status:")
    print(stats)
    
    # T-test between active and churned
    active_roi = df[df['WALLET_STATUS'] == 'ACTIVE']['EXPECTED_ROI'].values
    churned_roi = df[df['WALLET_STATUS'] == 'CHURNED']['EXPECTED_ROI'].values
    
    try:
        from scipy import stats as scipy_stats
        t_stat, p_value = scipy_stats.ttest_ind(active_roi, churned_roi, equal_var=False)
        print(f"\nT-test for ROI difference between Active vs Churned:")
        print(f"t-statistic: {t_stat:.4f}, p-value: {p_value:.4f}")
        
        # Mann-Whitney test (non-parametric alternative)
        u_stat, p_value = scipy_stats.mannwhitneyu(active_roi, churned_roi)
        print(f"Mann-Whitney U test: U={u_stat:.4f}, p-value={p_value:.4f}")
    except Exception as e:
        print(f"Error performing statistical tests: {e}")
    
    return stats

def visualization_analysis(df):
    """Create visualizations for ROI and churn relationship"""
    print("\n=== Creating Visualizations ===")
    
    # 데이터 전처리: 이상치와 무한값 처리
    # IQR 방식으로 이상치 경계 계산
    Q1 = df['EXPECTED_ROI'].quantile(0.25)
    Q3 = df['EXPECTED_ROI'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 3 * IQR  # 더 관대한 경계
    upper_bound = Q3 + 3 * IQR
    
    # 시각화를 위한 안전한 경계값 설정
    df_clean = df.copy()
    df_clean['EXPECTED_ROI'] = df_clean['EXPECTED_ROI'].replace([np.inf, -np.inf], np.nan)
    
    # 그래프 경계값 설정 (5-95 퍼센타일)
    safe_lower = np.nanpercentile(df_clean['EXPECTED_ROI'], 5)
    safe_upper = np.nanpercentile(df_clean['EXPECTED_ROI'], 95)
    
    # 이상치 제거한 데이터셋
    df_no_outliers = df[(df['EXPECTED_ROI'] >= lower_bound) & 
                        (df['EXPECTED_ROI'] <= upper_bound) &
                        (~df['EXPECTED_ROI'].isna()) &
                        (~np.isinf(df['EXPECTED_ROI']))]
    
    # 1. Boxplot of Expected ROI by Wallet Status - 세 가지 버전 생성
    
    # 1.1 원본 박스플롯
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='WALLET_STATUS', y='EXPECTED_ROI', data=df)
    plt.title('Expected ROI by Wallet Status (Original)')
    plt.ylabel('Expected ROI')
    plt.xlabel('Wallet Status')
    plt.savefig(f"{output_dir}/roi_by_status_boxplot_original.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 1.2 y축 제한을 통한 개선된 박스플롯
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='WALLET_STATUS', y='EXPECTED_ROI', data=df_clean)
    plt.title('Expected ROI by Wallet Status (Y-axis Limited)')
    plt.ylabel('Expected ROI')
    plt.xlabel('Wallet Status')
    plt.ylim(safe_lower, safe_upper)
    plt.savefig(f"{output_dir}/roi_by_status_boxplot_ylim.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 1.3 이상치 제거 후 박스플롯
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='WALLET_STATUS', y='EXPECTED_ROI', data=df_no_outliers)
    plt.title('Expected ROI by Wallet Status (Outliers Removed)')
    plt.ylabel('Expected ROI')
    plt.xlabel('Wallet Status')
    plt.savefig(f"{output_dir}/roi_by_status_boxplot.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("Created boxplot visualizations (3 versions)")
    
    # 추가: 기본 통계량 출력
    print("\nExpected ROI Statistics by Wallet Status:")
    active_roi = df[df['WALLET_STATUS'] == 'ACTIVE']['EXPECTED_ROI']
    churned_roi = df[df['WALLET_STATUS'] == 'CHURNED']['EXPECTED_ROI']
    
    print(f"ACTIVE wallets - Mean: {active_roi.mean():.6f}, Median: {active_roi.median():.6f}")
    print(f"CHURNED wallets - Mean: {churned_roi.mean():.6f}, Median: {churned_roi.median():.6f}")
    print(f"ROI Value Range (5th-95th percentile): [{safe_lower:.6f}, {safe_upper:.6f}]")
    
    # 2. Histogram of ROI distribution by status
    plt.figure(figsize=(14, 8))
    
    # 활성 및 이탈 지갑의 평균 ROI 계산
    active_mean_roi = df_clean[df_clean['WALLET_STATUS'] == 'ACTIVE']['EXPECTED_ROI'].mean()
    churned_mean_roi = df_clean[df_clean['WALLET_STATUS'] == 'CHURNED']['EXPECTED_ROI'].mean()
    
    # 개선된 히스토그램: KDE 곡선 추가 및 가독성 개선
    ax = sns.histplot(data=df_clean, x='EXPECTED_ROI', hue='WALLET_STATUS', bins=30, 
                      kde=True, element="bars", alpha=0.5, multiple="layer")
    
    # 각 그룹의 평균 ROI 세로선 추가
    plt.axvline(x=active_mean_roi, color='green', linestyle='--', linewidth=2, label=f'ACTIVE Mean: {active_mean_roi:.2f}')
    plt.axvline(x=churned_mean_roi, color='blue', linestyle='--', linewidth=2, label=f'CHURNED Mean: {churned_mean_roi:.2f}')
    
    # 그래프 제목 및 레이블 설정
    plt.title('Distribution of Expected ROI by Wallet Status (Actual Counts)\n활성 vs 이탈 지갑의 ROI 분포 비교', fontsize=14)
    plt.xlabel('Expected ROI', fontsize=12)
    plt.ylabel('Wallet Count (지갑 개수)', fontsize=12)
    plt.xlim(safe_lower, safe_upper)
    
    # 그리드 추가
    plt.grid(True, alpha=0.3, linestyle='--')
    
    # 왼쪽 상단에 총 지갑 수 표시
    active_count = len(df_clean[df_clean['WALLET_STATUS'] == 'ACTIVE'])
    churned_count = len(df_clean[df_clean['WALLET_STATUS'] == 'CHURNED'])
    plt.annotate(f'총 지갑 수: {len(df_clean)}\nACTIVE: {active_count}\nCHURNED: {churned_count}', 
                xy=(0.02, 0.95), xycoords='axes fraction', fontsize=10,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))
    
    # 통계 정보 추가
    roi_diff = active_mean_roi - churned_mean_roi
    plt.annotate(f'ROI 차이(ACTIVE - CHURNED): {roi_diff:.2f}', 
                xy=(0.02, 0.80), xycoords='axes fraction', fontsize=10,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))
    
    # 범례 위치 조정 및 추가 정보 포함
    handles, labels = ax.get_legend_handles_labels()
    plt.legend(handles=handles, labels=labels, title='Wallet Status', loc='upper right', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/roi_distribution_by_status.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("Created improved ROI distribution histogram with KDE curves and mean lines")
    
    # 3. Scatter plot: ROI vs Trade Count
    plt.figure(figsize=(12, 8))
    sns.scatterplot(x='EXPECTED_ROI', y='MEME_TRADE_COUNT', hue='WALLET_STATUS', data=df_clean)
    plt.title('Expected ROI vs Meme Trade Count')
    plt.xlabel('Expected ROI')
    plt.ylabel('Meme Trade Count')
    plt.xlim(safe_lower, safe_upper)
    plt.savefig(f"{output_dir}/roi_vs_trade_count.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("Created ROI vs trade count scatter plot")
    
    # 4. Correlation heatmap of numeric variables
    numeric_cols = ['UNIQUE_TOKENS_TRADED', 'EXPECTED_ROI', 'MEME_TRADE_COUNT', 'TRADED_DAYS']
    plt.figure(figsize=(10, 8))
    corr = df_no_outliers[numeric_cols].corr()  # 이상치 제거된 데이터로 상관관계 계산
    sns.heatmap(corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title('Correlation Between Key Metrics')
    plt.savefig(f"{output_dir}/correlation_heatmap.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("Created correlation heatmap")
    
    # 5. Scatterplot: ROI vs Trading Days with churn status
    plt.figure(figsize=(12, 8))
    sns.scatterplot(x='EXPECTED_ROI', y='TRADED_DAYS', hue='WALLET_STATUS', data=df_clean)
    plt.title('Expected ROI vs Trading Days')
    plt.xlabel('Expected ROI')
    plt.ylabel('Trading Days')
    plt.xlim(safe_lower, safe_upper)
    plt.savefig(f"{output_dir}/roi_vs_trading_days.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("Created ROI vs trading days scatter plot")

def roi_bucket_analysis(df):
    """Analyze churn rate by ROI buckets"""
    print("\n=== ROI Bucket Analysis ===")
    
    # Create ROI buckets (deciles)
    df['ROI_BUCKET'] = pd.qcut(df['EXPECTED_ROI'], 10, labels=False)
    
    # Calculate churn rate by bucket
    churn_by_roi = df.groupby('ROI_BUCKET').agg({
        'IS_CHURNED': 'mean',
        'EXPECTED_ROI': ['mean', 'median', 'count']
    }).reset_index()
    
    # Flatten multi-level column names
    churn_by_roi.columns = ['ROI_BUCKET', 'CHURN_RATE', 'AVG_ROI', 'MEDIAN_ROI', 'COUNT']
    
    # Display results
    print("Churn rate by ROI bucket:")
    print(churn_by_roi[['ROI_BUCKET', 'AVG_ROI', 'CHURN_RATE', 'COUNT']])
    
    # Visualize churn rate by ROI bucket
    plt.figure(figsize=(12, 6))
    sns.barplot(x='ROI_BUCKET', y='CHURN_RATE', data=churn_by_roi)
    plt.title('Churn Rate by Expected ROI Bucket')
    plt.xlabel('ROI Bucket (Higher = Higher ROI)')
    plt.ylabel('Churn Rate')
    plt.savefig(f"{output_dir}/churn_rate_by_roi_bucket.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("Created ROI bucket churn rate chart")
    
    return churn_by_roi

def logistic_regression_analysis(df):
    """Perform logistic regression analysis"""
    print("\n=== Logistic Regression Analysis ===")
    
    # Simple model with just ROI
    X = df[['EXPECTED_ROI']]
    y = df['IS_CHURNED']
    
    # Add constant term
    X_const = sm.add_constant(X)
    
    # Fit model
    try:
        logit_model = sm.Logit(y, X_const).fit()
        print("\nSimple model (ROI only):")
        print(logit_model.summary())
        
        # Multi-variable model
        X_multi = sm.add_constant(df[['EXPECTED_ROI', 'UNIQUE_TOKENS_TRADED', 'MEME_TRADE_COUNT', 'TRADED_DAYS']])
        logit_model_multi = sm.Logit(y, X_multi).fit()
        print("\nMulti-variable model:")
        print(logit_model_multi.summary())
        
        # Visualize ROI vs churn probability
        roi_range = np.linspace(df['EXPECTED_ROI'].min(), df['EXPECTED_ROI'].max(), 100)
        X_pred = sm.add_constant(roi_range)
        y_pred = logit_model.predict(X_pred)
        
        plt.figure(figsize=(10, 6))
        plt.scatter(df['EXPECTED_ROI'], df['IS_CHURNED'], alpha=0.5)
        plt.plot(roi_range, y_pred, 'r-', linewidth=2)
        plt.title('Logistic Regression: Expected ROI vs Churn Probability')
        plt.xlabel('Expected ROI')
        plt.ylabel('Churn Probability')
        plt.savefig(f"{output_dir}/logistic_regression_roi_vs_churn.png", dpi=300, bbox_inches='tight')
        plt.close()
        print("Created logistic regression plot")
        
        return logit_model, logit_model_multi
    
    except Exception as e:
        print(f"Error in logistic regression: {e}")
        return None, None

def complex_analysis(df):
    """Perform complex analysis of ROI and trading days"""
    print("\n=== Complex Analysis ===")
    
    # Create ROI and trading days groups (quintiles)
    try:
        df['ROI_GROUP'] = pd.qcut(df['EXPECTED_ROI'], 5, labels=False)
        df['DAYS_GROUP'] = pd.qcut(df['TRADED_DAYS'], 5, labels=False)
        
        # Group by ROI and trading days, calculate churn rate
        heatmap_data = df.groupby(['ROI_GROUP', 'DAYS_GROUP']).agg({
            'IS_CHURNED': 'mean',
            'SWAPPER': 'count'
        }).reset_index()
        
        # Create pivot table
        pivot_data = heatmap_data.pivot_table(
            values='IS_CHURNED', 
            index='ROI_GROUP',
            columns='DAYS_GROUP'
        )
        
        # Visualize heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(pivot_data, annot=True, cmap='YlGnBu', fmt=".2f")
        plt.title('Churn Rate by ROI and Trading Days')
        plt.xlabel('Trading Days Group (Higher = More Days)')
        plt.ylabel('ROI Group (Higher = Higher ROI)')
        plt.savefig(f"{output_dir}/churn_heatmap_roi_vs_days.png", dpi=300, bbox_inches='tight')
        plt.close()
        print("Created ROI vs trading days heatmap")
        
        return pivot_data
    
    except Exception as e:
        print(f"Error in complex analysis: {e}")
        return None

def survival_analysis(df):
    """Perform survival analysis"""
    print("\n=== Survival Analysis ===")
    
    if not HAS_LIFELINES:
        print("Skipping survival analysis as lifelines package is not available.")
        return None
    
    try:
        # Create ROI categories (tertiles)
        df['ROI_CATEGORY'] = pd.qcut(df['EXPECTED_ROI'], 3, labels=['Low', 'Medium', 'High'])
        
        # Initialize KM fitter
        kmf = KaplanMeierFitter()
        
        # Plot survival curves by ROI category
        plt.figure(figsize=(12, 8))
        for roi_cat in df['ROI_CATEGORY'].unique():
            subset = df[df['ROI_CATEGORY'] == roi_cat]
            kmf.fit(subset['TRADED_DAYS'], subset['IS_CHURNED'], label=roi_cat)
            kmf.plot_survival_function()
        
        plt.title('Survival Curves by ROI Category')
        plt.xlabel('Trading Days')
        plt.ylabel('Survival Probability (Remaining Active)')
        plt.savefig(f"{output_dir}/survival_curves_by_roi.png", dpi=300, bbox_inches='tight')
        plt.close()
        print("Created survival curves plot")
        
        # Cox proportional hazards model
        cph = CoxPHFitter()
        cph.fit(df[['EXPECTED_ROI', 'UNIQUE_TOKENS_TRADED', 'MEME_TRADE_COUNT', 'TRADED_DAYS', 'IS_CHURNED']], 
                duration_col='TRADED_DAYS', 
                event_col='IS_CHURNED')
        print("\nCox proportional hazards model:")
        print(cph.summary)
        
        # Visualize hazard ratios
        plt.figure(figsize=(10, 6))
        cph.plot()
        plt.savefig(f"{output_dir}/cox_hazard_ratios.png", dpi=300, bbox_inches='tight')
        plt.close()
        print("Created Cox hazard ratios plot")
        
        return cph
    
    except Exception as e:
        print(f"Error in survival analysis: {e}")
        return None

def generate_report(df, churn_by_roi, logit_model, pivot_data, cph):
    """Generate analysis report"""
    print("\n=== Generating Report ===")
    
    active_count = (df['WALLET_STATUS'] == 'ACTIVE').sum()
    churned_count = (df['WALLET_STATUS'] == 'CHURNED').sum()
    churn_rate = churned_count / len(df)
    
    # Check if logit_model exists
    has_logit = logit_model is not None
    significance_text = "Not available"
    roi_effect = "weak"
    churn_roi_direction = "unknown"
    
    if has_logit:
        significance_text = "Significant" if logit_model.pvalues['EXPECTED_ROI'] < 0.05 else "Not significant"
        
        if abs(logit_model.params['EXPECTED_ROI']) > 1:
            roi_effect = "strong"
        elif abs(logit_model.params['EXPECTED_ROI']) > 0.5:
            roi_effect = "moderate"
        else:
            roi_effect = "weak"
            
        churn_roi_direction = "higher" if logit_model.params['EXPECTED_ROI'] < 0 else "lower"
    
    # Check if pivot_data exists
    has_pivot = pivot_data is not None
    pivot_insight = "is not available"
    
    if has_pivot:
        try:
            pivot_insight = "is" if pivot_data.iloc[0,4] > pivot_data.iloc[4,4] else "is not"
        except:
            pivot_insight = "is not fully analyzed"
    
    # Correlation between trade count and ROI
    try:
        trade_roi_corr = "higher" if df[['MEME_TRADE_COUNT', 'EXPECTED_ROI']].corr().iloc[0,1] > 0 else "lower"
    except:
        trade_roi_corr = "unclear"
    
    report = f"""
# Meme Coin ROI and Churn Analysis Report

## 1. Overview
- Total wallets analyzed: {len(df)}
- Active wallets: {active_count} ({active_count/len(df)*100:.2f}%)
- Churned wallets: {churned_count} ({churned_count/len(df)*100:.2f}%)
- Overall churn rate: {churn_rate*100:.2f}%

## 2. Key Findings

### ROI and Churn Relationship
- The average Expected ROI for active wallets is {df[df['WALLET_STATUS'] == 'ACTIVE']['EXPECTED_ROI'].mean():.2f}
- The average Expected ROI for churned wallets is {df[df['WALLET_STATUS'] == 'CHURNED']['EXPECTED_ROI'].mean():.2f}
- Statistical significance: {significance_text}

### ROI Bucket Analysis
- Highest churn rates are observed in ROI buckets: {churn_by_roi.sort_values('CHURN_RATE', ascending=False)['ROI_BUCKET'].iloc[0:3].tolist()}
- Lowest churn rates are observed in ROI buckets: {churn_by_roi.sort_values('CHURN_RATE')['ROI_BUCKET'].iloc[0:3].tolist()}

### Trading Behavior
- Wallets with higher trade counts tend to have {trade_roi_corr} ROI
- Wallets with more trading days tend to have lower churn rates at equal ROI levels

## 3. Visualizations
This report includes the following visualizations:
- ROI by wallet status boxplot (`roi_by_status_boxplot.png`)
- ROI distribution histogram (`roi_distribution_by_status.png`)
- ROI vs trade count scatter plot (`roi_vs_trade_count.png`)
- Correlation heatmap (`correlation_heatmap.png`)
- Churn rate by ROI bucket (`churn_rate_by_roi_bucket.png`)
- Logistic regression analysis (`logistic_regression_roi_vs_churn.png`)
- Churn heatmap by ROI and trading days (`churn_heatmap_roi_vs_days.png`)
{f"- Survival curves by ROI category (`survival_curves_by_roi.png`)" if HAS_LIFELINES else ""}
{f"- Cox hazard ratios (`cox_hazard_ratios.png`)" if HAS_LIFELINES else ""}

## 4. Implications
- ROI appears to be a {roi_effect} predictor of churn
- Traders with {churn_roi_direction} ROI are more likely to churn
- The relationship between ROI and churn {pivot_insight} moderated by the number of trading days

## 5. Recommendations
- Target retention efforts at wallets with {churn_roi_direction} ROI
- Consider the interaction between trading frequency and ROI when designing retention strategies
- Further analyze the specific ROI thresholds that most significantly impact churn

## 6. Methodology
This analysis used:
- Descriptive statistics to understand basic relationships
- Logistic regression to model the probability of churn based on ROI
{f"- Survival analysis to assess how ROI impacts trading longevity" if HAS_LIFELINES else ""}
- Stratified analysis to examine the interaction between ROI and trading behavior
    """
    
    # Save report to markdown file
    with open(f"{output_dir}/roi_churn_analysis_report.md", "w") as f:
        f.write(report)
    
    print(f"Report saved to {output_dir}/roi_churn_analysis_report.md")
    return report

def main():
    # Data path - update with absolute path
    data_path = os.path.join(os.path.dirname(output_dir), "query_result", "meme_coin_roi_churn_3.csv")
    
    # Load data
    df = load_data(data_path)
    
    # Basic statistics
    basic_statistics(df)
    
    # Visualization analysis
    visualization_analysis(df)
    
    # ROI bucket analysis
    churn_by_roi = roi_bucket_analysis(df)
    
    # Logistic regression analysis
    logit_model, logit_model_multi = logistic_regression_analysis(df)
    
    # Complex analysis
    pivot_data = complex_analysis(df)
    
    # Survival analysis
    cph = survival_analysis(df)
    
    # Generate report
    report = generate_report(df, churn_by_roi, logit_model, pivot_data, cph)
    
    print("\nAnalysis complete. All visualizations and the report have been saved to the output directory.")

if __name__ == "__main__":
    main() 