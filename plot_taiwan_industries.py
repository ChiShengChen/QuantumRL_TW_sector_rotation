import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib import rcParams
from matplotlib.dates import YearLocator, DateFormatter
import warnings
warnings.filterwarnings('ignore')

# 設置中文字體和圖表樣式
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial Unicode MS', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 設置圖表樣式
plt.style.use('seaborn-v0_8-whitegrid')
rcParams['figure.figsize'] = 14, 10
rcParams['font.size'] = 12
rcParams['axes.titlesize'] = 16
rcParams['axes.labelsize'] = 14
rcParams['xtick.labelsize'] = 10
rcParams['ytick.labelsize'] = 10

def load_and_process_data():
    """載入並處理數據"""
    print("Loading data...")
    df = pd.read_csv('features_data.csv')
    
    # Convert date column
    df['date'] = pd.to_datetime(df['date'])
    
    # Group by date and industry, calculate daily capital share
    daily_industry_data = df.groupby(['date', 'industry'])['capital_share'].sum().reset_index()
    
    # Pivot table: each row is a date, each column is an industry
    pivot_data = daily_industry_data.pivot(index='date', columns='industry', values='capital_share')
    
    # Fill missing values
    pivot_data = pivot_data.fillna(method='ffill').fillna(0)
    
    print(f"Data loading completed! Time range: {pivot_data.index.min().strftime('%Y-%m-%d')} to {pivot_data.index.max().strftime('%Y-%m-%d')}")
    print(f"Number of industries: {len(pivot_data.columns)}")
    
    return pivot_data

def create_industry_price_chart(data):
    """Create industry price chart"""
    print("Creating chart...")
    
    # Create chart
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12), height_ratios=[2, 1])
    
    # Main chart: industry price trends
    # Select main industries (those with larger capital share)
    latest_values = data.iloc[-1].sort_values(ascending=False).head(10)
    main_industries = latest_values.index.tolist()
    
    # Plot main industries
    for industry in main_industries:
        if industry in data.columns:
            ax1.plot(data.index, data[industry], label=industry, linewidth=2, alpha=0.8)
    
    ax1.set_title('Taiwan Stock Market Industry Capital Share Trends (2007-2025)', fontsize=18, fontweight='bold', pad=20)
    ax1.set_ylabel('Capital Share', fontsize=14)
    ax1.set_xlabel('')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(bottom=0)
    
    # Format x-axis dates
    ax1.xaxis.set_major_locator(YearLocator(2))
    ax1.xaxis.set_major_formatter(DateFormatter('%Y'))
    
    # Secondary chart: industry distribution pie chart (latest data)
    latest_data = data.iloc[-1].sort_values(ascending=False).head(8)
    other_sum = data.iloc[-1].drop(latest_data.index).sum()
    
    if other_sum > 0:
        plot_data = pd.concat([latest_data, pd.Series({'Others': other_sum})])
    else:
        plot_data = latest_data
    
    # Create color mapping
    colors = plt.cm.Set3(np.linspace(0, 1, len(plot_data)))
    
    wedges, texts, autotexts = ax2.pie(plot_data.values, 
                                       labels=plot_data.index, 
                                       autopct='%1.1f%%',
                                       colors=colors,
                                       startangle=90,
                                       textprops={'fontsize': 10})
    
    ax2.set_title('Latest Industry Distribution (2025)', fontsize=16, fontweight='bold', pad=20)
    
    # Adjust layout
    plt.tight_layout()
    
    return fig

def create_alternative_chart(data):
    """Create alternative chart: stacked area chart"""
    print("Creating stacked area chart...")
    
    fig, ax = plt.subplots(figsize=(16, 10))
    
    # Select main industries
    latest_values = data.iloc[-1].sort_values(ascending=False).head(8)
    main_industries = latest_values.index.tolist()
    
    # Create stacked area chart
    ax.stackplot(data.index, 
                 [data[industry] for industry in main_industries],
                 labels=main_industries,
                 alpha=0.7)
    
    ax.set_title('Taiwan Stock Market Industry Capital Share Stacked Chart (2007-2025)', fontsize=18, fontweight='bold', pad=20)
    ax.set_ylabel('Capital Share', fontsize=14)
    ax.set_xlabel('Year', fontsize=14)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # Format x-axis
    ax.xaxis.set_major_locator(YearLocator(2))
    ax.xaxis.set_major_formatter(DateFormatter('%Y'))
    
    plt.tight_layout()
    
    return fig

def save_charts(fig1, fig2):
    """Save charts"""
    print("Saving charts...")
    
    # Save as high-resolution PNG
    fig1.savefig('taiwan_industries_main.png', dpi=300, bbox_inches='tight', 
                 facecolor='white', edgecolor='none')
    fig2.savefig('taiwan_industries_stacked.png', dpi=300, bbox_inches='tight',
                 facecolor='white', edgecolor='none')
    
    # Save as PDF (suitable for papers)
    fig1.savefig('taiwan_industries_main.pdf', bbox_inches='tight',
                 facecolor='white', edgecolor='none')
    fig2.savefig('taiwan_industries_stacked.pdf', bbox_inches='tight',
                 facecolor='white', edgecolor='none')
    
    print("Charts saved!")
    print("- taiwan_industries_main.png/pdf: Main chart (line chart + pie chart)")
    print("- taiwan_industries_stacked.png/pdf: Stacked area chart")

def main():
    """Main function"""
    try:
        # Load data
        data = load_and_process_data()
        
        # Create charts
        fig1 = create_industry_price_chart(data)
        fig2 = create_alternative_chart(data)
        
        # Save charts
        save_charts(fig1, fig2)
        
        # Display charts
        plt.show()
        
        print("\nCompleted! Charts have been created and saved.")
        print("These charts are suitable for papers, including:")
        print("1. Main chart: showing industry trends over time")
        print("2. Stacked chart: showing overall industry structure changes")
        print("3. High-resolution output: supporting paper printing requirements")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
