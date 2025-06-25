import pandas as pd
import os
import re
import numpy as np
import config

def calculate_industry_returns():
    """
    Calculates the average daily stock return for each industry.
    """
    print("Calculating daily industry returns from stock prices...")

    # Load data
    print("Loading stock prices, industry classifications, and translation map...")
    prices = pd.read_csv(config.PRICE_FILE, usecols=['date', 'stock_id', 'close'], parse_dates=['date'],
                         dtype={'stock_id': str})
    industry_df = pd.read_csv(config.INDUSTRY_FILE, usecols=['stock_id', 'category'],
                              dtype={'stock_id': str})
    translation_map = pd.read_csv(config.TRANSLATION_FILE)

    # Extract the primary industry (first one in the list)
    industry_df['industry_cn'] = industry_df['category'].apply(
        lambda x: re.search(r"'([^']*)'", str(x)).group(1) if re.search(r"'([^']*)'", str(x)) else None
    )
    industry_df = industry_df.dropna(subset=['industry_cn'])

    # Rename columns for merging
    translation_map.rename(columns={'chinese_name': 'industry_cn', 'english_name': 'industry'}, inplace=True)

    # Merge Chinese industry names with English translations
    industry_df = industry_df.merge(translation_map, on='industry_cn', how='left')
    industry_df = industry_df[['stock_id', 'industry']].dropna()

    # Calculate daily returns for each stock
    print("Calculating daily returns for each stock...")
    prices.sort_values(['stock_id', 'date'], inplace=True)
    prices['daily_return'] = prices.groupby('stock_id')['close'].pct_change().fillna(0)

    # Merge prices with industry information
    data = pd.merge(prices, industry_df, on='stock_id', how='inner')

    # Calculate equal-weighted average return for each industry per day
    print("Calculating average daily return for each industry...")
    industry_returns = data.groupby(['date', 'industry'])['daily_return'].mean().reset_index()
    industry_returns.rename(columns={'daily_return': 'industry_return'}, inplace=True)

    print("Industry return calculation complete.")
    return industry_returns


def prepare_capital_share_data():
    """
    Loads, pivots, and calculates the daily capital share for each industry.
    """
    source_file = config.CAPITAL_SHARE_FILE
    
    if not os.path.exists(source_file):
        print(f"Error: Source file not found at {source_file}")
        return None

    df = pd.read_csv(source_file, parse_dates=['date'])
    df['amount'] = df['amount'].fillna(0)

    # Group by date and industry and sum the amounts to remove duplicates
    print("Aggregating data to remove duplicates...")
    df = df.groupby(['date', 'industry'])['amount'].sum().reset_index()

    capital_df = df.pivot(index='date', columns='industry', values='amount')
    capital_df = capital_df.fillna(0)

    daily_total_capital = capital_df.sum(axis=1)
    capital_share_df = capital_df.div(daily_total_capital + 1e-9, axis=0)

    capital_share_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    capital_share_df.fillna(0, inplace=True)
    
    print("Capital share data preparation complete.")
    return capital_share_df 