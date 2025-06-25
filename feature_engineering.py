import pandas as pd
import numpy as np
import config

def create_capital_share_features(capital_share_df):
    """
    Engineers features for the capital share data, including moving averages,
    momentum, and volatility.
    """
    print("\nStarting feature engineering...")
    
    features = {}
    
    # --- Feature Engineering ---
    # Use the original capital share as a feature
    features['capital_share'] = capital_share_df

    # 1. Moving Averages (Trend)
    print("Calculating moving averages...")
    for n in [5, 10, 20]:
        features[f'share_MA_{n}'] = capital_share_df.rolling(window=n).mean()

    # 2. Momentum (Rate of Change)
    print("Calculating momentum...")
    for n in [5, 10, 20]:
        features[f'share_Momentum_{n}'] = capital_share_df.diff(periods=n)

    # 3. Volatility (Standard Deviation)
    print("Calculating volatility...")
    for n in [20]:
        features[f'share_Vol_{n}'] = capital_share_df.rolling(window=n).std()

    print("Combining all features...")
    # Combine all feature dataframes into a single multi-indexed dataframe
    feature_df = pd.concat(features, axis=1, names=['feature_name', 'industry'])
    
    # Reshape the dataframe to have one row per date and industry
    feature_df_long = feature_df.stack(level='industry').reset_index()
    
    print("Feature engineering complete.")
    print("Shape of the final feature dataframe:", feature_df_long.shape)
    
    return feature_df_long

def prepare_for_modeling(features_df):
    """
    Prepares the feature dataframe for model training. This includes creating the
    target variable, cleaning NaN values, and splitting into train/test sets.
    """
    print("\nPreparing data for modeling...")

    # --- Create Target Variable ---
    print(f"Creating target: Is industry in top {config.TOP_N} capital share tomorrow?")
    
    features_df['future_share'] = features_df.groupby('industry')['capital_share'].shift(-1)
    features_df['future_rank'] = features_df.groupby('date')['future_share'].rank(ascending=False)
    features_df['target'] = (features_df['future_rank'] <= config.TOP_N).astype(int)

    # --- Clean Data ---
    cleaned_df = features_df.dropna(subset=[col for col in features_df.columns if col not in ['future_share', 'future_rank']]).copy()
    cleaned_df = cleaned_df.dropna(subset=['target']) # Ensure target is not NaN
    cleaned_df.drop(columns=['future_share'], inplace=True)
    
    print(f"Original feature rows: {len(features_df)}, Cleaned rows: {len(cleaned_df)}")

    # --- Split Data ---
    print(f"Splitting data into training and testing sets at {config.TEST_START_DATE}...")
    
    train_df = cleaned_df[cleaned_df['date'] < config.TEST_START_DATE]
    test_df = cleaned_df[cleaned_df['date'] >= config.TEST_START_DATE]

    feature_cols = [col for col in cleaned_df.columns if col not in ['date', 'industry', 'target', 'future_rank']]
    
    X_train = train_df[feature_cols]
    y_train = train_df['target']
    
    X_test = test_df[feature_cols]
    y_test = test_df['target']
    
    print(f"Training set shape: {X_train.shape}")
    print(f"Testing set shape: {X_test.shape}")
    
    return X_train, y_train, X_test, y_test, train_df, test_df, feature_cols


def create_sequences(df, feature_cols, sequence_length):
    """
    Creates time series sequences from the dataframe for DL models.
    """
    df_sequences = []
    df_targets = []
    info_list = []

    industries = df['industry'].unique()
    
    for industry in industries:
        industry_df = df[df['industry'] == industry].sort_values('date').copy()
        features = industry_df[feature_cols].values
        targets = industry_df['target'].values
        
        if len(industry_df) < sequence_length:
            continue
            
        for i in range(len(features) - sequence_length + 1):
            seq_end_idx = i + sequence_length
            seq = features[i:seq_end_idx]
            
            target = targets[seq_end_idx - 1]
            info = industry_df.iloc[seq_end_idx - 1][['date', 'industry']]
            
            df_sequences.append(seq)
            df_targets.append(target)
            info_list.append(info)

    return np.array(df_sequences), np.array(df_targets), pd.DataFrame(info_list) 