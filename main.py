import torch
import pandas as pd
import numpy as np
import os
import sys
import logging
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from datetime import datetime

# Add the current directory to the python path
sys.path.insert(0, os.path.dirname(__file__))

import config
import data_loader
import feature_engineering
from env import SectorRotationEnv
from trainer import PPOAgent

def calculate_performance_metrics(daily_returns):
    """
    Calculates key performance metrics from a series of daily returns.
    """
    metrics = {}
    
    trading_days = 252
    
    # Cumulative and Annualized Return
    cumulative_return = (1 + daily_returns).prod() - 1
    metrics['Cumulative Return'] = f"{cumulative_return:.2%}"
    
    num_days = len(daily_returns)
    annualized_return = (1 + cumulative_return) ** (trading_days / num_days) - 1 if num_days > 0 else 0
    metrics['Annualized Return'] = f"{annualized_return:.2%}"
    
    # Volatility and Sharpe Ratio
    annualized_volatility = daily_returns.std() * np.sqrt(trading_days)
    metrics['Annualized Volatility'] = f"{annualized_volatility:.2%}"
    
    sharpe_ratio = annualized_return / (annualized_volatility + 1e-9)
    metrics['Sharpe Ratio'] = f"{sharpe_ratio:.2f}"
    
    # Max Drawdown and Calmar Ratio
    cumulative = (1 + daily_returns.fillna(0)).cumprod()
    peak = cumulative.expanding(min_periods=1).max()
    drawdown = (cumulative - peak) / peak
    max_drawdown = drawdown.min()
    metrics['Max Drawdown'] = f"{max_drawdown:.2%}"
    
    calmar_ratio = annualized_return / (abs(max_drawdown) + 1e-9)
    metrics['Calmar Ratio'] = f"{calmar_ratio:.2f}"
            
    return metrics

def main():
    """
    Main function to run the QRL PPO agent for sector rotation.
    """
    # --- Setup ---
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Create a unique directory for this run
    run_dir = os.path.join(config.MODEL_DIR, f"{config.MODEL_TYPE}_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    
    # Configure logging
    log_file_path = os.path.join(run_dir, 'training_log.log')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file_path),
            logging.StreamHandler(sys.stdout)
        ]
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info("--- Starting QRL PPO Training ---")
    logging.info(f"Using device: {device}")
    logging.info(f"Using model type: {config.MODEL_TYPE}")
    logging.info(f"Run directory: {run_dir}")


    # --- 1. Load or Create Features ---
    if os.path.exists(config.FEATURES_FILE):
        logging.info(f"Loading features from {config.FEATURES_FILE}")
        features_data = pd.read_csv(config.FEATURES_FILE, parse_dates=['date'])
    else:
        logging.info("Feature file not found. Running data preparation pipeline...")
        capital_share_data = data_loader.prepare_capital_share_data()
        if capital_share_data is not None:
            features_data = feature_engineering.create_capital_share_features(capital_share_data)
            features_data.to_csv(config.FEATURES_FILE, index=False)
            logging.info(f"Feature data saved to {config.FEATURES_FILE}")

    if 'features_data' not in locals():
        logging.error("Could not load or create feature data. Exiting.")
        return
        
    # --- 2. Prepare Data and Environment ---
    logging.info("Preparing data and environment...")
    
    # The 'target' column is needed by the environment to calculate rewards
    features_data['future_share'] = features_data.groupby('industry')['capital_share'].shift(-1)
    features_data['future_rank'] = features_data.groupby('date')['future_share'].rank(ascending=False)
    features_data['target'] = (features_data['future_rank'] <= config.TOP_N).astype(int)

    feature_cols = [col for col in features_data.columns if col not in ['date', 'industry', 'target', 'future_rank', 'future_share', 'capital_share']]
    
    # Drop rows with NaNs in feature columns or target from rolling windows
    features_data.dropna(subset=feature_cols + ['target'], inplace=True)

    train_df = features_data[features_data['date'] < config.TEST_START_DATE].copy()
    test_df = features_data[features_data['date'] >= config.TEST_START_DATE].copy()

    # Normalize features for better model stability
    scaler = StandardScaler()
    train_df.loc[:, feature_cols] = scaler.fit_transform(train_df[feature_cols])
    # Use the same scaler to transform the test set
    if not test_df.empty:
        test_df.loc[:, feature_cols] = scaler.transform(test_df[feature_cols])
    
    train_env = SectorRotationEnv(train_df, feature_cols, config.SEQUENCE_LENGTH)
    
    # --- 3. Configure and Instantiate Agent ---
    # Dynamically set input and output dimensions
    num_industries = train_df['industry'].nunique()
    
    # The QRWKV model's input_dim is the last dimension of the state tensor (T, C)
    # where C is num_industries * num_features
    input_dim = train_env.flat_features_dim
    
    if config.MODEL_TYPE == 'qrwkv':
        model_config = config.QRWKV_CONFIG
    elif config.MODEL_TYPE == 'qnn':
        model_config = config.QNN_CONFIG
    elif config.MODEL_TYPE == 'qasa':
        model_config = config.QASA_CONFIG
    elif config.MODEL_TYPE == 'lstm':
        model_config = config.LSTM_CONFIG
    elif config.MODEL_TYPE == 'transformer':
        model_config = config.TRANSFORMER_CONFIG
    else:
        raise ValueError(f"Unsupported model type: {config.MODEL_TYPE}")

    model_config['input_dim'] = input_dim
    # The output dimension for the base model is n_embd, not the final action space
    # The actor head will project to the action space
    config.QRL_PPO_CONFIG['output_dim'] = num_industries
    
    logging.info(f"Number of industries (action space): {num_industries}")
    logging.info(f"Number of features per industry: {len(feature_cols)}")
    logging.info(f"Model input dimension (T, C): C = {input_dim}")
    
    agent = PPOAgent(
        model_type=config.MODEL_TYPE,
        model_config=model_config,
        ppo_config=config.QRL_PPO_CONFIG, 
        device=device
    )

    # --- 4. Training Loop ---
    logging.info("\n--- Starting Training ---")
    
    for epoch in range(config.QRL_PPO_CONFIG['num_epochs']):
        state = train_env.reset()
        done = False
        total_reward = 0
        
        pbar = tqdm(total=len(train_env.unique_dates) - train_env.sequence_length, desc=f"Epoch {epoch+1}/{config.QRL_PPO_CONFIG['num_epochs']}")
        
        actor_hidden_state, critic_hidden_state = None, None

        while not done:
            action, actor_hidden_state, critic_hidden_state = agent.select_action(state, actor_hidden_state, critic_hidden_state)
            
            next_state, reward, done, _ = train_env.step(action)
            
            agent.memory.rewards.append(reward)
            agent.memory.is_terminals.append(done)
            
            state = next_state
            total_reward += reward
            pbar.update(1)

        pbar.close()
        agent.update()
        logging.info(f"Epoch {epoch+1}, Total Reward: {total_reward:.2f}")

    logging.info("\n--- Training Finished ---")

    # --- Save the trained models ---
    actor_model_path = os.path.join(run_dir, 'ppo_actor.pth')
    critic_model_path = os.path.join(run_dir, 'ppo_critic.pth')
    logging.info(f"Saving models to {run_dir}")
    torch.save(agent.actor.state_dict(), actor_model_path)
    torch.save(agent.critic.state_dict(), critic_model_path)
    logging.info("Models saved successfully.")
    
    # --- 5. Evaluation ---
    logging.info("\n--- Starting Evaluation ---")
    
    # Prepare data for backtest
    industry_returns_df = data_loader.calculate_industry_returns()
    eval_df = test_df.merge(industry_returns_df, on=['date', 'industry'], how='left')
    eval_df.sort_values(['industry', 'date'], inplace=True)
    eval_df['forward_return'] = eval_df.groupby('industry')['industry_return'].shift(-1).fillna(0)
    
    eval_env = SectorRotationEnv(eval_df, feature_cols, config.SEQUENCE_LENGTH)
    
    state = eval_env.reset()
    done = False
    
    agent.actor.eval() # Set actor to evaluation mode
    
    daily_returns = []
    actor_hidden_state = None
    
    with torch.no_grad():
        while not done:
            state_tensor = torch.from_numpy(state).float().to(device).unsqueeze(0)
            dist, actor_hidden_state = agent.actor(state_tensor, actor_hidden_state)
            
            probabilities = dist.probs
            top_k_actions = torch.topk(probabilities, config.TOP_N, dim=1).indices.squeeze(0)
            
            current_date = eval_env.unique_dates[eval_env.current_step]
            
            selected_industries = [eval_env.features_df['industry'].unique()[i] for i in top_k_actions.cpu().numpy()]
            
            daily_data = eval_df[eval_df['date'] == current_date]
            strategy_returns = daily_data[daily_data['industry'].isin(selected_industries)]['forward_return'].mean()
            daily_returns.append(strategy_returns)

            state, _, done, _ = eval_env.step(0) # Action doesn't matter here

    daily_returns_series = pd.Series(daily_returns, index=eval_env.unique_dates[eval_env.sequence_length:len(eval_env.unique_dates)])

    logging.info("\n--- QRL Strategy Performance Metrics ---")
    performance_metrics = calculate_performance_metrics(daily_returns_series.fillna(0))
    for name, value in performance_metrics.items():
        logging.info(f"{name:<25}: {value}")

if __name__ == '__main__':
    main()