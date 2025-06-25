# QRL for Industry Sector Rotation

This project implements a Quantum Reinforcement Learning (QRL) agent for the task of industry sector rotation in the stock market. The agent, based on Proximal Policy Optimization (PPO), learns a policy to select a portfolio of top-performing industry sectors to maximize investment returns.

## Features

- **Reinforcement Learning Agent**: Utilizes a PPO agent to learn an optimal investment strategy.
- **Multiple Model Architectures**: Supports various underlying models for the PPO agent's policy and value networks:
    - `qasa`: A Transformer-based model with a Quantum-enhanced Attention layer.
    - `qrwkv`: A Quantum-enhanced RWKV model.
    - `qnn`: A traditional Quantum Neural Network.
    - `lstm`: A classical Long Short-Term Memory (LSTM) network.
    - `transformer`: A classical Transformer encoder model.
- **Automated Feature Engineering**: Generates features from industry capital share data, including moving averages, momentum, and volatility.
- **Performance Evaluation**: Automatically backtests the trained agent and provides standard performance metrics like Cumulative Return, Annualized Return, Sharpe Ratio, and Max Drawdown.
- **Configurability**: Easily configure models, hyperparameters, and paths through the `config.py` file.

## Directory Structure

```
qrl_industry_sector_rotation/
│
├── checkpoints/              # Stores trained model checkpoints and logs
├── __pycache__/
├── main.py                   # Main script to run training and evaluation
├── config.py                 # All project configurations
├── models.py                 # PPO Actor/Critic network definitions
├── classical_transformer.py  # Classical Transformer base model
├── classical_lstm.py         # Classical LSTM base model
├── qasa_model.py             # QASA (Quantum Attention) base model
├── trainer.py                # PPO agent and training logic
├── traditional_qnn.py        # QNN base model
├── env.py                    # RL environment for sector rotation
├── qrwkv_model.py            # QRWKV base model
├── data_loader.py            # Data loading and preparation utilities
├── feature_engineering.py    # Feature creation logic
├── requirements.txt          # Python dependencies
└── ... (data files)
```

## Setup & Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository_url>
    cd qrl_industry_sector_rotation
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Prepare Data:**
    The model requires several data files, whose paths are configured in `config.py`. Make sure the following files are present in the correct locations:
    - `industry_daily_amount_en.csv`: Contains the daily capital share for each industry.
    - `all_stock_price_volume.csv`: Used during the evaluation phase to calculate portfolio returns.
    
    The script will automatically generate a `features_data.csv` file from your source data if it's not found.

## Usage

1.  **Configure the experiment:**
    Open `config.py` and set the `MODEL_TYPE` to the desired model (`qasa`, `transformer`, `lstm`, etc.). You can also adjust other parameters like PPO hyperparameters, sequence length, and test start date.

    ```python
    # Example config.py
    # --- Modeling Parameters ---
    MODEL_TYPE = 'qasa' # 'qrwkv', 'qnn', 'qasa', 'lstm', or 'transformer'
    TOP_N = 10
    TEST_START_DATE = '2020-01-01'
    SEQUENCE_LENGTH = 10
    
    # --- QRL PPO Agent Configuration ---
    QRL_PPO_CONFIG = {
        # ...
        'num_epochs': 100,
    }
    ```

2.  **Run the training and evaluation:**
    Execute the main script from the terminal:
    ```bash
    python main.py
    ```

3.  **View Results:**
    The script will stream logs to the console. A new directory will be created for each run inside `checkpoints/`, for example: `checkpoints/qasa_20250625_083943/`. This directory contains:
    - `training_log.log`: A detailed log of the training process and final evaluation metrics.
    - `ppo_actor.pth`: The saved weights for the trained actor model.
    - `ppo_critic.pth`: The saved weights for the trained critic model.

## Example Results

Here are some example performance metrics from training runs.

**QASA Model:**
```
--- QRL Strategy Performance Metrics ---
Cumulative Return        : 93.14%
Annualized Return        : 13.49%
Annualized Volatility    : 19.03%
Sharpe Ratio             : 0.71
Max Drawdown             : -31.73%
Calmar Ratio             : 0.43
```

**Transformer Model:**
```
--- QRL Strategy Performance Metrics ---
Cumulative Return        : 124.29%
Annualized Return        : 16.80%
Annualized Volatility    : 19.16%
Sharpe Ratio             : 0.88
Max Drawdown             : -35.31%
``` 