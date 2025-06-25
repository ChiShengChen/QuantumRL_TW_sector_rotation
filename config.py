import os

# --- File Paths ---
BENCHMARK_DIR = os.path.dirname(os.path.abspath(__file__))
SOURCE_DIR = os.path.dirname(BENCHMARK_DIR)
# The following files are assumed to be present in the benchmark directory
INDUSTRY_FILE = os.path.join(SOURCE_DIR, 'industry_classification.csv') # This might not be used anymore
TRANSLATION_FILE = os.path.join(BENCHMARK_DIR, 'industry_translation_map.csv')
CAPITAL_SHARE_FILE = os.path.join(BENCHMARK_DIR, 'industry_daily_amount_en.csv')
FEATURES_FILE = os.path.join(BENCHMARK_DIR, 'features_data.csv')
PRICE_FILE = os.path.join(SOURCE_DIR, 'all_stock_price_volume.csv')

# --- Output Paths ---
OUTPUT_DIR = BENCHMARK_DIR
MODEL_DIR = os.path.join(OUTPUT_DIR, 'checkpoints')
os.makedirs(MODEL_DIR, exist_ok=True)
ACTOR_MODEL_PATH = os.path.join(MODEL_DIR, 'ppo_actor.pth')
CRITIC_MODEL_PATH = os.path.join(MODEL_DIR, 'ppo_critic.pth')
# Output files for the QRL agent
# You might want to add paths for saving model checkpoints, logs, etc.
# For example:
# MODEL_CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, 'checkpoints')

# --- Modeling Parameters ---
MODEL_TYPE = 'transformer' # 'qrwkv', 'qnn', 'qasa', 'lstm', or 'transformer'
TOP_N = 10
TEST_START_DATE = '2020-01-01'
SEQUENCE_LENGTH = 10 # Sequence length for the QRWKV model

# --- QRL PPO Agent Configuration ---
QRL_PPO_CONFIG = {
    'input_dim': 8,      # Number of features from feature_engineering.py
    'output_dim': 48,    # Number of industries to choose from
    'gamma': 0.99,       # Discount factor for rewards
    'lr_actor': 0.0003,  # Learning rate for the actor
    'lr_critic': 0.001,  # Learning rate for the critic
    'ppo_epochs': 10,    # Number of epochs for PPO update
    'ppo_clip_eps': 0.2, # PPO clip parameter
    'entropy_beta': 0.01,# Coefficient for entropy bonus
    'batch_size': 64,    # Batch size for training
    'num_epochs': 100,   # Total training epochs
}

# --- QuantumRWKV Model Configuration ---
# This config is passed to the QuantumRWKVModel constructor
QRWKV_CONFIG = {
    'n_embd': 128,
    'n_head': 4,
    'n_layer': 4,
    'block_size': SEQUENCE_LENGTH,
    'n_intermediate': 128 * 4,
    'layer_norm_epsilon': 1e-5,
    'vocab_size': None, # Not used in waveform mode
    'input_dim': QRL_PPO_CONFIG['input_dim'],
    'output_dim': QRL_PPO_CONFIG['output_dim'], # For a base model, might be different for actor/critic
    'n_qubits': 4,
    'q_depth': 2,
}

# --- Traditional QNN Model Configuration ---
QNN_CONFIG = {
    'n_embd': 128,
    'block_size': SEQUENCE_LENGTH,
    'input_dim': QRL_PPO_CONFIG['input_dim'],
    'output_dim': QRL_PPO_CONFIG['output_dim'],
    'n_qubits': 4,
    'q_depth': 2,
}

# --- QASA Model Configuration ---
QASA_CONFIG = {
    'n_embd': 128,
    'seq_len': SEQUENCE_LENGTH,
    'input_dim': QRL_PPO_CONFIG['input_dim'],
    'output_dim': QRL_PPO_CONFIG['output_dim'],
    'n_qubits': 4,
    'n_layers': 2,
    'n_head': 4,
}

# --- Classical LSTM Model Configuration ---
LSTM_CONFIG = {
    'n_embd': 128,
    'input_dim': QRL_PPO_CONFIG['input_dim'],
    'output_dim': QRL_PPO_CONFIG['output_dim'],
    'n_layers': 2,
    'dropout': 0.1,
}

# --- Classical Transformer Model Configuration ---
TRANSFORMER_CONFIG = {
    'n_embd': 128,
    'input_dim': QRL_PPO_CONFIG['input_dim'],
    'output_dim': QRL_PPO_CONFIG['output_dim'],
    'n_head': 4,
    'n_layers': 2,
    'dropout': 0.1,
} 