import torch
import torch.nn as nn
from torch.distributions import Categorical
from qrwkv_model import QuantumRWKVModel, ModelConfig as QRWKVConfig
from traditional_qnn import TraditionalQNN, QNNConfig
from qasa_model import QASABaseModel, QASAConfig
from classical_lstm import ClassicalLSTM, LSTMConfig
from classical_transformer import ClassicalTransformer, TransformerConfig

class Actor(nn.Module):
    """
    The Actor network for PPO.
    It takes the state (sequence of features) and outputs a probability distribution
    over the actions (industries).
    """
    def __init__(self, model_type: str, model_config: dict, output_dim: int):
        super().__init__()
        self.model_type = model_type
        
        model_config_copy = model_config.copy()
        
        if self.model_type == 'qrwkv':
            base_output_dim = model_config_copy['n_embd']
            model_config_copy['output_dim'] = base_output_dim
            self.config = QRWKVConfig(**model_config_copy)
            self.base = QuantumRWKVModel(self.config)
        elif self.model_type == 'qnn':
            base_output_dim = model_config_copy['n_embd']
            model_config_copy['output_dim'] = base_output_dim
            self.config = QNNConfig(**model_config_copy)
            self.base = TraditionalQNN(self.config)
        elif self.model_type == 'qasa':
            base_output_dim = model_config_copy['n_embd']
            model_config_copy['output_dim'] = base_output_dim
            self.config = QASAConfig(**model_config_copy)
            self.base = QASABaseModel(self.config)
        elif self.model_type == 'lstm':
            base_output_dim = model_config_copy['n_embd']
            model_config_copy['output_dim'] = base_output_dim
            self.config = LSTMConfig(**model_config_copy)
            self.base = ClassicalLSTM(self.config)
        elif self.model_type == 'transformer':
            base_output_dim = model_config_copy['n_embd']
            model_config_copy['output_dim'] = base_output_dim
            self.config = TransformerConfig(**model_config_copy)
            self.base = ClassicalTransformer(self.config)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

        self.actor_head = nn.Linear(base_output_dim, output_dim)

    def forward(self, state, hidden_state=None):
        # The base model returns predictions (B, T, C) and new states
        base_out, new_hidden_state = self.base(state, hidden_state)
        
        # We only need the output from the last time step for decision making
        last_time_step_out = base_out[:, -1, :]
        
        logits = self.actor_head(last_time_step_out)
        dist = Categorical(logits=logits)
        
        return dist, new_hidden_state

class Critic(nn.Module):
    """
    The Critic network for PPO.
    It takes the state (sequence of features) and outputs a single value
    representing the value of that state.
    """
    def __init__(self, model_type: str, model_config: dict):
        super().__init__()
        self.model_type = model_type
        model_config_copy = model_config.copy()
        
        if self.model_type == 'qrwkv':
            base_output_dim = model_config_copy['n_embd']
            model_config_copy['output_dim'] = base_output_dim
            self.config = QRWKVConfig(**model_config_copy)
            self.base = QuantumRWKVModel(self.config)
        elif self.model_type == 'qnn':
            base_output_dim = model_config_copy['n_embd']
            model_config_copy['output_dim'] = base_output_dim
            self.config = QNNConfig(**model_config_copy)
            self.base = TraditionalQNN(self.config)
        elif self.model_type == 'qasa':
            base_output_dim = model_config_copy['n_embd']
            model_config_copy['output_dim'] = base_output_dim
            self.config = QASAConfig(**model_config_copy)
            self.base = QASABaseModel(self.config)
        elif self.model_type == 'lstm':
            base_output_dim = model_config_copy['n_embd']
            model_config_copy['output_dim'] = base_output_dim
            self.config = LSTMConfig(**model_config_copy)
            self.base = ClassicalLSTM(self.config)
        elif self.model_type == 'transformer':
            base_output_dim = model_config_copy['n_embd']
            model_config_copy['output_dim'] = base_output_dim
            self.config = TransformerConfig(**model_config_copy)
            self.base = ClassicalTransformer(self.config)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

        self.critic_head = nn.Linear(base_output_dim, 1)

    def forward(self, state, hidden_state=None):
        base_out, new_hidden_state = self.base(state, hidden_state)
        
        # We only need the output from the last time step
        last_time_step_out = base_out[:, -1, :]
        
        value = self.critic_head(last_time_step_out)
        
        return value, new_hidden_state 