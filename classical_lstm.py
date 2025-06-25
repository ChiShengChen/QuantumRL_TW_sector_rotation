import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Optional

@dataclass
class LSTMConfig:
    input_dim: int
    output_dim: int # This will be n_embd for the base model
    n_embd: int = 128
    n_layers: int = 2
    dropout: float = 0.1

class ClassicalLSTM(nn.Module):
    def __init__(self, config: LSTMConfig):
        super().__init__()
        self.config = config

        # Project input from flattened features to the LSTM's input size (n_embd)
        self.input_proj = nn.Linear(config.input_dim, config.n_embd)

        self.lstm = nn.LSTM(
            input_size=config.n_embd,
            hidden_size=config.n_embd, # Use n_embd as hidden_size for simplicity
            num_layers=config.n_layers,
            batch_first=True,
            dropout=config.dropout if config.n_layers > 1 else 0
        )
        
        # Project from LSTM's hidden state size to the base model's output dimension
        self.output_proj = nn.Linear(config.n_embd, config.output_dim)

    def forward(self, inputs, hidden_state=None):
        # inputs: [B, T, C], where C = num_industries * num_features
        
        # Project input to the embedding dimension expected by the LSTM
        x = self.input_proj(inputs.float()) # [B, T, n_embd]
        
        # Pass through LSTM
        # lstm_out: [B, T, n_embd]
        # new_hidden_state: tuple of (h_n, c_n), which is the format for the next call
        lstm_out, new_hidden_state = self.lstm(x, hidden_state)
        
        # Final projection to the required output dimension for the base model
        predictions = self.output_proj(lstm_out) # [B, T, output_dim]

        return predictions, new_hidden_state 