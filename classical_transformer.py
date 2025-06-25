import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Optional

@dataclass
class TransformerConfig:
    input_dim: int
    output_dim: int # This will be n_embd for the base model
    n_embd: int = 128
    n_head: int = 4
    n_layers: int = 2
    dropout: float = 0.1

class ClassicalTransformer(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config

        # Project input from flattened features to the Transformer's input size (n_embd)
        self.input_proj = nn.Linear(config.input_dim, config.n_embd)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.n_embd,
            nhead=config.n_head,
            dim_feedforward=config.n_embd * 4,
            dropout=config.dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.n_layers
        )
        
        # Project from Transformer's output to the base model's output dimension
        self.output_proj = nn.Linear(config.n_embd, config.output_dim)
        
        # Positional encoding
        # This can be a simple learned embedding or a fixed sinusoidal one.
        # For simplicity, we'll use a learned positional embedding.
        # Let's assume a max sequence length, e.g., 512
        self.pos_encoder = nn.Parameter(torch.zeros(1, 512, config.n_embd))

    def forward(self, inputs, hidden_state=None):
        # inputs: [B, T, C], where C = num_industries * num_features
        # hidden_state is ignored as Transformer is non-recurrent in this setup
        B, T, C = inputs.shape

        # Project input to the embedding dimension
        x = self.input_proj(inputs.float()) # [B, T, n_embd]
        
        # Add positional encoding
        x = x + self.pos_encoder[:, :T, :] # [B, T, n_embd]

        # Pass through Transformer Encoder
        transformer_out = self.transformer_encoder(x) # [B, T, n_embd]
        
        # Final projection
        predictions = self.output_proj(transformer_out) # [B, T, output_dim]

        # Return None for hidden_state to match the API
        return predictions, None 