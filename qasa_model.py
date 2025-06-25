import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import pennylane as qml
from dataclasses import dataclass
from typing import Optional

@dataclass
class QASAConfig:
    input_dim: int
    output_dim: int
    n_embd: int = 128
    n_qubits: int = 4
    n_layers: int = 2
    n_head: int = 4
    seq_len: int = 10 # This is block_size in other configs

# Global quantum device
dev = qml.device("default.qubit", wires=QASAConfig.n_qubits)

@qml.qnode(dev, interface="torch", diff_method="backprop")
def quantum_circuit(inputs, weights):
    n_qubits = QASAConfig.n_qubits
    n_layers = QASAConfig.n_layers
    
    qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation='X')
    
    for l in range(n_layers):
        for i in range(n_qubits):
            qml.CRZ(weights[l, i], wires=[i, (i + 1) % n_qubits])
        for i in range(n_qubits):
            qml.RY(weights[l, i + n_qubits], wires=i)

    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

class QuantumLayer(nn.Module):
    def __init__(self, config: QASAConfig):
        super().__init__()
        self.config = config
        self.weight_shapes = {"weights": (config.n_layers, config.n_qubits * 2)}
        self.qlayer = qml.qnn.TorchLayer(quantum_circuit, self.weight_shapes)
        self.q_proj = nn.Linear(config.n_embd, config.n_qubits)
        self.c_proj = nn.Linear(config.n_qubits, config.n_embd)

    def forward(self, x):
        # x: [B, n_embd]
        q_in = self.q_proj(x)
        q_out = self.qlayer(q_in) # [B, n_qubits]
        c_out = self.c_proj(q_out)
        return c_out

class QASABaseModel(nn.Module):
    def __init__(self, config: QASAConfig):
        super().__init__()
        self.config = config
        
        # This will project the flattened features (num_industries * num_features) to n_embd
        self.input_proj = nn.Linear(config.input_dim, config.n_embd)
        
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=config.n_embd, 
                nhead=config.n_head, 
                batch_first=True,
                dim_feedforward=config.n_embd * 4
            ),
            num_layers=1
        )
        self.q_layer = QuantumLayer(config)
        
        # This projects the final representation to the output_dim required by the Actor/Critic heads
        self.output_proj = nn.Linear(config.n_embd, config.output_dim)

    def forward(self, inputs, states=None):
        # inputs: [B, T, C], where C = num_industries * num_features
        # states is ignored for non-recurrent model
        
        B, T, C = inputs.shape
        
        # Project input features to embedding dimension
        x = self.input_proj(inputs.float()) # [B, T, n_embd]
        
        # The Transformer encoder processes the whole sequence
        x = self.encoder(x) # [B, T, n_embd]
        
        # Reshape for quantum layer: process each time step's representation
        x_reshaped = x.view(B * T, self.config.n_embd)
        q_out_reshaped = self.q_layer(x_reshaped) # [B * T, n_embd]
        
        # Add residual connection
        x = x + q_out_reshaped.view(B, T, self.config.n_embd)
        
        # Final projection to output dim (which is n_embd for base models)
        predictions = self.output_proj(x) # [B, T, output_dim]
        
        # Return None for the state, as this is not a recurrent model
        return predictions, None 