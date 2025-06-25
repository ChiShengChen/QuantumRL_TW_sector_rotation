# models/traditional_qnn.py
"""
TraditionalQNN: A non-recurrent Quantum Neural Network model
"""

import torch
import torch.nn as nn
from dataclasses import dataclass
import pennylane as qml
from pennylane.templates import AngleEmbedding, BasicEntanglerLayers
from typing import Optional

@dataclass
class QNNConfig:
    n_embd: int = 128
    input_dim: Optional[int] = None
    output_dim: Optional[int] = None
    n_qubits: int = 4
    q_depth: int = 2
    block_size: int = 10 # Just for compatibility, not used in the same way

class TraditionalQNN(nn.Module):
    def __init__(self, config: QNNConfig):
        super().__init__()
        self.config = config
        self.input_proj = nn.Linear(config.input_dim, config.n_embd)
        
        self.q_device = qml.device("default.qubit", wires=config.n_qubits)
        self.classical_input_projection = nn.Linear(config.n_embd, config.n_qubits, bias=False)

        def quantum_circuit(inputs, weights):
            AngleEmbedding(inputs, wires=range(config.n_qubits), rotation='X')
            BasicEntanglerLayers(weights, wires=range(config.n_qubits))
            return [qml.expval(qml.PauliZ(i)) for i in range(config.n_qubits)]

        weight_shapes = {"weights": (config.q_depth, config.n_qubits)}
        self.vqc_layer = qml.qnn.TorchLayer(
            qml.QNode(quantum_circuit, self.q_device, interface="torch", diff_method="backprop"),
            weight_shapes
        )
        
        self.classical_output_expansion = nn.Linear(config.n_qubits, config.n_embd, bias=False)
        self.lm_head = nn.Linear(config.n_embd, config.output_dim, bias=False)

    def forward(self, inputs, states=None):
        # states is ignored, present for API compatibility
        B, T, _ = inputs.shape
        
        # Project input features to embedding dimension
        x = self.input_proj(inputs.float()) # (B, T, n_embd)

        # Reshape for processing each time step
        x_reshaped = x.view(B * T, self.config.n_embd)

        # Quantum part
        q_input = self.classical_input_projection(x_reshaped) # (B*T, n_qubits)
        q_output = self.vqc_layer(q_input) # (B*T, n_qubits)
        
        # Expand quantum output and reshape back
        expanded_output = self.classical_output_expansion(q_output) # (B*T, n_embd)
        x = expanded_output.view(B, T, self.config.n_embd)

        # Final prediction head
        predictions = self.lm_head(x) # (B, T, output_dim)
        
        # Return None for state to match API
        return predictions, None 