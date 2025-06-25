# models/qrwkv_model.py
"""
QuantumRWKVModel: RWKV structure with quantum-enhanced channel mixing
"""

import torch
import torch.nn as nn
from dataclasses import dataclass
import pennylane as qml
from pennylane.templates import AngleEmbedding, BasicEntanglerLayers
from typing import Optional

@dataclass
class ModelConfig:
    n_embd: int = 768
    n_head: int = 12
    n_layer: int = 12
    block_size: int = 1024
    n_intermediate: int = 768 * 4
    layer_norm_epsilon: float = 1e-5
    vocab_size: Optional[int] = None
    input_dim: Optional[int] = None
    output_dim: Optional[int] = None
    n_qubits: int = 4
    q_depth: int = 2

class RWKVTimeMixing(nn.Module):
    def __init__(self, config: ModelConfig, layer_id: int):
        super().__init__()
        self.config = config
        self.layer_id = layer_id
        self.n_embd = config.n_embd
        self.n_head = config.n_head
        self.head_size = self.n_embd // self.n_head

        assert self.n_embd % self.n_head == 0

        with torch.no_grad():
            ratio_0_to_1 = torch.arange(self.n_head).float() / (self.n_head - 1 if self.n_head > 1 else 1)
            tmp = torch.zeros(self.n_embd)
            for i in range(self.n_head):
                tmp[i * self.head_size:(i + 1) * self.head_size] = ratio_0_to_1[i]

            self.time_mix_k = nn.Parameter(tmp.clone())
            self.time_mix_v = nn.Parameter(tmp.clone())
            self.time_mix_r = nn.Parameter(tmp.clone())
            self.time_decay = nn.Parameter(-5 + 8 * (torch.arange(self.n_embd).float() / (self.n_embd - 1)) ** 0.7)
            self.time_first = nn.Parameter(torch.ones(self.n_embd) * -3.0)

        self.key = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.value = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.receptance = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.output = nn.Linear(self.n_embd, self.n_embd, bias=False)

    def forward(self, x, state):
        B, T, C = x.size()
        k = self.key(x)
        v = self.value(x)
        r_in = self.receptance(x)
        r = torch.sigmoid(r_in)

        w = torch.exp(self.time_decay)
        u = self.time_first
        output_wkv = torch.zeros_like(k)

        current_aa, current_bb, current_pp = state if state is not None else (
            torch.zeros(B, C, device=x.device, dtype=x.dtype),
            torch.zeros(B, C, device=x.device, dtype=x.dtype),
            torch.full((B, C), -1e38, device=x.device, dtype=x.dtype)
        )

        for t_step in range(T):
            kt = k[:, t_step, :]
            vt = v[:, t_step, :]
            ww = u + kt
            p = torch.maximum(current_pp, ww)
            e1 = torch.exp(current_pp - p)
            e2 = torch.exp(ww - p)
            wkv_t_step = (e1 * current_aa + e2 * vt) / (e1 * current_bb + e2)
            output_wkv[:, t_step, :] = wkv_t_step

            ww = current_pp - w
            p = torch.maximum(ww, kt)
            e1 = torch.exp(ww - p)
            e2 = torch.exp(kt - p)
            current_aa = e1 * current_aa + e2 * vt
            current_bb = e1 * current_bb + e2
            current_pp = p

        rwkv_out = r * output_wkv
        new_wkv_state = (current_aa, current_bb, current_pp)
        return self.output(rwkv_out), new_wkv_state

class QuantumChannelMixing(nn.Module):
    def __init__(self, config: ModelConfig, layer_id: int):
        super().__init__()
        self.config = config
        self.layer_id = layer_id
        self.n_embd = config.n_embd
        self.n_intermediate = config.n_intermediate
        self.n_qubits = config.n_qubits
        self.q_depth = config.q_depth

        with torch.no_grad():
            ratio_0_to_1 = torch.arange(self.n_embd).float() / (self.n_embd - 1)
            self.time_mix_k = nn.Parameter(ratio_0_to_1.clone())
            self.time_mix_r = nn.Parameter(ratio_0_to_1.clone())

        self.receptance_linear = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.classical_fc1 = nn.Linear(self.n_embd, self.n_intermediate, bias=False)
        self.classical_act = nn.GELU()
        self.classical_fc2 = nn.Linear(self.n_intermediate, self.n_embd, bias=False)

        self.q_device = qml.device("default.qubit", wires=self.n_qubits)
        self.classical_input_projection = nn.Linear(self.n_embd, self.n_qubits, bias=False)

        def quantum_ffn_circuit(inputs, weights):
            AngleEmbedding(inputs, wires=range(self.n_qubits), rotation='X')
            BasicEntanglerLayers(weights, wires=range(self.n_qubits))
            return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]

        weight_shapes = {"weights": (self.q_depth, self.n_qubits)}
        self.vqc_layer = qml.qnn.TorchLayer(qml.QNode(quantum_ffn_circuit, self.q_device, interface="torch", diff_method="backprop"), weight_shapes)
        self.classical_output_expansion = nn.Linear(self.n_qubits, self.n_embd, bias=False)
        self.fusion_linear = nn.Linear(self.n_embd, self.n_embd, bias=False)

    def forward(self, x, prev_x_cm_state):
        B, T, C = x.size()
        if prev_x_cm_state is None:
            prev_x_cm_state = torch.zeros(B, C, device=x.device, dtype=x.dtype)

        prev_x_cm_state_expanded = prev_x_cm_state.unsqueeze(1).repeat(1, T, 1)
        xk_mixed = x * self.time_mix_k + prev_x_cm_state_expanded * (1 - self.time_mix_k)
        xr_mixed = x * self.time_mix_r + prev_x_cm_state_expanded * (1 - self.time_mix_r)
        r = torch.sigmoid(self.receptance_linear(xr_mixed))

        ffn_classical_output = self.classical_fc2(self.classical_act(self.classical_fc1(xk_mixed)))
        q_input_projected = self.classical_input_projection(xk_mixed)
        q_input_reshaped = q_input_projected.reshape(-1, self.n_qubits)
        vqc_output_reshaped = self.vqc_layer(q_input_reshaped)
        vqc_output = vqc_output_reshaped.reshape(B, T, self.n_qubits)
        ffn_quantum_output = self.classical_output_expansion(vqc_output)

        combined_signal = ffn_classical_output + ffn_quantum_output
        fused_signal = self.fusion_linear(combined_signal)
        output_val = r * fused_signal

        if T > 0:
            new_prev_x_cm_state = x[:, -1, :].clone()
        else:
            new_prev_x_cm_state = prev_x_cm_state[:, 0, :].clone()

        return output_val, new_prev_x_cm_state

class QuantumRWKVBlock(nn.Module):
    def __init__(self, config: ModelConfig, layer_id: int):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.ln2 = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.att = RWKVTimeMixing(config, layer_id)
        self.ffn = QuantumChannelMixing(config, layer_id)

    def forward(self, x, time_mix_state, channel_mix_state):
        x_norm_att = self.ln1(x)
        att_output, new_time_mix_state = self.att(x_norm_att, time_mix_state)
        x = x + att_output
        x_norm_ffn = self.ln2(x)
        ffn_output, new_channel_mix_state = self.ffn(x_norm_ffn, channel_mix_state)
        x = x + ffn_output
        return x, new_time_mix_state, new_channel_mix_state

class QuantumRWKVModel(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        if config.vocab_size is not None:
            self.wte = nn.Embedding(config.vocab_size, config.n_embd)
            self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
            self.model_mode = "token"
        elif config.input_dim is not None and config.output_dim is not None:
            self.input_proj = nn.Linear(config.input_dim, config.n_embd)
            self.lm_head = nn.Linear(config.n_embd, config.output_dim, bias=False)
            self.model_mode = "waveform"
        else:
            raise ValueError("Invalid ModelConfig")

        self.blocks = nn.ModuleList([QuantumRWKVBlock(config, i) for i in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)

    def forward(self, inputs, states=None):
        B, T = inputs.shape[0], inputs.shape[1]
        if self.model_mode == "token":
            x = self.wte(inputs)
        elif self.model_mode == "waveform":
            if inputs.ndim == 2:
                inputs = inputs.unsqueeze(-1)
            x = self.input_proj(inputs.float())
        else:
            raise RuntimeError("Invalid model mode.")

        new_states_out = []
        if states is None:
            states = []
            param_dtype = next(self.parameters()).dtype
            for _ in range(self.config.n_layer):
                initial_wkv_aa = torch.zeros(B, self.config.n_embd, device=inputs.device, dtype=param_dtype)
                initial_wkv_bb = torch.zeros(B, self.config.n_embd, device=inputs.device, dtype=param_dtype)
                initial_wkv_pp = torch.full((B, self.config.n_embd), -1e38, device=inputs.device, dtype=param_dtype)
                wkv_state = (initial_wkv_aa, initial_wkv_bb, initial_wkv_pp)
                cm_state = torch.zeros(B, self.config.n_embd, device=inputs.device, dtype=param_dtype)
                states.append((wkv_state, cm_state))

        for i, block in enumerate(self.blocks):
            layer_wkv_state, layer_cm_state = states[i]
            x, next_layer_wkv_state, next_layer_cm_state = block(x, layer_wkv_state, layer_cm_state)
            new_states_out.append((next_layer_wkv_state, next_layer_cm_state))

        x = self.ln_f(x)
        predictions = self.lm_head(x)
        return predictions, new_states_out
