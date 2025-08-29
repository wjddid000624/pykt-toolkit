import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Module, Embedding, Linear, Dropout, LayerNorm
from typing import Tuple, Optional
import math

class sLSTMBlock(nn.Module):
    """sLSTM block with exponential activation and normalizer state"""
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        self.W_i = nn.Linear(input_size, hidden_size, bias=False)
        self.U_i = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W_f = nn.Linear(input_size, hidden_size, bias=False)
        self.U_f = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W_o = nn.Linear(input_size, hidden_size, bias=False)
        self.U_o = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W_z = nn.Linear(input_size, hidden_size, bias=False)
        self.U_z = nn.Linear(hidden_size, hidden_size, bias=False)
        
        self.b_i = nn.Parameter(torch.zeros(hidden_size))
        self.b_f = nn.Parameter(torch.ones(hidden_size))
        self.b_o = nn.Parameter(torch.zeros(hidden_size))
        self.b_z = nn.Parameter(torch.zeros(hidden_size))
        
    def forward(self, x: torch.Tensor, states: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        h_prev, c_prev, n_prev = states  # n_prev is normalizer state
        
        # Gates with exponential activation
        i_tilde = self.W_i(x) + self.U_i(h_prev) + self.b_i
        f_tilde = self.W_f(x) + self.U_f(h_prev) + self.b_f
        o_tilde = self.W_o(x) + self.U_o(h_prev) + self.b_o
        z_tilde = self.W_z(x) + self.U_z(h_prev) + self.b_z
        
        # Use exponential activation for input gate, sigmoid for forget gate (OR operation)
        # Stabilization using max trick for numerical stability
        m_t = torch.maximum(f_tilde, i_tilde)
        i_t = torch.exp(i_tilde - m_t)
        f_t = torch.sigmoid(f_tilde) + torch.exp(f_tilde - m_t)  # OR operation approximation
        
        o_t = torch.sigmoid(o_tilde)
        z_t = torch.tanh(z_tilde)
        
        # Update cell state and normalizer state
        c_t = f_t * c_prev + i_t * z_t
        n_t = f_t * n_prev + i_t
        
        # Normalized hidden state
        h_tilde = c_t / (n_t + 1e-8)
        h_t = o_t * h_tilde
        
        return h_t, (h_t, c_t, n_t)

class mLSTMBlock(nn.Module):
    """mLSTM block with matrix memory"""
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # Key, value, query projections
        self.W_q = nn.Linear(input_size, hidden_size, bias=False)
        self.W_k = nn.Linear(input_size, hidden_size, bias=False)
        self.W_v = nn.Linear(input_size, hidden_size, bias=False)
        
        # Gates
        self.W_i = nn.Linear(input_size, hidden_size, bias=False)
        self.W_f = nn.Linear(input_size, hidden_size, bias=False)
        self.W_o = nn.Linear(input_size, hidden_size, bias=False)
        
        self.b_i = nn.Parameter(torch.zeros(hidden_size))
        self.b_f = nn.Parameter(torch.ones(hidden_size))
        self.b_o = nn.Parameter(torch.zeros(hidden_size))
        
        # Scaling factor
        self.scale = 1.0 / math.sqrt(hidden_size)
        
    def forward(self, x: torch.Tensor, states: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        h_prev, C_prev, n_prev = states  # C_prev is matrix memory, n_prev is normalizer
        
        # Compute key, value, query
        q_t = self.W_q(x) * self.scale
        k_t = self.W_k(x) * self.scale
        v_t = self.W_v(x)
        
        # Gates with exponential activation for input gate
        i_tilde = self.W_i(x) + self.b_i
        f_tilde = self.W_f(x) + self.b_f
        o_tilde = self.W_o(x) + self.b_o
        
        # Stabilized exponential for input gate
        i_t = torch.exp(i_tilde)
        f_t = torch.sigmoid(f_tilde)
        o_t = torch.sigmoid(o_tilde)
        
        # Update matrix memory using covariance rule
        # C_t = f_t * C_prev + i_t * (v_t ⊗ k_t)
        vk_outer = torch.bmm(v_t.unsqueeze(-1), k_t.unsqueeze(1))  # (B, H, H)
        f_gate_expanded = f_t.unsqueeze(-1)  # (B, H, 1)
        i_gate_expanded = i_t.unsqueeze(-1)  # (B, H, 1)
        
        C_t = f_gate_expanded * C_prev + i_gate_expanded * vk_outer
        
        # Update normalizer for key
        n_t = f_t * n_prev + i_t * k_t
        
        # Retrieve value using query
        # h_tilde = C_t @ q_t / max(|n_t^T @ q_t|, 1)
        retrieved = torch.bmm(C_t, q_t.unsqueeze(-1)).squeeze(-1)  # (B, H)
        normalizer = torch.abs(torch.sum(n_t * q_t, dim=-1, keepdim=True))  # (B, 1)
        normalizer = torch.maximum(normalizer, torch.ones_like(normalizer))
        
        h_tilde = retrieved / normalizer
        h_t = o_t * h_tilde
        
        return h_t, (h_t, C_t, n_t)

class xLSTMLayer(nn.Module):
    """Single xLSTM layer with alternating sLSTM/mLSTM blocks"""
    def __init__(self, input_size: int, hidden_size: int, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.hidden_size = hidden_size
        
        if layer_idx % 2 == 0:  # Even layers: sLSTM
            self.block = sLSTMBlock(input_size, hidden_size)
            self.is_slstm = True
        else:  # Odd layers: mLSTM
            self.block = mLSTMBlock(input_size, hidden_size)
            self.is_slstm = False
            
        self.layer_norm = LayerNorm(hidden_size)
        
        if input_size != hidden_size:
            self.input_proj = Linear(input_size, hidden_size)
        else:
            self.input_proj = nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        device = x.device
        
        # Initialize states
        if self.is_slstm:  # sLSTM
            h_0 = torch.zeros(batch_size, self.hidden_size, device=device)
            c_0 = torch.zeros(batch_size, self.hidden_size, device=device)
            n_0 = torch.ones(batch_size, self.hidden_size, device=device)  # Normalizer state
            state = (h_0, c_0, n_0)
        else:  # mLSTM
            h_0 = torch.zeros(batch_size, self.hidden_size, device=device)
            C_0 = torch.zeros(batch_size, self.hidden_size, self.hidden_size, device=device)
            n_0 = torch.zeros(batch_size, self.hidden_size, device=device)
            state = (h_0, C_0, n_0)
        
        outputs = []
        for t in range(seq_len):
            x_t = x[:, t, :]
            residual = self.input_proj(x_t)
            
            h_t, state = self.block(x_t, state)
            out = self.layer_norm(h_t + residual)  # Residual connection
            outputs.append(out.unsqueeze(1))
        
        return torch.cat(outputs, dim=1)

class DKT2(Module):
    def __init__(self, num_c, emb_size=64, dropout=0.1, hidden_size=256, num_layers=2, 
                 emb_type='qid', emb_path="", pretrain_dim=768):
        super().__init__()
        self.model_name = "dkt2"
        self.num_c = num_c
        self.emb_size = emb_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.emb_type = emb_type
        
        # Rasch model embeddings
        self.concept_emb = Embedding(num_c, emb_size)
        self.response_emb = Embedding(2, emb_size)  # 0 or 1
        
        # Question difficulty (scalar) and variation vector
        self.item_difficulty = nn.Parameter(torch.zeros(num_c))
        self.mu_c = Embedding(num_c, emb_size)  # Variation embedding for concepts
        self.g_r = Embedding(2, emb_size)  # Variant embedding for responses
        
        # xLSTM stack
        self.xlstm_layers = nn.ModuleList()
        for i in range(num_layers):
            input_dim = emb_size if i == 0 else hidden_size
            self.xlstm_layers.append(xLSTMLayer(input_dim, hidden_size, i))
        
        # Output layers for comprehensive knowledge state
        self.dropout = Dropout(dropout)
        self.fc1 = Linear(hidden_size * 4, hidden_size * 2)  # For integrated knowledge fusion
        self.fc2 = Linear(hidden_size * 2, num_c)  # Output all concepts
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0, std=0.02)
    
    def forward(self, q, r, target_q=None):
        """
        Args:
            q: question/concept IDs (batch_size, seq_len)
            r: responses (batch_size, seq_len)
            target_q: target questions for prediction (batch_size, seq_len) or None
        Returns:
            predictions: (batch_size, seq_len, num_concepts)
        """
        batch_size, seq_len = q.shape
        device = q.device
        
        # Rasch Embedding
        e_c = self.concept_emb(q)  # (B, T, D)
        e_r = self.response_emb(r)  # (B, T, D)
        
        # Question embedding: Q_t = e_ct + d_qt * μ_ct
        d_q = self.item_difficulty[q].unsqueeze(-1)  # (B, T, 1)
        mu_c = self.mu_c(q)  # (B, T, D)
        Q = e_c + d_q * mu_c  # (B, T, D)
        
        # Student skill embedding: S_t = e_(ct,rt) + d_qt * g_rt
        e_interaction = e_c + e_r  # (B, T, D)
        g_r = self.g_r(r)  # (B, T, D)
        S = e_interaction + d_q * g_r  # (B, T, D)
        
        # Process through xLSTM blocks
        A = S  # Student ability representation
        for layer in self.xlstm_layers:
            A = layer(A)  # (B, T, hidden_size)
        
        # IRT Prediction & Decompose
        # Knowledge: K = A - d_q (broadcasted)
        d_q_expanded = d_q.expand(-1, -1, self.hidden_size)
        K = A - d_q_expanded  # (B, T, hidden_size)
        
        # Decompose into familiar and unfamiliar knowledge
        r_expanded = r.unsqueeze(-1).float().expand(-1, -1, self.hidden_size)
        K_plus = r_expanded * K  # Familiar knowledge (correct responses)
        K_minus = (1 - r_expanded) * K  # Unfamiliar knowledge (incorrect responses)
        
        # Integrated Knowledge Fusion
        # For comprehensive output, we predict for all concepts at each time step
        # Shift for next-step prediction
        if seq_len > 1:
            # Use previous knowledge to predict next step
            K_prev = K[:, :-1, :]  # (B, T-1, hidden_size)
            K_plus_prev = K_plus[:, :-1, :]
            K_minus_prev = K_minus[:, :-1, :]
            
            # If target questions are provided, use them; otherwise use next questions
            if target_q is not None:
                Q_next = self.concept_emb(target_q[:, 1:]) + \
                         self.item_difficulty[target_q[:, 1:]].unsqueeze(-1) * self.mu_c(target_q[:, 1:])
            else:
                Q_next = Q[:, 1:, :]  # Use shifted Q for next-step prediction
            
            # Concatenate all information
            X = torch.cat([Q_next, K_prev, K_plus_prev, K_minus_prev], dim=-1)  # (B, T-1, 4*hidden_size)
        else:
            # For single time step, use current knowledge
            X = torch.cat([Q, K, K_plus, K_minus], dim=-1)
        
        # Generate comprehensive knowledge states
        h = self.dropout(X)
        h = F.relu(self.fc1(h))
        h = self.dropout(h)
        y = torch.sigmoid(self.fc2(h))  # (B, T-1, num_concepts) or (B, 1, num_concepts)
        
        # Pad the output to match input sequence length if needed
        if seq_len > 1:
            # Add a dummy prediction for the first time step (no previous knowledge)
            first_pred = torch.zeros(batch_size, 1, self.num_c, device=device)
            y = torch.cat([first_pred, y], dim=1)  # (B, T, num_concepts)
        
        return y