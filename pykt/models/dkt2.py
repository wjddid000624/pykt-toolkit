import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Module, Embedding, Linear, Dropout, LayerNorm
from typing import Tuple, Optional

class sLSTMBlock(nn.Module):
    """Simplified sLSTM block for DKT2"""
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
        self.W_g = nn.Linear(input_size, hidden_size, bias=False)
        self.U_g = nn.Linear(hidden_size, hidden_size, bias=False)
        
        self.b_i = nn.Parameter(torch.zeros(hidden_size))
        self.b_f = nn.Parameter(torch.ones(hidden_size))
        self.b_o = nn.Parameter(torch.zeros(hidden_size))
        
    def forward(self, x: torch.Tensor, states: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        h_prev, c_prev = states
        
        i_t = torch.sigmoid(self.W_i(x) + self.U_i(h_prev) + self.b_i)
        f_t = torch.sigmoid(self.W_f(x) + self.U_f(h_prev) + self.b_f)
        o_t = torch.sigmoid(self.W_o(x) + self.U_o(h_prev) + self.b_o)
        g_t = torch.tanh(self.W_g(x) + self.U_g(h_prev))
        
        i_tilde = torch.exp(i_t - i_t.max(dim=-1, keepdim=True)[0])
        i_tilde = i_tilde / (i_tilde.sum(dim=-1, keepdim=True) + 1e-8)
        
        c_t = f_t * c_prev + i_tilde * g_t
        h_t = o_t * torch.tanh(c_t)
        
        return h_t, (h_t, c_t)

class mLSTMBlock(nn.Module):
    """Simplified mLSTM block for DKT2"""
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        self.W_q = nn.Linear(input_size, hidden_size, bias=False)
        self.W_k = nn.Linear(input_size, hidden_size, bias=False)
        self.W_v = nn.Linear(input_size, hidden_size, bias=False)
        self.W_i = nn.Linear(input_size, hidden_size, bias=False)
        self.W_f = nn.Linear(input_size, hidden_size, bias=False)
        self.W_o = nn.Linear(input_size, hidden_size, bias=False)
        
        self.b_i = nn.Parameter(torch.zeros(hidden_size))
        self.b_f = nn.Parameter(torch.ones(hidden_size))
        self.b_o = nn.Parameter(torch.zeros(hidden_size))
        
    def forward(self, x: torch.Tensor, states: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        h_prev, C_prev = states
        
        q_t = self.W_q(x)
        k_t = self.W_k(x)
        v_t = self.W_v(x)
        i_t = torch.sigmoid(self.W_i(x) + self.b_i)
        f_t = torch.sigmoid(self.W_f(x) + self.b_f)
        o_t = torch.sigmoid(self.W_o(x) + self.b_o)
        
        outer_products = torch.bmm(k_t.unsqueeze(-1), v_t.unsqueeze(1))
        
        f_gate = f_t.unsqueeze(-1)
        i_gate = i_t.unsqueeze(-1)
        
        C_t = f_gate * C_prev + i_gate * outer_products
        
        h_t = o_t * torch.tanh(torch.bmm(C_t, q_t.unsqueeze(-1)).squeeze(-1))
        
        return h_t, (h_t, C_t)

class xLSTMLayer(nn.Module):
    """Single xLSTM layer with alternating sLSTM/mLSTM blocks"""
    def __init__(self, input_size: int, hidden_size: int, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.hidden_size = hidden_size
        
        if layer_idx % 2 == 0:  # Even layers: sLSTM
            self.block = sLSTMBlock(input_size, hidden_size)
        else:  # Odd layers: mLSTM
            self.block = mLSTMBlock(input_size, hidden_size)
            
        self.layer_norm = LayerNorm(hidden_size)
        
        if input_size != hidden_size:
            self.input_proj = Linear(input_size, hidden_size)
        else:
            self.input_proj = nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        device = x.device
        
        if self.layer_idx % 2 == 0:  # sLSTM
            h_0 = torch.zeros(batch_size, self.hidden_size, device=device)
            c_0 = torch.zeros(batch_size, self.hidden_size, device=device)
            state = (h_0, c_0)
        else:  # mLSTM
            h_0 = torch.zeros(batch_size, self.hidden_size, device=device)
            C_0 = torch.zeros(batch_size, self.hidden_size, self.hidden_size, device=device)
            state = (h_0, C_0)
        
        outputs = []
        for t in range(seq_len):
            x_t = x[:, t, :]
            residual = self.input_proj(x_t)
            
            h_t, state = self.block(x_t, state)
            out = self.layer_norm(h_t + residual)
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
        
        # Input embedding (concept + response interaction)
        if emb_type.startswith("qid"):
            self.interaction_emb = Embedding(num_c * 2, emb_size)
        
        # Item difficulty parameters (learnable)
        self.item_difficulty = nn.Parameter(torch.zeros(num_c))
        
        # xLSTM stack
        self.xlstm_layers = nn.ModuleList()
        for i in range(num_layers):
            input_dim = emb_size if i == 0 else hidden_size
            self.xlstm_layers.append(xLSTMLayer(input_dim, hidden_size, i))
        
        # Knowledge decomposition
        self.knowledge_proj = Linear(hidden_size, hidden_size)
        
        # Output layers
        self.dropout = Dropout(dropout)
        self.out_layer = Linear(hidden_size, num_c)
        
    def forward(self, q, r):
        """
        Args:
            q: question IDs (batch_size, seq_len)
            r: responses (batch_size, seq_len)
        Returns:
            predictions: (batch_size, seq_len, num_concepts)
        """
        batch_size, seq_len = q.shape
        
        # Input embedding
        if self.emb_type == "qid":
            x = q + self.num_c * r  # Interaction encoding
            x_emb = self.interaction_emb(x)  # (B, T, emb_size)
        
        # Process through xLSTM stack
        h = x_emb
        for layer in self.xlstm_layers:
            h = layer(h)  # (B, T, hidden_size)
        
        # Knowledge decomposition with item difficulty
        d_q = self.item_difficulty[q].unsqueeze(-1)  # (B, T, 1)
        knowledge = self.knowledge_proj(h) - d_q  # (B, T, hidden_size)
        
        # Output prediction
        h_out = self.dropout(knowledge)
        y = self.out_layer(h_out)  # (B, T, num_concepts)
        y = torch.sigmoid(y)
        
        return y