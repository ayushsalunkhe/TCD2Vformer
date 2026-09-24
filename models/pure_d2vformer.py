import torch
import torch.nn as nn
import math
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from layers.Revin import RevIN

class PureD2Vformer(nn.Module):
    """
    Exact mathematical implementation of D2Vformer as specified in arXiv:2409.11024v1 (Section 3, Eq. 1-10).
    Completely horizon-independent: zero trainable parameters depend on output horizon O.
    """
    def __init__(self, c_in: int, seq_len: int = 96, d_model: int = 128, d_ff: int = 256, k_freq: int = 16, dropout: float = 0.05):
        super().__init__()
        self.seq_len = seq_len
        self.c_in = c_in
        self.d_model = d_model
        self.k_freq = k_freq
        
        # 1. Reversible Instance Normalization (RevIN)
        self.revin = RevIN(c_in, affine=True, subtract_last=False)
        
        # 2. Temporal Feature Extraction (TFE) - Eq. 3
        self.tfe = nn.Linear(c_in, d_model)
        
        # 3. Date2Vec (D2V) - Eq. 4-7
        self.w_T = nn.Parameter(torch.randn(seq_len))
        self.b_T = nn.Parameter(torch.zeros(d_model))
        
        self.W_S = nn.Parameter(torch.randn(k_freq, seq_len))
        self.B_S = nn.Parameter(torch.zeros(k_freq, d_model))
        
        self.b_1 = nn.Parameter(torch.zeros(d_model, 1, 1))
        self.B_2 = nn.Parameter(torch.zeros(k_freq, d_model, 1, 1))
        self.b_3 = nn.Parameter(torch.zeros(d_model, 1, 1))
        self.B_4 = nn.Parameter(torch.zeros(k_freq, d_model, 1, 1))
        
        # 4. Fusion Block FeedForward - Eq. 10
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, c_in)
        )
        
    def forward(self, x_enc, x_mark_enc, y_mark_dec):
        B, L, D = x_enc.shape
        O = y_mark_dec.shape[1]
        
        x_norm = self.revin(x_enc, 'norm')
        T = self.tfe(x_norm)
        
        v_T = torch.einsum('l, blh -> bh', self.w_T, T) + self.b_T
        Omega_S = torch.einsum('kl, blh -> bkh', self.W_S, T) + self.B_S
        
        E_lin = (v_T.unsqueeze(2).unsqueeze(3) * x_mark_enc.unsqueeze(1) + self.b_1).unsqueeze(1)
        E_har = torch.sin(Omega_S.unsqueeze(3).unsqueeze(4) * x_mark_enc.unsqueeze(1).unsqueeze(2) + self.B_2)
        E = torch.cat([E_lin, E_har], dim=1)
        D_x_hat = E.mean(dim=-1)
        
        F_lin = (v_T.unsqueeze(2).unsqueeze(3) * y_mark_dec.unsqueeze(1) + self.b_3).unsqueeze(1)
        F_har = torch.sin(Omega_S.unsqueeze(3).unsqueeze(4) * y_mark_dec.unsqueeze(1).unsqueeze(2) + self.B_4)
        F = torch.cat([F_lin, F_har], dim=1)
        D_y_hat = F.mean(dim=-1)
        
        D_x_tilde = D_x_hat.permute(0, 2, 3, 1)
        D_y_tilde = D_y_hat.permute(0, 2, 3, 1)
        
        scale = 1.0 / math.sqrt(self.k_freq + 1)
        scores = torch.einsum('bhok, bhlk -> bhol', D_y_tilde, D_x_tilde) * scale
        A = torch.softmax(scores, dim=-1)
        
        T_transposed = T.transpose(1, 2)
        Y_tilde = torch.einsum('bhol, bhl -> bho', A, T_transposed)
        
        Y_hat = self.ffn(Y_tilde.transpose(1, 2))
        Y_out = self.revin(Y_hat, 'denorm')
        return Y_out, A
