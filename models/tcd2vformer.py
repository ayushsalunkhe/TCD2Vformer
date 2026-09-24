import math
import os
import sys
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from layers.Revin import RevIN
from utils.reproducibility import compute_parameter_checksum


class TCD2Vformer(nn.Module):
    """Temporal-Conditioned Date2Vecformer (TCD2Vformer).

    Extends PureD2Vformer (arXiv:2409.11024v1) with parameter-efficient,
    horizon-independent adaptive attention temperature mechanisms:

        S = (D_y @ D_x^T) / sqrt(k_freq + 1)
        A = Softmax(S / tau, dim=-1)
        Y_tilde = A @ T^T

    Supported temperature modes:
        - 'fixed': Constant buffer tau (baseline or validation-selected).
        - 'learned_global': Single scalar parameter tau = softplus(tau_raw) + eps.
        - 'temporal_context': tau = tau_min + softplus(MLP(mean(D_x))).
        - 'query_conditioned': tau_o = tau_min + softplus(MLP(mean(D_y[:, :, o, :]))).

    Strict Horizon Independence:
        Zero trainable parameters scale with forecast horizon O.
        Asserted via self.assert_horizon_independence().
    """

    def __init__(
        self,
        c_in: int,
        seq_len: int = 96,
        d_model: int = 128,
        d_ff: int = 256,
        k_freq: int = 16,
        dropout: float = 0.05,
        temperature_mode: str = "fixed",
        initial_temperature: float = 1.0,
        tau_min: float = 0.1,
        d_temp: int = 16,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.c_in = c_in
        self.d_model = d_model
        self.k_freq = k_freq
        self.temperature_mode = temperature_mode
        self.tau_min = tau_min
        self.d_temp = d_temp

        # 1. Reversible Instance Normalization (RevIN)
        self.revin = RevIN(c_in, affine=True, subtract_last=False)

        # 2. Temporal Feature Extraction (TFE) - Eq. 3
        self.tfe = nn.Linear(c_in, d_model)

        # 3. Date2Vec parameters - Eq. 4-7
        self.w_T = nn.Parameter(torch.randn(seq_len))
        self.b_T = nn.Parameter(torch.zeros(d_model))

        self.W_S = nn.Parameter(torch.randn(k_freq, seq_len))
        self.B_S = nn.Parameter(torch.zeros(k_freq, d_model))

        self.b_1 = nn.Parameter(torch.zeros(d_model, 1, 1))
        self.B_2 = nn.Parameter(torch.zeros(k_freq, d_model, 1, 1))
        self.b_3 = nn.Parameter(torch.zeros(d_model, 1, 1))
        self.B_4 = nn.Parameter(torch.zeros(k_freq, d_model, 1, 1))

        # 4. Position-wise FeedForward - Eq. 10
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, c_in),
        )

        # 5. Temperature Parameterization
        if self.temperature_mode == "fixed":
            self.register_buffer("tau_fixed", torch.tensor(float(initial_temperature), dtype=torch.float32))

        elif self.temperature_mode == "learned_global":
            init_raw = math.log(math.exp(initial_temperature) - 1.0) if initial_temperature > 1e-4 else -5.0
            self.tau_raw = nn.Parameter(torch.tensor(init_raw, dtype=torch.float32))

        elif self.temperature_mode == "temporal_context":
            self.temp_mlp = nn.Sequential(
                nn.Linear(k_freq + 1, d_temp),
                nn.GELU(),
                nn.Linear(d_temp, 1),
            )
            nn.init.normal_(self.temp_mlp[2].weight, mean=0.0, std=0.01)
            init_bias = math.log(math.exp(max(1e-4, initial_temperature - tau_min)) - 1.0)
            nn.init.constant_(self.temp_mlp[2].bias, init_bias)

        elif self.temperature_mode == "query_conditioned":
            self.temp_mlp = nn.Sequential(
                nn.Linear(k_freq + 1, d_temp),
                nn.GELU(),
                nn.Linear(d_temp, 1),
            )
            nn.init.normal_(self.temp_mlp[2].weight, mean=0.0, std=0.01)
            init_bias = math.log(math.exp(max(1e-4, initial_temperature - tau_min)) - 1.0)
            nn.init.constant_(self.temp_mlp[2].bias, init_bias)

        else:
            raise ValueError(
                f"Unknown temperature_mode: {temperature_mode}. "
                "Must be one of ['fixed', 'learned_global', 'temporal_context', 'query_conditioned']."
            )

    def get_parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def compute_temperature(
        self,
        Dx: torch.Tensor,
        Dy: torch.Tensor,
    ) -> torch.Tensor:
        if self.temperature_mode == "fixed":
            return self.tau_fixed

        elif self.temperature_mode == "learned_global":
            return F.softplus(self.tau_raw) + 1e-4

        elif self.temperature_mode == "temporal_context":
            dx_mean = Dx.mean(dim=(1, 2))
            raw_tau = self.temp_mlp(dx_mean)
            tau = self.tau_min + F.softplus(raw_tau)
            return tau.unsqueeze(1).unsqueeze(-1)

        elif self.temperature_mode == "query_conditioned":
            dy_mean = Dy.mean(dim=1)
            raw_tau = self.temp_mlp(dy_mean)
            tau = self.tau_min + F.softplus(raw_tau)
            return tau.unsqueeze(1)

        raise RuntimeError(f"Unhandled temperature_mode: {self.temperature_mode}")

    def forward(
        self,
        x_enc: torch.Tensor,
        x_mark_enc: torch.Tensor,
        y_mark_dec: torch.Tensor,
        return_attention: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        B, L, D = x_enc.shape
        O = y_mark_dec.shape[1]

        x_norm = self.revin(x_enc, "norm")
        T = self.tfe(x_norm)

        v_T = torch.einsum("l, blh -> bh", self.w_T, T) + self.b_T
        Omega_S = torch.einsum("kl, blh -> bkh", self.W_S, T) + self.B_S

        E_lin = (v_T.unsqueeze(2).unsqueeze(3) * x_mark_enc.unsqueeze(1) + self.b_1).unsqueeze(1)
        E_har = torch.sin(Omega_S.unsqueeze(3).unsqueeze(4) * x_mark_enc.unsqueeze(1).unsqueeze(2) + self.B_2)
        D_x = torch.cat([E_lin, E_har], dim=1).mean(dim=-1)

        F_lin = (v_T.unsqueeze(2).unsqueeze(3) * y_mark_dec.unsqueeze(1) + self.b_3).unsqueeze(1)
        F_har = torch.sin(Omega_S.unsqueeze(3).unsqueeze(4) * y_mark_dec.unsqueeze(1).unsqueeze(2) + self.B_4)
        D_y = torch.cat([F_lin, F_har], dim=1).mean(dim=-1)

        Dx = D_x.permute(0, 2, 3, 1)
        Dy = D_y.permute(0, 2, 3, 1)

        scale = 1.0 / math.sqrt(self.k_freq + 1)
        S = torch.einsum("bhok, bhlk -> bhol", Dy, Dx) * scale

        tau = self.compute_temperature(Dx, Dy)
        A = torch.softmax(S / tau, dim=-1)

        Y_tilde = torch.einsum("bhol, bhl -> bho", A, T.transpose(1, 2))
        Y_hat = self.ffn(Y_tilde.transpose(1, 2))
        Y_out = self.revin(Y_hat, "denorm")

        if return_attention:
            return Y_out, A, tau
        return Y_out, A

    def assert_horizon_independence(
        self,
        eval_horizons: List[int] = [24, 48, 96, 192, 336, 720],
        device: str = "cpu",
    ) -> Dict[int, int]:
        baseline_params = self.get_parameter_count()
        param_counts = {}

        self.eval()
        with torch.no_grad():
            for O in eval_horizons:
                dummy_x = torch.randn(2, self.seq_len, self.c_in, device=device)
                dummy_xm = torch.randn(2, self.seq_len, 4, device=device)
                dummy_ym = torch.randn(2, O, 4, device=device)

                out, A = self(dummy_x, dummy_xm, dummy_ym)
                assert out.shape == (2, O, self.c_in), f"Output shape mismatch for O={O}: got {out.shape}"
                assert A.shape == (2, self.d_model, O, self.seq_len), f"Attention shape mismatch for O={O}: got {A.shape}"

                curr_params = self.get_parameter_count()
                assert curr_params == baseline_params, (
                    f"Parameter count changed for horizon O={O}! Expected {baseline_params}, got {curr_params}."
                )
                param_counts[O] = curr_params

        return param_counts
