import torch
import numpy as np
import math

def compute_mse(preds: np.ndarray, trues: np.ndarray) -> float:
    return float(np.mean((preds - trues) ** 2))

def compute_mae(preds: np.ndarray, trues: np.ndarray) -> float:
    return float(np.mean(np.abs(preds - trues)))

def compute_batch_attention_entropy(attention_tensor: torch.Tensor, eps: float = 1e-12) -> tuple:
    L = attention_tensor.shape[-1]
    entropy = -torch.sum(attention_tensor * torch.log(attention_tensor + eps), dim=-1)
    mean_entropy = float(entropy.mean().item())
    max_entropy = math.log(L)
    norm_entropy = float(mean_entropy / max_entropy) if max_entropy > 0 else 0.0
    return mean_entropy, norm_entropy
