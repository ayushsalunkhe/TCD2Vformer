import torch
import numpy as np
import random
import hashlib
import os

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

def compute_parameter_checksum(model: torch.nn.Module) -> str:
    hasher = hashlib.sha256()
    for name, param in sorted(model.named_parameters()):
        param_bytes = param.detach().cpu().numpy().tobytes()
        hasher.update(name.encode('utf-8'))
        hasher.update(param_bytes)
    return hasher.hexdigest()

def verify_parameter_invariance(model: torch.nn.Module, baseline_hash: str) -> bool:
    current_hash = compute_parameter_checksum(model)
    return current_hash == baseline_hash
