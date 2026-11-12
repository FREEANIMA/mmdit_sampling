import torch
import time
from contextlib import contextmanager

def default_device():
    return "cuda" if torch.cuda.is_available() else "cpu"

def default_dtype():
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float32

def strip_orig_mod(state_dict):
    prefix = "_orig_mod."
    return {
        (k[len(prefix):] if k.startswith(prefix) else k): v
        for k, v in state_dict.items()
    }

def load_model(model_class, path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    params = checkpoint.get("param")
    state_dict = None
    for key in ("ema_state_dict", "model_state_dict", "model"):
        if key in checkpoint:
            state_dict = checkpoint[key]
            break
        
    state_dict = strip_orig_mod(state_dict)

    with torch.device("meta"):
        model = model_class(**params)

    model.load_state_dict(state_dict, assign=True)
    model.eval()

    del checkpoint, state_dict

    return model