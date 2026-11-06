import torch
import time
from contextlib import contextmanager

@contextmanager
def console(name="run", verbose=True):
    t0 = time.perf_counter()

    def log(*msg):
        if verbose:
            print(f"[{name}]", *msg)

    log("start")
    try:
        yield log
    finally:
        log(f"done ({time.perf_counter() - t0:.3f}s)")
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
    checkpoint = torch.load(
        path,
        map_location="cpu",
        weights_only=True,
        mmap=True
    )

    params = checkpoint.get("param")
    if params is None:
        params = checkpoint.get("config")
    if params is None:
        raise KeyError("param 또는 config 키가 필요합니다.")

    state_dict = None
    for key in ("ema_state_dict", "model_state_dict", "model"):
        if key in checkpoint:
            state_dict = checkpoint[key]
            break

    if state_dict is None:
        raise KeyError("모델 state_dict를 찾을 수 없습니다.")

    state_dict = strip_orig_mod(state_dict)

    with torch.device("meta"):
        model = model_class(**params)

    model.load_state_dict(state_dict, assign=True)
    model.eval()

    del checkpoint, state_dict

    return model

@contextmanager
def console(name="run", verbose=True):
    t0 = time.perf_counter()

    def log(*msg):
        if verbose:
            print(f"[{name}]", *msg)

    log("start")
    try:
        yield log
    finally:
        log(f"done ({time.perf_counter() - t0:.3f}s)")