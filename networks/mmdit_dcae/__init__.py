from common import load_model as _load_model, fetch_ckpt_from_hf
from .Model import Model
from version import VERSION, HUGGINGFACE

VERSION_CODE = sum(
    int(v) * (100, 10, 1)[i]
    for i, v in enumerate(VERSION.split("."))
)

def load():
    path = fetch_ckpt_from_hf(
        repo_id=f"{HUGGINGFACE}/VER{VERSION_CODE}",
        filename="model.pt",
    )
    return _load_model(Model, path)

__all__ = ["Model", "load"]