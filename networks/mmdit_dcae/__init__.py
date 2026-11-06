from common import load_model as _load_model, fetch_ckpt_from_hf
from .Model import Model

def load():
    path = fetch_ckpt_from_hf(
        repo_id="honghong3/VER8",
        filename="model.pt",
    )
    return _load_model(Model, path)

__all__ = ["Model", "load"]