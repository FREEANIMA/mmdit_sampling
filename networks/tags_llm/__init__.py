from common import load_model as _load_model, fetch_ckpt_from_hf
from .Model import Model
from version import VERSION, HUGGINGFACE

def load():
    path = fetch_ckpt_from_hf(
        repo_id=HUGGINGFACE,
        filename="model.pt",
    )
    vocab_path = fetch_ckpt_from_hf(
        repo_id=HUGGINGFACE,
        filename="vocab.json",
    )
    tags_type_path = fetch_ckpt_from_hf(
        repo_id=HUGGINGFACE,
        filename="tags_type.parquet",
    )

    return _load_model(Model, path), vocab_path, tags_type_path

__all__ = ["Model", "load"]
