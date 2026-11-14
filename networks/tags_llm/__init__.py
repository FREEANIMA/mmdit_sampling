from common import load_model as _load_model, fetch_ckpt_from_hf
from .Model import Model
from version import HUGGINGFACE

def load():
    model_path = fetch_ckpt_from_hf(
        repo_id=f"{HUGGINGFACE}/tags_llm",
        filename="model.pt",
    )
    vocab_path = fetch_ckpt_from_hf(
        repo_id=f"{HUGGINGFACE}/tags_llm",
        filename="vocab.json",
    )
    tags_type_path = fetch_ckpt_from_hf(
        repo_id=f"{HUGGINGFACE}/tags_llm",
        filename="tags_type.parquet",
    )

    return model_path, vocab_path, tags_type_path

__all__ = ["Model", "load"]
