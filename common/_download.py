from huggingface_hub import hf_hub_download

def fetch_ckpt_from_hf(repo_id, filename):
    return hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        repo_type="model",
    )
