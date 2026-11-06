# main.py
import os
from huggingface_hub import login
from app.cli import main

COLAB_ROOT = "/content/drive/MyDrive/model/mmdit_sampling"

def get_root():
    if os.path.exists("/content"):          # 코랩 환경
        from google.colab import drive
        if not os.path.exists("/content/drive/MyDrive"):
            drive.mount("/content/drive")
        os.makedirs(COLAB_ROOT, exist_ok=True)
        return COLAB_ROOT
    return os.path.dirname(os.path.abspath(__file__))

def hf_login():
    token = os.environ.get("HF_TOKEN")
    if token:
        login(token=token)

if __name__ == "__main__":
    ROOT = get_root()
    hf_login()
    main(base_root=ROOT)