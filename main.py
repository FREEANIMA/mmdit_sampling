# main.py
import os
from huggingface_hub import login
from app.cli import main

ROOT = os.path.dirname(os.path.abspath(__file__))

def hf_login():
    token = os.environ.get("HF_TOKEN")
    if token:
        login(token=token)

if __name__ == "__main__":
    hf_login()
    main(base_root=ROOT)