import os
import json
import pandas as pd
import numpy as np
from tqdm import tqdm
from itertools import combinations

tqdm.pandas()

def load(base_root):
    dup_meta_path = os.path.join(base_root, "data", "metadata", "dup_meta.parquet")
    vocab_path = os.path.join(base_root, "data", "metadata", "vocab.json")

    dup_meta_df = pd.read_parquet(dup_meta_path)
    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab = json.load(f)

    return dup_meta_df, vocab

def convert_vocab(dup_meta_df, vocab):
    return (
        dup_meta_df["tags"]
        .fillna("")
        .str.split(",")
        .progress_map(lambda xs: [vocab[t] for t in (x.strip() for x in xs) if t in vocab])
        .tolist()
    )

def build_vocab_matrix(vocab):
    tags = [t for t, _ in sorted(vocab.items(), key=lambda kv: kv[1])]
    return pd.DataFrame(0, index=tags, columns=tags, dtype="int32")

def accumulate_cooccurrence_np(arr, N):
    co = np.zeros((N, N), dtype=np.int32)
    for ids in tqdm(arr, desc="co-occur"):
        uniq = sorted(set(ids))
        for a, b in combinations(uniq, 2):
            co[a, b] += 1
            co[b, a] += 1
    return co

def vocab_matrix_run(base_root):
    dup_meta_df, vocab = load(base_root)
    arr = convert_vocab(dup_meta_df=dup_meta_df, vocab=vocab)
    co_df = build_vocab_matrix(vocab)

    N = len(vocab)
    co = accumulate_cooccurrence_np(arr, N)

    tags = [t for t, _ in sorted(vocab.items(), key=lambda kv: kv[1])]
    co_df = pd.DataFrame(co, index=tags, columns=tags)

    out_path = os.path.join(base_root, "data", "metadata", "vocab_matrix.parquet")
    co_df.to_parquet(out_path, index=True)
    print(f"saved {co_df.shape} -> {out_path}")

if __name__ == "__main__":
    vocab_matrix_run(base_root=r"C:\Users\HONG\Desktop\inquiry\mmdit_dcae_sampling")