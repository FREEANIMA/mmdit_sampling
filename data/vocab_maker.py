import os
import pandas as pd
import json

def load(base_root):
    tags_type_path = os.path.join(base_root, "data", "metadata", "tags_type.parquet")
    tags_df = pd.read_parquet(tags_type_path)

    return tags_df

def filter_vocab(tags_df, max_count=10_000):
    return (
        tags_df.loc[tags_df["count"] > max_count, ["dup", "count"]]
        .sort_values("count", ascending=False)
        .drop_duplicates(subset="dup")     # ← 추가
        .reset_index(drop=True)
    )

def vocab_maker_run(base_root):
    tags_df = load(base_root=base_root)
    filtered = filter_vocab(tags_df)

    special = ["general", "nsfw"]
    tags = special + filtered["dup"].tolist()
    tags = list(dict.fromkeys(tags))

    vocab = {tag: i for i, tag in enumerate(tags)}

    vocab_out_path = os.path.join(base_root, "data", "metadata", "vocab.json")
    with open(vocab_out_path, "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)
    print(f"saved {len(vocab)} -> {vocab_out_path}")

if __name__ == "__main__":
    vocab_maker_run(base_root=r"C:\Users\HONG\Desktop\inquiry\mmdit_dcae_sampling")