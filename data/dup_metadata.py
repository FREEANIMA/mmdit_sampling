import os
import pandas as pd
from tqdm import tqdm

base_root = r"C:\Users\HONG\Desktop\inquiry\mmdit_dcae_sampling\data"

def load(base_root):
    metadata_path = os.path.join(base_root, "data", "metadata", "metadata.parquet")
    tags_type_path = os.path.join(base_root, "data", "metadata", "tags_type.parquet")

    meta_df = pd.read_parquet(metadata_path)
    tags_df = pd.read_parquet(tags_type_path)

    return meta_df, tags_df

def convert_dup(meta_df, tags_df, BAD_FILTER, BAD_TAG):
    SPECIAL = {"nsfw", "general"}

    mapping = {
        t: d for t, d in zip(tags_df["tag"], tags_df["dup"])
        if d not in SPECIAL
    }

    def has_bad(s):
        if not isinstance(s, str):
            return True
        return any(t.strip() in BAD_FILTER for t in s.split(","))

    bad_mask = meta_df["tags"].map(has_bad)
    meta_df = meta_df.loc[~bad_mask].copy()

    def conv(s, nsfw):
        if not isinstance(s, str):
            s = ""
        tags = [t.strip() for t in s.split(",") if t.strip()]
        tags = {mapping.get(t, t) for t in tags if t not in BAD_TAG}

        # 기존 special 제거 후 하나만 추가 (general/nsfw 상극 보장)
        tags -= SPECIAL
        tags.add("nsfw" if bool(nsfw) else "general")

        return ", ".join(sorted(tags))

    meta_df["tags"] = [
        conv(s, n)
        for s, n in tqdm(
            zip(meta_df["tags"], meta_df["nsfw"]),
            total=len(meta_df),
            desc="convert_dup",
        )
    ]
    return meta_df

def dup_meatdata_run(base_root):
    BAD_FILTER = {"NO_TAGS", "LOAD_FAIL"} 
    BAD_TAG    = {"NO_TAGS", "LOAD_FAIL", "nsfw", "general"}

    meta_df, tags_df = load(base_root=base_root)
    print(f"loaded: {len(meta_df)} rows")

    meta_df = convert_dup(meta_df=meta_df, tags_df=tags_df, BAD_FILTER=BAD_FILTER, BAD_TAG=BAD_TAG)

    dup_meta_path = os.path.join(base_root, "data", "metadata", "dup_meta.parquet")
    meta_df.to_parquet(dup_meta_path, index=False)
    print(f"saved: {len(meta_df)} rows -> {dup_meta_path}")

if __name__ == "__main__":
    dup_meatdata_run(base_root=r"C:\Users\HONG\Desktop\inquiry\mmdit_dcae_sampling")