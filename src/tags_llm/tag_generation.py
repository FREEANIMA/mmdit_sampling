import os
import torch
import numpy as np
import pandas as pd
import json
import random
from dataclasses import asdict, is_dataclass
from networks import load_tags_llm
from networks.tags_llm import Model

NAS = [
    "style",
    "person_count",
    "gender",
    "species",
    "hair_color",
    "hair_length",
    "hair_style",
    "eye_color",
    "body_type",
    "clothing",
    "accessory",
    "background",
    "pose",
    "expression",
    "composition",
    "gaze",
    "sex_act"
]

INIT_PROMPT = "general img; "

class TagGenration:
    def __init__(self, device="cuda", dtype=torch.bfloat16):
        self.device = device
        self.dtype = dtype

        ckpt_path, vocab_path, tags_type_path = load_tags_llm()

        with open(vocab_path, "r", encoding="utf-8") as f:
            self.vocab = json.load(f)

        vocab_size = len(self.vocab)

        chk = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        config = chk["config"]
        param = asdict(config) if is_dataclass(config) else dict(config)
        param["vocab_size"] = vocab_size 

        self.text_model = Model(**param)
        self.text_model.load_state_dict(chk["model"])
        self.text_model = self.text_model.to(device=device, dtype=dtype)
        self.text_model.eval()

        tags_type_df = pd.read_parquet(tags_type_path)

        self.type_map, self.dup_map = self.build_tag_maps(tags_type_df=tags_type_df)


        self.PAD_ID = self.vocab["<pad>"]
        self.CLS_ID = self.vocab["<cls>"]
        self.EOS_ID = self.vocab["<eos>"]
        self.UNK_ID = self.vocab["<unk>"]

        self.id_to_tag = {v: k for k, v in self.vocab.items()}

    def build_tag_maps(self, tags_type_df):
        s = tags_type_df[["tag", "type", "dup"]].drop_duplicates(subset="tag", keep="first")
        type_map = dict(zip(s["tag"].tolist(), s["type"].tolist()))
        dup_map = dict(zip(s["tag"].tolist(), s["dup"].fillna(s["tag"]).tolist()))

        return type_map, dup_map

    def canonicalize(self, tag_list: list):
        out = []
        seen = set()
        for t in tag_list:
            t = t.strip()
            if not t:
                continue
            rep = self.dup_map.get(t, t)
            if rep not in seen:
                seen.add(rep)
                out.append(rep)
        return out

    def to_ids(self, xs, max_len=256):
        body = []
        for t in xs:
            body.append(self.vocab.get(t, self.UNK_ID))
            if len(body) >= max_len - 1:
                break
        return [self.CLS_ID] + body

    @torch.inference_mode()
    def sampling(self, batch, max_new=256, temperature=1.0, top_k=50, top_p=0.95):
        ids = batch
        generated = []

        with torch.amp.autocast(device_type="cuda", dtype=self.dtype):
            logits, past = self.text_model(x=ids, attn_mask=None, use_cache=True)
            next_logits = logits[:, -1, :]            # (1, V)

            for _ in range(max_new):
                next_logits = next_logits.float() / max(temperature, 1e-5)
                for bad in (self.PAD_ID, self.CLS_ID):
                    next_logits[:, bad] = float("-inf")

                if top_k is not None and top_k > 0:
                    kth = torch.topk(next_logits, min(top_k, next_logits.size(-1))).values[:, -1:]
                    next_logits = next_logits.masked_fill(next_logits < kth, float("-inf"))

                if top_p is not None and top_p < 1.0:
                    sorted_logits, sorted_idx = torch.sort(next_logits, descending=True)
                    probs = torch.softmax(sorted_logits, dim=-1)
                    cum = probs.cumsum(dim=-1)
                    remove = cum - probs > top_p      # 첫 토큰은 항상 유지
                    sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
                    next_logits = torch.full_like(next_logits, float("-inf")).scatter(1, sorted_idx, sorted_logits)

                probs = torch.softmax(next_logits, dim=-1)
                next_id = torch.multinomial(probs, num_samples=1)

                tok = next_id.item()
                if tok == self.EOS_ID:
                    break
                generated.append(tok)

                logits, past = self.text_model(x=next_id, attn_mask=None, past_kv=past, use_cache=True)
                next_logits = logits[:, -1, :]

        tags_infer = [self.id_to_tag.get(i, f"<{i}>") for i in generated]
        return tags_infer
    
    def text_generation(self, tags_infer: list, nsfw, H, W):
        nsfw_text = f"nsfw: {str(nsfw).lower()}; "
        size_text = f"size: {H}x{W}; "

        grouped = {}
        others = []
        for tag in tags_infer:
            _type = self.type_map.get(tag)
            if _type is None or pd.isna(_type) or not str(_type).strip():
                others.append(tag)
            else:
                grouped.setdefault(_type, []).append(tag)

        # NAS 순서대로 먼저, NAS에 없는 type은 뒤에 등장 순서대로
        ordered_types = [t for t in NAS if t in grouped]
        ordered_types += [t for t in grouped if t not in NAS]

        text = "".join(f"{t}: {', '.join(grouped[t])}; " for t in ordered_types)
        others_text = f"others: {', '.join(others)}; " if others else ""

        total = INIT_PROMPT + nsfw_text + size_text + text + others_text
        return [total]
    
    def tags_mapping_worker(self, tags: str, nsfw, H, W, device=None):
        device = device or self.device
        nsfw_str = "nsfw" if nsfw else "general"
        tag_list = tags.split(", ")
        dup_list = self.canonicalize(tag_list=tag_list) + [nsfw_str]
        random.shuffle(dup_list)
        ids = self.to_ids(dup_list)
        batch = torch.tensor([ids], dtype=torch.long, device=device)
        tags_infer = self.sampling(batch)

        return self.text_generation(tags_infer, nsfw, H, W)

def test():
    tag_g = TagGenration()

    for _ in range(5):
        tag = tag_g.tags_mapping_worker(tags="woman", H="1024", W="1024", nsfw=True)
        print(tag)