import os
import numpy as np
import pandas as pd

INIT_PROMPT = "general img; "
NAS = [
    "style", "person_count", "gender", "hair_color", "hair_length", "hair_style",
    "eye_color", "body_type", "clothing", "accessory", "background", "pose",
    "expression", "composition", "gaze", "sex_act",
]
AUTO, NONE = "__auto__", "__none__"

class TagsCoOC:
    def __init__(self, base_root):
        self.co_df = pd.read_parquet(os.path.join(base_root, "data", "metadata", "vocab_matrix.parquet"))
        self.tags_type_df = pd.read_parquet(os.path.join(base_root, "data", "metadata", "tags_type.parquet"))

        # 매 호출마다 만들던 것을 한 번만 생성
        self.dup_map = dict(zip(self.tags_type_df["tag"], self.tags_type_df["dup"]))
        self.type_map = dict(zip(self.tags_type_df["tag"], self.tags_type_df["type"]))
        self.type_series = pd.Series(self.type_map)

    def get_nas_options(self) -> dict[str, list[str]]:
        """UI 드롭다운용: type -> 태그 목록"""
        g = self.tags_type_df.groupby("type")["tag"].apply(lambda s: sorted(set(s))).to_dict()
        return {k: g.get(k, []) for k in NAS}

    def _scores(self, valid_tags):
        valid = [t for t in valid_tags if t in self.co_df.index]
        if not valid:
            return pd.Series(dtype=float)
        s = self.co_df.loc[valid].min(axis=0)      # 모든 조건 태그와 동시에 연관된 정도
        return s.drop(labels=valid, errors="ignore")

    def _sample_one(self, scores, ty, rng, top_n=20, temperature=1.0):
        if scores.empty:
            return None
        types = self.type_series.reindex(scores.index).values
        cand = scores[(types == ty) & (scores.values > 0)]
        cand = cand.sort_values(ascending=False).head(top_n)
        if cand.empty:
            return None
        w = cand.values.astype(float) ** (1.0 / temperature)
        w /= w.sum()
        return rng.choice(cand.index.to_numpy(), p=w)

    def tags_mapping_worker(self, user_tags: list, nsfw, H, W,
                            selection: dict | None = None, seed: int | None = None,
                            top_n: int = 20, temperature: float = 1.0):
        selection = selection or {}
        rng = np.random.default_rng(seed)

        nsfw_text = f"nsfw: {str(nsfw).lower()}; "
        size_text = f"size: {H}x{W}; "
        nsfw_str = "nsfw" if nsfw is True else "general"

        tag_list = [t.replace("_", " ") for t in user_tags]

        # 1) 유저가 직접 입력한 태그
        include_dict = {t: self.type_map[t] for t in tag_list if t in self.type_map}
        exclude_dict = {t: "others" for t in tag_list if t not in self.dup_map}

        # 2) UI에서 직접 고른 태그
        picked = {}
        for ty in NAS:
            v = selection.get(ty, AUTO)
            if v not in (AUTO, NONE, "", None):
                picked[v] = ty
        include_dict.update(picked)

        # 3) 자동으로 채울 type: auto 이면서 아직 비어있는 것
        filled = set(include_dict.values())
        auto_types = [ty for ty in NAS
                      if selection.get(ty, AUTO) in (AUTO, "", None) and ty not in filled]
        rng.shuffle(auto_types)   # 항상 같은 type이 먼저 뽑혀서 쏠리는 것 방지

        # 4) 하나 뽑을 때마다 조건에 추가 -> 뒤에 뽑는 태그가 앞선 선택과 어울리게
        cond = [self.dup_map[t] for t in include_dict if t in self.dup_map] + [nsfw_str]
        auto_dict = {}
        for ty in auto_types:
            tag = self._sample_one(self._scores(cond), ty, rng, top_n, temperature)
            if tag is not None:
                auto_dict[tag] = ty
                cond.append(self.dup_map.get(tag, tag))

        total = {**include_dict, **auto_dict, **exclude_dict}

        text = ""
        for key in NAS:
            tags = [t for t, ty in total.items() if ty == key]
            if tags:
                text += f"{key}: {', '.join(tags)}; "
        for key in set(total.values()) - set(NAS):
            tags = [t for t, ty in total.items() if ty == key]
            text += f"{key}: {', '.join(tags)}; "

        return INIT_PROMPT + nsfw_text + size_text + text