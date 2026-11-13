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
REQUIRED = {"style", "person_count", "gender", "hair_color"}


class TagsCoOC:
    def __init__(self, base_root):
        self.co_df = pd.read_parquet(os.path.join(base_root, "data", "metadata", "vocab_matrix.parquet"))
        self.tags_type_df = pd.read_parquet(os.path.join(base_root, "data", "metadata", "tags_type.parquet"))

        df = self.tags_type_df
        # tag -> 대표 tag (dup 가 비어 있으면 자기 자신)
        self.dup_map = {
            t: (d if isinstance(d, str) and d else t)
            for t, d in zip(df["tag"], df["dup"])
        }
        self.type_map = dict(zip(df["tag"], df["type"]))

        # 대표 tag -> type (대표 tag 가 행으로 없으면 별칭 행의 type 을 사용)
        self.canon_type = {}
        for t, ty in self.type_map.items():
            c = self.dup_map[t]
            if c == t or c not in self.canon_type:
                self.canon_type[c] = self.type_map.get(c, ty)
        self.type_series = pd.Series(self.canon_type)

    # ---------- 유틸 ----------
    def _canon(self, t: str) -> str:
        return self.dup_map.get(t, t)

    def get_nas_options(self) -> dict[str, list[str]]:
        """UI 드롭다운용: type -> 대표 tag 목록"""
        g = {}
        for c, ty in self.canon_type.items():
            g.setdefault(ty, []).append(c)
        return {k: sorted(set(g.get(k, []))) for k in NAS}

    def _scores(self, cond_tags):
        valid = [t for t in cond_tags if t in self.co_df.index]
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

    # ---------- 메인 ----------
    def tags_mapping_worker(self, user_tags: list, nsfw, H, W,
                            selection: dict | None = None, seed: int | None = None,
                            top_n: int = 20, temperature: float = 1.0):
        selection = dict(selection or {})
        for ty in REQUIRED:
            if selection.get(ty) == NONE:
                selection[ty] = AUTO            # 필수 type 은 none 불가
        rng = np.random.default_rng(seed)

        nsfw_text = f"nsfw: {str(nsfw).lower()}; "
        size_text = f"size: {H}x{W}; "
        nsfw_str = "nsfw" if nsfw is True else "general"

        tag_list = [t.replace("_", " ") for t in user_tags]

        # 1) 유저가 입력한 tag -> 대표 tag 로 정규화 (female/girl/woman -> woman, 자동 중복 제거)
        include_dict = {}
        for t in tag_list:
            if t in self.type_map:
                c = self._canon(t)
                include_dict[c] = self.canon_type.get(c, self.type_map[t])
        exclude_dict = {t: "others" for t in tag_list if t not in self.type_map}

        # 2) UI 에서 직접 고른 tag -> 대표 tag 로 변환, 같은 type 의 Tags 입력보다 우선
        for ty in NAS:
            v = selection.get(ty, AUTO)
            if v in (AUTO, NONE, "", None):
                continue
            c = self._canon(str(v).replace("_", " "))
            include_dict = {k: t2 for k, t2 in include_dict.items() if t2 != ty}
            include_dict[c] = ty

        # 3) 자동으로 채울 type: auto 이면서 아직 비어있는 것
        filled = set(include_dict.values())
        auto_types = [ty for ty in NAS
                      if selection.get(ty, AUTO) in (AUTO, "", None) and ty not in filled]
        rng.shuffle(auto_types)   # 항상 같은 type 이 먼저 뽑혀서 쏠리는 것 방지

        # 4) 하나 뽑을 때마다 조건에 추가 -> 뒤에 뽑는 tag 가 앞선 선택과 어울리게
        cond = list(include_dict) + [nsfw_str]
        auto_dict = {}
        for ty in auto_types:
            tag = self._sample_one(self._scores(cond), ty, rng, top_n, temperature)
            if tag is not None and tag not in include_dict:
                auto_dict[tag] = ty
                cond.append(tag)

        total = {**include_dict, **auto_dict, **exclude_dict}

        text = ""
        for key in NAS:
            tags = [t for t, ty in total.items() if ty == key]
            if tags:
                text += f"{key}: {', '.join(tags)}; "
        for key in sorted(set(total.values()) - set(NAS)):
            tags = [t for t, ty in total.items() if ty == key]
            text += f"{key}: {', '.join(tags)}; "

        return INIT_PROMPT + nsfw_text + size_text + text


def test_code():
    tag_co = TagsCoOC(base_root=r"C:\Users\HONG\Desktop\inquiry\mmdit_dcae_sampling")
    print(tag_co.tags_mapping_worker(user_tags=["female", "girl", "solo"], nsfw=False, H=1024, W=768, seed=0))
    print(tag_co.tags_mapping_worker(user_tags=["female", "girl", "solo"], nsfw=False, H=1024, W=768, seed=0,
                                     selection={"gender": "man"}))