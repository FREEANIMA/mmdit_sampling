import os
import pandas as pd

INIT_PROMPT = "general img; "
NAS = [
    "style", "person_count", "gender", "hair_color", "hair_length", "hair_style",
    "eye_color","body_type", "clothing", "accessory", "background", "pose", "expression", "composition",
    "gaze", "sex_act"
]

class TagsCoOC:
    def __init__(self, base_root):
        vocab_matrix_path = os.path.join(base_root, "data", "metadata", "vocab_matrix.parquet")
        tags_type_path = os.path.join(base_root, "data", "metadata", "tags_type.parquet")
        self.co_df = pd.read_parquet(vocab_matrix_path)
        self.tags_type_df = pd.read_parquet(tags_type_path)        

    def recommend(self, user_tags, top_k=30, special_top_k=500, debug=False):
        valid = [t for t in user_tags if t in self.co_df.index]
        if not valid:
            return []

        SPECIAL = {"general", "nsfw"}

        per_tag_sets = []
        for t in valid:
            row = self.co_df.loc[t].drop(labels=valid, errors="ignore")
            row = row[row > 0].sort_values(ascending=False)
            k = special_top_k if t in SPECIAL else top_k   # special은 넓게
            per_tag_sets.append(set(row.head(k).index))

        common = set.intersection(*per_tag_sets) if per_tag_sets else set()

        scores = self.co_df.loc[valid].min(axis=0)
        scores = scores[scores.index.isin(common)]
        scores = scores.sort_values(ascending=False).head(top_k)

        if debug:
            print("valid:", valid)
            for t, s in zip(valid, per_tag_sets):
                print(f"[{t}] set size={len(s)}")
            print("common size:", len(common))
            print("top:", scores.index.tolist()[:20])

        return scores.index.tolist()
    
    def tags_mapping_worker(self, user_tags: list, nsfw, H, W):
        nsfw_text = f"nsfw: {str(nsfw).lower()}; "
        size_text = f"size: {H}x{W}; "
        nsfw_str = "nsfw" if nsfw is True else "general"

        tag_list = [t.replace("_", " ") for t in user_tags]

        mapping = dict(zip(self.tags_type_df["tag"], self.tags_type_df["dup"]))
        type_mapping = dict(zip(self.tags_type_df["tag"], self.tags_type_df["type"]))
        
        # vocab에 있는 태그
        include_tags = [t for t in tag_list if t in mapping]
        include_type = [type_mapping[t] for t in tag_list if t in type_mapping]
        include_tags_dict = {t: type_mapping[t] for t in tag_list if t in type_mapping}

        # 한번도 보지 못한 태그
        exclude_tags_dict = {t: "others" for t in tag_list if t not in mapping}

        # 한번도 사용하지 않은 필수 type
        unused_type = [x for x in NAS if x not in include_type]

        # co oc로 측정한 연관된 top tag (자기 자신 제외)
        mapped_include = [mapping[t] for t in include_tags] + [nsfw_str]
        top_tags_list = self.recommend(user_tags=mapped_include, top_k=100, special_top_k=500 ,debug=False)

        # top tag 중 type별 하나 씩 필터
        top_tags_type_dict = {t: type_mapping[t] for t in top_tags_list if t in type_mapping}

        unused_set = set(unused_type)

        seen = set()
        dedup_dict = {}
        for t, ty in top_tags_type_dict.items():
            if ty in unused_set and ty not in seen:
                seen.add(ty)
                dedup_dict[t] = ty

        total = {**include_tags_dict, **dedup_dict, **exclude_tags_dict}

        text = ""
        for key in NAS:
            tags = [t for t, ty in total.items() if ty == key]
            if tags:
                text += f"{key}: {', '.join(tags)}; "
                        
        for key in set(total.values()) - set(NAS):
            tags = [t for t, ty in total.items() if ty == key]
            text += f"{key}: {', '.join(tags)}; "

        total = INIT_PROMPT + nsfw_text + size_text + text

        return total

def test_code():
    tag_co = TagsCoOC(base_root=r"C:\Users\HONG\Desktop\inquiry\mmdit_dcae_sampling")
    user_tags = ["woman", "solo"]
    total = tag_co.tags_mapping_worker(user_tags=user_tags, nsfw=False, H="1024", W="768")
    print(total)
