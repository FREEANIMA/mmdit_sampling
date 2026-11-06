# app/handler.py
from scripts.sample import Sampling

_sampler = None

def get_sampler(base_root: str = "."):
    global _sampler
    if _sampler is None: 
        _sampler = Sampling(base_root=base_root)
    return _sampler

def generate(base_root, tags_text, nsfw, calc_type, cfg_scale, H, W, steps, seed):
    sampler = get_sampler(base_root=base_root)
    user_tags = [t.strip() for t in tags_text.split(",") if t.strip()]
    seed = None if seed in (None, "", -1) else int(seed)

    img, path = sampler.sampling(
        user_tags=user_tags,
        nsfw=bool(nsfw),
        caculate_type=calc_type,
        cfg_scale=float(cfg_scale),
        H=int(H),
        W=int(W),
        steps=int(steps),
        seed=seed,
    )
    return img, path