# app/handler.py
from src.generation.sample import Sampling
from src.generation.inpainting import InPaint

_sampler = None
_inpainter = None

def get_sampler(base_root="."):
    global _sampler
    if _sampler is None:
        _sampler = Sampling(base_root=base_root)
    return _sampler

def get_inpainter(base_root="."):
    global _inpainter
    if _inpainter is None:
        _inpainter = InPaint(base_root=base_root)
    return _inpainter


def generate_txt2img(base_root, tags_text, nsfw, calc_type, cfg_scale,
                     H, W, steps, seed):
    sampler = get_sampler(base_root=base_root)
    user_tags = [t.strip() for t in tags_text.split(",") if t.strip()]
    seed = None if seed in (None, "", -1) else int(seed)
    return sampler.sampling(
        user_tags=user_tags, nsfw=bool(nsfw),
        caculate_type=calc_type, cfg_scale=float(cfg_scale),
        H=int(H), W=int(W), steps=int(steps), seed=seed,
    )

def generate_inpainting(base_root, tags_text, nsfw, calc_type, cfg_scale,
                        steps, seed, img_path, start_t=0.9): # ✅ 기본값 0.9 추가
    inpainter = get_inpainter(base_root=base_root)
    user_tags = [t.strip() for t in tags_text.split(",") if t.strip()]
    seed = None if seed in (None, "", -1) else int(seed)
    return inpainter.sampling(
        user_tags=user_tags, nsfw=bool(nsfw),
        caculate_type=calc_type, cfg_scale=float(cfg_scale),
        steps=int(steps), seed=seed,
        path=img_path,
        start_t=float(start_t), # ✅ 넘겨주기
    )