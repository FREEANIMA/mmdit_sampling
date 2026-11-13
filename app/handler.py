from typing import Optional
import torch
from data.cooc import AUTO

from src.generation import Sampling, InPaint
from common import T5TextModel, VaeModel
from networks import load_mmdit_dcae

def _parse_tags(tags_text: str) -> list[str]:
    return [t.strip() for t in tags_text.split(",") if t.strip()]

def _parse_seed(seed) -> Optional[int]:
    return None if seed in (None, "", -1) else int(seed)

class Handler:
    def __init__(self, base_root: str = ".", device: str = "cuda", dtype: torch.dtype = torch.bfloat16,):
        self.base_root = base_root
        self.device = device
        self.dtype = dtype

        self.model, self.config = load_mmdit_dcae()
        self.model = self.model.to(device=device, dtype=dtype)
        self.text_encoder = T5TextModel(device=device, dtype=dtype)
        self.vae = VaeModel(device=device, dtype=torch.float32)

        self._sampler: Optional[Sampling] = None
        self._inpainter: Optional[InPaint] = None


    def get_nas_options(self):
        return self.sampler.cooc.get_nas_options()

    def generate_txt2img(self, tags_text, nsfw, calc_type, cfg_scale, H, W, steps, seed, nas=None):
        return self.sampler.sampling(
            user_tags=_parse_tags(tags_text), nsfw=bool(nsfw), calculate_type=calc_type,
            cfg_scale=float(cfg_scale), H=int(H), W=int(W), steps=int(steps),
            seed=_parse_seed(seed), nas=nas or {},
        )

    @property
    def sampler(self) -> Sampling:
        if self._sampler is None:
            self._sampler = Sampling(base_root=self.base_root, model=self.model, text_encoder=self.text_encoder, vae=self.vae, config=self.config)
        return self._sampler

    @property
    def inpainter(self) -> InPaint:
        if self._inpainter is None:
            self._inpainter = InPaint(base_root=self.base_root, model=self.model, text_encoder=self.text_encoder, vae=self.vae, config=self.config)
        return self._inpainter

    def generate_txt2img(self, tags_text, nsfw, calc_type, cfg_scale, H, W, steps, seed):
        return self.sampler.sampling(
            user_tags=_parse_tags(tags_text),
            nsfw=bool(nsfw),
            calculate_type=calc_type,
            cfg_scale=float(cfg_scale),
            H=int(H),
            W=int(W),
            steps=int(steps),
            seed=_parse_seed(seed),
        )

    def generate_inpainting(self, tags_text, nsfw, calc_type, cfg_scale, steps, seed, img_path, start_t: float = 0.9):
        return self.inpainter.sampling(
            user_tags=_parse_tags(tags_text),
            nsfw=bool(nsfw),
            caculate_type=calc_type,
            cfg_scale=float(cfg_scale),
            steps=int(steps),
            seed=_parse_seed(seed),
            path=img_path,
            start_t=float(start_t),
        )