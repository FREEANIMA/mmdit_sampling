import torch
from networks import load_mmdit_dcae
from data import TagsCoOC
from .caclulater import (
    euler, heun, ddim, ddpm_plus
)

import os
from tqdm.auto import tqdm
from datetime import datetime
from common import T5TextModel
from common import VaeModel
from .attn_stats import (
    attn_summary, tag_attn_map, stack_log, print_stats
)

INIT_PROMPT = "general img; "

class Sampling:
    def __init__(self, base_root):
        self.base_root = base_root
        self.cooc = TagsCoOC(base_root=base_root)
        self.shift: float = 3.0
        self.device = "cuda"
        self.dtype=torch.bfloat16
        self.f, self.in_channels, self.patch_size = 32, 32, 2
        self.model = load_mmdit_dcae()
        
        self.model = self.model.to(device=self.device, dtype=self.dtype)   # prepare_offload 대신
        self.text_encoder = T5TextModel(device=self.device, dtype=self.dtype)
        self.vae = VaeModel(device=self.device, dtype=torch.float32)

    def sigma_schedular(self, step, device):
        t = torch.linspace(0.999, 1e-3, step + 1, device=device)
        s = self.shift
        return s * t / (1 + (s - 1) * t)

    def _snap(self, v, unit):
        return max(round(v / unit), 1) * unit

    @torch.inference_mode()
    def predict(self, model, x, sigma, cond_emb, cond_mask, uncond_emb, uncond_mask, cfg_scale, collect=True):
        x_in = x.to(self.dtype)
        sig = sigma.expand(x.shape[0])
        with torch.amp.autocast(device_type="cuda", dtype=self.dtype):
            cond_out, cond_stats = model.forward(
                x=x_in, sigma=sig, txt_emb=cond_emb, txt_mask=cond_mask,
                attn_collects_stats=collect,
            )
            if cfg_scale == 1.0:
                return cond_out.float(), cond_stats
            uncond_out, _ = model.forward(
                x=x_in, sigma=sig, txt_emb=uncond_emb, txt_mask=uncond_mask,
                attn_collects_stats=False,   # uncond는 통계 불필요 -> SDPA 경로(빠름)
            )
        c, u = cond_out.float(), uncond_out.float()
        return u + cfg_scale * (c - u), cond_stats

    @torch.inference_mode()
    def text_encode(self, tags):
        cond_emb, cond_mask = self.text_encoder.encoder(tags)
        uncond_emb, uncond_mask = self.text_encoder.encoder([INIT_PROMPT])

        return cond_emb, uncond_emb, cond_mask, uncond_mask
        
    @torch.inference_mode()
    def steps_sampling(self, x, steps, caculate_type, cfg_scale, cond_emb, uncond_emb, cond_mask, uncond_mask, gen, sigmas, collect_stats=True):
        self.model.eval()

        MAX = self.model.max_side
        n_txt = int(cond_mask.sum().item())
        h_tok, w_tok = x.shape[2] // self.patch_size, x.shape[3] // self.patch_size
        log = []   # log[step] = {"step", "sigma", "blocks": {block: summary}}
        
        try:
            for i in tqdm(range(steps), desc="sampling"):
                sigma, sigma_next = sigmas[i], sigmas[i + 1]

                v, stats = self.predict(
                    model=self.model,
                    x=x, sigma=sigma,
                    cond_emb=cond_emb, cond_mask=cond_mask,
                    uncond_emb=uncond_emb, uncond_mask=uncond_mask,
                    cfg_scale=cfg_scale,
                    collect=collect_stats,
                )

                # 통계는 매 step 첫 번째 predict(cond)에서만 수집
                if collect_stats:
                    row = {
                        b: attn_summary(s["attn_weights"], h_tok, w_tok, MAX, n_txt)
                        for b, s in stats.items()
                    }
                    log.append({"step": i, "sigma": float(sigma), "blocks": row})
                del stats

                if caculate_type == "euler":
                    x = euler(x, v, sigma, sigma_next)

                elif caculate_type == "ddim":
                    x = ddim(x, v, sigma, sigma_next)

                elif caculate_type == "heun":
                    if i == steps - 1:
                        x = euler(x, v, sigma, sigma_next)
                    else:
                        x_e = euler(x, v, sigma, sigma_next)

                        v2, _ = self.predict(
                            model=self.model,
                            x=x_e, sigma=sigma_next,
                            cond_emb=cond_emb, cond_mask=cond_mask,
                            uncond_emb=uncond_emb, uncond_mask=uncond_mask,
                            cfg_scale=cfg_scale,
                            collect=False,
                        )

                        x = heun(x, v, v2, sigma, sigma_next)
                        del x_e, v2

                elif caculate_type == "ddpm_plus":
                    z = torch.randn(
                        x.shape, generator=gen,
                        device=x.device, dtype=x.dtype
                    )
                    x = ddpm_plus(x, v, z, sigma, sigma_next)
                    del z

                else:
                    raise ValueError(f"Unsupported sampling method: {caculate_type}")

                del v

        finally:
            torch.cuda.empty_cache()

        return x, log
    
    @torch.inference_mode()
    def vae_decode(self, x):
        with torch.inference_mode():
            decoded = self.vae.decoder(x)        

        img = self.vae._tensor_to_pil(decoded)
        return img

    @torch.inference_mode()
    def sampling(self, user_tags, nsfw, caculate_type, cfg_scale, H, W, steps, seed=None, collect_stats=False):       
        unit = self.f * self.patch_size

        if H >= W:
            scale = 1024 / H
            H, W = 1024, self._snap(W * scale, unit=unit)
        else:
            scale = 1024 / W
            H, W = self._snap(H * scale, unit=unit), 1024

        tags = self.cooc.tags_mapping_worker(
            user_tags=user_tags, 
            nsfw=nsfw,
            H=H, W=W
        )

        print(f"tags: {tags}")
        
        cond_emb, uncond_emb, cond_mask, uncond_mask = self.text_encode(tags=tags)
        print(f"cond_emb: {cond_emb.shape}")
        print(f"uncond_emb: {uncond_emb.shape}")


        gen = torch.Generator(device=self.device)
        if seed is not None:
            gen.manual_seed(seed)
        else:
            gen.seed()

        # dc ae 전용
        noise = torch.randn(
            (1, self.in_channels, H // self.f, W // self.f),
            generator=gen,
            device=self.device,
            dtype=torch.float32,
        )

        x = noise
        sigmas = self.sigma_schedular(step=steps, device=self.device)

        x, log = self.steps_sampling(
            x=x,
            steps=steps,
            caculate_type=caculate_type,
            cfg_scale=cfg_scale,
            cond_emb=cond_emb,
            cond_mask=cond_mask,
            uncond_emb=uncond_emb,
            uncond_mask=uncond_mask,
            gen=gen,
            sigmas=sigmas,
            collect_stats=collect_stats,
        )
        img = self.vae_decode(x)


        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_dir = os.path.join(self.base_root, "generation", stamp)
        os.makedirs(save_dir, exist_ok=True)
        save_img_name = os.path.join(save_dir, "img.png")

        img.save(save_img_name)
        print(f"img saved: {save_img_name}")

        if collect_stats:
            stacked = stack_log(log)
            print_stats(stacked, stride=1)
        
        return img, save_img_name