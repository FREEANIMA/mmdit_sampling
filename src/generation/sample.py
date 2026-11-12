import os
from tqdm.auto import tqdm
from datetime import datetime
import torch
from data import TagsCoOC
from src.utilities import (
    euler, heun, ddim, ddpm_plus
)

INIT_PROMPT = "general img; "

class Sampling:
    def __init__(self, base_root, model, text_encoder, vae):
        self.base_root = base_root
        self.cooc = TagsCoOC(base_root=base_root)
        self.shift: float = 3.0
        self.device = "cuda"
        self.dtype=torch.bfloat16
        self.f, self.in_channels, self.patch_size = 32, 32, 2
        self.model = model
        self.text_encoder = text_encoder
        self.vae = vae

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
    def steps_sampling(self, x, steps, calculate_type, cfg_scale, cond_emb, uncond_emb, cond_mask, uncond_mask, gen, sigmas):
        self.model.eval()

        try:
            for i in tqdm(range(steps), desc="sampling"):
                sigma, sigma_next = sigmas[i], sigmas[i + 1]

                v, _ = self.predict(
                    model=self.model,
                    x=x, sigma=sigma,
                    cond_emb=cond_emb, cond_mask=cond_mask,
                    uncond_emb=uncond_emb, uncond_mask=uncond_mask,
                    cfg_scale=cfg_scale,
                )

                if calculate_type == "euler":
                    x = euler(x, v, sigma, sigma_next)

                elif calculate_type == "ddim":
                    x = ddim(x, v, sigma, sigma_next)

                elif calculate_type == "heun":
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

                elif calculate_type == "ddpm_plus":
                    z = torch.randn(
                        x.shape, generator=gen,
                        device=x.device, dtype=x.dtype
                    )
                    x = ddpm_plus(x, v, z, sigma, sigma_next)
                    del z

                del v

        finally:
            torch.cuda.empty_cache()

        return x
    
    @torch.inference_mode()
    def vae_decode(self, x):
        with torch.inference_mode():
            decoded = self.vae.decoder(x)        

        img = self.vae._tensor_to_pil(decoded)
        return img

    @torch.inference_mode()
    def sampling(self, user_tags, nsfw, calculate_type, cfg_scale, H, W, steps, seed=None):       
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

        x = self.steps_sampling(
            x=x,
            steps=steps,
            calculate_type=calculate_type,
            cfg_scale=cfg_scale,
            cond_emb=cond_emb,
            cond_mask=cond_mask,
            uncond_emb=uncond_emb,
            uncond_mask=uncond_mask,
            gen=gen,
            sigmas=sigmas,
        )
        img = self.vae_decode(x)

        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_dir = os.path.join(self.base_root, "generation", stamp)
        os.makedirs(save_dir, exist_ok=True)
        save_img_name = os.path.join(save_dir, "img.png")

        img.save(save_img_name)
        print(f"img saved: {save_img_name}")
        
        return img, save_img_name