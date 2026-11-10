import torch
import torch.nn.functional as F
import numpy as np
import os
from datetime import datetime
from tqdm.auto import tqdm

from networks import load_mmdit_dcae
from data import TagsCoOC
from .caclulater import euler, heun, ddim, ddpm_plus
from .resize_worker import img_resize
from .draw_colab import draw_mask_colab
from common import T5TextModel, VaeModel

INIT_PROMPT = "general img; "

class InPaint:
    def __init__(self, base_root):
        self.base_root = base_root
        self.cooc = TagsCoOC(base_root=base_root)
        self.shift: float = 3.0
        self.device = "cuda"
        self.dtype = torch.bfloat16
        self.f, self.in_channels, self.patch_size = 32, 32, 2

        self.model = load_mmdit_dcae().to(device=self.device, dtype=self.dtype)
        self.text_encoder = T5TextModel(device=self.device, dtype=self.dtype)
        self.vae = VaeModel(device=self.device, dtype=torch.float32)

    # ---------------- sigma ----------------
    def sigma_schedular(self, step, device):
        t = torch.linspace(0.999, 1e-3, step + 1, device=device)
        s = self.shift
        return s * t / (1 + (s - 1) * t)

    # ---------------- VAE ----------------
    @torch.inference_mode()
    def vae_encode(self, img):
        return self.vae.encoder(img)          # ✅ 수정

    @torch.inference_mode()
    def vae_decode(self, latent):
        decoded = self.vae.decoder(latent)
        return self.vae._tensor_to_pil(decoded)

    # ---------------- text ----------------
    @torch.inference_mode()
    def text_encode(self, tags):
        cond_emb, cond_mask = self.text_encoder.encoder(tags)
        uncond_emb, uncond_mask = self.text_encoder.encoder([INIT_PROMPT])
        return cond_emb, uncond_emb, cond_mask, uncond_mask

    # ---------------- predict (CFG) ----------------
    @torch.inference_mode()
    def predict(self, model, x, sigma, cond_emb, cond_mask,
                uncond_emb, uncond_mask, cfg_scale):
        x_in = x.to(self.dtype)
        sig = sigma.expand(x.shape[0])
        with torch.amp.autocast(device_type="cuda", dtype=self.dtype):
            cond_out, _ = model.forward(
                x=x_in, sigma=sig, txt_emb=cond_emb, txt_mask=cond_mask,
            )
            if cfg_scale == 1.0:
                return cond_out.float()
            uncond_out, _ = model.forward(
                x=x_in, sigma=sig, txt_emb=uncond_emb, txt_mask=uncond_mask,
            )
        c, u = cond_out.float(), uncond_out.float()
        return u + cfg_scale * (c - u)

    @torch.inference_mode()
    def steps_sampling(self, x, sigmas, latent, noise, mask_lat, caculate_type, cfg_scale, cond_emb, cond_mask, uncond_emb, uncond_mask, gen,):
        self.model.eval()
        n = len(sigmas) - 1

        try:
            for i in tqdm(range(n), desc="sampling"):
                sigma, sigma_next = sigmas[i], sigmas[i + 1]

                v = self.predict(
                    model=self.model,
                    x=x, sigma=sigma,
                    cond_emb=cond_emb, cond_mask=cond_mask,
                    uncond_emb=uncond_emb, uncond_mask=uncond_mask,
                    cfg_scale=cfg_scale,
                )

                if caculate_type == "euler":
                    x = euler(x, v, sigma, sigma_next)
                elif caculate_type == "ddim":
                    x = ddim(x, v, sigma, sigma_next)
                elif caculate_type == "heun":
                    if i == n - 1:
                        x = euler(x, v, sigma, sigma_next)
                    else:
                        x_e = euler(x, v, sigma, sigma_next)
                        v2 = self.predict(
                            model=self.model,
                            x=x_e, sigma=sigma_next,
                            cond_emb=cond_emb, cond_mask=cond_mask,
                            uncond_emb=uncond_emb, uncond_mask=uncond_mask,
                            cfg_scale=cfg_scale,
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
                sigma_view = sigma.view(1, 1, 1, 1)
                orig_latent_t = (1 - sigma_view) * latent + sigma_view * noise
                x = (1 - mask_lat) * orig_latent_t + mask_lat * x

        finally:
            torch.cuda.empty_cache()

        return x

    def mask_generation(self, mask_arr, latent):
        if isinstance(mask_arr, np.ndarray):
            m = torch.from_numpy(mask_arr).float()
        else:
            m = mask_arr.float()

        if m.max() > 1.0:
            m = m / 255.0

        if m.ndim == 2:
            m = m[None, None]
        elif m.ndim == 3:
            m = m[None]

        m = m.to(latent.device, latent.dtype)
        m_lat = F.interpolate(m, size=latent.shape[-2:], mode="nearest")
        return (m_lat > 0.5).float()

    @torch.inference_mode()
    def sampling(self, user_tags, nsfw, caculate_type, cfg_scale, steps, seed=None, path=None):
        if path is None:
            path = input("경로를 입력해 주세요 : ")        
        with open(path, "rb") as f:
            img_bytes = f.read()

        arr, H, W = img_resize(img_bytes, 1024, 2, 32)
        mask_arr = draw_mask_colab(arr, brush=30)

        img = torch.from_numpy(arr).permute(2, 0, 1).float().div_(255.0).unsqueeze(0)
        img = img.to(device=self.device, dtype=self.dtype)

        latent = self.vae_encode(img)                      # (1,C,h,w)
        mask_lat = self.mask_generation(mask_arr, latent)  # (1,1,h,w)

        noise = torch.randn_like(latent)

        # ✅ 수정: sigmas를 먼저 자르고, x_t를 start_t 레벨의 노이즈로 초기화
        sigmas = self.sigma_schedular(step=steps, device=self.device)
        start_t = 0.6
        start_idx = int(torch.argmin((sigmas - start_t).abs()).item())
        sigmas = sigmas[start_idx:]
        start_sigma = sigmas[0]                            # 실제 시작 sigma

        # 마스크 영역을 start_sigma 레벨의 노이즈로 초기화 (불일치 해소)
        orig_at_start = (1 - start_sigma) * latent + start_sigma * noise
        x_t = (1 - mask_lat) * latent + mask_lat * orig_at_start

        tags = self.cooc.tags_mapping_worker(
            user_tags=user_tags, nsfw=nsfw, H=H, W=W
        )
        cond_emb, uncond_emb, cond_mask, uncond_mask = self.text_encode(tags=tags)

        gen = torch.Generator(device=self.device)
        if seed is not None:
            gen.manual_seed(seed)

        x = self.steps_sampling(
            x=x_t,
            sigmas=sigmas,
            latent=latent,
            noise=noise,
            mask_lat=mask_lat,
            caculate_type=caculate_type,
            cfg_scale=cfg_scale,
            cond_emb=cond_emb, cond_mask=cond_mask,
            uncond_emb=uncond_emb, uncond_mask=uncond_mask,
            gen=gen,
        )

        img_out = self.vae_decode(x)

        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_dir = os.path.join(self.base_root, "inpainting", stamp)
        os.makedirs(save_dir, exist_ok=True)
        save_img_name = os.path.join(save_dir, "img.png")
        img_out.save(save_img_name)
        print(f"img saved: {save_img_name}")

        return img_out, save_img_name