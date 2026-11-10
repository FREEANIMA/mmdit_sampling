import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
import torch.nn.functional as F

from .patch_embedding import PatchEmbedding
from .sinusoidal_pos_emb import SinusoidalPosEmb
from .Rope2D import Rope2D
from .Block import Block

class Model(nn.Module):
    def __init__(self, hidden_size, nhead, num_layers, dropout, 
                 sigma_emb_hz, in_channels, patch_size, rope_hz, sigma_scale, txt_hidden_size, allowable_size, compression_ratio,
                 check=False, **kwargs):
        super().__init__()
        dim_feedforward = 3 * hidden_size
        self.patch_size = patch_size
        self.max_side = allowable_size // (compression_ratio * patch_size)
        self.in_channels = in_channels
        self.sigma_scale = sigma_scale

        # txt
        self.txt_emb = nn.Linear(txt_hidden_size, hidden_size)
        
        
        # patch
        self.patch_embedding = PatchEmbedding(
            in_channels=in_channels,
            patch_size=patch_size,
            d_model=hidden_size
        )

        self.rope2d = Rope2D(
            rope_hz=rope_hz,
            check=check
        )
        
        # sigma
        self.sigma_proj = SinusoidalPosEmb(
            hidden_size, 
            sigma_emb_hz,
            check
        )
        self.sigma_embed = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size)
        )
        self.sigma_norm = nn.LayerNorm(hidden_size)

        self.blocks = nn.ModuleList([
            Block(
                hidden_size=hidden_size,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                rope=self.rope2d,
            )
            for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(hidden_size)

        self.final_mod = nn.Linear(hidden_size, hidden_size * 2)

        self.output_proj = nn.Linear(
            hidden_size,
            patch_size * patch_size * in_channels
        )

        nn.init.zeros_(self.final_mod.weight)
        nn.init.zeros_(self.final_mod.bias)

    def forward(self, x, sigma, txt_emb, txt_mask=None, use_checkpoint=False, collects_stats=False, attn_collects_stats=False):
        B, C, H, W = x.shape
        p = self.patch_size
        h, w = H // p, W // p
        MAX = self.max_side
        assert h <= MAX and w <= MAX, f"token grid {h}x{w} > {MAX}x{MAX}"

        pad_h, pad_w = (MAX - h) * p, (MAX - w) * p
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h))

        img_emb = self.patch_embedding(x)
        T_img = MAX * MAX
        txt_emb = self.txt_emb(txt_emb)

        sigma = sigma.to(device=x.device, dtype=torch.float32)
        sigma_emb = self.sigma_proj(sigma * self.sigma_scale)
        sigma_norm = self.sigma_norm(self.sigma_embed(sigma_emb))

        # 2) key mask: 유효 img 토큰 + 텍스트 (기존 [img, txt] 순서 유지)
        img_mask = torch.zeros(MAX, MAX, dtype=torch.bool, device=x.device)
        img_mask[:h, :w] = True
        img_mask = img_mask.reshape(1, T_img).expand(B, -1)

        if txt_mask is None:
            txt_mask = torch.ones(B, txt_emb.shape[1], dtype=torch.bool, device=x.device)
        full_mask = torch.cat([img_mask, txt_mask.bool()], dim=1)    # (B, N)

        attn_mask = torch.zeros(full_mask.shape, dtype=img_emb.dtype, device=x.device)
        attn_mask = attn_mask.masked_fill(~full_mask, torch.finfo(img_emb.dtype).min)
        attn_mask = attn_mask[:, None, None, :]                       # (B, 1, 1, N)

        param = {
            "vec"       : sigma_norm,
            "H"         : MAX,
            "W"         : MAX,
            "attn_mask" : attn_mask,
            "collects_stats" : collects_stats,
            "attn_collects_stats" : attn_collects_stats
        }

        _stats = {}
        for key, block in enumerate(self.blocks):
            img_emb, txt_emb, stats = block(
                img_emb=img_emb,
                txt_emb=txt_emb,
                **param
            )
            _stats[key] = stats
            
        shift, scale = self.final_mod(sigma_norm).chunk(2, dim=-1)
        x_final = self.norm(img_emb)
        x_final = x_final * (1 + scale[:, None, :]) + shift[:, None, :]

        pred_velocity = self.output_proj(x_final)
        pred_velocity = self.patch_embedding.unpatchify(
            pred_velocity, MAX * p, MAX * p, out_channels=self.in_channels
        )
        pred_velocity = pred_velocity[:, :, :H, :W]

        return pred_velocity, _stats