import torch
import torch.nn as nn
import torch.nn.functional as F
from .Rope2D import Rope2D

class Block(nn.Module):
    def __init__(self, hidden_size, nhead, dim_feedforward, rope:Rope2D, dropout, **kwargs):
        super().__init__()

        self.nhead = nhead
        self.hidden_size = hidden_size
        self.head_dim = hidden_size // nhead
        self.rope = rope
        self.dropout = nn.Dropout(dropout)

        self.adaLN_modulation_img = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size * 6)
        )

        self.adaLN_modulation_txt = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size * 6)
        )

        self.img_norm = nn.RMSNorm(hidden_size, elementwise_affine=False)
        self.img_qkv = nn.Linear(hidden_size, hidden_size * 3)
        self.img_out_proj = nn.Linear(hidden_size, hidden_size)
        self.img_q_norm = nn.RMSNorm(self.head_dim)
        self.img_k_norm = nn.RMSNorm(self.head_dim)

        self.txt_norm = nn.RMSNorm(hidden_size, elementwise_affine=False)
        self.txt_qkv = nn.Linear(hidden_size, hidden_size * 3)
        self.txt_out_proj = nn.Linear(hidden_size, hidden_size)
        self.txt_q_norm = nn.RMSNorm(self.head_dim)
        self.txt_k_norm = nn.RMSNorm(self.head_dim)

        self.ffn_img_norm = nn.RMSNorm(hidden_size, elementwise_affine=False)
        self.ffn_img_in = nn.Linear(hidden_size, dim_feedforward)
        self.ffn_img_out = nn.Linear(dim_feedforward, hidden_size)

        self.ffn_txt_norm = nn.RMSNorm(hidden_size, elementwise_affine=False)
        self.ffn_txt_in = nn.Linear(hidden_size, dim_feedforward)
        self.ffn_txt_out = nn.Linear(dim_feedforward, hidden_size)

        nn.init.zeros_(self.adaLN_modulation_img[-1].weight)
        nn.init.zeros_(self.adaLN_modulation_img[-1].bias)
        nn.init.zeros_(self.adaLN_modulation_txt[-1].weight)
        nn.init.zeros_(self.adaLN_modulation_txt[-1].bias)

    def _split_heads(self, x):
        b, n, _ = x.shape
        return x.view(b, n, self.nhead, self.head_dim).transpose(1, 2)
            
    def joint_attention(self, img_norm, txt_norm, H=None, W=None, attn_mask=None, attn_collects_stats=False):
        B, N_img, _ = img_norm.shape
        B, N_txt, _ = txt_norm.shape

        q_img, k_img, v_img = self.img_qkv(img_norm).chunk(3, dim=-1)
        q_img, k_img, v_img = map(self._split_heads, (q_img, k_img, v_img))
        q_img, k_img = self.img_q_norm(q_img), self.img_k_norm(k_img)
        q_img, k_img = self.rope.forward(
            q=q_img,
            k=k_img,
            H=H,
            W=W
        )

        q_txt, k_txt, v_txt = self.txt_qkv(txt_norm).chunk(3, dim=-1)
        q_txt, k_txt, v_txt = map(self._split_heads, (q_txt, k_txt, v_txt))
        q_txt, k_txt = self.txt_q_norm(q_txt), self.txt_k_norm(k_txt)

        Q = torch.cat([q_img, q_txt], dim=2)
        K = torch.cat([k_img, k_txt], dim=2)
        V = torch.cat([v_img, v_txt], dim=2)

        attn_weights = None
        if attn_collects_stats:
            scale = 1.0 / (self.head_dim ** 0.5)
            scores = (Q.float() @ K.float().transpose(-2, -1)) * scale
            if attn_mask is not None:
                scores = scores + attn_mask.float()
            attn_weights = scores.softmax(dim=-1)
            attn_out = attn_weights.to(V.dtype) @ V
        else:
            if attn_mask is not None:
                attn_mask = attn_mask.to(dtype=Q.dtype)

            attn_out = F.scaled_dot_product_attention(
                Q, K, V,
                attn_mask=attn_mask, 
                dropout_p=0.0 # fixed
            )

        attn_out = attn_out.transpose(1, 2).reshape(B, N_img + N_txt, self.hidden_size)
        img_out = attn_out[:, :N_img]
        txt_out = attn_out[:, N_img:]
        attn_img = self.img_out_proj(img_out)
        attn_txt = self.txt_out_proj(txt_out)

        return attn_img, attn_txt, attn_weights
    
    def forward(self, img_emb: torch.Tensor, vec: torch.Tensor,  txt_emb: torch.Tensor=None, H=None, W=None, attn_mask=None, collects_stats=False, attn_collects_stats=False):
        (shift_msa_i, scale_msa_i, gate_msa_i,
         shift_mlp_i, scale_mlp_i, gate_mlp_i) = self.adaLN_modulation_img(vec).chunk(6, dim=-1)

        (shift_msa_t, scale_msa_t, gate_msa_t,
         shift_mlp_t, scale_mlp_t, gate_mlp_t) = self.adaLN_modulation_txt(vec).chunk(6, dim=-1)
                
        if collects_stats:
            origin_img_std = img_emb.std().item()

        img_norm = self.img_norm(img_emb) * (1 + scale_msa_i[:, None, :]) + shift_msa_i[:, None, :]
        txt_norm = self.txt_norm(txt_emb) * (1 + scale_msa_t[:, None, :]) + shift_msa_t[:, None, :]
    
        attn_img, attn_txt, attn_weights = self.joint_attention(
            img_norm, txt_norm, 
            H=H, W=W, 
            attn_mask=attn_mask, 
            attn_collects_stats=attn_collects_stats
        )

        img_emb = img_emb + gate_msa_i[:, None, :] * self.dropout(attn_img)
        txt_emb = txt_emb + gate_msa_t[:, None, :] * self.dropout(attn_txt)

        img_ffn_in = self.ffn_img_norm(img_emb) * (1 + scale_mlp_i[:, None, :]) + shift_mlp_i[:, None, :]
        img_ffn_out = self.ffn_img_out(F.gelu(self.ffn_img_in(img_ffn_in)))
        img_emb = img_emb + gate_mlp_i[:, None, :] * self.dropout(img_ffn_out)

        txt_ffn_in = self.ffn_txt_norm(txt_emb) * (1 + scale_mlp_t[:, None, :]) + shift_mlp_t[:, None, :]
        txt_ffn_out = self.ffn_txt_out(F.gelu(self.ffn_txt_in(txt_ffn_in)))
        txt_emb = txt_emb + gate_mlp_t[:, None, :] * self.dropout(txt_ffn_out)

        stats = {}
        if collects_stats:
            stats = {
                "origin_img": origin_img_std,
                "img_emb": img_emb.std().item(),
            }

        if attn_collects_stats:
            stats["attn_weights"] = attn_weights.detach()

        return img_emb, txt_emb, stats