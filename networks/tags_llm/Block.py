import torch
import torch.nn as nn
import torch.nn.functional as F

from .Rope1D import Rope1D

class Block(nn.Module):
    def __init__(self, hidden_size, nhead, dim_forward, drop_out, rope: Rope1D):
        super().__init__()
        assert hidden_size % nhead == 0
        self.nhead = nhead
        self.head_dim = hidden_size // nhead
        self.rope = rope
        self.p = drop_out

        # self attn
        self.emb_norm = nn.RMSNorm(hidden_size)
        self.qkv = nn.Linear(hidden_size, hidden_size * 3)
        self.q_norm = nn.RMSNorm(hidden_size)
        self.k_norm = nn.RMSNorm(hidden_size)
        self.self_out_proj = nn.Linear(hidden_size, hidden_size)

        # ffn
        self.ffn_norm = nn.RMSNorm(hidden_size)
        self.ffn_in = nn.Linear(hidden_size, dim_forward)
        self.ffn_out = nn.Linear(dim_forward, hidden_size)

    def _split_heads(self, x):
        b, n, _ = x.shape
        return x.view(b, n, self.nhead, self.head_dim).transpose(1, 2)

    def _merge_heads(self, x):
        b, h, n, d = x.shape
        return x.transpose(1, 2).reshape(b, n, h * d)

    def self_attention(self, x, attn_mask, past_kv=None, offset=0, use_cache=False):
        q, k, v = self.qkv(x).chunk(3, -1)
        q, k = self.q_norm(q), self.k_norm(k)
        Q, K, V = map(self._split_heads, (q, k, v))
        Q, K = self.rope(Q, K, offset=offset)

        if past_kv is not None:                   
            K = torch.cat([past_kv[0], K], dim=2)
            V = torch.cat([past_kv[1], V], dim=2)
        new_kv = (K, V) if use_cache else None

        is_causal = attn_mask is None and Q.shape[2] > 1

        out = F.scaled_dot_product_attention(
            Q, K, V,
            is_causal=is_causal,
            attn_mask=attn_mask,
            dropout_p=self.p if self.training else 0.0,
        )
        return self.self_out_proj(self._merge_heads(out)), new_kv

    def forward(self, x, attn_mask, past_kv=None, offset=0, use_cache=False):
        a, new_kv = self.self_attention(self.emb_norm(x), attn_mask, past_kv, offset, use_cache)
        x = x + a
        x = x + self.ffn_out(F.gelu(self.ffn_in(self.ffn_norm(x))))
        return x, new_kv