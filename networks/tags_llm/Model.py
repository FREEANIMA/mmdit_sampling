import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint

from .Block import Block
from .Rope1D import Rope1D

class Model(nn.Module):
    def __init__(self, hidden_size, vocab_size, nhead, num_layers, rope_hz, drop_out, max_len, **kwargs):
        super().__init__()
        dim_forward = 4 * hidden_size

        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.rope = Rope1D(head_dim=hidden_size // nhead, rope_hz=rope_hz, max_len=max_len)

        self.blocks = nn.ModuleList([
            Block(
                hidden_size=hidden_size, 
                nhead=nhead,
                dim_forward=dim_forward,
                drop_out=drop_out,
                rope=self.rope,
            )
            for _ in range(num_layers)
        ])

        self.final_norm = nn.LayerNorm(hidden_size)
        self.out_proj = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, attn_mask=None, past_kv=None, use_cache=False, use_checkpoint=False):
        x_emb = self.embedding(x)

        emb = x_emb
        offset = 0 if past_kv is None else past_kv[0][0].shape[2]

        new_past = [] if use_cache else None
        for i, block in enumerate(self.blocks):
            if use_checkpoint and self.training and not use_cache:
                emb, _ = checkpoint.checkpoint(
                    block, 
                    emb,
                    attn_mask, None, 0, False,
                    use_reentrant=False)
            else:
                emb, kv = block(
                    emb, 
                    past_kv=None if past_kv is None else past_kv[i],
                    offset=offset, 
                    use_cache=use_cache,
                    attn_mask=attn_mask)
                if use_cache:
                    new_past.append(kv)

        out = self.final_norm(emb)
        logits = self.out_proj(out)
        return (logits, new_past) if use_cache else logits