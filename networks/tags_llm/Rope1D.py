import torch
import torch.nn as nn

class Rope1D(nn.Module):
    def __init__(self, head_dim, rope_hz=10000.0, max_len=8192):
        super().__init__()
        assert head_dim % 2 == 0, f"head_dim({head_dim})은 짝수여야 함"
        self.head_dim = head_dim
        self.rope_hz = rope_hz
        self.max_len = max_len
        self._build(max_len)

    def _build(self, n):
        inv_freq = 1.0 / (
            self.rope_hz ** (torch.arange(0, self.head_dim, 2, dtype=torch.float32) / self.head_dim)
        )
        pos = torch.arange(n, dtype=torch.float32)
        f = torch.outer(pos, inv_freq)
        f = torch.cat((f, f), dim=-1)
        cos, sin = f.cos(), f.sin()
        if hasattr(self, "cos"):
            dev = self.cos.device
            self.cos, self.sin = cos.to(dev), sin.to(dev)
        else:
            self.register_buffer("cos", cos, persistent=False)
            self.register_buffer("sin", sin, persistent=False)

    @staticmethod
    def _rotate_half(x):
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)

    def forward(self, q, k, offset: int = 0):
        # q, k: [B, nh, L, D]
        L = q.shape[-2]
        end = offset + L
        if end > self.cos.shape[0]:                         # 길이 초과 시 확장
            self._build(max(end, self.cos.shape[0] * 2))

        cos = self.cos[offset:end][None, None]              # [1, 1, L, D]
        sin = self.sin[offset:end][None, None]

        def rot(x):
            xf = x.float()
            return (xf * cos + self._rotate_half(xf) * sin).to(x.dtype)

        return rot(q), rot(k)