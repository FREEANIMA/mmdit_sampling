import torch
import torch.nn as nn

class Rope2D(nn.Module):
    def __init__(self, rope_hz, check=False):
        super().__init__()
        self.rope_hz = rope_hz
        self.check = check

    def _rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)

    def _cos_sin(self, n_pos: torch.Tensor, inv_freq: torch.Tensor):
        f = torch.outer(n_pos, inv_freq)
        f = torch.cat((f, f), dim=-1)
        return f.cos()[None, None], f.sin()[None, None]

    def forward(self, q: torch.Tensor, k: torch.Tensor, H: int, W: int) -> tuple[torch.Tensor, torch.Tensor]:
        B, nh, N, D = q.shape
        assert N == H * W, f"토큰 수 {N} != H*W {H * W}"
        assert D % 4 == 0, f"head_dim({D})은 4의 배수여야 함"

        d = D // 2
        dev = q.device

        with torch.autocast(device_type=dev.type, enabled=False):
            inv_freq = 1.0 / (
                self.rope_hz ** (torch.arange(0, d, 2, device=dev, dtype=torch.float32) / d)
            )
            h_idx = torch.arange(H, device=dev, dtype=torch.float32)[:, None].expand(H, W).reshape(-1)
            w_idx = torch.arange(W, device=dev, dtype=torch.float32)[None, :].expand(H, W).reshape(-1)

            cos_h, sin_h = self._cos_sin(h_idx, inv_freq)
            cos_w, sin_w = self._cos_sin(w_idx, inv_freq)

        def rot(x):
            xf = x.float()
            x_h, x_w = xf.chunk(2, dim=-1)
            x_h = x_h * cos_h + self._rotate_half(x_h) * sin_h
            x_w = x_w * cos_w + self._rotate_half(x_w) * sin_w
            return torch.cat((x_h, x_w), dim=-1).to(x.dtype)

        if self.check:
            q_out = rot(q)
            k_out = rot(k)

            for name, before, after in (
                ("Q", q, q_out),
                ("K", k, k_out)
            ):
                before = before.float()
                after = after.float()

                cosine = torch.nn.functional.cosine_similarity(
                    before.flatten(),
                    after.flatten(),
                    dim=0
                ).item()

                print(f"\n[{name}]")
                print(f"Before: mean={before.mean().item():.6f}, std={before.std().item():.6f}")
                print(f"After : mean={after.mean().item():.6f}, std={after.std().item():.6f}")
                print(f"Mean abs diff: {(after - before).abs().mean().item():.6f}")
                print(f"Max abs diff : {(after - before).abs().max().item():.6f}")
                print(f"Cosine sim   : {cosine:.6f}")

            return q_out, k_out

        return rot(q), rot(k)