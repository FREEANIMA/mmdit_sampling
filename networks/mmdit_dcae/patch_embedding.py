import torch
import torch.nn as nn

class PatchEmbedding(nn.Module):
    def __init__(self, in_channels: int, patch_size: int, d_model: int):
        super().__init__()
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.proj = nn.Conv2d(
            in_channels,
            d_model,
            kernel_size=patch_size,
            stride=patch_size
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        x = x.flatten(2)
        x = x.transpose(1, 2).contiguous()
        return x

    def patchify(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        p = self.patch_size
        x = x.reshape(B, C, H // p, p, W // p, p)
        x = x.permute(0, 2, 4, 1, 3, 5).contiguous()
        x = x.reshape(B, -1, p * p * C)
        return x

    def unpatchify(self, x: torch.Tensor, H, W, out_channels: int = None):
        B = x.shape[0]
        p = self.patch_size
        C = out_channels if out_channels is not None else self.in_channels
        h, w = H // p, W // p
        x = x.reshape(B, h, w, C, p, p)
        x = x.permute(0, 3, 1, 4, 2, 5).contiguous()
        x = x.reshape(B, C, h * p, w * p)
        return x