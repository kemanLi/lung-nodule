from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SwinTinyLayer(nn.Module):
    """A compact Swin-style window attention layer that preserves C/H/W."""

    def __init__(
        self,
        channels: int,
        num_heads: int = 4,
        window_size: int = 7,
        shift_size: int = 0,
        mlp_ratio: float = 2.0,
    ) -> None:
        super().__init__()
        if channels % num_heads != 0:
            raise ValueError(f"channels={channels} must be divisible by num_heads={num_heads}")
        self.channels = channels
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.head_dim = channels // num_heads
        self.scale = self.head_dim**-0.5

        self.norm1 = nn.LayerNorm(channels)
        self.qkv = nn.Linear(channels, channels * 3, bias=True)
        self.proj = nn.Linear(channels, channels)
        self.norm2 = nn.LayerNorm(channels)
        hidden = int(channels * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.GELU(),
            nn.Linear(hidden, channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        b, c, h, w = x.shape
        x_windows, pad_hw = self._window_partition(x)
        x_windows = self.norm1(x_windows)

        qkv = self.qkv(x_windows).reshape(x_windows.shape[0], -1, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        x_windows = (attn @ v).transpose(1, 2).reshape(x_windows.shape[0], -1, c)
        x_windows = self.proj(x_windows)
        x = self._window_reverse(x_windows, pad_hw, h, w)
        x = shortcut + x

        y = x.flatten(2).transpose(1, 2)
        y = y + self.mlp(self.norm2(y))
        return y.transpose(1, 2).reshape(b, c, h, w)

    def _window_partition(self, x: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int]]:
        if self.shift_size:
            x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(2, 3))
        b, c, h, w = x.shape
        pad_h = (self.window_size - h % self.window_size) % self.window_size
        pad_w = (self.window_size - w % self.window_size) % self.window_size
        x = F.pad(x, (0, pad_w, 0, pad_h))
        hp, wp = h + pad_h, w + pad_w
        x = x.view(b, c, hp // self.window_size, self.window_size, wp // self.window_size, self.window_size)
        x = x.permute(0, 2, 4, 3, 5, 1).reshape(-1, self.window_size * self.window_size, c)
        return x, (hp, wp)

    def _window_reverse(self, windows: torch.Tensor, pad_hw: tuple[int, int], h: int, w: int) -> torch.Tensor:
        hp, wp = pad_hw
        num_windows = (hp // self.window_size) * (wp // self.window_size)
        b = windows.shape[0] // num_windows
        x = windows.view(b, hp // self.window_size, wp // self.window_size, self.window_size, self.window_size, self.channels)
        x = x.permute(0, 5, 1, 3, 2, 4).reshape(b, self.channels, hp, wp)
        x = x[:, :, :h, :w]
        if self.shift_size:
            x = torch.roll(x, shifts=(self.shift_size, self.shift_size), dims=(2, 3))
        return x


class MSPA(nn.Module):
    """Multi-scale spatial pyramid attention, channel-preserving."""

    def __init__(self, channels: int, kernels: tuple[int, int, int] = (3, 5, 7)) -> None:
        super().__init__()
        self.channels = channels
        self.branches = nn.ModuleList(
            nn.Conv2d(2, 1, kernel_size=k, padding=k // 2, bias=False) for k in kernels
        )
        self.fuse = nn.Conv2d(len(kernels), 1, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = x.mean(dim=1, keepdim=True)
        maxv = x.amax(dim=1, keepdim=True)
        pooled = torch.cat([avg, maxv], dim=1)
        pyramid = torch.cat([branch(pooled) for branch in self.branches], dim=1)
        attention = torch.sigmoid(self.fuse(pyramid))
        return x * attention + x
