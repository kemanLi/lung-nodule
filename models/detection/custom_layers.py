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
    """Multi-scale spatial pyramid attention with HPC and SPR branches."""

    def __init__(
        self,
        channels: int,
        kernels: tuple[int, int, int] = (3, 5, 7),
        reduction: int = 4,
        pool_sizes: tuple[int, int] = (1, 2),
    ) -> None:
        super().__init__()
        self.channels = channels
        groups = 4 if channels % 4 == 0 else max(1, min(len(kernels), channels))
        split_channels = [channels // groups for _ in range(groups)]
        split_channels[-1] += channels - sum(split_channels)
        self.split_channels = split_channels
        self.hpc = nn.ModuleList(
            nn.Sequential(
                nn.Conv2d(c, c, kernel_size=kernels[idx % len(kernels)], padding=kernels[idx % len(kernels)] // 2, bias=False),
                nn.BatchNorm2d(c),
                nn.SiLU(inplace=True),
            )
            for idx, c in enumerate(split_channels)
        )
        hidden = max(channels // reduction, 8)
        self.pool_sizes = pool_sizes
        self.channel_interaction = nn.Sequential(
            nn.Conv2d(channels * len(pool_sizes), hidden, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1, bias=True),
        )
        self.out = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        chunks = torch.split(x, self.split_channels, dim=1)
        refined = []
        residual = None
        for chunk, branch in zip(chunks, self.hpc):
            y = branch(chunk)
            if residual is not None:
                y = y + F.interpolate(residual, size=y.shape[-2:], mode="nearest")
            residual = y
            refined.append(y)
        hpc = torch.cat(refined, dim=1)

        pyramid = []
        for size in self.pool_sizes:
            pooled = F.adaptive_avg_pool2d(hpc, output_size=(size, size))
            pyramid.append(F.interpolate(pooled, size=hpc.shape[-2:], mode="bilinear", align_corners=False))
        weights = self.channel_interaction(torch.cat(pyramid, dim=1)).softmax(dim=1)
        return x + self.out(hpc * weights)


class KANConv(nn.Module):
    """KAN-style convolution with a base branch and learnable spline-like basis branch."""

    def __init__(self, channels: int, kernel_size: int = 3, grid_size: int = 5) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.base = nn.Sequential(
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(channels),
        )
        centers = torch.linspace(-1.0, 1.0, grid_size).view(1, grid_size, 1, 1, 1)
        self.register_buffer("centers", centers)
        self.log_scale = nn.Parameter(torch.zeros(1, grid_size, channels, 1, 1))
        self.spline_weight = nn.Parameter(torch.zeros(1, grid_size, channels, 1, 1))
        self.spline_conv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.out = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        nn.init.normal_(self.spline_weight, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_norm = torch.tanh(x).unsqueeze(1)
        scale = self.log_scale.exp().clamp_min(1e-3)
        basis = torch.exp(-((x_norm - self.centers) ** 2) * scale)
        spline = (basis * self.spline_weight).sum(dim=1)
        return self.out(self.base(x) + self.spline_conv(spline))


class KANBottleneck(nn.Module):
    """Two-layer KAN bottleneck with residual feature preservation."""

    def __init__(self, channels: int, shortcut: bool = True) -> None:
        super().__init__()
        self.shortcut = shortcut
        self.net = nn.Sequential(KANConv(channels), KANConv(channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.net(x)
        return x + y if self.shortcut else y


class KANC3k2(nn.Module):
    """Channel-preserving KAN-C3k2 enhancement block for YOLO neck ablations."""

    def __init__(self, channels: int, depth: int = 2, shortcut: bool = True, expansion: float = 0.5) -> None:
        super().__init__()
        hidden = max(8, int(channels * expansion))
        self.cv1 = nn.Conv2d(channels, hidden, kernel_size=1, bias=False)
        self.cv2 = nn.Conv2d(channels, hidden, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(hidden)
        self.bn2 = nn.BatchNorm2d(hidden)
        self.blocks = nn.Sequential(*(KANBottleneck(hidden, shortcut=shortcut) for _ in range(depth)))
        self.cv3 = nn.Sequential(
            nn.Conv2d(hidden * 2, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        main = self.blocks(F.silu(self.bn1(self.cv1(x))))
        skip = F.silu(self.bn2(self.cv2(x)))
        return self.cv3(torch.cat([main, skip], dim=1))
