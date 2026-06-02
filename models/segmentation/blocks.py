from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class FDConvBlock(nn.Module):
    """Frequency-aware dynamic convolution block for segmentation ablations."""

    def __init__(self, in_channels: int, out_channels: int, reduction: int = 4) -> None:
        super().__init__()
        self.spatial = DoubleConv(in_channels, out_channels)
        hidden = max(out_channels // reduction, 8)
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_channels, hidden, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, out_channels, kernel_size=1),
            nn.Sigmoid(),
        )
        self.out = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        spatial = self.spatial(x)
        freq = torch.fft.rfft2(spatial, norm="ortho")
        gate = self.gate(spatial).to(dtype=freq.real.dtype)
        freq = freq * (1.0 + gate)
        enhanced = torch.fft.irfft2(freq, s=spatial.shape[-2:], norm="ortho")
        return self.out(spatial + enhanced)


class CapsulePath(nn.Module):
    """Lightweight capsule-style feature path with squash non-linearity."""

    def __init__(self, channels: int, capsules: int = 4) -> None:
        super().__init__()
        self.capsules = capsules
        self.pose = nn.Conv2d(channels, channels * capsules, kernel_size=3, padding=1, groups=1, bias=False)
        self.proj = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        pose = self.pose(x).view(b, self.capsules, c, h, w)
        norm = torch.linalg.vector_norm(pose, dim=2, keepdim=True)
        scale = norm.square() / (1.0 + norm.square())
        pose = scale * pose / (norm + 1e-6)
        fused = pose.mean(dim=1)
        return x + self.proj(fused)


class RFAPM(nn.Module):
    """Receptive-field adaptive perception module."""

    def __init__(self, channels: int, dilations: tuple[int, int, int] = (1, 2, 3)) -> None:
        super().__init__()
        self.branches = nn.ModuleList(
            nn.Sequential(
                nn.Conv2d(channels, channels, kernel_size=3, padding=d, dilation=d, bias=False),
                nn.BatchNorm2d(channels),
                nn.ReLU(inplace=True),
            )
            for d in dilations
        )
        self.attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels * len(dilations), channels, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, len(dilations), kernel_size=1),
        )
        self.out = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = [branch(x) for branch in self.branches]
        weights = self.attn(torch.cat(features, dim=1)).softmax(dim=1)
        fused = sum(feat * weights[:, idx : idx + 1] for idx, feat in enumerate(features))
        return x + self.out(fused)


def match_size(x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    if x.shape[-2:] == ref.shape[-2:]:
        return x
    return F.interpolate(x, size=ref.shape[-2:], mode="bilinear", align_corners=False)
