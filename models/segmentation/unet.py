import torch
import torch.nn as nn

from .blocks import CapsulePath, DoubleConv, FDConvBlock, RFAPM, match_size


class UNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        base: int = 32,
        use_fdconv: bool = False,
        use_capsule: bool = False,
        use_rfapm: bool = False,
    ) -> None:
        super().__init__()
        block = FDConvBlock if use_fdconv else DoubleConv
        self.down1 = block(in_channels, base)
        self.down2 = block(base, base * 2)
        self.down3 = block(base * 2, base * 4)
        self.down4 = block(base * 4, base * 8)
        self.pool = nn.MaxPool2d(2)

        self.bridge = block(base * 8, base * 16)
        self.capsule = CapsulePath(base * 16) if use_capsule else nn.Identity()
        self.rfapm = RFAPM(base * 16) if use_rfapm else nn.Identity()

        self.up4 = nn.ConvTranspose2d(base * 16, base * 8, kernel_size=2, stride=2)
        self.dec4 = block(base * 16, base * 8)
        self.up3 = nn.ConvTranspose2d(base * 8, base * 4, kernel_size=2, stride=2)
        self.dec3 = block(base * 8, base * 4)
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, kernel_size=2, stride=2)
        self.dec2 = block(base * 4, base * 2)
        self.up1 = nn.ConvTranspose2d(base * 2, base, kernel_size=2, stride=2)
        self.dec1 = block(base * 2, base)
        self.head = nn.Conv2d(base, out_channels, kernel_size=1)

    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(self.pool(d1))
        d3 = self.down3(self.pool(d2))
        d4 = self.down4(self.pool(d3))
        b = self.bridge(self.pool(d4))
        b = self.rfapm(self.capsule(b))

        x = self.up4(b)
        x = self.dec4(torch.cat([match_size(x, d4), d4], dim=1))
        x = self.up3(x)
        x = self.dec3(torch.cat([match_size(x, d3), d3], dim=1))
        x = self.up2(x)
        x = self.dec2(torch.cat([match_size(x, d2), d2], dim=1))
        x = self.up1(x)
        x = self.dec1(torch.cat([match_size(x, d1), d1], dim=1))
        return self.head(x)


def build_segmentation_model(model_type: str = "unet", base: int = 32) -> UNet:
    configs = {
        "unet": {},
        "fdconv": {"use_fdconv": True},
        "fdconv_capsule": {"use_fdconv": True, "use_capsule": True},
        "fdconv_rfapm": {"use_fdconv": True, "use_rfapm": True},
        "caps_fdrnet_lite": {"use_fdconv": True, "use_capsule": True, "use_rfapm": True},
    }
    if model_type not in configs:
        raise ValueError(f"Unknown segmentation model_type: {model_type}")
    return UNet(base=base, **configs[model_type])
