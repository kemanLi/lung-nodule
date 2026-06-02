import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from models.segmentation import UNet


class PatchDataset(Dataset):
    def __init__(self, root: str, split: str) -> None:
        self.root = Path(root)
        self.patches = sorted((self.root / "patches" / split).glob("*.png"))
        self.masks = [self.root / "masks" / split / p.name for p in self.patches]

    def __len__(self) -> int:
        return len(self.patches)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        image = cv2.imread(str(self.patches[idx]), cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(str(self.masks[idx]), cv2.IMREAD_GRAYSCALE)
        if image is None or mask is None:
            raise FileNotFoundError(self.patches[idx])
        image = image.astype(np.float32) / 255.0
        mask = (mask.astype(np.float32) / 255.0 > 0.5).astype(np.float32)
        return torch.from_numpy(image[None]), torch.from_numpy(mask[None])


def dice_loss(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    inter = (probs * target).sum(dim=(1, 2, 3))
    denom = probs.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    return 1.0 - ((2 * inter + eps) / (denom + eps)).mean()


def train(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    train_ds = PatchDataset(args.data, "train")
    val_ds = PatchDataset(args.data, "val")
    if len(train_ds) == 0:
        raise SystemExit("No training patches found. Run scripts/build_seg_patches.py first.")

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=args.workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=args.workers) if len(val_ds) else None
    model = UNet(base=args.base).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    bce = nn.BCEWithLogitsLoss()

    best_val = float("inf")
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for image, mask in tqdm(train_loader, desc=f"epoch {epoch}/{args.epochs}"):
            image, mask = image.to(device), mask.to(device)
            logits = model(image)
            loss = bce(logits, mask) + dice_loss(logits, mask)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))

        val_loss = float(np.mean(losses))
        if val_loader is not None:
            model.eval()
            vals = []
            with torch.no_grad():
                for image, mask in val_loader:
                    image, mask = image.to(device), mask.to(device)
                    logits = model(image)
                    vals.append(float((bce(logits, mask) + dice_loss(logits, mask)).item()))
            val_loss = float(np.mean(vals))

        print(f"epoch={epoch} train_loss={np.mean(losses):.4f} val_loss={val_loss:.4f}")
        torch.save({"model": model.state_dict(), "base": args.base}, out_dir / "last.pt")
        if val_loss <= best_val:
            best_val = val_loss
            torch.save({"model": model.state_dict(), "base": args.base}, out_dir / "best.pt")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a small U-Net segmentation baseline.")
    parser.add_argument("--data", default="datasets/segmentation")
    parser.add_argument("--output", default="runs/segmentation/unet_mvp")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--base", type=int, default=32)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
