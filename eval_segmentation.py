import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

from models.segmentation import UNet
from train_segmentation import PatchDataset


def dice_iou_sen(pred: np.ndarray, target: np.ndarray) -> tuple[float, float, float]:
    pred = pred.astype(bool)
    target = target.astype(bool)
    inter = np.logical_and(pred, target).sum()
    union = np.logical_or(pred, target).sum()
    dice = (2 * inter) / (pred.sum() + target.sum() + 1e-6)
    iou = inter / (union + 1e-6)
    sen = inter / (target.sum() + 1e-6)
    return float(dice), float(iou), float(sen)


def evaluate(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    ckpt = torch.load(args.weights, map_location=device)
    model = UNet(base=int(ckpt.get("base", 32))).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    ds = PatchDataset(args.data, args.split)
    loader = DataLoader(ds, batch_size=1, shuffle=False)
    dices, ious, sens = [], [], []
    with torch.no_grad():
        for image, mask in loader:
            logits = model(image.to(device))
            pred = (torch.sigmoid(logits).cpu().numpy()[0, 0] > args.threshold).astype(np.uint8)
            target = mask.numpy()[0, 0].astype(np.uint8)
            dice, iou, sen = dice_iou_sen(pred, target)
            dices.append(dice)
            ious.append(iou)
            sens.append(sen)
    print(f"DSC: {np.mean(dices):.4f}")
    print(f"IoU: {np.mean(ious):.4f}")
    print(f"SEN: {np.mean(sens):.4f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate U-Net segmentation weights.")
    parser.add_argument("--weights", required=True)
    parser.add_argument("--data", default="datasets/segmentation")
    parser.add_argument("--split", default="val")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())
