import argparse

import numpy as np
import torch
from scipy import ndimage
from torch.utils.data import DataLoader

from models.segmentation import build_segmentation_model
from train_segmentation import PatchDataset


def segmentation_metrics(pred: np.ndarray, target: np.ndarray, spacing: tuple[float, float]) -> tuple[float, float, float, float]:
    pred = pred.astype(bool)
    target = target.astype(bool)
    inter = np.logical_and(pred, target).sum()
    union = np.logical_or(pred, target).sum()
    dice = (2 * inter) / (pred.sum() + target.sum() + 1e-6)
    iou = inter / (union + 1e-6)
    sen = inter / (target.sum() + 1e-6)
    asd = average_symmetric_surface_distance(pred, target, spacing)
    return float(dice), float(iou), float(sen), float(asd)


def average_symmetric_surface_distance(pred: np.ndarray, target: np.ndarray, spacing: tuple[float, float]) -> float:
    if not pred.any() and not target.any():
        return 0.0
    if not pred.any() or not target.any():
        return float("nan")

    pred_surface = surface_pixels(pred)
    target_surface = surface_pixels(target)
    target_distance = ndimage.distance_transform_edt(~target_surface, sampling=spacing)
    pred_distance = ndimage.distance_transform_edt(~pred_surface, sampling=spacing)

    pred_to_target = target_distance[pred_surface]
    target_to_pred = pred_distance[target_surface]
    return float(np.concatenate([pred_to_target, target_to_pred]).mean())


def surface_pixels(mask: np.ndarray) -> np.ndarray:
    eroded = ndimage.binary_erosion(mask, structure=np.ones((3, 3), dtype=bool), border_value=0)
    return np.logical_xor(mask, eroded)


def evaluate(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    ckpt = torch.load(args.weights, map_location=device)
    model_type = args.model_type or ckpt.get("model_type", "unet")
    model = build_segmentation_model(model_type, base=int(ckpt.get("base", 32))).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    ds = PatchDataset(args.data, args.split)
    loader = DataLoader(ds, batch_size=1, shuffle=False)
    dices, ious, sens, asds = [], [], [], []
    spacing = (args.spacing_y, args.spacing_x)
    with torch.no_grad():
        for image, mask in loader:
            logits = model(image.to(device))
            pred = (torch.sigmoid(logits).cpu().numpy()[0, 0] > args.threshold).astype(np.uint8)
            target = mask.numpy()[0, 0].astype(np.uint8)
            dice, iou, sen, asd = segmentation_metrics(pred, target, spacing)
            dices.append(dice)
            ious.append(iou)
            sens.append(sen)
            asds.append(asd)

    asds_array = np.asarray(asds, dtype=np.float32)
    skipped_asd = int(np.isnan(asds_array).sum())
    valid_asds = asds_array[~np.isnan(asds_array)]
    mean_asd = float(valid_asds.mean()) if len(valid_asds) else float("nan")
    print(f"DSC(%): {np.mean(dices) * 100:.2f}")
    print(f"IoU(%): {np.mean(ious) * 100:.2f}")
    print(f"SEN(%): {np.mean(sens) * 100:.2f}")
    print(f"ASD(mm): {mean_asd:.4f}")
    if skipped_asd:
        print(f"ASD skipped empty one-sided masks: {skipped_asd}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate U-Net segmentation weights.")
    parser.add_argument("--weights", required=True)
    parser.add_argument("--data", default="datasets/segmentation")
    parser.add_argument("--split", default="val")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--spacing-x", type=float, default=1.0, help="Pixel spacing along x/columns in mm.")
    parser.add_argument("--spacing-y", type=float, default=1.0, help="Pixel spacing along y/rows in mm.")
    parser.add_argument(
        "--model-type",
        default=None,
        choices=["unet", "fdconv", "fdconv_capsule", "fdconv_rfapm", "caps_fdrnet_lite"],
        help="Override model type. By default this is read from the checkpoint.",
    )
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())
