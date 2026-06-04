from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from eval_segmentation import segmentation_metrics
from models.segmentation import build_segmentation_model
from train_segmentation import PatchDataset


EXPERIMENTS = {
    "baseline": "unet_baseline",
    "fdconv": "unet_fdconv",
    "fdconv_capsule": "unet_fdconv_capsule",
    "fdconv_rfapm": "unet_fdconv_rfapm",
    "full": "caps_fdrnet_lite",
}

ORDER = ("baseline", "fdconv", "fdconv_capsule", "fdconv_rfapm", "full")


def resolve_weights(args: argparse.Namespace, experiment: str) -> Path:
    run_name = EXPERIMENTS[experiment]
    if args.prefix:
        run_name = f"{args.prefix}_{run_name}"
    return Path(args.project) / run_name / "best.pt"


def evaluate_weights(weights: Path, args: argparse.Namespace) -> dict[str, float | str]:
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    ckpt = torch.load(weights, map_location=device)
    model_type = ckpt.get("model_type", "unet")
    model = build_segmentation_model(model_type, base=int(ckpt.get("base", 32))).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    ds = PatchDataset(args.data, args.split)
    loader = DataLoader(ds, batch_size=1, shuffle=False)
    spacing = (args.spacing_y, args.spacing_x)
    dices, ious, sens, asds = [], [], [], []
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
    valid_asds = asds_array[~np.isnan(asds_array)]
    return {
        "weights": str(weights),
        "model_type": model_type,
        "dsc_percent": float(np.mean(dices) * 100),
        "iou_percent": float(np.mean(ious) * 100),
        "sen_percent": float(np.mean(sens) * 100),
        "asd_mm": float(valid_asds.mean()) if len(valid_asds) else float("nan"),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate segmentation ablations.")
    parser.add_argument("--project", default="runs/segmentation_ablation")
    parser.add_argument("--data", default="datasets/segmentation")
    parser.add_argument("--split", default="val")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--spacing-x", type=float, default=1.0)
    parser.add_argument("--spacing-y", type=float, default=1.0)
    parser.add_argument("--prefix", default="")
    parser.add_argument("--output", default="", help="Optional CSV path for ablation metrics.")
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    parsed = parse_args()
    rows = []
    for experiment in ORDER:
        weights = resolve_weights(parsed, experiment)
        if not weights.exists():
            print(f"{experiment}: missing {weights}")
            continue
        row = {"experiment": experiment, **evaluate_weights(weights, parsed)}
        rows.append(row)
        print(
            f"{experiment}: "
            f"DSC={row['dsc_percent']:.2f}% IoU={row['iou_percent']:.2f}% "
            f"SEN={row['sen_percent']:.2f}% ASD={row['asd_mm']:.4f}mm"
        )

    if parsed.output and rows:
        output = Path(parsed.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"wrote {output}")
