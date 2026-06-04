from __future__ import annotations

import argparse
import csv
from pathlib import Path


EXPERIMENTS = {
    "baseline": "yolo11n_baseline",
    "swin": "yolo11n_swin",
    "kan": "yolo11n_kan",
    "mspa": "yolo11n_mspa",
    "wo_swin": "yolo11n_kan_mspa",
    "swin_mspa": "yolo11n_swin_mspa",
    "swin_kan": "yolo11n_swin_kan",
    "ours": "skm_yolo",
}

ORDER = ("baseline", "swin", "kan", "mspa", "wo_swin", "swin_mspa", "swin_kan", "ours")


def evaluate_weights(weights: Path, args: argparse.Namespace) -> dict[str, float | str]:
    try:
        from models.detection import register_yolo_custom_layers
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit("Please install ultralytics before evaluating detection ablations.") from exc

    register_yolo_custom_layers()
    model = YOLO(str(weights))
    metrics = model.val(data=args.data, imgsz=args.imgsz, split=args.split)
    params_m = sum(param.numel() for param in model.model.parameters()) / 1e6
    return {
        "weights": str(weights),
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
        "map50": float(metrics.box.map50),
        "map50_95": float(metrics.box.map),
        "params_m": float(params_m),
    }


def resolve_weights(args: argparse.Namespace, experiment: str) -> Path:
    run_name = EXPERIMENTS[experiment]
    if args.prefix:
        run_name = f"{args.prefix}_{run_name}"
    return Path(args.project) / run_name / "weights" / "best.pt"


def default_output_path(args: argparse.Namespace) -> Path:
    stem = f"{args.prefix}_" if args.prefix else ""
    return Path(args.project) / f"{stem}{args.split}_metrics.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate detection ablations without KAN-C3k2.")
    parser.add_argument("--project", default="runs/detection_ablation")
    parser.add_argument("--data", default="configs/luna16_subset.yaml")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--split", default="val")
    parser.add_argument("--prefix", default="")
    parser.add_argument(
        "--output",
        default="",
        help="CSV path for ablation metrics. Defaults to <project>/<prefix><split>_metrics.csv.",
    )
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
            f"P={row['precision']:.4f} R={row['recall']:.4f} "
            f"mAP@0.5={row['map50']:.4f} mAP@0.5:0.95={row['map50_95']:.4f} "
            f"Params={row['params_m']:.2f}M"
        )

    if rows:
        output = Path(parsed.output) if parsed.output else default_output_path(parsed)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"wrote {output}")
