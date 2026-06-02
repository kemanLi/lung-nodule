import argparse
import csv
from pathlib import Path


EXPERIMENTS = {
    "baseline": "yolo11n_baseline",
    "swin": "yolo11n_swin",
    "mspa": "yolo11n_mspa",
    "swin_mspa": "yolo11n_swin_mspa",
}

ORDER = ("baseline", "swin", "mspa", "swin_mspa")


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate detection ablations without KAN-C3k2.")
    parser.add_argument("--project", default="runs/detection_ablation")
    parser.add_argument("--data", default="configs/luna16_subset.yaml")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--split", default="val")
    parser.add_argument("--prefix", default="")
    parser.add_argument("--output", default="", help="Optional CSV path for ablation metrics.")
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

    if parsed.output and rows:
        output = Path(parsed.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"wrote {output}")
