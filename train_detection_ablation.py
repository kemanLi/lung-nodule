import argparse
from pathlib import Path


EXPERIMENTS = {
    "baseline": {
        "model": "yolo11n.pt",
        "name": "yolo11n_baseline",
        "note": "Official YOLOv11n baseline.",
    },
    "swin": {
        "model": "configs/models/yolo11n_swin.yaml",
        "name": "yolo11n_swin",
        "note": "YOLOv11n-style model with SwinTinyLayer blocks in the backbone.",
    },
    "mspa": {
        "model": "configs/models/yolo11n_mspa.yaml",
        "name": "yolo11n_mspa",
        "note": "YOLOv11n-style model with MSPA before Detect.",
    },
    "swin_mspa": {
        "model": "configs/models/yolo11n_swin_mspa.yaml",
        "name": "yolo11n_swin_mspa",
        "note": "Ablation final model without KAN-C3k2: SwinTinyLayer + MSPA.",
    },
}

ORDER = ("baseline", "swin", "mspa", "swin_mspa")


def train_one(args: argparse.Namespace, experiment: str) -> None:
    try:
        from models.detection import register_yolo_custom_layers
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit("Please install ultralytics before training detection ablations.") from exc

    register_yolo_custom_layers()

    spec = EXPERIMENTS[experiment]
    model_path = Path(args.model).as_posix() if args.model else spec["model"]
    run_name = args.name or spec["name"]
    if args.prefix:
        run_name = f"{args.prefix}_{run_name}"

    print(f"[{experiment}] {spec['note']}")
    print(f"[{experiment}] model={model_path}")
    print(f"[{experiment}] name={run_name}")

    model = YOLO(model_path)
    model.train(
        data=args.data,
        imgsz=args.imgsz,
        epochs=args.epochs,
        batch=args.batch,
        project=args.project,
        name=run_name,
        workers=args.workers,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train detection ablations without KAN-C3k2.")
    parser.add_argument(
        "--experiment",
        choices=[*ORDER, "all"],
        default="baseline",
        help="Ablation to train. Use 'all' to run baseline -> swin -> mspa -> swin_mspa.",
    )
    parser.add_argument("--data", default="configs/luna16_subset.yaml")
    parser.add_argument("--model", default=None, help="Optional override for a single experiment model path.")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--project", default="runs/detection_ablation")
    parser.add_argument("--name", default=None, help="Optional override for a single experiment run name.")
    parser.add_argument("--prefix", default="", help="Optional prefix for run names, e.g. luna16.")
    return parser.parse_args()


if __name__ == "__main__":
    parsed = parse_args()
    experiments = ORDER if parsed.experiment == "all" else (parsed.experiment,)
    for exp in experiments:
        train_one(parsed, exp)
