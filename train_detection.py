import argparse
from pathlib import Path


def train(args: argparse.Namespace) -> None:
    try:
        from models.detection import register_yolo_custom_layers
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit("Please install ultralytics before training detection.") from exc

    register_yolo_custom_layers()

    data_yaml = Path(args.data)
    if not data_yaml.exists():
        raise SystemExit(f"Missing data yaml: {data_yaml}")

    model = YOLO(args.model)
    model.train(
        data=str(data_yaml),
        imgsz=args.imgsz,
        epochs=args.epochs,
        batch=args.batch,
        project=args.project,
        name=args.name,
        workers=args.workers,
        lr0=args.lr0,
        lrf=args.lrf,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        seed=args.seed,
        deterministic=args.deterministic,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train YOLOv11/SKM-YOLO for lung nodule detection.")
    parser.add_argument("--data", default="configs/luna16_subset.yaml")
    parser.add_argument("--model", default="yolo11n.pt", help="Use yolo11n.pt for baseline or a config under configs/models.")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--lr0", type=float, default=0.01)
    parser.add_argument("--lrf", type=float, default=0.2)
    parser.add_argument("--momentum", type=float, default=0.937)
    parser.add_argument("--weight-decay", type=float, default=0.0005)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--project", default="runs/detection")
    parser.add_argument("--name", default="yolo11_luna16_full")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
