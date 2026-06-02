import argparse


def evaluate(args: argparse.Namespace) -> None:
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit("Please install ultralytics before evaluating detection.") from exc

    model = YOLO(args.weights)
    metrics = model.val(data=args.data, imgsz=args.imgsz, split=args.split)
    print(f"mAP@0.5: {metrics.box.map50:.4f}")
    print(f"mAP@0.5:0.95: {metrics.box.map:.4f}")
    print(f"Precision: {metrics.box.mp:.4f}")
    print(f"Recall: {metrics.box.mr:.4f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate YOLO detection weights.")
    parser.add_argument("--weights", required=True)
    parser.add_argument("--data", default="configs/luna16_subset.yaml")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--split", default="val")
    return parser.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())
