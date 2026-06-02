import argparse
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


def read_yolo_labels(path: Path, width: int, height: int) -> list[tuple[int, int, int, int]]:
    boxes = []
    if not path.exists():
        return boxes
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) != 5:
            continue
        _, cx, cy, bw, bh = map(float, parts)
        x1 = int((cx - bw / 2) * width)
        y1 = int((cy - bh / 2) * height)
        x2 = int((cx + bw / 2) * width)
        y2 = int((cy + bh / 2) * height)
        boxes.append((x1, y1, x2, y2))
    return boxes


def crop_square(image: np.ndarray, cx: int, cy: int, size: int = 64) -> np.ndarray:
    half = size // 2
    padded = cv2.copyMakeBorder(image, half, half, half, half, cv2.BORDER_CONSTANT, value=0)
    cx += half
    cy += half
    return padded[cy - half : cy + half, cx - half : cx + half]


def ellipse_mask(box: tuple[int, int, int, int], image_shape: tuple[int, int], patch_size: int = 64) -> np.ndarray:
    x1, y1, x2, y2 = box
    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2
    rx = max(3, (x2 - x1) // 2)
    ry = max(3, (y2 - y1) // 2)
    full = np.zeros(image_shape, dtype=np.uint8)
    cv2.ellipse(full, (cx, cy), (rx, ry), 0, 0, 360, 255, -1)
    return crop_square(full, cx, cy, patch_size)


def build(args: argparse.Namespace) -> None:
    source = Path(args.detection_dir)
    output = Path(args.output_dir)
    for split in ["train", "val", "test"]:
        images = sorted((source / "images" / split).glob("*.png"))
        for image_path in tqdm(images, desc=f"patches/{split}"):
            image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
            if image is None:
                continue
            labels = read_yolo_labels(source / "labels" / split / f"{image_path.stem}.txt", image.shape[1], image.shape[0])
            for idx, box in enumerate(labels):
                x1, y1, x2, y2 = box
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2
                patch = crop_square(image, cx, cy, args.patch_size)
                mask = ellipse_mask(box, image.shape, args.patch_size)
                stem = f"{image_path.stem}_{idx:02d}"
                (output / "patches" / split).mkdir(parents=True, exist_ok=True)
                (output / "masks" / split).mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(output / "patches" / split / f"{stem}.png"), patch)
                cv2.imwrite(str(output / "masks" / split / f"{stem}.png"), mask)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build quick 64x64 segmentation patches from YOLO labels.")
    parser.add_argument("--detection-dir", default="datasets/detection")
    parser.add_argument("--output-dir", default="datasets/segmentation")
    parser.add_argument("--patch-size", type=int, default=64)
    return parser.parse_args()


if __name__ == "__main__":
    build(parse_args())
