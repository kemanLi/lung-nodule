from __future__ import annotations

import argparse
from pathlib import Path


def inspect_split(root: Path, split: str) -> dict[str, float | int | str]:
    image_dir = root / "images" / split
    label_dir = root / "labels" / split
    images = sorted(image_dir.glob("*.png"))
    labels = [label_dir / f"{image.stem}.txt" for image in images]
    box_count = 0
    empty_count = 0
    invalid_count = 0
    widths: list[float] = []
    heights: list[float] = []

    for label_path in labels:
        if not label_path.exists():
            empty_count += 1
            continue
        rows = [row.strip() for row in label_path.read_text(encoding="utf-8").splitlines() if row.strip()]
        if not rows:
            empty_count += 1
            continue
        for row in rows:
            parts = row.split()
            if len(parts) != 5:
                invalid_count += 1
                continue
            _, cx, cy, w, h = parts
            values = [float(cx), float(cy), float(w), float(h)]
            if any(value < 0.0 or value > 1.0 for value in values) or values[2] <= 0.0 or values[3] <= 0.0:
                invalid_count += 1
            else:
                box_count += 1
                widths.append(values[2])
                heights.append(values[3])

    return {
        "split": split,
        "images": len(images),
        "boxes": box_count,
        "empty_labels": empty_count,
        "invalid_rows": invalid_count,
        "avg_box_w": sum(widths) / len(widths) if widths else 0.0,
        "avg_box_h": sum(heights) / len(heights) if heights else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect a YOLO-format LUNA16 detection dataset.")
    parser.add_argument("--root", default="datasets/detection")
    args = parser.parse_args()

    root = Path(args.root)
    for split in ("train", "val", "test"):
        row = inspect_split(root, split)
        print(
            f"{row['split']}: images={row['images']} boxes={row['boxes']} "
            f"empty_labels={row['empty_labels']} invalid_rows={row['invalid_rows']} "
            f"avg_box=({row['avg_box_w']:.4f}, {row['avg_box_h']:.4f})"
        )


if __name__ == "__main__":
    main()
