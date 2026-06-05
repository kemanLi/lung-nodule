"""
Build D3: center + adjacent slices with sphere-consistent 2D cross-section diameters.

Adjacent layer box diameter:
  slice_diameter_mm = 2 * sqrt(radius_mm^2 - dz_mm^2)
Skip layer if dz_mm >= radius_mm (nodule not visible on that slice).

Does not modify D0/D1/D2.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

try:
    import SimpleITK as sitk
except ImportError as exc:
    raise SystemExit("Please install SimpleITK before preprocessing LUNA16.") from exc


def window_to_uint8(volume: np.ndarray, hu_min: int, hu_max: int) -> np.ndarray:
    volume = np.clip(volume, hu_min, hu_max)
    volume = (volume - hu_min) / float(hu_max - hu_min)
    return (volume * 255).astype(np.uint8)


def world_to_voxel(world_xyz: np.ndarray, origin_xyz: np.ndarray, spacing_xyz: np.ndarray) -> np.ndarray:
    return np.rint((world_xyz - origin_xyz) / spacing_xyz).astype(int)


def make_label(cx: float, cy: float, diameter_px: float, width: int, height: int, min_box_size: int) -> str:
    box = max(float(diameter_px), float(min_box_size))
    half = box / 2.0
    x1 = max(0.0, cx - half)
    y1 = max(0.0, cy - half)
    x2 = min(float(width - 1), cx + half)
    y2 = min(float(height - 1), cy + half)
    clipped_w = max(1.0, x2 - x1)
    clipped_h = max(1.0, y2 - y1)
    clipped_cx = x1 + clipped_w / 2.0
    clipped_cy = y1 + clipped_h / 2.0
    return (
        f"0 {clipped_cx / width:.6f} {clipped_cy / height:.6f} "
        f"{clipped_w / width:.6f} {clipped_h / height:.6f}"
    )


def split_for_subset(subset_name: str) -> str:
    if subset_name == "subset9":
        return "test"
    if subset_name == "subset8":
        return "val"
    return "train"


def slice_diameter_mm(diameter_mm: float, dz_mm: float) -> float | None:
    radius_mm = diameter_mm / 2.0
    if abs(dz_mm) >= radius_mm:
        return None
    return 2.0 * math.sqrt(radius_mm * radius_mm - dz_mm * dz_mm)


def save_yolo_sample(image: np.ndarray, label_rows: list[str], out_image: Path, out_label: Path) -> None:
    out_image.parent.mkdir(parents=True, exist_ok=True)
    out_label.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_image), image)
    out_label.write_text("\n".join(label_rows), encoding="utf-8")


def split_row_stats(root: Path, split: str) -> tuple[int, int, float, float]:
    img_dir = root / "images" / split
    lbl_dir = root / "labels" / split
    n_img = len(list(img_dir.glob("*.png")))
    n_box = 0
    widths: list[float] = []
    heights: list[float] = []
    for lbl in lbl_dir.glob("*.txt"):
        for line in lbl.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            parts = line.split()
            if len(parts) == 5:
                n_box += 1
                widths.append(float(parts[3]))
                heights.append(float(parts[4]))
    avg_w = sum(widths) / len(widths) if widths else 0.0
    avg_h = sum(heights) / len(heights) if heights else 0.0
    return n_img, n_box, avg_w, avg_h


def preprocess(args: argparse.Namespace) -> None:
    data_dir = Path(args.data_dir)
    out_dir = Path(args.output_dir) / "detection"
    annotations = pd.read_csv(args.annotations)
    annotations_by_uid = {uid: rows for uid, rows in annotations.groupby("seriesuid")}

    mhd_files = sorted(data_dir.glob("subset*/*.mhd"))
    if not mhd_files:
        raise SystemExit(f"No .mhd files found under {data_dir}")

    stats = {
        "nodules": 0,
        "layers_written": 0,
        "layers_skipped_invisible": 0,
        "center_layers": 0,
        "adjacent_layers": 0,
    }

    for mhd_path in tqdm(mhd_files, desc="D3 preprocess"):
        seriesuid = mhd_path.stem
        if seriesuid not in annotations_by_uid:
            continue

        image = sitk.ReadImage(str(mhd_path))
        volume = sitk.GetArrayFromImage(image)
        origin_xyz = np.array(image.GetOrigin(), dtype=np.float32)
        spacing_xyz = np.array(image.GetSpacing(), dtype=np.float32)
        spacing_z = float(spacing_xyz[2])
        spacing_xy = float(np.mean(spacing_xyz[:2]))
        volume_u8 = window_to_uint8(volume, args.hu_min, args.hu_max)
        split = split_for_subset(mhd_path.parent.name)
        h, w = volume_u8.shape[1], volume_u8.shape[2]

        labels_by_z: dict[int, list[str]] = {}

        for row in annotations_by_uid[seriesuid].itertuples(index=False):
            stats["nodules"] += 1
            world_xyz = np.array([row.coordX, row.coordY, row.coordZ], dtype=np.float32)
            voxel_xyz = world_to_voxel(world_xyz, origin_xyz, spacing_xyz)
            x, y, center_z = int(voxel_xyz[0]), int(voxel_xyz[1]), int(voxel_xyz[2])
            diameter_mm = float(row.diameter_mm)

            for dz in range(-args.adjacent_slices, args.adjacent_slices + 1):
                zz = center_z + dz
                if zz < 0 or zz >= volume_u8.shape[0]:
                    continue
                if dz == 0:
                    d_mm = diameter_mm
                    stats["center_layers"] += 1
                else:
                    dz_mm = abs(dz) * spacing_z
                    d_mm = slice_diameter_mm(diameter_mm, dz_mm)
                    if d_mm is None:
                        stats["layers_skipped_invisible"] += 1
                        continue
                    stats["adjacent_layers"] += 1

                d_px = d_mm / spacing_xy
                label = make_label(float(x), float(y), d_px, w, h, args.min_box_size)
                labels_by_z.setdefault(zz, []).append(label)

        for z, label_rows in labels_by_z.items():
            image2d = volume_u8[z]
            if args.output_size and (image2d.shape[0] != args.output_size or image2d.shape[1] != args.output_size):
                image2d = cv2.resize(image2d, (args.output_size, args.output_size), interpolation=cv2.INTER_LINEAR)
            stem = f"{seriesuid}_z{z:04d}"
            save_yolo_sample(
                image2d,
                label_rows,
                out_dir / "images" / split / f"{stem}.png",
                out_dir / "labels" / split / f"{stem}.txt",
            )
            stats["layers_written"] += 1

    print("\n=== D3 build summary ===")
    print(f"nodules processed: {stats['nodules']}")
    print(f"layers written (images): {stats['layers_written']}")
    print(f"center layer labels: {stats['center_layers']}")
    print(f"adjacent layer labels (visible): {stats['adjacent_layers']}")
    print(f"adjacent layers skipped (dz >= radius): {stats['layers_skipped_invisible']}")
    if stats["nodules"]:
        print(f"avg layers per nodule: {stats['layers_written'] / stats['nodules']:.3f}")

    print("\n=== D3 dataset splits ===")
    for split in ("train", "val", "test"):
        n_img, n_box, avg_w, avg_h = split_row_stats(out_dir, split)
        print(f"{split}: images={n_img} instances={n_box} avg_box=({avg_w:.4f}, {avg_h:.4f})")

    print(f"\nD3 written to: {out_dir.resolve()}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build D3 spherical adjacent-slice LUNA16 dataset.")
    parser.add_argument("--data-dir", default="data/LUNA16")
    parser.add_argument("--annotations", default="data/LUNA16/annotations.csv")
    parser.add_argument("--output-dir", default="datasets_D3_spherical_adjacent")
    parser.add_argument("--hu-min", type=int, default=-1200)
    parser.add_argument("--hu-max", type=int, default=600)
    parser.add_argument("--output-size", type=int, default=640)
    parser.add_argument("--adjacent-slices", type=int, default=1)
    parser.add_argument("--min-box-size", type=int, default=8)
    return parser.parse_args()


if __name__ == "__main__":
    preprocess(parse_args())
