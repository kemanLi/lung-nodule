"""
Build D2 dataset: D1 center-positive slices + train-only negative background slices.

Does not modify D0/D1. Copies val/test from D1 exactly; augments train with empty-label negatives.
"""

from __future__ import annotations

import argparse
import random
import re
import shutil
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

try:
    import SimpleITK as sitk
except ImportError as exc:
    raise SystemExit("Please install SimpleITK before building D2.") from exc

STEM_RE = re.compile(r"^(?P<uid>.+)_z(?P<z>\d{4})$")


def window_to_uint8(volume: np.ndarray, hu_min: int, hu_max: int) -> np.ndarray:
    volume = np.clip(volume, hu_min, hu_max)
    volume = (volume - hu_min) / float(hu_max - hu_min)
    return (volume * 255).astype(np.uint8)


def world_to_voxel_z(world_z: float, origin_z: float, spacing_z: float) -> int:
    return int(np.rint((world_z - origin_z) / spacing_z))


def parse_stem(stem: str) -> tuple[str, int] | None:
    match = STEM_RE.match(stem)
    if not match:
        return None
    return match.group("uid"), int(match.group("z"))


def is_lung_like_slice(slice_u8: np.ndarray, min_mean: float, max_mean: float, min_std: float) -> bool:
    mean = float(slice_u8.mean())
    std = float(slice_u8.std())
    if mean < min_mean or mean > max_mean:
        return False
    if std < min_std:
        return False
    if mean < 5.0:
        return False
    return True


def copy_split_files(src_root: Path, dst_root: Path, split: str) -> int:
    count = 0
    for sub in ("images", "labels"):
        src_dir = src_root / sub / split
        dst_dir = dst_root / sub / split
        dst_dir.mkdir(parents=True, exist_ok=True)
        for src_file in sorted(src_dir.glob("*")):
            if not src_file.is_file():
                continue
            shutil.copy2(src_file, dst_dir / src_file.name)
            count += 1
    return count // 2


def inspect_split_stats(root: Path, split: str) -> dict[str, int | float]:
    image_dir = root / "images" / split
    label_dir = root / "labels" / split
    images = sorted(image_dir.glob("*.png"))
    pos_images = 0
    neg_images = 0
    boxes = 0
    missing_labels = 0
    invalid_rows = 0

    for image_path in images:
        label_path = label_dir / f"{image_path.stem}.txt"
        if not label_path.exists():
            missing_labels += 1
            continue
        rows = [row.strip() for row in label_path.read_text(encoding="utf-8").splitlines() if row.strip()]
        if not rows:
            neg_images += 1
            continue
        pos_images += 1
        for row in rows:
            parts = row.split()
            if len(parts) != 5:
                invalid_rows += 1
                continue
            _, cx, cy, w, h = parts
            vals = [float(cx), float(cy), float(w), float(h)]
            if any(v < 0.0 or v > 1.0 for v in vals) or vals[2] <= 0.0 or vals[3] <= 0.0:
                invalid_rows += 1
            else:
                boxes += 1

    return {
        "split": split,
        "images": len(images),
        "positive_images": pos_images,
        "negative_images": neg_images,
        "boxes": boxes,
        "missing_labels": missing_labels,
        "invalid_rows": invalid_rows,
    }


def build_d2(args: argparse.Namespace) -> None:
    d1_root = Path(args.d1_root)
    out_root = Path(args.output_root)
    data_dir = Path(args.data_dir)
    annotations = pd.read_csv(args.annotations)

    if not d1_root.exists():
        raise SystemExit(f"D1 dataset not found: {d1_root}")

    out_root.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val", "test"):
        (out_root / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_root / "labels" / split).mkdir(parents=True, exist_ok=True)

    # --- val/test: exact copy from D1 ---
    for split in ("val", "test"):
        n = copy_split_files(d1_root, out_root, split)
        print(f"copied {split}: {n} image/label pairs from D1")

    # --- train positives: copy from D1 ---
    train_pos = 0
    train_series: set[str] = set()
    annotations_by_uid = {uid: rows for uid, rows in annotations.groupby("seriesuid")}

    d1_train_images = sorted((d1_root / "images" / "train").glob("*.png"))
    for src_img in d1_train_images:
        stem = src_img.stem
        parsed = parse_stem(stem)
        if parsed:
            train_series.add(parsed[0])
        shutil.copy2(src_img, out_root / "images" / "train" / src_img.name)
        src_lbl = d1_root / "labels" / "train" / f"{stem}.txt"
        shutil.copy2(src_lbl, out_root / "labels" / "train" / f"{stem}.txt")
        train_pos += 1

    target_neg = int(train_pos * args.negative_ratio)
    print(f"train positives copied: {train_pos}")
    print(f"target negatives (ratio {args.negative_ratio}:1): {target_neg}")

    # Map seriesuid -> mhd path (train subsets only)
    train_subsets = {f"subset{i}" for i in range(8)}
    mhd_by_uid: dict[str, Path] = {}
    for mhd_path in sorted(data_dir.glob("subset*/*.mhd")):
        if mhd_path.parent.name not in train_subsets:
            continue
        mhd_by_uid[mhd_path.stem] = mhd_path

    rng = random.Random(args.seed)
    candidates: list[tuple[str, int, np.ndarray]] = []

    for seriesuid in tqdm(sorted(train_series), desc="scan negatives"):
        if seriesuid not in mhd_by_uid:
            print(f"warning: no mhd for train series {seriesuid}")
            continue
        if seriesuid not in annotations_by_uid:
            continue

        image = sitk.ReadImage(str(mhd_by_uid[seriesuid]))
        volume = sitk.GetArrayFromImage(image)
        origin = image.GetOrigin()
        spacing = image.GetSpacing()
        volume_u8 = window_to_uint8(volume, args.hu_min, args.hu_max)
        depth = volume_u8.shape[0]

        forbidden: set[int] = set()
        for row in annotations_by_uid[seriesuid].itertuples(index=False):
            zz = world_to_voxel_z(float(row.coordZ), float(origin[2]), float(spacing[2]))
            for dz in range(-args.min_z_distance, args.min_z_distance + 1):
                forbidden.add(zz + dz)

        d1_train_z = {
            parse_stem(p.stem)[1]
            for p in d1_train_images
            if parse_stem(p.stem) and parse_stem(p.stem)[0] == seriesuid
        }

        z_margin = max(1, int(depth * args.edge_z_fraction))
        for z in range(z_margin, depth - z_margin):
            if z in forbidden or z in d1_train_z:
                continue
            sl = volume_u8[z]
            if not is_lung_like_slice(sl, args.min_slice_mean, args.max_slice_mean, args.min_slice_std):
                continue
            candidates.append((seriesuid, z, sl))

    rng.shuffle(candidates)
    selected = candidates[:target_neg]
    if len(selected) < target_neg:
        print(
            f"warning: only found {len(selected)} negative candidates "
            f"(requested {target_neg}). Using all available."
        )

    for seriesuid, z, sl in tqdm(selected, desc="write negatives"):
        if args.output_size and (sl.shape[0] != args.output_size or sl.shape[1] != args.output_size):
            sl = cv2.resize(sl, (args.output_size, args.output_size), interpolation=cv2.INTER_LINEAR)
        stem = f"{seriesuid}_z{z:04d}_neg"
        out_img = out_root / "images" / "train" / f"{stem}.png"
        out_lbl = out_root / "labels" / "train" / f"{stem}.txt"
        cv2.imwrite(str(out_img), sl)
        out_lbl.write_text("", encoding="utf-8")

    print("\n=== D2 dataset statistics ===")
    for split in ("train", "val", "test"):
        row = inspect_split_stats(out_root, split)
        print(
            f"{row['split']}: images={row['images']} "
            f"positive_images={row['positive_images']} negative_images={row['negative_images']} "
            f"boxes={row['boxes']} missing_labels={row['missing_labels']} "
            f"invalid_rows={row['invalid_rows']}"
        )
    print(f"\nD2 written to: {out_root.resolve()}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build D2 = D1 center positives + train-only negatives.")
    parser.add_argument("--d1-root", default="datasets_D1_center/detection")
    parser.add_argument("--output-root", default="datasets_D2_center_negative/detection")
    parser.add_argument("--data-dir", default="data/LUNA16")
    parser.add_argument("--annotations", default="data/LUNA16/annotations.csv")
    parser.add_argument("--negative-ratio", type=float, default=1.0, help="negatives per positive in train.")
    parser.add_argument("--min-z-distance", type=int, default=10)
    parser.add_argument("--hu-min", type=int, default=-1200)
    parser.add_argument("--hu-max", type=int, default=600)
    parser.add_argument("--output-size", type=int, default=640)
    parser.add_argument("--min-slice-mean", type=float, default=25.0)
    parser.add_argument("--max-slice-mean", type=float, default=230.0)
    parser.add_argument("--min-slice-std", type=float, default=12.0)
    parser.add_argument("--edge-z_fraction", type=float, default=0.05, help="Skip top/bottom fraction of z.")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    build_d2(parse_args())
