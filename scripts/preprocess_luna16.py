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


def save_yolo_sample(image: np.ndarray, label_rows: list[str], out_image: Path, out_label: Path) -> None:
    out_image.parent.mkdir(parents=True, exist_ok=True)
    out_label.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_image), image)
    out_label.write_text("\n".join(label_rows), encoding="utf-8")


def make_label(cx: float, cy: float, diameter_px: float, width: int, height: int, min_box_size: int) -> str:
    box = max(float(diameter_px), float(min_box_size))
    return f"0 {cx / width:.6f} {cy / height:.6f} {box / width:.6f} {box / height:.6f}"


def split_for_subset(subset_name: str) -> str:
    if subset_name == "subset9":
        return "test"
    if subset_name == "subset8":
        return "val"
    return "train"


def preprocess(args: argparse.Namespace) -> None:
    data_dir = Path(args.data_dir)
    out_dir = Path(args.output_dir)
    annotations = pd.read_csv(args.annotations)
    annotations_by_uid = {uid: rows for uid, rows in annotations.groupby("seriesuid")}

    mhd_files = sorted(data_dir.glob("subset*/*.mhd"))
    if not mhd_files:
        raise SystemExit(f"No .mhd files found under {data_dir}")

    for mhd_path in tqdm(mhd_files, desc="preprocess"):
        seriesuid = mhd_path.stem
        if seriesuid not in annotations_by_uid:
            continue

        image = sitk.ReadImage(str(mhd_path))
        volume = sitk.GetArrayFromImage(image)
        origin_xyz = np.array(image.GetOrigin(), dtype=np.float32)
        spacing_xyz = np.array(image.GetSpacing(), dtype=np.float32)
        volume_u8 = window_to_uint8(volume, args.hu_min, args.hu_max)

        split = split_for_subset(mhd_path.parent.name)
        labels_by_z: dict[int, list[str]] = {}
        for row in annotations_by_uid[seriesuid].itertuples(index=False):
            world_xyz = np.array([row.coordX, row.coordY, row.coordZ], dtype=np.float32)
            voxel_xyz = world_to_voxel(world_xyz, origin_xyz, spacing_xyz)
            x, y, z = int(voxel_xyz[0]), int(voxel_xyz[1]), int(voxel_xyz[2])
            diameter_px = float(row.diameter_mm) / float(np.mean(spacing_xyz[:2]))
            for dz in range(-args.adjacent_slices, args.adjacent_slices + 1):
                zz = z + dz
                if 0 <= zz < volume_u8.shape[0]:
                    label = make_label(x, y, diameter_px, volume_u8.shape[2], volume_u8.shape[1], args.min_box_size)
                    labels_by_z.setdefault(zz, []).append(label)

        for z, label_rows in labels_by_z.items():
            image2d = volume_u8[z]
            if args.output_size and image2d.shape[0] != args.output_size:
                image2d = cv2.resize(image2d, (args.output_size, args.output_size), interpolation=cv2.INTER_LINEAR)
            stem = f"{seriesuid}_z{z:04d}"
            save_yolo_sample(
                image2d,
                label_rows,
                out_dir / "detection" / "images" / split / f"{stem}.png",
                out_dir / "detection" / "labels" / split / f"{stem}.txt",
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a 2D YOLO dataset from LUNA16 subsets.")
    parser.add_argument("--data-dir", default="data/LUNA16", help="Directory containing subset0/subset1/subset2 folders.")
    parser.add_argument("--annotations", default="data/LUNA16/annotations.csv")
    parser.add_argument("--output-dir", default="datasets")
    parser.add_argument("--hu-min", type=int, default=-1200)
    parser.add_argument("--hu-max", type=int, default=600)
    parser.add_argument("--output-size", type=int, default=640)
    parser.add_argument("--adjacent-slices", type=int, default=1)
    parser.add_argument("--min-box-size", type=int, default=8)
    return parser.parse_args()


if __name__ == "__main__":
    preprocess(parse_args())
