"""
Build 10-fold LUNA16 YOLO data YAMLs from an existing detection dataset (no image copy).

Official LUNA16: fold N holds out subsetN for test; other 9 subsets form train.
For YOLO workflow compatibility, val and test both point to the held-out subset images.
WARNING: using held-out subset as val enables checkpoint selection leakage — diagnostic only.
"""

from __future__ import annotations

import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path

import yaml

STEM_RE = re.compile(r"^(?P<uid>.+)_z(?P<z>\d{4})(?:_neg)?$")


def build_seriesuid_to_subset(luna16_dir: Path) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for mhd in sorted(luna16_dir.glob("subset*/*.mhd")):
        subset = mhd.parent.name
        mapping[mhd.stem] = subset
    return mapping


def parse_seriesuid(stem: str) -> str | None:
    match = STEM_RE.match(stem)
    return match.group("uid") if match else None


def collect_images(detection_root: Path) -> list[tuple[str, Path]]:
    """Return (seriesuid, absolute_path) for every png under images/."""
    items: list[tuple[str, Path]] = []
    for split in ("train", "val", "test"):
        img_dir = detection_root / "images" / split
        if not img_dir.exists():
            continue
        for img in sorted(img_dir.glob("*.png")):
            uid = parse_seriesuid(img.stem)
            if uid:
                items.append((uid, img))
    return items


def image_list_path(detection_root: Path, img_path: Path, absolute: bool) -> str:
    if absolute:
        return img_path.resolve().as_posix()
    return img_path.relative_to(detection_root).as_posix()


def count_instances(detection_root: Path, image_paths: list[str]) -> int:
    total = 0
    for entry in image_paths:
        img = Path(entry)
        if not img.is_absolute():
            img = detection_root / entry
        label = Path(str(img).replace("/images/", "/labels/").replace("\\images\\", "\\labels\\")).with_suffix(".txt")
        if not label.exists():
            continue
        rows = [r.strip() for r in label.read_text(encoding="utf-8").splitlines() if r.strip()]
        total += len(rows)
    return total


def write_list_file(path: Path, rel_paths: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(rel_paths) + ("\n" if rel_paths else ""), encoding="utf-8")


def build_folds(
    detection_root: Path,
    luna16_dir: Path,
    output_config_dir: Path,
    name_prefix: str,
    absolute_paths: bool,
) -> list[dict[str, str | int | float]]:
    uid_to_subset = build_seriesuid_to_subset(luna16_dir)
    all_images = collect_images(detection_root)

    by_subset: dict[str, list[str]] = defaultdict(list)
    unknown = 0
    for uid, img_path in all_images:
        subset = uid_to_subset.get(uid)
        if not subset:
            unknown += 1
            continue
        by_subset[subset].append(image_list_path(detection_root, img_path, absolute_paths))

    if unknown:
        print(f"warning: {unknown} images could not be mapped to a LUNA16 subset")

    list_dir = detection_root / "lists"
    output_config_dir.mkdir(parents=True, exist_ok=True)
    stats_rows: list[dict[str, str | int | float]] = []

    for fold in range(10):
        held_subset = f"subset{fold}"
        train_paths: list[str] = []
        for s in range(10):
            if s == fold:
                continue
            train_paths.extend(sorted(by_subset.get(f"subset{s}", [])))
        held_paths = sorted(by_subset.get(held_subset, []))

        train_txt = list_dir / f"fold{fold}_train.txt"
        val_txt = list_dir / f"fold{fold}_val.txt"
        test_txt = list_dir / f"fold{fold}_test.txt"
        write_list_file(train_txt, train_paths)
        write_list_file(val_txt, held_paths)
        write_list_file(test_txt, held_paths)

        yaml_path = output_config_dir / f"{name_prefix}_fold{fold}.yaml"
        cfg = {
            "path": detection_root.as_posix(),
            "train": f"lists/fold{fold}_train.txt",
            "val": f"lists/fold{fold}_val.txt",
            "test": f"lists/fold{fold}_test.txt",
            "names": {0: "nodule"},
            "fold": fold,
            "held_out_subset": held_subset,
            "note": "fold yaml uses held-out subset as val/test for diagnostic only",
        }
        yaml_path.write_text(yaml.dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")

        stats_rows.append(
            {
                "fold": fold,
                "subset": held_subset,
                "train_images": len(train_paths),
                "val_images": len(held_paths),
                "test_images": len(held_paths),
                "train_instances": count_instances(detection_root, train_paths),
                "val_instances": count_instances(detection_root, held_paths),
            }
        )
        print(
            f"fold{fold} ({held_subset}): train_images={len(train_paths)} "
            f"held_out_images={len(held_paths)} train_instances={stats_rows[-1]['train_instances']} "
            f"held_instances={stats_rows[-1]['val_instances']}"
        )

    return stats_rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Build 10-fold LUNA16 YOLO configs from existing data.")
    parser.add_argument(
        "--variant",
        choices=("d1_center", "d0_adjacent", "both"),
        default="both",
        help="Which dataset variant to process.",
    )
    parser.add_argument("--luna16-dir", default="data/LUNA16")
    parser.add_argument("--project-root", default=".", help="Project root for relative paths in YAML.")
    parser.add_argument(
        "--relative-paths",
        action="store_true",
        help="Write paths relative to detection root (default: absolute paths in list files).",
    )
    args = parser.parse_args()
    use_absolute = not args.relative_paths

    root = Path(args.project_root).resolve()
    luna16 = (root / args.luna16_dir).resolve()

    variants = []
    if args.variant in ("d1_center", "both"):
        variants.append(
            (
                root / "datasets_D1_center" / "detection",
                root / "configs" / "folds_D1_center",
                "luna16_D1",
                use_absolute,
            )
        )
    if args.variant in ("d0_adjacent", "both"):
        variants.append(
            (
                root / "datasets" / "detection",
                root / "configs" / "folds_D0_adjacent",
                "luna16_D0",
                use_absolute,
            )
        )

    print(
        "NOTE: fold yaml uses held-out subset as val/test for diagnostic only "
        "(checkpoint selection leakage if used for training)."
    )

    for detection_root, config_dir, prefix, abs_paths in variants:
        if not detection_root.exists():
            print(f"skip missing: {detection_root}")
            continue
        print(f"\n=== {prefix} ({detection_root}) absolute_paths={abs_paths} ===")
        stats = build_folds(detection_root, luna16, config_dir, prefix, abs_paths)
        csv_path = config_dir / f"{prefix}_fold_stats.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(stats[0].keys()))
            writer.writeheader()
            writer.writerows(stats)
        print(f"wrote configs under {config_dir}")
        print(f"wrote stats {csv_path}")


if __name__ == "__main__":
    main()
