"""
Evaluate a fixed detector on each LUNA16 10-fold YAML (held-out subset as val).
No training — fold difficulty diagnostic only.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import yaml


def main() -> None:
    parser = argparse.ArgumentParser(description="Per-fold val metrics for fold YAMLs.")
    parser.add_argument("--weights", required=True)
    parser.add_argument("--fold-config-dir", default="configs/folds_D1_center")
    parser.add_argument("--name-prefix", default="luna16_D1")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--output", default="runs/detection_ablation/fold_difficulty_M0_on_D1_center.csv")
    args = parser.parse_args()

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit("Install ultralytics first.") from exc

    config_dir = Path(args.fold_config_dir)
    model = YOLO(args.weights)
    rows: list[dict[str, str | int | float]] = []

    for fold in range(10):
        yaml_path = config_dir / f"{args.name_prefix}_fold{fold}.yaml"
        if not yaml_path.exists():
            print(f"missing {yaml_path}")
            continue
        cfg = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
        subset = cfg.get("held_out_subset", f"subset{fold}")
        data_root = Path(cfg["path"])
        val_list = data_root / cfg["val"]
        val_images = [ln for ln in val_list.read_text(encoding="utf-8").splitlines() if ln.strip()]
        instances = 0
        for entry in val_images:
            img = Path(entry)
            if not img.is_absolute():
                img = data_root / entry
            lbl = img.with_suffix(".txt")
            lbl = Path(str(lbl).replace("/images/", "/labels/").replace("\\images\\", "\\labels\\"))
            if lbl.exists():
                instances += sum(1 for ln in lbl.read_text(encoding="utf-8").splitlines() if ln.strip())

        print(f"\nfold{fold} ({subset}) val on {yaml_path.name} ...")
        metrics = model.val(data=str(yaml_path.resolve()), imgsz=args.imgsz, split="val")
        row = {
            "fold": fold,
            "subset": subset,
            "images": len(val_images),
            "instances": instances,
            "precision": float(metrics.box.mp),
            "recall": float(metrics.box.mr),
            "map50": float(metrics.box.map50),
            "map50_95": float(metrics.box.map),
        }
        rows.append(row)
        print(
            f"  P={row['precision']:.4f} R={row['recall']:.4f} "
            f"mAP50={row['map50']:.4f} mAP50-95={row['map50_95']:.4f}"
        )

    if rows:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        means = {k: sum(float(r[k]) for r in rows) / len(rows) for k in ("precision", "recall", "map50", "map50_95")}
        print(f"\nwrote {out}")
        print(
            f"10-fold mean: P={means['precision']:.4f} R={means['recall']:.4f} "
            f"mAP50={means['map50']:.4f} mAP50-95={means['map50_95']:.4f}"
        )
        print(f"subset8 fold: mAP50={next(r['map50'] for r in rows if r['subset']=='subset8'):.4f}")


if __name__ == "__main__":
    main()
