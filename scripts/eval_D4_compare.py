"""
Evaluate D4 weights on D4, D1 center, and D0 adjacent validation sets.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from ultralytics import YOLO


EVAL_SPECS = [
    ("D4_val", "configs/luna16_D4_2p5d_center_stack.yaml", "D4 2.5D center-stack val"),
    ("D4_on_D1_center", "configs/luna16_D1_center.yaml", "D1 center val (compare M0/V1)"),
    ("D4_on_D0_adjacent", "configs/luna16_subset.yaml", "D0 adjacent val (diagnostic only)"),
]

BASELINE_REF = [
    ("M0/V1_ref", "M0 on D1 center", 0.8005, 0.7822, 0.8359, 0.4629),
    ("M1/V1_ref", "M1 on D1 center", 0.7997, 0.7781, 0.8130, 0.4383),
    ("M0/V0_ref", "M0 on D0 adjacent", 0.7729, 0.6695, 0.7503, 0.3957),
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--weights",
        default="runs/detection_ablation/E4_D4_2p5d_center_stack_yolo11n_auto/weights/best.pt",
    )
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--output", default="runs/detection_ablation/E4_D4_eval_compare.csv")
    args = parser.parse_args()

    model = YOLO(args.weights)
    rows: list[dict[str, str | float]] = []

    for tag, data_yaml, note in EVAL_SPECS:
        print(f"\n=== {tag} ({data_yaml}) ===")
        m = model.val(data=data_yaml, imgsz=args.imgsz, split="val")
        row = {
            "tag": tag,
            "data": data_yaml,
            "note": note,
            "precision": float(m.box.mp),
            "recall": float(m.box.mr),
            "map50": float(m.box.map50),
            "map50_95": float(m.box.map),
        }
        rows.append(row)
        print(
            f"P={row['precision']:.4f} R={row['recall']:.4f} "
            f"mAP50={row['map50']:.4f} mAP50-95={row['map50_95']:.4f}"
        )

    for tag, note, p, r, m50, m95 in BASELINE_REF:
        rows.append(
            {
                "tag": tag,
                "data": "reference",
                "note": note,
                "precision": p,
                "recall": r,
                "map50": m50,
                "map50_95": m95,
            }
        )

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
