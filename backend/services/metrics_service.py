from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from backend.config import OUTPUT_DIR, PROJECT_ROOT, RUNS_DIR

# Candidate CSV files produced by eval_detection_ablation.py.
DETECTION_ABLATION_CSVS = [
    PROJECT_ROOT / "outputs" / "ablation_detection_full.csv",
    PROJECT_ROOT / "outputs" / "ablation_detection.csv",
]
SEGMENTATION_ABLATION_CSVS = [
    PROJECT_ROOT / "outputs" / "ablation_segmentation_full.csv",
    PROJECT_ROOT / "outputs" / "ablation_segmentation.csv",
]


def _read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows: list[dict[str, Any]] = []
        for row in reader:
            parsed: dict[str, Any] = {}
            for key, value in row.items():
                try:
                    parsed[key] = float(value)
                except (TypeError, ValueError):
                    parsed[key] = value
            rows.append(parsed)
        return rows


def detection_metrics() -> dict[str, Any]:
    for csv_path in DETECTION_ABLATION_CSVS:
        if csv_path.exists():
            return {"source": csv_path.name, "rows": _read_csv(csv_path)}
    return {"source": None, "rows": _scan_results_csv(RUNS_DIR / "detection_ablation")}


def segmentation_metrics() -> dict[str, Any]:
    for csv_path in SEGMENTATION_ABLATION_CSVS:
        if csv_path.exists():
            return {"source": csv_path.name, "rows": _read_csv(csv_path)}
    return {"source": None, "rows": []}


def _scan_results_csv(base: Path) -> list[dict[str, Any]]:
    """Fallback: read the last row of each run's results.csv (Ultralytics)."""
    rows: list[dict[str, Any]] = []
    if not base.exists():
        return rows
    for results in sorted(base.glob("*/results.csv")):
        data = _read_csv(results)
        if not data:
            continue
        last = data[-1]
        rows.append(
            {
                "experiment": results.parent.name,
                "precision": last.get("metrics/precision(B)"),
                "recall": last.get("metrics/recall(B)"),
                "map50": last.get("metrics/mAP50(B)"),
                "map50_95": last.get("metrics/mAP50-95(B)"),
            }
        )
    return rows
