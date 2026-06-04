from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

from backend.config import OUTPUT_DIR, ensure_dirs, to_file_url
from backend.services import model_registry
from pipeline import run_pipeline


class InferenceService:
    """Wraps the existing run_pipeline into a web-friendly service."""

    def __init__(self) -> None:
        ensure_dirs()

    def analyze(
        self,
        image_path: Path,
        det_weight_id: str | None = None,
        seg_weight_id: str | None = None,
        conf: float = 0.25,
    ) -> dict[str, Any]:
        case_id = self._make_case_id(image_path)
        case_dir = OUTPUT_DIR / case_id
        case_dir.mkdir(parents=True, exist_ok=True)

        det_weights = model_registry.resolve_weight(det_weight_id)
        seg_weights = model_registry.resolve_weight(seg_weight_id)

        result = run_pipeline(
            image_path=str(image_path),
            output_dir=str(case_dir),
            det_weights=det_weights,
            seg_weights=seg_weights,
            conf=conf,
        )

        detections = result.get("detections", [])
        mask_urls = [to_file_url(p) for p in sorted(case_dir.glob("mask_*.png"))]
        patch_urls = [to_file_url(p) for p in result.get("patch_paths", [])]
        max_conf = max((float(d.get("confidence", 0.0)) for d in detections), default=0.0)

        input_path = result.get("input_path")
        return {
            "case_id": case_id,
            "image_name": image_path.name,
            "image_url": to_file_url(image_path),
            "input_url": to_file_url(input_path) if input_path else to_file_url(image_path),
            "overlay_url": to_file_url(result["overlay_path"]),
            "detections": detections,
            "patch_urls": patch_urls,
            "mask_urls": mask_urls,
            "nodule_count": len(detections),
            "max_confidence": round(max_conf, 4),
            "mode": result.get("mode", "heuristic_demo"),
            "det_weights": det_weight_id,
            "seg_weights": seg_weight_id,
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }

    @staticmethod
    def _make_case_id(image_path: Path) -> str:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        stem = image_path.stem[:40]
        return f"{stamp}_{stem}"
