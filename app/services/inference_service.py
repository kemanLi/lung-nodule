from __future__ import annotations

from pathlib import Path

from pipeline import run_pipeline


class InferenceService:
    def __init__(self, output_root: str = "outputs/app") -> None:
        self.output_root = Path(output_root)
        self.output_root.mkdir(parents=True, exist_ok=True)

    def analyze(self, image_path: str, det_weights: str | None = None, seg_weights: str | None = None) -> dict:
        case_dir = self.output_root / Path(image_path).stem
        case_dir.mkdir(parents=True, exist_ok=True)
        return run_pipeline(
            image_path=image_path,
            output_dir=str(case_dir),
            det_weights=det_weights or None,
            seg_weights=seg_weights or None,
        )
