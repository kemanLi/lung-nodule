from __future__ import annotations

from pathlib import Path
from typing import Any

from backend.config import PROJECT_ROOT, RUNS_DIR, WEIGHTS_RELEASE_DIR


def _entry(path: Path, kind: str) -> dict[str, Any]:
    rel = path.resolve().relative_to(PROJECT_ROOT).as_posix()
    return {
        "id": rel,
        "name": _display_name(path, kind),
        "path": rel,
        "size_mb": round(path.stat().st_size / 1024 / 1024, 2) if path.exists() else 0.0,
    }


def _display_name(path: Path, kind: str) -> str:
    if WEIGHTS_RELEASE_DIR in path.parents:
        return f"release/{path.stem}"
    # runs/detection/<run>/weights/best.pt -> <run>
    parts = path.parts
    if "weights" in parts:
        idx = parts.index("weights")
        if idx > 0:
            return parts[idx - 1]
    # runs/segmentation/<run>/best.pt -> <run>
    return path.parent.name or path.stem


def list_detection_weights() -> list[dict[str, Any]]:
    found: list[Path] = []
    for base in (RUNS_DIR / "detection", RUNS_DIR / "detection_ablation"):
        if base.exists():
            found.extend(sorted(base.glob("*/weights/best.pt")))
    if WEIGHTS_RELEASE_DIR.exists():
        found.extend(sorted(WEIGHTS_RELEASE_DIR.glob("*yolo*best*.pt")))
    return [_entry(p, "detection") for p in _dedupe(found)]


def list_segmentation_weights() -> list[dict[str, Any]]:
    found: list[Path] = []
    base = RUNS_DIR / "segmentation"
    if base.exists():
        found.extend(sorted(base.glob("*/best.pt")))
    if WEIGHTS_RELEASE_DIR.exists():
        found.extend(sorted(WEIGHTS_RELEASE_DIR.glob("*unet*best*.pt")))
    return [_entry(p, "segmentation") for p in _dedupe(found)]


def resolve_weight(weight_id: str | None) -> str | None:
    """Resolve a registry id (relative path) to an absolute path string."""
    if not weight_id:
        return None
    candidate = (PROJECT_ROOT / weight_id).resolve()
    if candidate.exists():
        return str(candidate)
    return None


def _dedupe(paths: list[Path]) -> list[Path]:
    seen: set[Path] = set()
    result: list[Path] = []
    for p in paths:
        rp = p.resolve()
        if rp not in seen:
            seen.add(rp)
            result.append(rp)
    return result
