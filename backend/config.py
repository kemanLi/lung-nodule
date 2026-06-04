from __future__ import annotations

from pathlib import Path

# Project root = parent of the backend package directory.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

UPLOAD_DIR = PROJECT_ROOT / "uploads"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "api"
DB_PATH = PROJECT_ROOT / "outputs" / "cases.sqlite"

RUNS_DIR = PROJECT_ROOT / "runs"
WEIGHTS_RELEASE_DIR = PROJECT_ROOT / "weights_release"

# Directories that are exposed read-only through the /files static mount.
FILES_ROOT = PROJECT_ROOT

ALLOWED_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".dcm", ".mhd"}


def ensure_dirs() -> None:
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)


def to_file_url(path: Path | str) -> str:
    """Convert an absolute path under the project root into a /files/... URL."""
    p = Path(path).resolve()
    rel = p.relative_to(FILES_ROOT)
    return "/files/" + rel.as_posix()
