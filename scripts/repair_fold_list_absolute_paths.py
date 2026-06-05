"""Convert fold list txt files to absolute image paths (fixes Ultralytics val access warnings)."""

from __future__ import annotations

import argparse
from pathlib import Path


def repair(detection_root: Path) -> None:
    list_dir = detection_root / "lists"
    if not list_dir.exists():
        raise SystemExit(f"No lists/ under {detection_root}")
    root = detection_root.resolve()
    for list_file in sorted(list_dir.glob("fold*_*.txt")):
        lines = [ln.strip() for ln in list_file.read_text(encoding="utf-8").splitlines() if ln.strip()]
        out_lines = []
        for ln in lines:
            p = Path(ln)
            if not p.is_absolute():
                p = root / ln
            if not p.exists():
                print(f"warning: missing {p}")
            out_lines.append(p.resolve().as_posix())
        list_file.write_text("\n".join(out_lines) + ("\n" if out_lines else ""), encoding="utf-8")
        print(f"repaired {list_file.name} ({len(out_lines)} paths)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--detection-root", default="datasets_D1_center/detection")
    args = parser.parse_args()
    repair(Path(args.detection_root))


if __name__ == "__main__":
    main()
