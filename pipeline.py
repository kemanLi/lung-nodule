from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass
class Detection:
    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float


def load_grayscale(path: str) -> np.ndarray:
    suffix = Path(path).suffix.lower()
    if suffix in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}:
        # cv2.imread can fail on Windows paths containing non-ASCII characters.
        image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(path)
        return image
    if suffix == ".dcm":
        import pydicom

        ds = pydicom.dcmread(path)
        arr = ds.pixel_array.astype(np.float32)
        slope = float(getattr(ds, "RescaleSlope", 1.0))
        intercept = float(getattr(ds, "RescaleIntercept", 0.0))
        return window_to_uint8(arr * slope + intercept)
    if suffix == ".mhd":
        import SimpleITK as sitk

        image = sitk.ReadImage(path)
        volume = sitk.GetArrayFromImage(image)
        return window_to_uint8(volume[volume.shape[0] // 2])
    raise ValueError(f"Unsupported image format: {path}")


def window_to_uint8(image: np.ndarray, hu_min: int = -1200, hu_max: int = 600) -> np.ndarray:
    image = np.clip(image.astype(np.float32), hu_min, hu_max)
    image = (image - hu_min) / float(hu_max - hu_min)
    return (image * 255).astype(np.uint8)


def yolo_detect(image: np.ndarray, weights: str, conf: float) -> list[Detection]:
    from models.detection import register_yolo_custom_layers
    from ultralytics import YOLO

    register_yolo_custom_layers()
    model = YOLO(weights)
    results = model.predict(cv2.cvtColor(image, cv2.COLOR_GRAY2BGR), conf=conf, verbose=False)
    detections: list[Detection] = []
    for result in results:
        if result.boxes is None:
            continue
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy.cpu().numpy()[0].astype(int).tolist()
            score = float(box.conf.cpu().numpy()[0])
            detections.append(Detection(x1, y1, x2, y2, score))
    return detections


def heuristic_detect(image: np.ndarray, max_boxes: int = 5) -> list[Detection]:
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates: list[tuple[float, Detection]] = []
    h, w = image.shape[:2]
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 20 or area > (h * w * 0.05):
            continue
        x, y, bw, bh = cv2.boundingRect(contour)
        ratio = bw / max(bh, 1)
        if ratio < 0.4 or ratio > 2.5:
            continue
        pad = max(6, int(max(bw, bh) * 0.35))
        x1, y1 = max(0, x - pad), max(0, y - pad)
        x2, y2 = min(w - 1, x + bw + pad), min(h - 1, y + bh + pad)
        center_bias = 1.0 - min(abs((x + bw / 2) - w / 2) / (w / 2), 1.0) * 0.25
        score = float(min(0.95, 0.35 + area / 500.0) * center_bias)
        candidates.append((score, Detection(x1, y1, x2, y2, score)))

    candidates.sort(key=lambda item: item[0], reverse=True)
    if candidates:
        return [det for _, det in candidates[:max_boxes]]

    size = min(h, w) // 8
    cx, cy = w // 2, h // 2
    return [Detection(cx - size, cy - size, cx + size, cy + size, 0.25)]


def crop_patch(image: np.ndarray, det: Detection, size: int = 64) -> np.ndarray:
    cx = (det.x1 + det.x2) // 2
    cy = (det.y1 + det.y2) // 2
    half = size // 2
    padded = cv2.copyMakeBorder(image, half, half, half, half, cv2.BORDER_CONSTANT, value=0)
    cx += half
    cy += half
    return padded[cy - half : cy + half, cx - half : cx + half]


def segment_patch(patch: np.ndarray, weights: str | None = None) -> np.ndarray:
    if weights and Path(weights).exists():
        import torch
        from models.segmentation import build_segmentation_model

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ckpt = torch.load(weights, map_location=device)
        model = build_segmentation_model(ckpt.get("model_type", "unet"), base=int(ckpt.get("base", 32))).to(device)
        model.load_state_dict(ckpt["model"])
        model.eval()
        tensor = torch.from_numpy((patch.astype(np.float32) / 255.0)[None, None]).to(device)
        with torch.no_grad():
            pred = torch.sigmoid(model(tensor)).cpu().numpy()[0, 0]
        return (pred > 0.5).astype(np.uint8) * 255

    blur = cv2.GaussianBlur(patch, (3, 3), 0)
    _, mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask


def draw_results(image: np.ndarray, detections: list[Detection], masks: list[np.ndarray]) -> np.ndarray:
    canvas = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    for idx, det in enumerate(detections):
        cv2.rectangle(canvas, (det.x1, det.y1), (det.x2, det.y2), (0, 220, 255), 1)
        cv2.putText(canvas, f"{det.confidence:.2f}", (det.x1, max(14, det.y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 220, 255), 1)
        if idx < len(masks):
            mask = masks[idx]
            cx = (det.x1 + det.x2) // 2
            cy = (det.y1 + det.y2) // 2
            half = mask.shape[0] // 2
            x1, y1 = max(0, cx - half), max(0, cy - half)
            x2, y2 = min(image.shape[1], cx + half), min(image.shape[0], cy + half)
            mx1, my1 = half - (cx - x1), half - (cy - y1)
            mx2, my2 = mx1 + (x2 - x1), my1 + (y2 - y1)
            local = mask[my1:my2, mx1:mx2] > 0
            canvas[y1:y2, x1:x2][local] = (120, 200, 80)
    return canvas


def run_pipeline(image_path: str, output_dir: str = "outputs", det_weights: str | None = None, seg_weights: str | None = None, conf: float = 0.25) -> dict:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    image = load_grayscale(image_path)
    image = cv2.resize(image, (640, 640), interpolation=cv2.INTER_LINEAR) if image.shape != (640, 640) else image

    if det_weights and Path(det_weights).exists():
        detections = yolo_detect(image, det_weights, conf)
    else:
        detections = heuristic_detect(image)

    patches, masks = [], []
    for idx, det in enumerate(detections):
        patch = crop_patch(image, det)
        mask = segment_patch(patch, seg_weights)
        patch_path = out_dir / f"patch_{idx:02d}.png"
        mask_path = out_dir / f"mask_{idx:02d}.png"
        cv2.imwrite(str(patch_path), patch)
        cv2.imwrite(str(mask_path), mask)
        patches.append(str(patch_path))
        masks.append(mask)

    input_path = out_dir / "input.png"
    cv2.imwrite(str(input_path), image)

    overlay = draw_results(image, detections, masks)
    overlay_path = out_dir / "overlay.png"
    cv2.imwrite(str(overlay_path), overlay)
    result = {
        "image_path": image_path,
        "input_path": str(input_path),
        "overlay_path": str(overlay_path),
        "patch_paths": patches,
        "detections": [asdict(det) for det in detections],
        "mode": "model" if det_weights and Path(det_weights).exists() else "heuristic_demo",
    }
    (out_dir / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run detection -> patch -> segmentation demo pipeline.")
    parser.add_argument("image")
    parser.add_argument("--output-dir", default="outputs/demo")
    parser.add_argument("--det-weights", default=None)
    parser.add_argument("--seg-weights", default=None)
    parser.add_argument("--conf", type=float, default=0.25)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(json.dumps(run_pipeline(args.image, args.output_dir, args.det_weights, args.seg_weights, args.conf), ensure_ascii=False, indent=2))
