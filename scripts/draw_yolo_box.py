"""Draw a YOLO label on an image. Usage: python scripts/draw_yolo_box.py <image> <label_line> [output]"""
import sys
from pathlib import Path

try:
    import cv2
    import numpy as np

    def load_gray(path: str):
        data = np.fromfile(path, dtype=np.uint8)
        return cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)

    def save_bgr(path: str, bgr):
        cv2.imencode(".png", bgr)[1].tofile(path)

    def draw_box(gray, x1, y1, x2, y2):
        canvas = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 255, 255), 2)
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        cv2.circle(canvas, (cx, cy), 4, (0, 0, 255), -1)
        cv2.putText(canvas, "YOLO", (x1, max(18, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
        return canvas

except ImportError:
    from PIL import Image, ImageDraw

    def load_gray(path: str):
        return Image.open(path).convert("L")

    def save_bgr(path: str, img):
        img.save(path)

    def draw_box(gray, x1, y1, x2, y2):
        rgb = gray.convert("RGB")
        draw = ImageDraw.Draw(rgb)
        draw.rectangle([x1, y1, x2, y2], outline=(255, 255, 0), width=2)
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        draw.ellipse([cx - 4, cy - 4, cx + 4, cy + 4], fill=(255, 0, 0))
        return rgb


def main() -> None:
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)
    img_path = sys.argv[1]
    parts = sys.argv[2].split()
    if len(parts) != 5:
        raise SystemExit("Label must be: class cx cy w h")
    _, cx, cy, w, h = map(float, parts)
    out = sys.argv[3] if len(sys.argv) > 3 else "outputs/label_bbox_demo.png"

    gray = load_gray(img_path)
    if hasattr(gray, "shape"):
        H, W = gray.shape[:2]
    else:
        W, H = gray.size
    x1 = int((cx - w / 2) * W)
    y1 = int((cy - h / 2) * H)
    x2 = int((cx + w / 2) * W)
    y2 = int((cy + h / 2) * H)
    canvas = draw_box(gray, x1, y1, x2, y2)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    save_bgr(out, canvas)
    print(f"Image: {W}x{H}")
    print(f"Box pixels: ({x1},{y1})-({x2},{y2}), size {x2-x1}x{y2-y1}")
    print(f"Center: ({int(cx*W)}, {int(cy*H)})")
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
