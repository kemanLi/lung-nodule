from __future__ import annotations

import shutil
import sys
from datetime import datetime
from pathlib import Path

# Project root on sys.path (same as backend.main).
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from flask import (  # noqa: E402
    Flask,
    abort,
    flash,
    redirect,
    render_template,
    request,
    send_from_directory,
    url_for,
)

from backend.config import ALLOWED_IMAGE_SUFFIXES, FILES_ROOT, UPLOAD_DIR, ensure_dirs  # noqa: E402
from backend.database.db import CaseDatabase  # noqa: E402
from backend.services import metrics_service, model_registry  # noqa: E402
from backend.services.inference_service import InferenceService  # noqa: E402

_inference = InferenceService()
_db = CaseDatabase()


def create_app() -> Flask:
    app = Flask(__name__, template_folder="templates", static_folder="static")
    app.secret_key = "lung-nodule-mvp-dev"  # noqa: S105 — local demo only

    @app.before_request
    def _ensure_dirs() -> None:
        ensure_dirs()

    @app.route("/files/<path:filepath>")
    def files(filepath: str):
        target = (FILES_ROOT / filepath).resolve()
        try:
            target.relative_to(FILES_ROOT.resolve())
        except ValueError:
            abort(404)
        if not target.is_file():
            abort(404)
        return send_from_directory(FILES_ROOT, filepath)

    @app.route("/")
    def index():
        recent = _db.list_cases(limit=10)
        det_models = model_registry.list_detection_weights()
        seg_models = model_registry.list_segmentation_weights()
        return render_template(
            "index.html",
            recent=recent,
            det_models=det_models,
            seg_models=seg_models,
        )

    @app.route("/analyze", methods=["POST"])
    def analyze():
        upload = request.files.get("file")
        if not upload or not upload.filename:
            flash("请选择要上传的 CT 图像。", "error")
            return redirect(url_for("index"))

        suffix = Path(upload.filename).suffix.lower()
        if suffix not in ALLOWED_IMAGE_SUFFIXES:
            flash(f"不支持的文件类型: {suffix or '未知'}", "error")
            return redirect(url_for("index"))

        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_name = Path(upload.filename).stem[:40]
        saved_path = UPLOAD_DIR / f"{stamp}_{safe_name}{suffix}"
        upload.save(saved_path)

        try:
            result = _inference.analyze(
                image_path=saved_path,
                det_weight_id=request.form.get("det_weights") or None,
                seg_weight_id=request.form.get("seg_weights") or None,
                conf=float(request.form.get("conf") or 0.25),
            )
        except Exception as exc:  # noqa: BLE001
            flash(f"推理失败: {exc}", "error")
            return redirect(url_for("index"))

        _db.add_case(result)
        return redirect(url_for("case_detail", case_id=result["case_id"]))

    @app.route("/case/<case_id>")
    def case_detail(case_id: str):
        row = _db.get_case(case_id)
        if not row:
            abort(404)
        return render_template("case.html", case=row)

    @app.route("/history")
    def history():
        search = request.args.get("q", "").strip() or None
        cases = _db.list_cases(limit=100, search=search)
        return render_template("history.html", cases=cases, search=search or "")

    @app.route("/history/<case_id>/delete", methods=["POST"])
    def history_delete(case_id: str):
        if not _db.delete_case(case_id):
            flash("未找到该病例。", "error")
        else:
            flash("已删除病例记录。", "ok")
        return redirect(url_for("history"))

    @app.route("/models")
    def models_page():
        return render_template(
            "models.html",
            det_models=model_registry.list_detection_weights(),
            seg_models=model_registry.list_segmentation_weights(),
        )

    @app.route("/experiments")
    def experiments():
        det = metrics_service.detection_metrics()
        seg = metrics_service.segmentation_metrics()
        return render_template(
            "experiments.html",
            det_source=det.get("source"),
            det_rows=det.get("rows") or [],
            seg_source=seg.get("source"),
            seg_rows=seg.get("rows") or [],
        )

    @app.route("/health")
    def health():
        return {"status": "ok"}

    return app


app = create_app()


if __name__ == "__main__":
    ensure_dirs()
    app.run(host="127.0.0.1", port=8000, debug=True)
