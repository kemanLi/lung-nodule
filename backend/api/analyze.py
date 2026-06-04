from __future__ import annotations

import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from backend.config import ALLOWED_IMAGE_SUFFIXES, UPLOAD_DIR, ensure_dirs
from backend.database.db import CaseDatabase
from backend.schemas import AnalyzeResponse
from backend.services.inference_service import InferenceService

router = APIRouter(prefix="/api", tags=["analyze"])

_service = InferenceService()
_db = CaseDatabase()


@router.post("/analyze", response_model=AnalyzeResponse)
async def analyze(
    file: UploadFile = File(...),
    det_weights: Optional[str] = Form(None),
    seg_weights: Optional[str] = Form(None),
    conf: float = Form(0.25),
) -> AnalyzeResponse:
    ensure_dirs()
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in ALLOWED_IMAGE_SUFFIXES:
        raise HTTPException(status_code=400, detail=f"不支持的文件类型: {suffix or '未知'}")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_name = Path(file.filename or "image").stem[:40]
    saved_path = UPLOAD_DIR / f"{stamp}_{safe_name}{suffix}"
    with saved_path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        result = _service.analyze(
            image_path=saved_path,
            det_weight_id=det_weights or None,
            seg_weight_id=seg_weights or None,
            conf=conf,
        )
    except Exception as exc:  # noqa: BLE001 - surface inference errors to the client
        raise HTTPException(status_code=500, detail=f"推理失败: {exc}") from exc

    _db.add_case(result)
    return AnalyzeResponse(**result)
