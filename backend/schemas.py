from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel


class Detection(BaseModel):
    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float


class AnalyzeResponse(BaseModel):
    case_id: str
    image_name: str
    image_url: str
    input_url: str
    overlay_url: str
    detections: list[Detection]
    patch_urls: list[str]
    mask_urls: list[str]
    nodule_count: int
    max_confidence: float
    mode: str
    det_weights: Optional[str] = None
    seg_weights: Optional[str] = None
    created_at: str


class WeightInfo(BaseModel):
    id: str
    name: str
    path: str
    size_mb: float


class CaseSummary(BaseModel):
    case_id: str
    image_name: str
    image_url: str
    input_url: str
    overlay_url: str
    nodule_count: int
    max_confidence: float
    mode: str
    det_weights: Optional[str] = None
    seg_weights: Optional[str] = None
    detections: list[Any] = []
    created_at: str


class MetricsResponse(BaseModel):
    source: Optional[str]
    rows: list[dict[str, Any]]
