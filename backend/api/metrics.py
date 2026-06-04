from __future__ import annotations

from fastapi import APIRouter

from backend.schemas import MetricsResponse
from backend.services import metrics_service

router = APIRouter(prefix="/api/metrics", tags=["metrics"])


@router.get("/detection", response_model=MetricsResponse)
def detection_metrics() -> MetricsResponse:
    return MetricsResponse(**metrics_service.detection_metrics())


@router.get("/segmentation", response_model=MetricsResponse)
def segmentation_metrics() -> MetricsResponse:
    return MetricsResponse(**metrics_service.segmentation_metrics())
