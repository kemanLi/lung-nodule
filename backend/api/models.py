from __future__ import annotations

from fastapi import APIRouter

from backend.schemas import WeightInfo
from backend.services import model_registry

router = APIRouter(prefix="/api/models", tags=["models"])


@router.get("/detection", response_model=list[WeightInfo])
def detection_models() -> list[WeightInfo]:
    return [WeightInfo(**item) for item in model_registry.list_detection_weights()]


@router.get("/segmentation", response_model=list[WeightInfo])
def segmentation_models() -> list[WeightInfo]:
    return [WeightInfo(**item) for item in model_registry.list_segmentation_weights()]
