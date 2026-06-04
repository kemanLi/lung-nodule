from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, HTTPException

from backend.database.db import CaseDatabase
from backend.schemas import CaseSummary

router = APIRouter(prefix="/api/history", tags=["history"])

_db = CaseDatabase()


@router.get("", response_model=list[CaseSummary])
def list_history(limit: int = 50, search: Optional[str] = None) -> list[CaseSummary]:
    return [CaseSummary(**row) for row in _db.list_cases(limit=limit, search=search)]


@router.get("/{case_id}", response_model=CaseSummary)
def get_history(case_id: str) -> CaseSummary:
    row = _db.get_case(case_id)
    if not row:
        raise HTTPException(status_code=404, detail="未找到该病例")
    return CaseSummary(**row)


@router.delete("/{case_id}")
def delete_history(case_id: str) -> dict[str, bool]:
    ok = _db.delete_case(case_id)
    if not ok:
        raise HTTPException(status_code=404, detail="未找到该病例")
    return {"deleted": True}
