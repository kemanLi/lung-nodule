from __future__ import annotations

import sys
from pathlib import Path

# Ensure the project root is importable so we can reuse pipeline.py / models/.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi import FastAPI  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402

from backend.api import analyze, history, metrics, models  # noqa: E402
from backend.config import FILES_ROOT, ensure_dirs  # noqa: E402

app = FastAPI(
    title="肺结节检测与分割辅助诊断系统 API",
    description="基于 YOLO + U-Net 的肺结节检测与分割推理服务",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def _startup() -> None:
    ensure_dirs()


@app.get("/api/health", tags=["health"])
def health() -> dict[str, str]:
    return {"status": "ok"}


app.include_router(analyze.router)
app.include_router(history.router)
app.include_router(models.router)
app.include_router(metrics.router)

# Serve uploads / outputs / runs artifacts read-only under /files.
app.mount("/files", StaticFiles(directory=str(FILES_ROOT)), name="files")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)
