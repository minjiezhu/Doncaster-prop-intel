from __future__ import annotations

import httpx
from fastapi import APIRouter
from fastapi.responses import JSONResponse

from backend.app.config import get_settings

router = APIRouter(tags=["health"])


@router.get("/health")
def health() -> JSONResponse:
    settings = get_settings()
    checks: dict[str, str] = {}

    # Weaviate readiness
    try:
        r = httpx.get(f"{settings.weaviate_url}/v1/.well-known/ready", timeout=3.0)
        checks["weaviate"] = "ok" if r.status_code == 200 else f"http_{r.status_code}"
    except httpx.RequestError as exc:
        checks["weaviate"] = f"unreachable: {exc}"

    # Ollama reachability (list local models)
    try:
        r = httpx.get(f"{settings.ollama_base_url}/api/tags", timeout=3.0)
        checks["ollama"] = "ok" if r.status_code == 200 else f"http_{r.status_code}"
    except httpx.RequestError as exc:
        checks["ollama"] = f"unreachable: {exc}"

    all_ok = all(v == "ok" for v in checks.values())
    status_code = 200 if all_ok else 503
    return JSONResponse(
        status_code=status_code,
        content={"status": "ok" if all_ok else "degraded", "checks": checks},
    )
