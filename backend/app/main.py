from fastapi import FastAPI

from backend.app.api.router import api_router
from backend.app.config import get_settings

settings = get_settings()

app = FastAPI(title=settings.app_name)
app.include_router(api_router)
