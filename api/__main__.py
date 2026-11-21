"""Run: uvicorn api.main:app --reload --port 8000"""

from api.main import app

__all__ = ["app"]
