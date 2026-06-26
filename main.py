"""Run from project root: uvicorn main:app --reload --port 8000"""
from backend.main import app

__all__ = ["app"]
