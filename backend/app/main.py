"""
NeuroMetabolic Dashboard - FastApi Application Entry Point
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import settings

app = FastAPI(
    title="NeuroMetabolic Dashboard API",
    description="AI-driven glycemic predictedion system for type 1 diabetes management",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", tags=["Health"])
async def root():
    return {"status": "ok", "service": "NeuroMetabolic Dashboard API", "version": "0.1.0"}


@app.get("/health", tags=["Health"])
async def health_check():
    return {"status": "healthy"}