"""
NeuroMetabolic Dashboard – FastAPI Application Entry Point
"""
import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.glucose import router as glucose_router
from app.core.config import settings
from app.services.carelink_scraper import CareLinkScraper
from app.services.influxdb_service import get_influxdb

logger = logging.getLogger(__name__)

SCRAPER_INTERVAL = 5 * 60    # fetch every 5 minutes
AUTH_TTL = 60 * 60           # re-authenticate every 60 minutes


async def carelink_background_task() -> None:
    """
    Background task: fetches CGM data from CareLink every 5 minutes.
    Re-authenticates every 60 minutes instead of on every cycle.
    """
    if not settings.CARELINK_USERNAME or not settings.CARELINK_PASSWORD:
        logger.warning("CareLink credentials not set – background scraper disabled.")
        return

    scraper = CareLinkScraper()
    seconds_since_auth: float = AUTH_TTL  # trigger auth immediately on first run

    logger.info("CareLink background scraper started.")

    while True:
        try:
            if seconds_since_auth >= AUTH_TTL:
                if not scraper.authenticate():
                    logger.warning("Background scraper: CareLink auth failed.")
                    await asyncio.sleep(SCRAPER_INTERVAL)
                    seconds_since_auth += SCRAPER_INTERVAL
                    continue
                seconds_since_auth = 0
                logger.info("Background scraper: re-authenticated with CareLink.")

            readings = scraper.fetch_recent_readings()
            if readings:
                db = get_influxdb()
                count = db.write_glucose_readings(readings)
                logger.info(f"Background scraper: saved {count} readings.")
            else:
                logger.info("Background scraper: no new readings.")

        except Exception as e:
            logger.error(f"Background scraper error: {e}")

        await asyncio.sleep(SCRAPER_INTERVAL)
        seconds_since_auth += SCRAPER_INTERVAL


@asynccontextmanager
async def lifespan(app: FastAPI):
    asyncio.create_task(carelink_background_task())
    yield


app = FastAPI(
    title="NeuroMetabolic Dashboard API",
    description="AI-driven glycemic prediction system for type 1 diabetes management",
    version="0.2.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
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
    return {
        "status": "ok",
        "service": "NeuroMetabolic Dashboard API",
        "version": "0.2.0",
    }


@app.get("/health", tags=["Health"])
async def health_check():
    return {"status": "healthy"}


app.include_router(glucose_router, prefix="/api/v1")
