"""
Application Configuration – loaded from environment variables / .env file
"""
from pydantic_settings import BaseSettings, SettingsConfigDict  # <- zmiana
from typing import List


class Settings(BaseSettings):
    model_config = SettingsConfigDict(          # <- zmiana
        env_file=".env",
        env_file_encoding="utf-8",
    )

    # App
    APP_NAME: str = "NeuroMetabolic Dashboard"
    DEBUG: bool = False
    API_V1_STR: str = "/api/v1"


    # CORS
    ALLOWED_ORIGINS: List[str] = ["http://localhost:3000", "http://localhost:5173"]

    # InfluxDB
    INFLUXDB_URL: str = "http://localhost:8086"
    INFLUXDB_TOKEN: str = ""
    INFLUXDB_ORG: str = "nmd"
    INFLUXDB_BUCKET: str = "cgm_data"


    # CareLink
    CARELINK_USERNAME: str = ""
    CARELINK_PASSWORD: str = ""


settings = Settings()
