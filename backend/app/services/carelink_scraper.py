"""
CareLink API Scraper – automated CGM data ingestion.
Authenticates with Medtronic CareLink EU and fetches recent glucose readings.
"""
import requests
from datetime import datetime, timezone
from app.core.config import settings
from app.services.carelink_parser import GlucoseReading

CARELINK_BASE_URL = "https://carelink.minimed.eu"
AUTH_URL = f"{CARELINK_BASE_URL}/patient/sso/login"
DATA_URL = f"{CARELINK_BASE_URL}/patient/connect/data"


class CareLinkScraper:
    """Handles authentication and data fetching from CareLink EU API."""

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0",
            "Accept": "application/json",
        })
        self._authenticated = False

    def authenticate(self) -> bool:
        """
        Authenticate with CareLink EU using credentials from settings.
        Returns True if successful, False otherwise.
        """
        try:
            payload = {
                "username": settings.CARELINK_USERNAME,
                "password": settings.CARELINK_PASSWORD,
            }
            response = self.session.post(AUTH_URL, data=payload, timeout=10)
            self._authenticated = response.status_code == 200
            return self._authenticated
        except requests.RequestException:
            self._authenticated = False
            return False

    def fetch_recent_readings(self) -> list[GlucoseReading]:
        """
        Fetch recent glucose readings from CareLink API.
        Returns a list of GlucoseReading objects.
        """
        if not self._authenticated:
            raise RuntimeError("Not authenticated. Call authenticate() first.")

        try:
            response = self.session.get(DATA_URL, timeout=10)
            response.raise_for_status()
            data = response.json()
            return self._parse_response(data)
        except requests.RequestException as e:
            raise RuntimeError(f"Failed to fetch data from CareLink: {e}")

    def _parse_response(self, data: dict) -> list[GlucoseReading]:
        """
        Parse CareLink API JSON response into GlucoseReading objects.
        """
        readings = []

        sgs = data.get("sgs", [])

        for entry in sgs:
            try:
                timestamp_ms = entry.get("timestamp")
                glucose_value = entry.get("sg")

                if not timestamp_ms or not glucose_value:
                    continue

                timestamp = datetime.fromtimestamp(
                    timestamp_ms / 1000,
                    tz=timezone.utc
                ).replace(tzinfo=None)

                glucose = float(glucose_value)

                if glucose <= 0:
                    continue

                readings.append(
                    GlucoseReading(
                        timestamp=timestamp,
                        glucose_mg_dl=glucose,
                    )
                )
            except (ValueError, TypeError, KeyError):
                continue

        return readings
