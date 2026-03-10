"""
Unit tests – CareLink Scraper
"""
from unittest.mock import MagicMock, patch
from datetime import datetime
from app.services.carelink_scraper import CareLinkScraper


def test_authenticate_success():
    scraper = CareLinkScraper()

    with patch.object(scraper.session, "post") as mock_post:
        mock_post.return_value.status_code = 200
        result = scraper.authenticate()

    assert result is True
    assert scraper._authenticated is True


def test_authenticate_failure():
    scraper = CareLinkScraper()

    with patch.object(scraper.session, "post") as mock_post:
        mock_post.return_value.status_code = 401
        result = scraper.authenticate()

    assert result is False
    assert scraper._authenticated is False


def test_fetch_without_auth_raises():
    scraper = CareLinkScraper()

    try:
        scraper.fetch_recent_readings()
        assert False, "Should have raised RuntimeError"
    except RuntimeError as e:
        assert "Not authenticated" in str(e)


def test_parse_response_valid():
    scraper = CareLinkScraper()

    data = {
        "sgs": [
            {"timestamp": 1741500000000, "sg": 120},
            {"timestamp": 1741500300000, "sg": 115},
        ]
    }

    readings = scraper._parse_response(data)

    assert len(readings) == 2
    assert readings[0].glucose_mg_dl == 120.0
    assert readings[1].glucose_mg_dl == 115.0


def test_parse_response_skips_zero_glucose():
    scraper = CareLinkScraper()

    data = {
        "sgs": [
            {"timestamp": 1741500000000, "sg": 0},
            {"timestamp": 1741500300000, "sg": 120},
        ]
    }

    readings = scraper._parse_response(data)

    assert len(readings) == 1
    assert readings[0].glucose_mg_dl == 120.0


def test_parse_response_empty():
    scraper = CareLinkScraper()
    readings = scraper._parse_response({"sgs": []})
    assert len(readings) == 0
