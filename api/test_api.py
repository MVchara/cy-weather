# tests/test_main.py
import pytest
from fastapi.testclient import TestClient
from httpx import Response
import respx
from main import app

client = TestClient(app)


# ----------------------------
# Health check
# ----------------------------
def test_health_check():
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


# ----------------------------
# Weather endpoints (mock API)
# ----------------------------
@pytest.mark.asyncio
@respx.mock
async def test_get_current_weather():
    # Mock de l'API géocoding
    respx.get("https://geocoding-api.open-meteo.com/v1/search").mock(
        return_value=Response(
            200,
            json={
                "results": [
                    {
                        "name": "Paris",
                        "latitude": 48.8566,
                        "longitude": 2.3522,
                        "country_code": "FR",
                    }
                ]
            },
        )
    )

    # Mock de l'API météo
    respx.get("https://api.open-meteo.com/v1/forecast").mock(
        return_value=Response(
            200,
            json={
                "current": {
                    "temperature_2m": 15.5,
                    "apparent_temperature": 14.0,
                    "relative_humidity_2m": 60,
                    "pressure_msl": 1012,
                    "wind_speed_10m": 5.5,
                    "weather_code": 1,
                    "time": "2026-01-13T12:00:00",
                }
            },
        )
    )

    response = client.get("/api/weather/current?city=Paris")
    assert response.status_code == 200
    data = response.json()
    assert data["city"] == "Paris"
    assert data["country"] == "FR"
    assert "weather" in data
    assert data["weather"]["temperature"] == 15.5
    assert data["weather"]["description"] == "Principalement dégagé"


@pytest.mark.asyncio
@respx.mock
async def test_get_weather_forecast():
    # Mock de l'API géocoding
    respx.get("https://geocoding-api.open-meteo.com/v1/search").mock(
        return_value=Response(
            200,
            json={
                "results": [
                    {
                        "name": "Paris",
                        "latitude": 48.8566,
                        "longitude": 2.3522,
                        "country_code": "FR",
                    }
                ]
            },
        )
    )

    # Mock de l'API météo pour le forecast
    respx.get("https://api.open-meteo.com/v1/forecast").mock(
        return_value=Response(
            200,
            json={
                "daily": {
                    "time": ["2026-01-13", "2026-01-14", "2026-01-15"],
                    "temperature_2m_max": [16, 17, 18],
                    "temperature_2m_min": [6, 7, 8],
                    "wind_speed_10m_max": [5.0, 4.5, 6.0],
                    "weather_code": [0, 1, 3],
                    "precipitation_probability_max": [10, 20, 0],
                }
            },
        )
    )

    response = client.get("/api/weather/forecast?city=Paris")
    assert response.status_code == 200
    data = response.json()
    assert data["city"] == "Paris"
    assert data["country"] == "FR"
    assert len(data["forecast"]) == 3
    assert data["forecast"][0]["temp_max"] == 16
    assert data["forecast"][1]["description"] == "Principalement dégagé"
