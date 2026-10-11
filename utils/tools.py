import json
import os
import urllib.parse
import urllib.request
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from langchain_core.tools import tool

from utils.db import sqlite_connection

NOTES_FILE = Path("notes.json")

WEATHER_CODES = {
    0: "clear sky", 1: "mostly clear", 2: "partly cloudy", 3: "overcast",
    45: "foggy", 48: "foggy",
    51: "light drizzle", 53: "drizzle", 55: "heavy drizzle",
    61: "light rain", 63: "rain", 65: "heavy rain",
    71: "light snow", 73: "snow", 75: "heavy snow",
    80: "rain showers", 81: "rain showers", 82: "heavy rain showers",
    95: "thunderstorm", 96: "thunderstorm with hail", 99: "thunderstorm with hail",
}


def _get_json(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=10) as resp:
        return json.load(resp)


# NOTE: the docstring is NOT just documentation. The model reads it to decide
# when to call the tool, so write it for the model.
@tool
def get_current_time() -> str:
    """Get the current date and time. Use this whenever the user asks what
    time it is, what day it is, or what today's date is."""
    tz_name = os.environ.get("IVY_TIMEZONE")  # e.g. "Asia/Kolkata"
    now = datetime.now(
        ZoneInfo(tz_name)) if tz_name else datetime.now().astimezone()
    return now.strftime("%A, %d %B %Y, %I:%M %p")


@tool
def get_weather(city: str | None = None) -> str:
    """Get the current weather for a city. If the user doesn't name a city,
    leave city empty and their home city will be used."""
    city = city or os.environ.get("IVY_HOME_CITY")
    if not city:
        return "No city was given and no home city is set. Ask the user which city."
    try:
        geo = _get_json(
            "https://geocoding-api.open-meteo.com/v1/search?"
            + urllib.parse.urlencode({"name": city, "count": 1})
        )
        results = geo.get("results")
        if not results:
            return f"Couldn't find a place called {city}."
        place = results[0]

        params = urllib.parse.urlencode({
            "latitude": place["latitude"],
            "longitude": place["longitude"],
            "current": "temperature_2m,apparent_temperature,weather_code,wind_speed_10m",
        })
        cur = _get_json(
            "https://api.open-meteo.com/v1/forecast?" + params)["current"]
        desc = WEATHER_CODES.get(cur["weather_code"], "unknown conditions")
        return (
            f"{place['name']}: {cur['temperature_2m']} degrees Celsius, "
            f"feels like {cur['apparent_temperature']}, {desc}, "
            f"wind {cur['wind_speed_10m']} kilometres per hour."
        )
    except Exception as e:  # return errors as text so the model can react to them
        return f"Weather lookup failed: {e}"


def _load_notes() -> list[str]:
    if NOTES_FILE.exists():
        return json.loads(NOTES_FILE.read_text(encoding="utf-8"))
    return []


@tool
def save_note(text: str) -> str:
    """Save a short note for the user. Use when they ask you to remember,
    note down, or jot something."""
    notes = _load_notes()
    notes.append(text)
    NOTES_FILE.write_text(json.dumps(notes, indent=2), encoding="utf-8")
    return "Saved."


@tool
def list_notes() -> str:
    """Read back all of the user's saved notes."""
    notes = _load_notes()
    if not notes:
        return "There are no saved notes."
    return "\n".join(f"{i}. {n}" for i, n in enumerate(notes, 1))


TOOLS = [
    get_current_time,
    get_weather,
    save_note,
    list_notes,
]
