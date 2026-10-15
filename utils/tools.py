import json
import os
from uuid import uuid4
from typing_extensions import Annotated
import urllib.parse
import urllib.request
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from langchain_core.tools import tool
from langgraph.store.base import BaseStore
from langgraph.prebuilt import InjectedState, InjectedStore

from utils.credentials import Credentials
from utils.datamodels import IvyState, Memory, Note

WEATHER_CODES = {
    0: "clear sky", 1: "mostly clear", 2: "partly cloudy", 3: "overcast",
    45: "foggy", 48: "foggy",
    51: "light drizzle", 53: "drizzle", 55: "heavy drizzle",
    61: "light rain", 63: "rain", 65: "heavy rain",
    71: "light snow", 73: "snow", 75: "heavy snow",
    80: "rain showers", 81: "rain showers", 82: "heavy rain showers",
    95: "thunderstorm", 96: "thunderstorm with hail", 99: "thunderstorm with hail",
}


# ---------- Helper Functions ---------- #
def _get_json(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=10) as resp:
        return json.load(resp)


# --------- Tools ---------- #
@tool
def get_current_time() -> str:
    """Get the current date and time. Use this whenever the user asks what
    time it is, what day it is, or what today's date is."""
    tz_name = Credentials.IVY_TIMEZONE 
    now = datetime.now(
        ZoneInfo(tz_name)) if tz_name else datetime.now().astimezone()
    return now.strftime("%A, %d %B %Y, %I:%M %p")


@tool
def get_weather(city: str | None = None) -> str:
    """Get the current weather for a city. If the user doesn't name a city,
    leave city empty and their home city will be used."""
    city = city or Credentials.IVY_HOME_CITY
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

@tool
async def save_note(text: str, store: Annotated[BaseStore, InjectedStore()], state: Annotated[IvyState, InjectedState()]) -> str:
    """Save a short note for the user. Use when they ask you to remember,
    note down, or jot something."""
    id = str(uuid4())
    now = datetime.now().isoformat()
    await store.aput(
        ("notes", state["person"]),
        id,
        {
            "id:": id,
            "user_id": state["person"],
            "content": text,
            "created_at": now,
            "updated_at": now
        },
    )
    return "Saved."


@tool
async def list_notes(num_of_notes: int, store: Annotated[BaseStore, InjectedStore()], state: Annotated[IvyState, InjectedState()], id = None) -> list[Note] | str:
    '''Read the notes that have been saved for the user. Use when they ask you to list, show, or read their notes.
    
    Parameters:
    - num_of_notes: The number of notes to retrieve. If there are fewer notes than this number, return all available notes.
    - id: Optional. If provided, retrieve the note with this specific ID. If not provided, retrieve the most recent notes up to num_of_notes.
    '''
    if not id:
        notes: list[Note] = [Note(**note.value) for note in await store.asearch(("notes", state["person"]), limit=num_of_notes)]
        if not notes:
            return "You have no saved notes."
        return notes
    else:
        note = await store.aget(("notes", state["person"]), id)
        if note:
            return Note(**note.value)
        else:
            return f"No note found with ID: {id}"
    

@tool
async def remember(memory: str, store: Annotated[BaseStore, InjectedStore()], state: Annotated[IvyState, InjectedState()]) -> str:
    '''Store a memory for the user. Use when they ask you to remember something for later. Or use it when they something about themselves. Use this to remember their preferences and who they are'''
    if memory:
        id = str(uuid4())
        await store.aput(
            ("memories", state["person"]),
            id,
            {
                "id": id,
                "content": memory,
                "datetime": datetime.now().isoformat(),
                "source": "User",
                "category": "Intentional",
            },
            index=["content"],
        )
        return f"Memory saved. With the ID {id}. You can recall it later."
    else:
        return "No memory provided to remember."
        

@tool
async def recall(faintMemory: str, store: Annotated[BaseStore, InjectedStore()], state: Annotated[IvyState, InjectedState()]) -> list[str]:
    '''Recall memories that match the given faint memory. Use when the user asks you to recall something they told you before.'''
    if faintMemory:
        memories = await store.asearch(("memories", state["person"]), query=faintMemory, limit=5)
        if memories:
            return [Memory(**item.value) for item in memories]
        else:
            return "No matching memories found."
    else:
        return "No faint memory provided to recall."
    


TOOLS = [
    get_current_time,
    get_weather,
    save_note,
    list_notes,
    remember,
    recall
]
