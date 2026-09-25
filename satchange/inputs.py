"""Helpers that turn simple user choices into analysis inputs."""

import datetime as dt
import math

import requests

from .engine import Grid

NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
USER_AGENT = "satellite-change-map/1.0 (https://github.com/Inasjackw321/satellite-change-)"
MAX_PIXELS = 80e6  # ~8,000 km2 at 10 m
MIN_SIDE_KM = 4.0


def search_places(query, limit=6):
    """[(name, (west, south, east, north))] from OpenStreetMap's place search."""
    r = requests.get(NOMINATIM_URL, params={"q": query, "format": "jsonv2", "limit": limit},
                     headers={"User-Agent": USER_AGENT}, timeout=15)
    r.raise_for_status()
    places = []
    for item in r.json():
        south, north, west, east = (float(v) for v in item["boundingbox"])
        places.append((item["display_name"], (west, south, east, north)))
    return places


def area_km2(bounds):
    west, south, east, north = bounds
    lat = math.radians((south + north) / 2)
    return (east - west) * 111.32 * math.cos(lat) * (north - south) * 110.57


def fit_area(bounds, max_pixels=MAX_PIXELS, min_side_km=MIN_SIDE_KM):
    """Grow tiny areas (a village's point-like box) to ``min_side_km`` and
    shrink huge ones around their centre until they fit ``max_pixels``.
    Returns (bounds, note) where note explains any adjustment."""
    west, south, east, north = bounds
    cx, cy = (west + east) / 2, (south + north) / 2
    km_per_deg_x = 111.32 * math.cos(math.radians(cy))
    half_w = max((east - west) / 2, min_side_km / 2 / km_per_deg_x)
    half_h = max((north - south) / 2, min_side_km / 2 / 110.57)
    note = None
    if (half_w, half_h) != ((east - west) / 2, (north - south) / 2):
        note = f"Enlarged to at least {min_side_km:g} km across."
    g = Grid.for_bounds((cx - half_w, cy - half_h, cx + half_w, cy + half_h))
    if g.width * g.height > max_pixels:
        shrink = math.sqrt(max_pixels / (g.width * g.height)) * 0.99
        half_w, half_h = half_w * shrink, half_h * shrink
        note = "Too big for a 10 m analysis, so trimmed to the central area."
    fitted = (cx - half_w, cy - half_h, cx + half_w, cy + half_h)
    return tuple(round(v, 5) for v in fitted), note


def _year_earlier(day):
    try:
        return day.replace(year=day.year - 1)
    except ValueError:  # 29 February
        return day.replace(year=day.year - 1, day=28)


def periods_for_event(after_start, weeks=8):
    """After period: ``weeks`` from ``after_start``. Before period: the same
    weeks one year earlier, so seasonal changes (leaves, snow, soil moisture)
    cancel out. Both as inclusive (start, end) dates."""
    after_end = after_start + dt.timedelta(weeks=weeks) - dt.timedelta(days=1)
    return (_year_earlier(after_start), _year_earlier(after_end)), (after_start, after_end)
