"""Sentinel-1 images from Microsoft Planetary Computer: free, no account.

Uses the ``sentinel-1-rtc`` collection: Sentinel-1 IW scenes, radiometrically
terrain corrected (gamma0, linear power), as cloud-optimised GeoTIFFs at
10 m. Only the part of each image covering the area is downloaded.
"""

import datetime as dt
import threading
from dataclasses import dataclass

import numpy as np
import rasterio
import requests
from affine import Affine
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT

STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1/search"
TOKEN_URL = "https://planetarycomputer.microsoft.com/api/sas/v1/token/{collection}"
COLLECTION = "sentinel-1-rtc"
BANDS = ("VV", "VH")

# Read only the needed byte ranges of each cloud-optimised GeoTIFF.
GDAL_ENV = {
    "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
    "GDAL_HTTP_MERGE_CONSECUTIVE_RANGES": "YES",
    "GDAL_HTTP_MULTIPLEX": "YES",
    "GDAL_HTTP_MAX_RETRY": "5",
    "GDAL_HTTP_RETRY_DELAY": "1",
    "VSI_CACHE": "TRUE",
}


class SourceError(Exception):
    """A problem reaching or reading the image archive, worded for users."""


@dataclass(frozen=True)
class Scene:
    id: str
    day: dt.date
    orbit: int  # relative orbit
    orbit_state: str  # "ascending" / "descending"
    geometry: dict  # GeoJSON footprint, lon/lat
    hrefs: dict  # band -> URL


def _scene(item):
    p = item["properties"]
    assets = {k.upper(): v["href"] for k, v in item["assets"].items() if k.upper() in BANDS}
    if p.get("sar:instrument_mode", "IW") != "IW" or set(BANDS) - set(assets):
        return None
    return Scene(
        id=item["id"],
        day=dt.date.fromisoformat(p["datetime"][:10]),
        orbit=int(p["sat:relative_orbit"]),
        orbit_state=str(p.get("sat:orbit_state", "")).lower(),
        geometry=item["geometry"],
        hrefs=assets,
    )


def search(bounds, start, end, session=None):
    """Scenes over ``bounds`` acquired from ``start`` up to (not including) ``end``."""
    session = session or requests.Session()
    last = dt.date.fromisoformat(str(end)) - dt.timedelta(days=1)
    body = {"collections": [COLLECTION], "bbox": list(bounds),
            "datetime": f"{start}T00:00:00Z/{last}T23:59:59Z", "limit": 250}
    scenes, request = [], ("POST", STAC_URL, body)
    try:
        while request:
            method, url, payload = request
            r = (session.post(url, json=payload, timeout=60) if method == "POST"
                 else session.get(url, timeout=60))
            r.raise_for_status()
            page = r.json()
            scenes += [s for s in map(_scene, page.get("features", [])) if s]
            nxt = next((link for link in page.get("links", []) if link.get("rel") == "next"), None)
            request = (nxt.get("method", "GET").upper(), nxt["href"], nxt.get("body")) if nxt else None
    except requests.ConnectionError as e:
        raise SourceError("Can't reach Microsoft Planetary Computer. Check your internet connection.") from e
    except requests.HTTPError as e:
        raise SourceError(f"The image catalogue returned an error ({e.response.status_code}). "
                          "Try again in a minute.") from e
    return scenes


class Signer:
    """Adds Planetary Computer's free, anonymous access token to image URLs,
    refreshing it before it expires."""

    def __init__(self, session=None):
        self.session = session or requests.Session()
        self.lock = threading.Lock()
        self.token, self.expires = None, None

    def _refresh(self):
        try:
            r = self.session.get(TOKEN_URL.format(collection=COLLECTION), timeout=30)
            r.raise_for_status()
        except requests.ConnectionError as e:
            raise SourceError("Can't reach Microsoft Planetary Computer. Check your internet connection.") from e
        except requests.HTTPError as e:
            raise SourceError(f"Planetary Computer refused access to the images ({e.response.status_code}). "
                              "Try again later.") from e
        data = r.json()
        self.token = data["token"]
        self.expires = dt.datetime.fromisoformat(data["msft:expiry"].replace("Z", "+00:00"))

    def sign(self, href):
        if not href.startswith("http"):
            return href  # local file (tests)
        with self.lock:
            margin = dt.timedelta(minutes=5)
            if self.expires is None or dt.datetime.now(dt.timezone.utc) > self.expires - margin:
                self._refresh()
            token = self.token
        return f"{href}{'&' if '?' in href else '?'}{token}"


def read_band(href, crs, transform, width, height):
    """One band resampled (nearest neighbour, so speckle statistics are kept)
    onto the given grid window. Invalid pixels are NaN."""
    with rasterio.Env(**GDAL_ENV), rasterio.open(href) as src, WarpedVRT(
        src, crs=crs, transform=transform, width=width, height=height,
        resampling=Resampling.nearest,
    ) as vrt:
        a = vrt.read(1, out_dtype="float32")
        nodata = vrt.nodata
    invalid = ~np.isfinite(a) | (a <= 0)
    if nodata is not None and np.isfinite(nodata):
        invalid |= a == nodata
    a[invalid] = np.nan
    return a


def read_day(scenes, signer, crs, transform, width, height, bands=BANDS):
    """Mosaic of one acquisition day (a pass can be split into several scenes).
    Returns {band: array}; a pixel is NaN in every band unless all bands are valid."""
    out = {b: np.full((height, width), np.nan, dtype=np.float32) for b in bands}
    for scene in scenes:
        todo = np.isnan(out[bands[0]])
        if not todo.any():
            break
        try:
            layers = {b: read_band(signer.sign(scene.hrefs[b]), crs, transform, width, height)
                      for b in bands}
        except rasterio.errors.RasterioIOError as e:
            raise SourceError(f"Couldn't download image {scene.id}. Check your internet "
                              "connection and try again.") from e
        valid = todo & np.logical_and.reduce([np.isfinite(layers[b]) for b in bands])
        for b in bands:
            out[b][valid] = layers[b][valid]
    return out


def tile_transform(x0, y0, scale, col, row):
    return Affine(scale, 0, x0 + col * scale, 0, -scale, y0 - row * scale)
