"""Earth Engine side: find Sentinel-1 scenes, run the change test, download it.

The test statistic is computed at full 10 m resolution on a fixed Web
Mercator grid and downloaded, rather than viewed as Earth Engine map tiles.
Map tiles are computed from averaged (pyramid) pixels when zoomed out, which
would silently change the number of looks and hence the statistics.
Downloading once also lets the app change the significance level instantly.
"""

import datetime as dt
import hashlib
import json
import math
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from pathlib import Path

import ee
import numpy as np

from . import stats

COLLECTION = "COPERNICUS/S1_GRD_FLOAT"  # linear power, as in the tutorial
CRS = "EPSG:3857"  # Web Mercator: the downloaded grid overlays web maps exactly
EARTH_RADIUS = 6378137.0
GROUND_RES = 10.0  # metres, Sentinel-1 GRDH pixel spacing
TILE = 1024
NULL_MAX, NULL_BINS = 50.0, 500
CACHE_DIR = Path(__file__).resolve().parent.parent / "cache"
CACHE_VERSION = 1  # bump when the computation changes

# (west, south, east, north) in degrees.
CITIES = {
    # Kyiv city plus the north-western suburbs (Irpin, Bucha, Hostomel)
    # that were fought over in February-March 2022.
    "Kyiv": (30.15, 50.20, 30.85, 50.62),
}


# --- Grid ---------------------------------------------------------------

def _merc(lon, lat):
    return (EARTH_RADIUS * math.radians(lon),
            EARTH_RADIUS * math.log(math.tan(math.pi / 4 + math.radians(lat) / 2)))


def _lat(y):
    return math.degrees(2 * math.atan(math.exp(y / EARTH_RADIUS)) - math.pi / 2)


@dataclass(frozen=True)
class Grid:
    """Pixel grid in EPSG:3857, top-left origin, ~10 m ground pixels."""
    x0: float
    y0: float
    scale: float  # EPSG:3857 metres per pixel
    width: int
    height: int

    @classmethod
    def for_bounds(cls, bounds, ground_res=GROUND_RES):
        west, south, east, north = bounds
        # Mercator stretches distances by 1/cos(lat).
        scale = ground_res / math.cos(math.radians((south + north) / 2))
        x0, y0 = _merc(west, north)
        x1, y1 = _merc(east, south)
        return cls(x0, y0, scale, math.ceil((x1 - x0) / scale), math.ceil((y0 - y1) / scale))

    def latlon_bounds(self):
        """[[south, west], [north, east]] of the whole grid."""
        west = math.degrees(self.x0 / EARTH_RADIUS)
        east = math.degrees((self.x0 + self.width * self.scale) / EARTH_RADIUS)
        return [[_lat(self.y0 - self.height * self.scale), west], [_lat(self.y0), east]]

    def row_area_km2(self):
        """Ground area of one pixel in each row (varies with latitude)."""
        y = self.y0 - (np.arange(self.height) + 0.5) * self.scale
        lat = 2 * np.arctan(np.exp(y / EARTH_RADIUS)) - np.pi / 2
        return (self.scale * np.cos(lat)) ** 2 / 1e6

    def ee_grid(self, col, row, width, height):
        return {
            "dimensions": {"width": width, "height": height},
            "affineTransform": {
                "scaleX": self.scale, "shearX": 0, "translateX": self.x0 + col * self.scale,
                "shearY": 0, "scaleY": -self.scale, "translateY": self.y0 - row * self.scale,
            },
            "crsCode": CRS,
        }


# --- Parameters and results -------------------------------------------------

@dataclass(frozen=True)
class Params:
    label: str
    bounds: tuple  # (west, south, east, north)
    before: tuple  # (start, end) YYYY-MM-DD, end exclusive
    after: tuple
    orbit_pass: str | None = None
    orbit: int | None = None
    max_images: int = 0  # per window; 0 = all
    enl: float | None = None  # None = estimate from the data
    bands: tuple = ("VV", "VH")

    def cache_key(self):
        blob = json.dumps([CACHE_VERSION, asdict(self)], sort_keys=True).encode()
        return hashlib.sha1(blob).hexdigest()[:16]


@dataclass
class Result:
    params: Params
    grid: Grid
    signed: np.ndarray  # int16, see stats.decode
    orbit: int
    before_days: list
    after_days: list
    enl: float
    enl_estimated: bool
    null_hist: np.ndarray | None  # counts, bins of NULL_MAX / NULL_BINS from 0
    orbits: list = field(default_factory=list)  # candidates considered

    def save(self, path):
        meta = {k: v for k, v in asdict(self).items() if k not in ("signed", "null_hist")}
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path, signed=self.signed, meta=json.dumps(meta),
            null_hist=self.null_hist if self.null_hist is not None else np.zeros(0),
        )

    @classmethod
    def load(cls, path):
        with np.load(path) as f:
            meta = json.loads(str(f["meta"]))
            meta["params"] = Params(**{k: tuple(v) if isinstance(v, list) else v
                                       for k, v in meta["params"].items()})
            meta["grid"] = Grid(**meta["grid"])
            hist = f["null_hist"]
            return cls(signed=f["signed"], null_hist=hist if hist.size else None, **meta)


# --- Earth Engine pipeline -------------------------------------------------

def s1_collection(aoi, start, end, orbit_pass):
    col = (
        ee.ImageCollection(COLLECTION)
        .filterBounds(aoi)
        .filterDate(start, end)
        .filter(ee.Filter.eq("instrumentMode", "IW"))
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
    )
    if orbit_pass:
        col = col.filter(ee.Filter.eq("orbitProperties_pass", orbit_pass))
    return col


def acquisition_days(col):
    """{relative orbit: sorted acquisition days}. A pass over the AOI can be
    split into several adjacent scenes, so count days rather than images."""
    rows = col.reduceColumns(
        ee.Reducer.toList(2), ["relativeOrbitNumber_start", "system:time_start"]
    ).get("list").getInfo()
    days = defaultdict(set)
    for orbit, millis in rows:
        days[int(orbit)].add(dt.datetime.fromtimestamp(millis / 1000, dt.timezone.utc).date())
    return {orbit: sorted(d) for orbit, d in days.items()}


def choose_orbit(before, after, aoi, days_before, days_after):
    """Rank relative orbits by AOI coverage in both windows, then by number of
    acquisitions. Change detection needs the same viewing geometry."""
    candidates = sorted(set(days_before) & set(days_after))
    if not candidates:
        raise ValueError("No satellite orbit has images in both periods. Try wider date ranges.")

    def coverage(col, orbit):
        footprint = col.filter(ee.Filter.eq("relativeOrbitNumber_start", orbit)).geometry()
        return footprint.intersection(aoi, 100).area(100).divide(aoi.area(100))

    cover = ee.Dictionary({
        str(o): ee.Number(coverage(before, o)).min(coverage(after, o)) for o in candidates
    }).getInfo()
    table = [{"orbit": o, "coverage": cover[str(o)], "before": len(days_before[o]),
              "after": len(days_after[o])} for o in candidates]

    def score(row):
        full = row["coverage"] > 0.98
        return (full, min(row["before"], row["after"]) if full else row["coverage"])

    return max(table, key=score)["orbit"], table


def day_image(col, day, bands):
    """Mosaic of one acquisition day. Non-positive values (no-data borders)
    are masked so that log() stays finite."""
    im = col.filterDate(str(day), str(day + dt.timedelta(days=1))).mosaic().select(list(bands))
    return im.updateMask(im.gt(0).reduce(ee.Reducer.min()))


def window_mean(col, days, bands):
    """Mean intensity of several acquisitions and the per-pixel count.

    Averaging n independent acquisitions multiplies the number of looks by n;
    the test uses the count per pixel, so partial coverage is handled.
    """
    daily = ee.ImageCollection([day_image(col, d, bands) for d in days])
    return daily.mean(), daily.select(bands[0]).count()


def lrt_image(mean1, n1, mean2, n2, bands, enl):
    """Summed Bartlett-corrected -2 log Q over the bands (chi2, len(bands) dof)."""
    looks1, looks2 = n1.toFloat().multiply(enl), n2.toFloat().multiply(enl)
    per_band = [
        mean1.expression(stats.LRT_EXPRESSION, {
            "s1": mean1.select(b), "s2": mean2.select(b), "L1": looks1, "L2": looks2,
        })
        for b in bands
    ]
    return ee.Image.cat(per_band).reduce(ee.Reducer.sum()).max(0).rename("stat")


def sample_log_ratios(col, days, bands, aoi, grid, max_pairs=3, per_pair=4000):
    """log(s_a / s_b) of consecutive single acquisitions, at native resolution."""
    pairs = list(zip(days, days[1:]))[:max_pairs]
    samples = [
        day_image(col, a, bands).divide(day_image(col, b, bands)).log()
        .sample(region=aoi, projection=CRS, scale=grid.scale, numPixels=per_pair,
                seed=i, dropNulls=True, geometries=False)
        for i, (a, b) in enumerate(pairs)
    ]
    fc = ee.FeatureCollection(samples).flatten()
    values = ee.Dictionary({b: fc.aggregate_array(b) for b in bands}).getInfo()
    return np.concatenate([np.asarray(values[b], dtype=float) for b in bands])


def null_histogram(col, days, bands, enl, aoi, grid):
    """Histogram of the statistic when comparing two halves of the before
    period (alternate acquisitions), where no real change is expected."""
    m1, n1 = window_mean(col, days[0::2], bands)
    m2, n2 = window_mean(col, days[1::2], bands)
    stat = lrt_image(m1, n1, m2, n2, bands, enl).min(NULL_MAX - 1e-6)
    hist = stat.reduceRegion(
        ee.Reducer.fixedHistogram(0, NULL_MAX, NULL_BINS), aoi,
        crs=CRS, scale=grid.scale, maxPixels=1e10, tileScale=4,
    ).get("stat").getInfo()
    return np.array([count for _, count in hist], dtype=np.int64)


def _fetch(image, grid, col, row, width, height, attempts=4):
    for attempt in range(attempts):
        try:
            arr = ee.data.computePixels({
                "expression": image, "fileFormat": "NUMPY_NDARRAY",
                "grid": grid.ee_grid(col, row, width, height), "bandIds": ["signed"],
            })
            return arr["signed"]
        except ee.EEException:
            if attempt == attempts - 1:
                raise
            time.sleep(2 ** attempt)


def run(params, progress=lambda fraction, message: None, use_cache=True):
    cache = CACHE_DIR / f"{params.cache_key()}.npz"
    if use_cache and cache.exists():
        return Result.load(cache)

    aoi = ee.Geometry.Rectangle(list(params.bounds))
    grid = Grid.for_bounds(params.bounds)
    bands = list(params.bands)

    progress(0.02, "Searching Sentinel-1 archive…")
    before_col = s1_collection(aoi, *params.before, params.orbit_pass)
    after_col = s1_collection(aoi, *params.after, params.orbit_pass)
    days_before, days_after = acquisition_days(before_col), acquisition_days(after_col)

    progress(0.08, "Choosing satellite orbit…")
    if params.orbit is None:
        orbit, table = choose_orbit(before_col, after_col, aoi, days_before, days_after)
    else:
        orbit, table = params.orbit, []
        if orbit not in days_before or orbit not in days_after:
            raise ValueError(f"Orbit {orbit} has no images in one of the periods.")
    by_orbit = ee.Filter.eq("relativeOrbitNumber_start", orbit)
    before_col, after_col = before_col.filter(by_orbit), after_col.filter(by_orbit)
    all_before = days_before[orbit]
    before_days, after_days = all_before, days_after[orbit]
    if params.max_images:
        # Keep the acquisitions closest to the event.
        before_days = before_days[-params.max_images:]
        after_days = after_days[:params.max_images]

    enl, estimated = params.enl or stats.NOMINAL_ENL, False
    if params.enl is None and len(all_before) >= 2:
        progress(0.14, "Estimating speckle (equivalent number of looks)…")
        enl = stats.estimate_enl(sample_log_ratios(before_col, all_before, bands, aoi, grid))
        estimated = True

    mean_b, n_b = window_mean(before_col, before_days, bands)
    mean_a, n_a = window_mean(after_col, after_days, bands)
    stat = lrt_image(mean_b, n_b, mean_a, n_a, bands, enl)
    span = lambda im: im.reduce(ee.Reducer.sum())
    direction = span(mean_a).gt(span(mean_b)).multiply(2).subtract(1)
    signed = (
        stat.min(stats.STAT_MAX).multiply(stats.STAT_SCALE).multiply(direction)
        .round().toInt16().unmask(stats.NODATA, False).rename("signed")
    )

    tiles = [(c, r, min(TILE, grid.width - c), min(TILE, grid.height - r))
             for r in range(0, grid.height, TILE) for c in range(0, grid.width, TILE)]
    out = np.full((grid.height, grid.width), stats.NODATA, dtype=np.int16)
    null_hist = None
    with ThreadPoolExecutor(max_workers=6) as pool:
        null_job = (pool.submit(null_histogram, before_col, before_days, bands, enl, aoi, grid)
                    if len(before_days) >= 2 else None)
        jobs = {pool.submit(_fetch, signed, grid, *t): t for t in tiles}
        for done, job in enumerate(as_completed(jobs), 1):
            c, r, w, h = jobs[job]
            out[r:r + h, c:c + w] = job.result()
            progress(0.2 + 0.7 * done / len(tiles), f"Computing change at 10 m: tile {done}/{len(tiles)}")
        if null_job:
            progress(0.93, "Running the no-change check…")
            null_hist = null_job.result()

    result = Result(
        params=params, grid=grid, signed=out, orbit=orbit,
        before_days=[str(d) for d in before_days], after_days=[str(d) for d in after_days],
        enl=float(enl), enl_estimated=estimated, null_hist=null_hist, orbits=table,
    )
    result.save(cache)
    progress(1.0, "Done")
    return result


def saved_results():
    """[(path, description)] of cached results, newest first."""
    out = []
    for path in sorted(CACHE_DIR.glob("*.npz"), key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            with np.load(path) as f:
                meta = json.loads(str(f["meta"]))
        except (OSError, ValueError, KeyError):
            continue
        saved = dt.datetime.fromtimestamp(path.stat().st_mtime).strftime("%d %b %Y")
        out.append((path, f"{meta['params']['label']}: {meta['before_days'][0][:7]} → "
                          f"{meta['after_days'][0][:7]} (saved {saved})"))
    return out


def delete_saved():
    for path in CACHE_DIR.glob("*.npz"):
        path.unlink(missing_ok=True)


def display_layers(result):
    """Earth Engine tile URLs for the before/after backscatter (display only)."""
    p = result.params
    aoi = ee.Geometry.Rectangle(list(p.bounds))
    by_orbit = ee.Filter.eq("relativeOrbitNumber_start", result.orbit)
    layers = {}
    for name, window, days in (("Before", p.before, result.before_days),
                               ("After", p.after, result.after_days)):
        col = s1_collection(aoi, *window, p.orbit_pass).filter(by_orbit)
        mean, _ = window_mean(col, [dt.date.fromisoformat(d) for d in days], ["VV"])
        db = mean.log10().multiply(10).clip(aoi)
        layers[name] = db.getMapId({"min": -20, "max": 0})["tile_fetcher"].url_format
    return layers
