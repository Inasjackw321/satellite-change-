"""Find Sentinel-1 images, run the change test locally, and cache the result.

Images come from Microsoft Planetary Computer (free, no account; see
source.py). Only the pixels covering the area are downloaded. Everything
is computed at full 10 m resolution on a fixed Web Mercator grid, so the
result overlays web maps exactly and the statistics never depend on zoom.
"""

import datetime as dt
import hashlib
import json
import math
import warnings
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
from shapely.geometry import box, shape
from shapely.ops import unary_union

from . import source, stats

CRS = "EPSG:3857"  # Web Mercator: the grid overlays web maps exactly
EARTH_RADIUS = 6378137.0
GROUND_RES = 10.0  # metres, Sentinel-1 pixel spacing
TILE = 1024
DISPLAY_FACTOR = 4  # before/after background layers at 40 m
NULL_MAX, NULL_BINS = 50.0, 500
BYTES_PER_PIXEL = 3.2  # compressed float32 speckle, per band, for download estimates
MAX_PIXELS = 30e6  # ~3,000 km2 at 10 m
CACHE_DIR = Path(__file__).resolve().parent.parent / "cache"
CACHE_VERSION = 2  # bump when the computation changes

# (west, south, east, north) in degrees.
CITIES = {
    # Fought over in February-March 2022.
    "Kyiv – Irpin, Bucha & Hostomel": (30.15, 50.48, 30.40, 50.62),
    "Kyiv – city centre": (30.40, 50.38, 30.62, 50.50),
    "Kyiv – whole city (large download)": (30.15, 50.20, 30.85, 50.62),
}


# --- Grid ---------------------------------------------------------------

def _merc(lon, lat):
    return (EARTH_RADIUS * math.radians(lon),
            EARTH_RADIUS * math.log(math.tan(math.pi / 4 + math.radians(lat) / 2)))


def _lat(y):
    return math.degrees(2 * math.atan(math.exp(y / EARTH_RADIUS)) - math.pi / 2)


def _lon(x):
    return math.degrees(x / EARTH_RADIUS)


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

    @property
    def pixels(self):
        return self.width * self.height

    def latlon_bounds(self):
        """[[south, west], [north, east]] of the whole grid."""
        return self.window_latlon(0, 0, self.width, self.height)

    def window_latlon(self, col, row, width, height):
        x0, y0 = self.x0 + col * self.scale, self.y0 - row * self.scale
        return [[_lat(y0 - height * self.scale), _lon(x0)], [_lat(y0), _lon(x0 + width * self.scale)]]

    def row_area_km2(self):
        """Ground area of one pixel in each row (varies with latitude)."""
        y = self.y0 - (np.arange(self.height) + 0.5) * self.scale
        lat = 2 * np.arctan(np.exp(y / EARTH_RADIUS)) - np.pi / 2
        return (self.scale * np.cos(lat)) ** 2 / 1e6

    def transform(self, col, row):
        return source.tile_transform(self.x0, self.y0, self.scale, col, row)

    def tiles(self, size=None):
        size = size or TILE
        return [(c, r, min(size, self.width - c), min(size, self.height - r))
                for r in range(0, self.height, size) for c in range(0, self.width, size)]


# --- Parameters, plan and result -------------------------------------------

@dataclass(frozen=True)
class Params:
    label: str
    bounds: tuple  # (west, south, east, north)
    before: tuple  # (start, end) YYYY-MM-DD, end exclusive
    after: tuple
    orbit_pass: str | None = None  # "ASCENDING" / "DESCENDING"
    orbit: int | None = None
    max_images: int = 0  # per period; 0 = all
    enl: float | None = None  # None = estimate from the data
    bands: tuple = ("VV", "VH")

    def cache_key(self):
        blob = json.dumps([CACHE_VERSION, asdict(self)], sort_keys=True).encode()
        return hashlib.sha1(blob).hexdigest()[:16]


@dataclass
class Plan:
    """Which images an analysis will use, found before downloading anything."""
    orbit: int
    orbit_state: str
    before: dict  # day -> [Scene], selected acquisitions
    after: dict
    all_before: dict  # every before acquisition of the orbit (for the ENL estimate)
    orbits: list  # candidates considered
    download_mb: float


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
    orbits: list = field(default_factory=list)
    before_db: np.ndarray | None = None  # uint8 VV backscatter at 40 m, 0 = no data
    after_db: np.ndarray | None = None

    ARRAYS = ("signed", "null_hist", "before_db", "after_db")

    def save(self, path):
        meta = {k: v for k, v in asdict(self).items() if k not in self.ARRAYS}
        arrays = {k: getattr(self, k) for k in self.ARRAYS if getattr(self, k) is not None}
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, meta=json.dumps(meta), **arrays)

    @classmethod
    def load(cls, path):
        with np.load(path) as f:
            meta = json.loads(str(f["meta"]))
            meta["params"] = Params(**{k: tuple(v) if isinstance(v, list) else v
                                       for k, v in meta["params"].items()})
            meta["grid"] = Grid(**meta["grid"])
            arrays = {k: f[k] if k in f and f[k].size else None for k in cls.ARRAYS}
            return cls(**meta, **arrays)


# --- Planning ----------------------------------------------------------------

def _by_orbit_day(scenes):
    out = defaultdict(lambda: defaultdict(list))
    for s in scenes:
        out[s.orbit][s.day].append(s)
    return out


def _coverage(scenes, aoi):
    footprint = unary_union([shape(s.geometry) for s in scenes])
    return footprint.intersection(aoi).area / aoi.area


def make_plan(params, session=None):
    """Search the archive and pick the images to use."""
    before = source.search(params.bounds, *params.before, session=session)
    after = source.search(params.bounds, *params.after, session=session)
    if params.orbit_pass:
        keep = params.orbit_pass.lower()
        before = [s for s in before if s.orbit_state == keep]
        after = [s for s in after if s.orbit_state == keep]
    b, a = _by_orbit_day(before), _by_orbit_day(after)
    candidates = sorted(set(b) & set(a))
    if params.orbit is not None:
        candidates = [o for o in candidates if o == params.orbit]
    if not candidates:
        raise ValueError("No radar images were taken from the same orbit in both periods. "
                         "Try longer periods or different dates.")

    aoi = box(*params.bounds)
    table = []
    for o in candidates:
        cover = min(_coverage([s for d in b[o].values() for s in d], aoi),
                    _coverage([s for d in a[o].values() for s in d], aoi))
        table.append({"orbit": o, "coverage": cover, "before": len(b[o]), "after": len(a[o])})

    def score(row):
        full = row["coverage"] > 0.98
        return (full, min(row["before"], row["after"]) if full else row["coverage"])

    orbit = max(table, key=score)["orbit"]
    before_days, after_days = sorted(b[orbit]), sorted(a[orbit])
    if params.max_images:
        # Keep the acquisitions closest to the event.
        before_days, after_days = before_days[-params.max_images:], after_days[:params.max_images]

    grid = Grid.for_bounds(params.bounds)
    reads = len(before_days) + len(after_days)
    mb = grid.pixels * len(params.bands) * reads * BYTES_PER_PIXEL / 1e6
    state = next(iter(b[orbit].values()))[0].orbit_state
    return Plan(orbit=orbit, orbit_state=state,
                before={d: b[orbit][d] for d in before_days},
                after={d: a[orbit][d] for d in after_days},
                all_before=dict(sorted(b[orbit].items())), orbits=table, download_mb=mb)


# --- Computation -------------------------------------------------------------

def _read(grid, scenes, signer, bands, col, row, width, height):
    """One day's mosaic for a window, skipping scenes that don't touch it."""
    (south, west), (north, east) = grid.window_latlon(col, row, width, height)
    window = box(west, south, east, north)
    touching = [s for s in scenes if shape(s.geometry).intersects(window)]
    if not touching:
        return {b: np.full((height, width), np.nan, dtype=np.float32) for b in bands}
    return source.read_day(touching, signer, CRS, grid.transform(col, row), width, height, bands)


def estimate_enl(plan, grid, signer, bands, pairs=3, size=256):
    """ENL from consecutive single before acquisitions, sampled in four
    windows spread over the area. Returns None if there's too little data."""
    days = list(plan.all_before)[: pairs + 1]
    if len(days) < 2:
        return None
    s = min(size, grid.width, grid.height)
    windows = {(max(0, int(fx * grid.width) - s // 2), max(0, int(fy * grid.height) - s // 2))
               for fx in (0.3, 0.7) for fy in (0.3, 0.7)}
    samples = []
    for c, r in windows:
        w, h = min(s, grid.width - c), min(s, grid.height - r)
        imgs = [_read(grid, plan.all_before[d], signer, bands, c, r, w, h) for d in days]
        for a, b in zip(imgs, imgs[1:]):
            for band in bands:
                lr = np.log(a[band] / b[band])
                samples.append(lr[np.isfinite(lr)])
    samples = np.concatenate(samples)
    return stats.estimate_enl(samples) if samples.size >= 1000 else None


def _accumulate(days, scenes_by_day, grid, signer, bands, tile, split=False):
    """Sum and count per band over acquisitions; with split, separately for
    even and odd acquisitions (for the no-change check)."""
    c, r, w, h = tile
    groups = 2 if split else 1
    sums = [{b: np.zeros((h, w)) for b in bands} for _ in range(groups)]
    counts = [np.zeros((h, w), dtype=np.int32) for _ in range(groups)]
    for i, day in enumerate(days):
        img = _read(grid, scenes_by_day[day], signer, bands, c, r, w, h)
        valid = np.logical_and.reduce([np.isfinite(img[b]) for b in bands])
        g = i % groups
        for b in bands:
            sums[g][b] += np.where(valid, img[b], 0)
        counts[g] += valid
    return sums, counts


def _test(sum1, n1, sum2, n2, bands, enl):
    """Summed statistic over bands, plus whether the total intensity rose."""
    ok = (n1 > 0) & (n2 > 0)
    safe1, safe2 = np.maximum(n1, 1), np.maximum(n2, 1)
    mean1 = {b: np.where(ok, sum1[b] / safe1, 1) for b in bands}
    mean2 = {b: np.where(ok, sum2[b] / safe2, 1) for b in bands}
    stat = sum(stats.lrt(mean1[b], mean2[b], safe1 * enl, safe2 * enl) for b in bands)
    increased = sum(mean2[b] for b in bands) > sum(mean1[b] for b in bands)
    return stat, increased, ok, mean1, mean2


def _to_display(intensity, valid):
    """Block-average to 40 m and scale -25..0 dB to 1..255 (0 = no data)."""
    f = DISPLAY_FACTOR
    h, w = intensity.shape
    H, W = -(-h // f) * f, -(-w // f) * f
    padded = np.full((H, W), np.nan)
    padded[:h, :w] = np.where(valid, intensity, np.nan)
    blocks = padded.reshape(H // f, f, W // f, f)
    with np.errstate(invalid="ignore", divide="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-empty blocks
        db = 10 * np.log10(np.nanmean(blocks, axis=(1, 3)))
    out = 1 + np.round((np.clip(db, -25, 0) + 25) / 25 * 254)
    return np.where(np.isfinite(db), out, 0).astype(np.uint8)


def process_tile(tile, plan, grid, signer, bands, enl):
    c, r, w, h = tile
    b_sums, b_counts = _accumulate(list(plan.before), plan.before, grid, signer, bands, tile, split=True)
    a_sums, a_counts = _accumulate(list(plan.after), plan.after, grid, signer, bands, tile)

    before_sum = {b: b_sums[0][b] + b_sums[1][b] for b in bands}
    n_before = b_counts[0] + b_counts[1]
    stat, increased, ok, mean_b, mean_a = _test(before_sum, n_before, a_sums[0], a_counts[0], bands, enl)
    direction = np.where(increased, 1, -1)
    signed = np.round(np.minimum(stat, stats.STAT_MAX) * stats.STAT_SCALE * direction)
    signed = np.where(ok, signed, stats.NODATA).astype(np.int16)

    # No-change check: alternate before acquisitions against each other.
    hist = np.zeros(NULL_BINS, dtype=np.int64)
    if len(plan.before) >= 2:
        null_stat, _, null_ok, _, _ = _test(b_sums[0], b_counts[0], b_sums[1], b_counts[1], bands, enl)
        hist, _ = np.histogram(np.minimum(null_stat[null_ok], NULL_MAX - 1e-6),
                               bins=NULL_BINS, range=(0, NULL_MAX))

    return (signed, hist, _to_display(mean_b["VV"], ok) if "VV" in bands else None,
            _to_display(mean_a["VV"], ok) if "VV" in bands else None)


def run(params, progress=lambda fraction, message: None, use_cache=True, plan=None,
        signer=None, workers=6):
    cache = CACHE_DIR / f"{params.cache_key()}.npz"
    if use_cache and cache.exists():
        return Result.load(cache)

    progress(0.02, "Searching the Sentinel-1 archive…")
    plan = plan or make_plan(params)
    signer = signer or source.Signer()
    grid = Grid.for_bounds(params.bounds)
    bands = list(params.bands)

    enl, estimated = params.enl or stats.NOMINAL_ENL, False
    if params.enl is None:
        progress(0.06, "Measuring the speckle level…")
        measured = estimate_enl(plan, grid, signer, bands)
        if measured is not None:
            enl, estimated = measured, True

    signed = np.full((grid.height, grid.width), stats.NODATA, dtype=np.int16)
    dh, dw = -(-grid.height // DISPLAY_FACTOR), -(-grid.width // DISPLAY_FACTOR)
    before_db, after_db = np.zeros((dh, dw), np.uint8), np.zeros((dh, dw), np.uint8)
    null_hist = np.zeros(NULL_BINS, dtype=np.int64)
    tiles = grid.tiles()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        jobs = {pool.submit(process_tile, t, plan, grid, signer, bands, enl): t for t in tiles}
        for done, job in enumerate(as_completed(jobs), 1):
            c, r, w, h = jobs[job]
            tile_signed, hist, disp_b, disp_a = job.result()
            signed[r:r + h, c:c + w] = tile_signed
            null_hist += hist
            if disp_b is not None:
                f = DISPLAY_FACTOR
                before_db[r // f:r // f + disp_b.shape[0], c // f:c // f + disp_b.shape[1]] = disp_b
                after_db[r // f:r // f + disp_a.shape[0], c // f:c // f + disp_a.shape[1]] = disp_a
            progress(0.1 + 0.88 * done / len(tiles),
                     f"Downloading and comparing images: part {done} of {len(tiles)}")

    if (signed == stats.NODATA).all():
        raise ValueError("The satellite images don't cover this area. Try a different area "
                         "or orbit direction.")
    result = Result(
        params=params, grid=grid, signed=signed, orbit=plan.orbit,
        before_days=[str(d) for d in plan.before], after_days=[str(d) for d in plan.after],
        enl=float(enl), enl_estimated=estimated,
        null_hist=null_hist if len(plan.before) >= 2 else None, orbits=plan.orbits,
        before_db=before_db if "VV" in bands else None, after_db=after_db if "VV" in bands else None,
    )
    result.save(cache)
    progress(1.0, "Done")
    return result


# --- Saved results ------------------------------------------------------------

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
