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
from scipy import ndimage
from shapely.geometry import box, shape
from shapely.ops import unary_union

from . import source, stats

CRS = "EPSG:3857"  # Web Mercator: the grid overlays web maps exactly
EARTH_RADIUS = 6378137.0
GROUND_RES = 10.0  # metres, Sentinel-1 pixel spacing
TILE = 1024
DISPLAY_FACTOR = 4  # before/after background layers at 40 m
LOOKBACK_DAYS = 90  # how far before the start date to look for images
BYTES_PER_PIXEL = 3.2  # compressed float32 speckle, per band, for download estimates
MAX_PIXELS = 30e6  # ~3,000 km2 at 10 m
CACHE_DIR = Path(__file__).resolve().parent.parent / "cache"
CACHE_VERSION = 3  # bump when the computation or file format changes

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
    """An analysis: compare the latest images up to ``start`` with the latest
    images up to ``end`` (and after ``start``)."""
    label: str
    bounds: tuple  # (west, south, east, north)
    start: str  # YYYY-MM-DD
    end: str
    images: int = 3  # images averaged on each side
    smooth: int = 3  # speckle filter window in pixels (1 = none)
    orbit_pass: str | None = None  # "ASCENDING" / "DESCENDING"
    orbit: int | None = None
    enl: float | None = None  # None = measure from the data
    bands: tuple = ("VV", "VH")

    @property
    def before_window(self):
        start = dt.date.fromisoformat(self.start)
        return str(start - dt.timedelta(days=LOOKBACK_DAYS)), str(start + dt.timedelta(days=1))

    @property
    def after_window(self):
        start, end = dt.date.fromisoformat(self.start), dt.date.fromisoformat(self.end)
        return str(start + dt.timedelta(days=1)), str(end + dt.timedelta(days=1))

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
    signed: np.ndarray  # int16 test statistic signed by direction, see stats.decode
    change_db: np.ndarray  # int16 change of total backscatter in 0.01 dB
    orbit: int
    before_days: list
    after_days: list
    enl: float  # per image, after the speckle filter
    enl_estimated: bool
    orbits: list = field(default_factory=list)
    null_signed: np.ndarray | None = None  # same, for before-vs-before (no-change check)
    null_db: np.ndarray | None = None
    before_db: np.ndarray | None = None  # uint8 VV backscatter at 40 m, 0 = no data
    after_db: np.ndarray | None = None
    version: int = CACHE_VERSION

    ARRAYS = ("signed", "change_db", "null_signed", "null_db", "before_db", "after_db")

    def save(self, path):
        meta = {k: v for k, v in asdict(self).items() if k not in self.ARRAYS}
        arrays = {k: getattr(self, k) for k in self.ARRAYS if getattr(self, k) is not None}
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, meta=json.dumps(meta), **arrays)

    @classmethod
    def load(cls, path):
        with np.load(path) as f:
            meta = json.loads(str(f["meta"]))
            if meta.get("version") != CACHE_VERSION:
                raise ValueError("saved with an older version of the app")
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
    """Search the archive and pick the images: the latest ``images`` up to the
    start date, and the latest ``images`` up to the end date (after the start),
    all from one orbit so the viewing geometry is identical."""
    before = source.search(params.bounds, *params.before_window, session=session)
    after = source.search(params.bounds, *params.after_window, session=session)
    if params.orbit_pass:
        keep = params.orbit_pass.lower()
        before = [s for s in before if s.orbit_state == keep]
        after = [s for s in after if s.orbit_state == keep]
    b, a = _by_orbit_day(before), _by_orbit_day(after)
    candidates = sorted(set(b) & set(a))
    if params.orbit is not None:
        candidates = [o for o in candidates if o == params.orbit]
    if not candidates:
        raise ValueError("No radar images were taken from the same orbit before the start date "
                         "and between the start and end dates. Try a longer date range.")

    start, end = dt.date.fromisoformat(params.start), dt.date.fromisoformat(params.end)
    aoi = box(*params.bounds)
    table = []
    for o in candidates:
        cover = min(_coverage([s for d in b[o].values() for s in d], aoi),
                    _coverage([s for d in a[o].values() for s in d], aoi))
        gap = (start - max(b[o])).days + (end - max(a[o])).days
        table.append({"orbit": o, "coverage": cover, "before": len(b[o]), "after": len(a[o]),
                      "gap_days": gap})

    def score(row):
        # Full coverage first, then images closest to the two dates, then more images.
        full = row["coverage"] > 0.98
        return (full, row["coverage"] if not full else 0, -row["gap_days"],
                min(row["before"], row["after"]))

    orbit = max(table, key=score)["orbit"]
    n = params.images
    before_days, after_days = sorted(b[orbit])[-n:], sorted(a[orbit])[-n:]

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


def _boxcar(a, k):
    """Mean over a k x k window (zero outside the array)."""
    return a if k <= 1 else ndimage.uniform_filter(a.astype(np.float64), size=k, mode="constant")


def estimate_enl(plan, grid, signer, bands, smooth=1, pairs=3, size=256):
    """ENL of single images after the speckle filter, from consecutive before
    acquisitions sampled in four windows spread over the area. Returns None if
    there's too little data."""
    days = list(plan.all_before)[-(pairs + 1):]
    if len(days) < 2:
        return None
    s = min(size, grid.width, grid.height)
    m = smooth // 2
    windows = {(max(0, int(fx * grid.width) - s // 2), max(0, int(fy * grid.height) - s // 2))
               for fx in (0.3, 0.7) for fy in (0.3, 0.7)}
    samples = []
    for c, r in windows:
        w, h = min(s, grid.width - c), min(s, grid.height - r)
        filtered = []
        for d in days:
            img = _read(grid, plan.all_before[d], signer, bands, c, r, w, h)
            valid = np.logical_and.reduce([np.isfinite(img[b]) for b in bands])
            share = _boxcar(valid.astype(float), smooth)
            full = share > 1 - 1e-9  # only pixels whose whole window is valid
            if m:
                full[:m], full[-m:], full[:, :m], full[:, -m:] = False, False, False, False
            filtered.append({b: np.where(full, _boxcar(np.where(valid, img[b], 0), smooth), np.nan)
                             for b in bands})
        for a, b in zip(filtered, filtered[1:]):
            for band in bands:
                lr = np.log(a[band] / b[band])
                samples.append(lr[np.isfinite(lr)])
    samples = np.concatenate(samples)
    return stats.estimate_enl(samples) if samples.size >= 1000 else None


def _accumulate(days, scenes_by_day, grid, signer, bands, window, split=False):
    """Sum and count per band over acquisitions; with split, separately for
    even and odd acquisitions (for the no-change check)."""
    c, r, w, h = window
    groups = 2 if split else 1
    sums = [{b: np.zeros((h, w)) for b in bands} for _ in range(groups)]
    counts = [np.zeros((h, w)) for _ in range(groups)]
    for i, day in enumerate(days):
        img = _read(grid, scenes_by_day[day], signer, bands, c, r, w, h)
        valid = np.logical_and.reduce([np.isfinite(img[b]) for b in bands])
        g = i % groups
        for b in bands:
            sums[g][b] += np.where(valid, img[b], 0)
        counts[g] += valid
    return sums, counts


def _compare(sums1, n1, sums2, n2, ok, bands, enl, smooth, crop):
    """Speckle-filter both sides, then test. Returns (signed, change_db) as
    int16 arrays, plus the filtered means for display."""
    mean = {}
    for side, sums, n in ((1, sums1, n1), (2, sums2, n2)):
        count = _boxcar(n, smooth)[crop]  # images x share of the window that is valid
        safe = np.maximum(count, 1e-9)
        mean[side] = ({b: _boxcar(sums[b], smooth)[crop] / safe for b in bands}, safe)
    (m1, c1), (m2, c2) = mean[1], mean[2]
    m1 = {b: np.where(ok, m1[b], 1) for b in bands}
    m2 = {b: np.where(ok, m2[b], 1) for b in bands}
    stat = sum(stats.lrt(m1[b], m2[b], c1 * enl, c2 * enl) for b in bands)
    span1, span2 = sum(m1[b] for b in bands), sum(m2[b] for b in bands)
    change = 10 * np.log10(span2 / span1)
    direction = np.where(change > 0, 1, -1)
    signed = np.round(np.minimum(stat, stats.STAT_MAX) * stats.STAT_SCALE * direction)
    signed = np.where(ok, signed, stats.NODATA).astype(np.int16)
    db = np.where(ok, np.round(np.clip(change, -300, 300) * 100), stats.NODATA).astype(np.int16)
    return signed, db, m1, m2


def process_tile(tile, plan, grid, signer, bands, enl, smooth=1):
    c, r, w, h = tile
    # Read a margin around the tile so the speckle filter is seamless.
    m = smooth // 2
    c0, r0 = max(0, c - m), max(0, r - m)
    c1, r1 = min(grid.width, c + w + m), min(grid.height, r + h + m)
    window = (c0, r0, c1 - c0, r1 - r0)
    crop = (slice(r - r0, r - r0 + h), slice(c - c0, c - c0 + w))

    b_sums, b_counts = _accumulate(list(plan.before), plan.before, grid, signer, bands, window, split=True)
    a_sums, a_counts = _accumulate(list(plan.after), plan.after, grid, signer, bands, window)
    before_sum = {b: b_sums[0][b] + b_sums[1][b] for b in bands}
    before_n = b_counts[0] + b_counts[1]

    ok = (before_n[crop] > 0) & (a_counts[0][crop] > 0)
    signed, db, mean_b, mean_a = _compare(before_sum, before_n, a_sums[0], a_counts[0], ok,
                                          bands, enl, smooth, crop)

    # No-change check: alternate before acquisitions against each other.
    null_signed = null_db = None
    if len(plan.before) >= 2:
        null_ok = (b_counts[0][crop] > 0) & (b_counts[1][crop] > 0)
        null_signed, null_db, _, _ = _compare(b_sums[0], b_counts[0], b_sums[1], b_counts[1], null_ok,
                                              bands, enl, smooth, crop)

    display = (_to_display(mean_b["VV"], ok), _to_display(mean_a["VV"], ok)) if "VV" in bands else None
    return signed, db, null_signed, null_db, display


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


def run(params, progress=lambda fraction, message: None, use_cache=True, plan=None,
        signer=None, workers=6):
    cache = CACHE_DIR / f"{params.cache_key()}.npz"
    if use_cache and cache.exists():
        try:
            return Result.load(cache)
        except ValueError:
            pass  # older format: recompute

    progress(0.02, "Searching the Sentinel-1 archive…")
    plan = plan or make_plan(params)
    signer = signer or source.Signer()
    grid = Grid.for_bounds(params.bounds)
    bands = list(params.bands)

    enl, estimated = params.enl or stats.NOMINAL_ENL * params.smooth ** 2, False
    if params.enl is None:
        progress(0.06, "Measuring the speckle level…")
        measured = estimate_enl(plan, grid, signer, bands, smooth=params.smooth)
        if measured is not None:
            enl, estimated = measured, True

    shape_ = (grid.height, grid.width)
    signed, change_db = np.full(shape_, stats.NODATA, np.int16), np.full(shape_, stats.NODATA, np.int16)
    has_null = len(plan.before) >= 2
    null_signed = np.full(shape_, stats.NODATA, np.int16) if has_null else None
    null_db = np.full(shape_, stats.NODATA, np.int16) if has_null else None
    dh, dw = -(-grid.height // DISPLAY_FACTOR), -(-grid.width // DISPLAY_FACTOR)
    before_db, after_db = np.zeros((dh, dw), np.uint8), np.zeros((dh, dw), np.uint8)
    tiles = grid.tiles()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        jobs = {pool.submit(process_tile, t, plan, grid, signer, bands, enl, params.smooth): t
                for t in tiles}
        for done, job in enumerate(as_completed(jobs), 1):
            c, r, w, h = jobs[job]
            t_signed, t_db, t_null, t_null_db, display = job.result()
            signed[r:r + h, c:c + w], change_db[r:r + h, c:c + w] = t_signed, t_db
            if has_null:
                null_signed[r:r + h, c:c + w], null_db[r:r + h, c:c + w] = t_null, t_null_db
            if display is not None:
                f = DISPLAY_FACTOR
                for target, d in zip((before_db, after_db), display):
                    target[r // f:r // f + d.shape[0], c // f:c // f + d.shape[1]] = d
            progress(0.1 + 0.88 * done / len(tiles),
                     f"Downloading and comparing images: part {done} of {len(tiles)}")

    if (signed == stats.NODATA).all():
        raise ValueError("The satellite images don't cover this area. Try a different area "
                         "or orbit direction.")
    result = Result(
        params=params, grid=grid, signed=signed, change_db=change_db, orbit=plan.orbit,
        before_days=[str(d) for d in plan.before], after_days=[str(d) for d in plan.after],
        enl=float(enl), enl_estimated=estimated, orbits=plan.orbits,
        null_signed=null_signed, null_db=null_db,
        before_db=before_db if "VV" in bands else None, after_db=after_db if "VV" in bands else None,
    )
    result.save(cache)
    progress(1.0, "Done")
    return result


# --- Saved results ------------------------------------------------------------

def saved_results():
    """[(path, description)] of cached results, newest first. Results saved by
    older versions of the app are skipped."""
    out = []
    for path in sorted(CACHE_DIR.glob("*.npz"), key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            with np.load(path) as f:
                meta = json.loads(str(f["meta"]))
            if meta.get("version") != CACHE_VERSION:
                continue
        except (OSError, ValueError, KeyError):
            continue
        p = meta["params"]
        out.append((path, f"{p['label']}: {short_date(p['start'])} → {short_date(p['end'])}"))
    return out


def short_date(day):
    d = dt.date.fromisoformat(day)
    return f"{d.day} {d:%b %Y}"


def delete_saved():
    for path in CACHE_DIR.glob("*.npz"):
        path.unlink(missing_ok=True)
