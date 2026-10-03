"""Monitor a site over time: find objects that appear in individual passes.

Every satellite pass between two dates is compared with the usual state of
each pixel: the lower median of the other passes from the same orbit (see
``stats.pass_threshold_db``). Objects that are present in a pass but not
usually (aircraft on a stand, vehicles, ships) stand out as brighter than
usual. This suits things that come and go, which averaging two periods would
blur away.

Two targets:
* activity on land (airports, bases, ports, car parks): 3 x 3 speckle filter;
* ships: full 10 m detail, only on water. Water is found from the data: large
  areas that are consistently dark.
"""

import datetime as dt
import hashlib
import json
import math
import warnings
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field

import numpy as np
from scipy import ndimage
from shapely.geometry import box

from . import engine, source, stats
from .engine import Grid

VERSION = 2
MAX_PASSES = 32
MIN_PASSES = 3  # per orbit, to have a usual state to compare with
MAX_PIXEL_PASSES = 1e8  # pixels x passes kept in memory (two int8 each)
DB_SCALE = 5  # stored as int8 in 0.2 dB steps: +-25.4 dB
REF_SIDE = 4  # each pass is compared with up to 4 passes before it and 4 after it
NO_SIDE = 127  # stored when a pass has no passes on one side (first/last pass)
NODATA = -128
DISPLAY_FACTOR = 2  # per-pass radar images at 20 m
WATER_DB = -17.0  # usual total backscatter below this is water (or tarmac)
WATER_MIN_KM2 = 0.25
WATER_OPENING_M = 110  # removes runways and roads, which are dark but narrow

TARGETS = {"land": "Activity on land", "water": "Ships on water", "all": "Everywhere"}

# Example sites, (west, south, east, north). Draw any other area on the map.
SITES = {
    "land": {
        "Boryspil airport (Kyiv)": (30.86, 50.325, 30.93, 50.365),
        "Hostomel airport (Kyiv)": (30.165, 50.590, 30.225, 50.620),
    },
    "water": {
        "Odesa port": (30.72, 46.46, 30.80, 46.52),
        "Bosphorus, south (Istanbul)": (28.97, 41.00, 29.08, 41.12),
        "Kerch Strait": (36.45, 45.25, 36.65, 45.40),
    },
}


# --- Parameters, plan, result ---------------------------------------------------

@dataclass(frozen=True)
class TSParams:
    label: str
    bounds: tuple
    start: str
    end: str
    target: str = "land"  # "land" | "water" | "all"
    smooth: int = 3
    all_orbits: bool = True  # use every orbit that covers the area (more passes)
    orbit_pass: str | None = None
    enl: float | None = None
    bands: tuple = ("VV", "VH")

    def cache_key(self):
        blob = json.dumps(["ts", VERSION, engine.CACHE_VERSION, asdict(self)], sort_keys=True).encode()
        return "ts_" + hashlib.sha1(blob).hexdigest()[:16]


@dataclass
class TSPlan:
    passes: list  # [(day, orbit, [Scene])] in date order
    orbits: list  # candidates considered
    download_mb: float

    @property
    def days(self):
        return [d for d, _, _ in self.passes]


@dataclass
class TSResult:
    params: TSParams
    grid: Grid
    days: list  # ISO date of each pass
    pass_orbits: list  # relative orbit of each pass
    before_db: np.ndarray  # int8 (passes, H, W): pass vs the passes just before it, DB_SCALE per dB
    after_db: np.ndarray  # same, vs the passes just after it (NO_SIDE: no such passes)
    water: np.ndarray  # bool (H, W)
    pass_images: np.ndarray  # uint8 (passes, H/2, W/2): radar image of each pass, 0 = no data
    enl: float
    enl_estimated: bool
    orbits: list = field(default_factory=list)
    kind: str = "timeseries"
    version: int = VERSION

    ARRAYS = ("before_db", "after_db", "water", "pass_images")

    def groups(self):
        """{orbit: [pass indices]}"""
        out = defaultdict(list)
        for i, o in enumerate(self.pass_orbits):
            out[o].append(i)
        return dict(out)

    def save(self, path):
        meta = {k: v for k, v in asdict(self).items() if k not in self.ARRAYS}
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, meta=json.dumps(meta), **{k: getattr(self, k) for k in self.ARRAYS})

    @classmethod
    def load(cls, path):
        with np.load(path) as f:
            meta = json.loads(str(f["meta"]))
            if meta.get("kind") != "timeseries" or meta.get("version") != VERSION:
                raise ValueError("not a current time-series result")
            meta["params"] = TSParams(**{k: tuple(v) if isinstance(v, list) else v
                                         for k, v in meta["params"].items()})
            meta["grid"] = Grid(**meta["grid"])
            return cls(**meta, **{k: f[k] for k in cls.ARRAYS})


# --- Planning ---------------------------------------------------------------------

def make_plan(params, session=None):
    end = dt.date.fromisoformat(params.end) + dt.timedelta(days=1)
    scenes = source.search(params.bounds, params.start, str(end), session=session)
    if params.orbit_pass:
        scenes = [s for s in scenes if s.orbit_state == params.orbit_pass.lower()]
    by_orbit = engine._by_orbit_day(scenes)
    aoi = box(*params.bounds)
    table = []
    for o, days in by_orbit.items():
        cover = engine._coverage([s for d in days.values() for s in d], aoi)
        table.append({"orbit": o, "coverage": cover, "passes": len(days),
                      "direction": next(iter(days.values()))[0].orbit_state})
    usable = [r for r in table if r["coverage"] > 0.98 and r["passes"] >= MIN_PASSES]
    if not usable:
        raise ValueError(f"Need at least {MIN_PASSES} passes from one satellite orbit between the "
                         "dates. Try a longer date range.")
    if not params.all_orbits:
        usable = [max(usable, key=lambda r: r["passes"])]
    keep = {r["orbit"] for r in usable}
    passes = sorted((d, o, sc) for o in keep for d, sc in by_orbit[o].items())
    if len(passes) > MAX_PASSES:  # keep the latest, then drop orbits left with too few
        passes = passes[-MAX_PASSES:]
        counts = defaultdict(int)
        for _, o, _ in passes:
            counts[o] += 1
        passes = [p for p in passes if counts[p[1]] >= MIN_PASSES]
    grid = Grid.for_bounds(params.bounds)
    mb = grid.pixels * len(params.bands) * len(passes) * engine.BYTES_PER_PIXEL / 1e6
    return TSPlan(passes=passes, orbits=sorted(table, key=lambda r: -r["passes"]), download_mb=mb)


def fits_in_memory(grid, n_passes):
    return grid.pixels * n_passes <= MAX_PIXEL_PASSES


# --- Computation --------------------------------------------------------------------

def sides(n, i, size=REF_SIDE):
    """Indices of up to ``size`` passes before pass i and after it."""
    return list(range(max(0, i - size), i)), list(range(i + 1, min(n, i + 1 + size)))


def _usual_ratio_db(stack):
    """For a (passes, H, W) stack of one orbit: each pass against the lower
    median of the passes just before it, and of those just after it, in dB.

    Something that is only there for a while (aircraft, vehicle, ship) is
    brighter than both. A lasting change (a building demolished or built) is
    not: on one side the passes already show the new state. Local references
    also follow slow trends such as crops growing. Returns (before, after);
    NaN where data is missing, None for a missing side."""
    n = stack.shape[0]
    missing = ~np.isfinite(stack).all(axis=0)
    out = ([], [])
    for i in range(n):
        for side, idx in enumerate(sides(n, i)):
            if not idx:
                out[side].append(None)
                continue
            ref = np.sort(stack[idx], axis=0)[stats.median_rank(len(idx)) - 1]
            with np.errstate(divide="ignore", invalid="ignore"):
                ratio = (10 * np.log10(stack[i] / ref)).astype(np.float32)
            ratio[missing] = np.nan
            out[side].append(ratio)
    return out


def _display(intensity):
    """Block-average to 20 m and scale -30..0 dB to 1..255 (0 = no data)."""
    f = DISPLAY_FACTOR
    h, w = intensity.shape
    H, W = -(-h // f) * f, -(-w // f) * f
    padded = np.full((H, W), np.nan)
    padded[:h, :w] = intensity
    with np.errstate(invalid="ignore", divide="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-empty blocks
        db = 10 * np.log10(np.nanmean(padded.reshape(H // f, f, W // f, f), axis=(1, 3)))
    out = 1 + np.round((np.clip(db, -30, 0) + 30) / 30 * 254)
    return np.where(np.isfinite(db), out, 0).astype(np.uint8)


def process_tile(tile, plan, grid, signer, bands, smooth):
    c, r, w, h = tile
    m = smooth // 2
    c0, r0 = max(0, c - m), max(0, r - m)
    c1, r1 = min(grid.width, c + w + m), min(grid.height, r + h + m)
    crop = (slice(r - r0, r - r0 + h), slice(c - c0, c - c0 + w))
    stack = np.full((len(plan.passes), h, w), np.nan, np.float32)
    for i, (_, _, scenes) in enumerate(plan.passes):
        img = engine._read(grid, scenes, signer, bands, c0, r0, c1 - c0, r1 - r0)
        valid = np.logical_and.reduce([np.isfinite(img[b]) for b in bands])
        span = np.where(valid, sum(img[b] for b in bands), 0)
        share = engine._boxcar(valid.astype(float), smooth)[crop]
        filtered = engine._boxcar(span, smooth)[crop] / np.maximum(share, 1e-9)
        stack[i] = np.where(share > 1 - 1e-9, filtered, np.nan)  # whole window valid

    before_db = np.full(stack.shape, NODATA, np.int8)
    after_db = np.full(stack.shape, NODATA, np.int8)
    usual = np.full((h, w), np.nan)
    groups = defaultdict(list)
    for i, (_, o, _) in enumerate(plan.passes):
        groups[o].append(i)
    for o, idx in sorted(groups.items(), key=lambda kv: -len(kv[1])):
        for target, ratios in zip((before_db, after_db), _usual_ratio_db(stack[idx])):
            for i, ratio in zip(idx, ratios):
                if ratio is None:
                    target[i] = np.where(np.isfinite(stack[i]), NO_SIDE, NODATA)
                else:
                    target[i] = np.where(np.isfinite(ratio), np.clip(np.round(ratio * DB_SCALE), -126, 126),
                                         NODATA)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # pixels with no data
            med = np.nanmedian(stack[idx], axis=0)
        usual = np.where(np.isfinite(usual), usual, med)  # largest orbit group first
    with np.errstate(divide="ignore", invalid="ignore"):
        water = 10 * np.log10(usual) < WATER_DB
    images = np.stack([_display(stack[i]) for i in range(len(plan.passes))])
    return before_db, after_db, water, images


def clean_water(raw):
    """Keep large dark areas; drop runways, roads and speckle."""
    size = max(3, int(round(WATER_OPENING_M / engine.GROUND_RES)) | 1)
    water = ndimage.binary_opening(raw, structure=np.ones((size, size), bool))
    labels, n = ndimage.label(water)
    if n:
        min_px = WATER_MIN_KM2 * 1e6 / engine.GROUND_RES ** 2
        keep = np.bincount(labels.ravel()) >= min_px
        keep[0] = False
        water = keep[labels]
    return ndimage.binary_dilation(water, iterations=3)  # ships sit at the water's edge too


def run(params, progress=lambda f, m: None, use_cache=True, plan=None, signer=None, workers=4):
    cache = engine.CACHE_DIR / f"{params.cache_key()}.npz"
    if use_cache and cache.exists():
        try:
            return TSResult.load(cache)
        except ValueError:
            pass
    progress(0.02, "Searching the Sentinel-1 archive…")
    plan = plan or make_plan(params)
    signer = signer or source.Signer()
    grid = Grid.for_bounds(params.bounds)
    if not fits_in_memory(grid, len(plan.passes)):
        raise ValueError("That's too much to analyse at once. Choose a smaller area or a shorter "
                         "date range.")
    bands = list(params.bands)

    enl, estimated = params.enl or stats.NOMINAL_ENL * params.smooth ** 2, False
    if params.enl is None:
        progress(0.05, "Measuring the speckle level…")
        biggest = max({o for _, o, _ in plan.passes},
                      key=lambda o: sum(1 for _, oo, _ in plan.passes if oo == o))
        by_day = {d: sc for d, o, sc in plan.passes if o == biggest}
        measured = engine.measure_enl(by_day, grid, signer, bands, params.smooth, span=True)
        if measured is not None:
            enl, estimated = measured, True

    n = len(plan.passes)
    before_db = np.full((n, grid.height, grid.width), NODATA, np.int8)
    after_db = np.full((n, grid.height, grid.width), NODATA, np.int8)
    water_raw = np.zeros((grid.height, grid.width), bool)
    f = DISPLAY_FACTOR
    images = np.zeros((n, -(-grid.height // f), -(-grid.width // f)), np.uint8)
    tiles = grid.tiles()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        jobs = {pool.submit(process_tile, t, plan, grid, signer, bands, params.smooth): t for t in tiles}
        for done, job in enumerate(as_completed(jobs), 1):
            c, r, w, h = jobs[job]
            t_before, t_after, t_water, t_images = job.result()
            before_db[:, r:r + h, c:c + w], after_db[:, r:r + h, c:c + w] = t_before, t_after
            water_raw[r:r + h, c:c + w] = t_water
            images[:, r // f:r // f + t_images.shape[1], c // f:c // f + t_images.shape[2]] = t_images
            progress(0.08 + 0.9 * done / len(tiles),
                     f"Downloading {n} passes and comparing: part {done} of {len(tiles)}")
    if (before_db == NODATA).all():
        raise ValueError("The satellite images don't cover this area. Try a different area.")
    # Remove each pass's overall offset (e.g. everything wetter after rain).
    for arr in (before_db, after_db):
        for i in range(n):
            layer = arr[i]
            valid = (layer != NODATA) & (layer != NO_SIDE)
            if valid.any():
                offset = int(np.median(layer[valid]))
                layer[valid] = np.clip(layer[valid].astype(np.int16) - offset, -126, 126)

    result = TSResult(params=params, grid=grid, days=[str(d) for d in plan.days],
                      pass_orbits=[o for _, o, _ in plan.passes], before_db=before_db, after_db=after_db,
                      water=clean_water(water_raw), pass_images=images, enl=float(enl),
                      enl_estimated=estimated, orbits=plan.orbits)
    result.save(cache)
    progress(1.0, "Done")
    return result


# --- Analysis ------------------------------------------------------------------------

@dataclass(frozen=True)
class Sensitivity:
    alpha: float  # chance per pixel and pass of flagging speckle
    min_db: float  # at least this much brighter than usual
    min_area_m2: float  # objects at least this big
    max_area_m2: float = 20_000  # and at most this big (a field brightening after rain isn't an object)


# Aircraft, vehicles and ships are strong radar reflectors: typically 10 dB or
# more above tarmac, and far more above water.
PRESETS = {
    "land": {"Sensitive": Sensitivity(1e-4, 4.0, 200), "Balanced": Sensitivity(1e-5, 6.0, 300),
             "Strict": Sensitivity(1e-7, 8.0, 800)},
    "water": {"Sensitive": Sensitivity(1e-5, 6.0, 100, 60_000), "Balanced": Sensitivity(1e-6, 8.0, 200, 60_000),
              "Strict": Sensitivity(1e-8, 10.0, 500, 60_000)},
}
PRESETS["all"] = PRESETS["land"]


@dataclass
class Detection:
    """An object seen in one pass."""
    pass_index: int
    day: str
    lat: float
    lon: float
    area_m2: float
    length_m: float  # longest extent (ships: roughly the hull length)
    brightness_db: float  # peak brightness above usual
    on_water: bool
    row: float = 0.0
    col: float = 0.0


@dataclass
class Hotspot:
    """A place where objects were seen in one or more passes."""
    lat: float
    lon: float
    area_m2: float  # typical size of the object seen there
    seen: int  # passes with an object here
    of: int  # passes that covered it
    days: list
    row: float = 0.0
    col: float = 0.0


@dataclass
class Analysis:
    detections: list  # [Detection], all passes
    per_pass: list  # number of objects in each pass
    count: np.ndarray  # uint8: passes with an object at each pixel
    masks: list  # bool mask of objects, per pass
    hotspots: list
    thresholds_db: list  # (before, after) dB thresholds used for each pass


def thresholds(result, sens):
    """dB thresholds (before side, after side) for each pass: the statistical
    threshold for the number of passes on that side, but at least ``min_db``.
    A side with no passes is None. Requiring both sides keeps the chance of
    flagging speckle below alpha."""
    out = [None] * len(result.days)
    for idx in result.groups().values():
        for pos, i in enumerate(idx):
            out[i] = tuple(max(stats.pass_threshold_db(sens.alpha, result.enl, len(side)), sens.min_db)
                           if side else None for side in sides(len(idx), pos))
    return out


def _length_m(rows, cols):
    """Longest extent of a pixel blob from its second moments: a straight run
    of n pixels has variance (n^2 - 1) / 12 along it."""
    if len(rows) < 2:
        return engine.GROUND_RES
    var = np.linalg.eigvalsh(np.cov(np.vstack([rows, cols]), bias=True)).max()
    return math.sqrt(12 * max(var, 0) + 1) * engine.GROUND_RES


def analyse(result, sens, target=None):
    target = target or result.params.target
    grid = result.grid
    th = thresholds(result, sens)
    min_px = max(1, int(np.ceil(sens.min_area_m2 / engine.GROUND_RES ** 2)))
    max_px = sens.max_area_m2 / engine.GROUND_RES ** 2
    blur = result.params.smooth // 2
    area_row = grid.row_area_km2() * 1e6
    detections, per_pass, masks = [], [], []
    count = np.zeros((grid.height, grid.width), np.uint8)
    for i, day in enumerate(result.days):
        level = np.minimum(result.before_db[i], result.after_db[i])  # the weaker side
        flag = result.before_db[i] != NODATA
        for arr, t in zip((result.before_db[i], result.after_db[i]), th[i]):
            if t is not None:
                flag &= arr >= t * DB_SCALE
        if target == "water":
            flag &= result.water
        elif target == "land":
            flag &= ~result.water
        labels, n = ndimage.label(flag, structure=np.ones((3, 3)))
        kept = np.zeros_like(flag)
        found = 0
        if n:
            sizes = np.bincount(labels.ravel())
            objects = ndimage.find_objects(labels)
            for lab in np.flatnonzero((sizes >= min_px) & (sizes <= max_px)):
                if lab == 0:
                    continue
                sl = objects[lab - 1]
                blob = labels[sl] == lab
                rr, cc = np.nonzero(blob)
                rr, cc = rr + sl[0].start, cc + sl[1].start
                kept[rr, cc] = True
                row, col = rr.mean(), cc.mean()
                # The speckle filter spreads each object by its half-width:
                # shrink it back before measuring size and length.
                core = ndimage.binary_erosion(np.pad(blob, 1), iterations=blur)[1:-1, 1:-1] if blur else blob
                if core.any():
                    sr, sc = np.nonzero(core)
                    sr, sc = sr + sl[0].start, sc + sl[1].start
                else:
                    sr, sc = rr, cc
                lat, lon = grid.pixel_latlon(row, col)
                detections.append(Detection(
                    pass_index=i, day=day, lat=float(lat), lon=float(lon),
                    area_m2=float(area_row[sr].sum()), length_m=float(_length_m(sr, sc)),
                    brightness_db=float(level[rr, cc].max() / DB_SCALE),
                    on_water=bool(result.water[int(row), int(col)]), row=float(row), col=float(col)))
                found += 1
        per_pass.append(found)
        masks.append(kept)
        count += kept
    return Analysis(detections, per_pass, count, masks, _hotspots(result, masks, count, detections), th)


def _hotspots(result, masks, count, detections):
    grid = result.grid
    labels, n = ndimage.label(count > 0, structure=np.ones((3, 3)))
    if not n:
        return []
    ids = np.arange(1, n + 1)
    covered = np.stack([result.before_db[i] != NODATA for i in range(len(masks))])
    present = np.array([ndimage.maximum(m, labels, ids) for m in masks]).astype(bool)  # (passes, n)
    seen_by = np.array([ndimage.maximum(c, labels, ids) for c in covered]).astype(bool)
    area = ndimage.sum(np.broadcast_to((grid.row_area_km2() * 1e6)[:, None], labels.shape), labels, ids)
    rows, cols = np.array(ndimage.center_of_mass(count > 0, labels, ids)).T
    # Typical size: median of the (blur-corrected) objects seen at each hotspot.
    sizes = defaultdict(list)
    for d in detections:
        sizes[labels[int(round(d.row)), int(round(d.col))]].append(d.area_m2)
    spots = []
    for j in range(n):
        lat, lon = grid.pixel_latlon(rows[j], cols[j])
        typical = float(np.median(sizes[j + 1])) if sizes[j + 1] else float(area[j])
        spots.append(Hotspot(lat=float(lat), lon=float(lon), area_m2=typical,
                             seen=int(present[:, j].sum()), of=int(seen_by[:, j].sum()),
                             days=[result.days[i] for i in np.flatnonzero(present[:, j])],
                             row=float(rows[j]), col=float(cols[j])))
    return sorted(spots, key=lambda s: (-s.seen, -s.area_m2))
