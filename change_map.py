"""Map Sentinel-1 radar change over a city with Google Earth Engine.

Implements the bivariate (VV + VH) likelihood ratio test from "Detecting
Changes in Sentinel-1 Imagery (Part 1)" and writes an interactive HTML map.
Pixels whose backscatter changed significantly between a "before" and an
"after" window are coloured by direction (increase / decrease).

Example:
    python change_map.py --project my-gcp-project
    python change_map.py --project my-gcp-project --max-images 1  # tutorial's single pair
    python change_map.py --project my-gcp-project --bbox 36.10 49.90 36.45 50.10 --name Kharkiv
"""

import argparse
import datetime as dt
import sys
from collections import defaultdict

import ee
import folium

from sar_stats import DEFAULT_ENL, LRT_EXPRESSION, chi2_threshold

# (west, south, east, north) in degrees.
CITIES = {
    # Kyiv city plus the north-western suburbs (Irpin, Bucha, Hostomel)
    # that were fought over in February-March 2022.
    "kyiv": ("Kyiv", (30.15, 50.20, 30.85, 50.62)),
}

# Same season a year apart, so seasonal vegetation/soil moisture changes
# mostly cancel. The after window starts once Russian forces withdrew from
# the Kyiv region at the start of April 2022.
DEFAULT_BEFORE = ("2021-04-01", "2021-05-31")
DEFAULT_AFTER = ("2022-04-01", "2022-05-31")

COLLECTION = "COPERNICUS/S1_GRD_FLOAT"  # linear power, as in the tutorial
DECREASE_COLOR = "ff3b30"
INCREASE_COLOR = "00c8ff"


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--project", help="Google Cloud project registered for Earth Engine")
    p.add_argument("--city", default="kyiv", choices=sorted(CITIES), help="preset area of interest")
    p.add_argument("--bbox", nargs=4, type=float, metavar=("W", "S", "E", "N"), help="custom area (overrides --city)")
    p.add_argument("--name", help="label for a custom --bbox")
    p.add_argument("--before", nargs=2, default=DEFAULT_BEFORE, metavar=("START", "END"), help="before window, YYYY-MM-DD (end exclusive)")
    p.add_argument("--after", nargs=2, default=DEFAULT_AFTER, metavar=("START", "END"), help="after window, YYYY-MM-DD (end exclusive)")
    p.add_argument("--orbit-pass", choices=["ASCENDING", "DESCENDING"], help="restrict to one orbit direction")
    p.add_argument("--orbit", type=int, help="relative orbit number (default: best one found)")
    p.add_argument("--max-images", type=int, default=0, help="acquisitions per window; 1 = the tutorial's single pair, 0 = all")
    p.add_argument("--alpha", type=float, default=0.01, help="significance level = expected false alarm rate")
    p.add_argument("--enl", type=float, default=DEFAULT_ENL, help="equivalent number of looks of one scene")
    p.add_argument("--bands", nargs="+", default=["VV", "VH"], choices=["VV", "VH"])
    p.add_argument("--out", help="output HTML (default: <name>_change_map.html)")
    p.add_argument("--no-stats", action="store_true", help="skip the changed-area summary")
    args = p.parse_args(argv)

    if args.bbox:
        args.label, args.bounds = args.name or "Custom area", tuple(args.bbox)
    else:
        args.label, args.bounds = CITIES[args.city]
    if args.out is None:
        args.out = args.label.lower().replace(" ", "_") + "_change_map.html"
    return args


def initialize(project):
    try:
        ee.Initialize(project=project)
    except ee.EEException:  # no stored credentials yet
        ee.Authenticate()
        ee.Initialize(project=project)


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
    """Pick the relative orbit that covers the AOI in both windows with the
    most acquisitions. Change detection needs the same viewing geometry."""
    candidates = sorted(set(days_before) & set(days_after))
    if not candidates:
        sys.exit("No relative orbit has acquisitions in both windows; widen the date ranges.")

    def coverage(col, orbit):
        footprint = col.filter(ee.Filter.eq("relativeOrbitNumber_start", orbit)).geometry()
        return footprint.intersection(aoi, 100).area(100).divide(aoi.area(100))

    cover = ee.Dictionary({
        str(o): ee.Number(coverage(before, o)).min(coverage(after, o)) for o in candidates
    }).getInfo()

    def score(o):
        full = cover[str(o)] > 0.98
        return (full, min(len(days_before[o]), len(days_after[o])) if full else cover[str(o)])

    best = max(candidates, key=score)
    for o in candidates:
        print(f"  orbit {o:3d}: {cover[str(o)]:5.0%} of AOI, "
              f"{len(days_before[o])} before / {len(days_after[o])} after acquisitions")
    return best


def window_mean(col, days, bands):
    """Average the acquisitions of one window.

    Scenes from the same day are mosaicked first. Averaging n independent
    acquisitions multiplies the number of looks by n, which the test accounts
    for per pixel via the count image.
    """
    daily = ee.ImageCollection([
        col.filterDate(str(d), str(d + dt.timedelta(days=1))).mosaic().select(bands)
        for d in days
    ])
    # Masking non-positive values drops no-data borders and keeps log() finite.
    daily = daily.map(lambda im: im.updateMask(im.gt(0).reduce(ee.Reducer.min())))
    return daily.mean(), daily.select(bands[0]).count()


def change_images(before, after, n_before, n_after, bands, enl, alpha):
    looks_before = n_before.toFloat().multiply(enl)
    looks_after = n_after.toFloat().multiply(enl)
    stat = ee.Image.cat([
        before.expression(LRT_EXPRESSION, {
            "s1": before.select(b), "s2": after.select(b),
            "L1": looks_before, "L2": looks_after,
        })
        for b in bands
    ]).reduce(ee.Reducer.sum())

    changed = stat.gt(chi2_threshold(alpha, dof=len(bands)))
    span = lambda im: im.select(bands).reduce(ee.Reducer.sum())
    increased = span(after).gt(span(before))
    # +1 increase, -1 decrease; unchanged pixels masked.
    direction = increased.multiply(2).subtract(1).updateMask(changed).rename("direction")
    return changed.rename("changed"), direction


def area_summary(changed, direction, aoi):
    px = ee.Image.pixelArea().divide(1e6)
    areas = ee.Image.cat([
        px.updateMask(changed.mask()).rename("valid"),
        px.updateMask(direction.eq(1)).rename("increase"),
        px.updateMask(direction.eq(-1)).rename("decrease"),
    ]).reduceRegion(ee.Reducer.sum(), aoi, scale=20, maxPixels=1e10, tileScale=4)
    return areas.getInfo()


def add_ee_layer(m, image, vis, name, show=True):
    url = image.getMapId(vis)["tile_fetcher"].url_format
    folium.TileLayer(tiles=url, attr="Google Earth Engine, Copernicus Sentinel-1", name=name, overlay=True, show=show).add_to(m)


def build_map(args, aoi, before, after, direction, info):
    west, south, east, north = args.bounds
    m = folium.Map(location=[(south + north) / 2, (west + east) / 2], zoom_start=11, tiles=None)
    folium.TileLayer(
        "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri World Imagery", name="Satellite (Esri)",
    ).add_to(m)
    folium.TileLayer("OpenStreetMap", name="OpenStreetMap").add_to(m)

    db = lambda im: im.select("VV").log10().multiply(10).clip(aoi)
    gray = {"min": -20, "max": 0}
    add_ee_layer(m, db(before), gray, "Before: VV backscatter (dB)", show=False)
    add_ee_layer(m, db(after), gray, "After: VV backscatter (dB)", show=False)
    # Classic change composite: red = decrease, cyan = increase.
    add_ee_layer(m, ee.Image.cat([db(before), db(after), db(after)]), {"min": -20, "max": 0},
                 "Before/after composite (VV)", show=False)
    add_ee_layer(m, direction.clip(aoi), {"min": -1, "max": 1, "palette": [DECREASE_COLOR, "000000", INCREASE_COLOR]},
                 f"Significant change (alpha={args.alpha})")
    folium.Rectangle([[south, west], [north, east]], name="Area of interest",
                     fill=False, color="#ffffff", weight=1.5).add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)

    legend = f"""
    <div style="position:fixed;bottom:24px;left:12px;z-index:9999;max-width:320px;
                background:rgba(20,20,20,.85);color:#eee;padding:10px 12px;border-radius:6px;
                font:12px/1.45 system-ui,sans-serif">
      <b style="font-size:14px">{args.label}: Sentinel-1 change</b><br>
      Before {info['before']} &rarr; after {info['after']}<br>
      Relative orbit {info['orbit']}, {info['n_before']} + {info['n_after']} acquisitions,
      bands {'+'.join(args.bands)}, &alpha; = {args.alpha}<br>
      <span style="color:#{DECREASE_COLOR}">&#9632;</span> backscatter decrease &nbsp;
      <span style="color:#{INCREASE_COLOR}">&#9632;</span> backscatter increase<br>
      <span style="opacity:.7">~{args.alpha:.0%} of unchanged pixels are flagged by chance.</span>
    </div>"""
    m.get_root().html.add_child(folium.Element(legend))
    return m


def main(argv=None):
    args = parse_args(argv)
    initialize(args.project)
    aoi = ee.Geometry.Rectangle(list(args.bounds))

    before_col = s1_collection(aoi, *args.before, args.orbit_pass)
    after_col = s1_collection(aoi, *args.after, args.orbit_pass)
    days_before, days_after = acquisition_days(before_col), acquisition_days(after_col)

    if args.orbit is None:
        print("Candidate orbits:")
        orbit = choose_orbit(before_col, after_col, aoi, days_before, days_after)
    else:
        orbit = args.orbit
        if orbit not in days_before or orbit not in days_after:
            sys.exit(f"Relative orbit {orbit} has no acquisitions in one of the windows.")

    by_orbit = ee.Filter.eq("relativeOrbitNumber_start", orbit)
    before_days, after_days = days_before[orbit], days_after[orbit]
    if args.max_images:
        # Keep the acquisitions closest to the event.
        before_days, after_days = before_days[-args.max_images:], after_days[:args.max_images]
    print(f"Using relative orbit {orbit}:")
    print(f"  before: {', '.join(map(str, before_days))}")
    print(f"  after:  {', '.join(map(str, after_days))}")

    before, n_before = window_mean(before_col.filter(by_orbit), before_days, args.bands)
    after, n_after = window_mean(after_col.filter(by_orbit), after_days, args.bands)
    changed, direction = change_images(before, after, n_before, n_after, args.bands, args.enl, args.alpha)

    if not args.no_stats:
        print("Computing changed area (20 m)...")
        a = area_summary(changed, direction, aoi)
        valid = a["valid"] or float("nan")
        print(f"  area analysed:        {valid:8.1f} km2")
        print(f"  backscatter increase: {a['increase']:8.1f} km2 ({a['increase'] / valid:.1%})")
        print(f"  backscatter decrease: {a['decrease']:8.1f} km2 ({a['decrease'] / valid:.1%})")
        print(f"  expected by chance:   {args.alpha * valid:8.1f} km2 ({args.alpha:.1%})")

    info = {
        "before": f"{before_days[0]}..{before_days[-1]}", "after": f"{after_days[0]}..{after_days[-1]}",
        "orbit": orbit, "n_before": len(before_days), "n_after": len(after_days),
    }
    build_map(args, aoi, before, after, direction, info).save(args.out)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
