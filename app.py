"""Satellite Change Map: Streamlit app. Start it with launch.py or the
"Start Change Map" launcher for your system."""

import datetime as dt
import json
import os
from pathlib import Path

import altair as alt
import folium
import pandas as pd
import streamlit as st
import streamlit_folium
from branca.element import MacroElement
from folium.plugins import Draw
from jinja2 import Template

from satchange import engine, export, inputs, render, source
from satchange import timeseries as ts

ROOT = Path(__file__).resolve().parent
CONFIG = Path(os.environ.get("SATCHANGE_CONFIG", ROOT / "config.json"))
ALPHAS = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8]
AREAS = [100, 200, 400, 1000, 2000, 5000, 10000]
SMOOTHING = {"Off (10 m detail)": 1, "Medium (recommended)": 3, "Strong": 5}
SEARCH, DRAWN, COORDS = "Search for a place…", "Drawn on the map", "Enter coordinates…"
CHOOSE, RESULT = "✏️ Choose area", "🗺️ Results"
SATELLITE = "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
BIG_DOWNLOAD_MB = 500
DEFAULT_WATERMARK = "@Kaldockhi"

MODES = {
    "activity": ("✈️ Activity at a site", "Aircraft, vehicles and other objects that come and go: "
                 "airports, bases, ports, car parks."),
    "ships": ("🚢 Ships", "Vessels at sea, in straits, rivers and harbours."),
    "compare": ("🏚️ Changes between two dates", "What changed for good: damage, demolition, "
                "construction, cleared land."),
}
TARGET = {"activity": "land", "ships": "water"}
RUN_LABEL = {"activity": "Find activity", "ships": "Find ships", "compare": "Find changes"}

st.set_page_config(page_title="Satellite Change Map", page_icon="🛰️", layout="wide")
ss = st.session_state


# --- Helpers ---------------------------------------------------------------------

def load_config():
    try:
        return json.loads(CONFIG.read_text())
    except (OSError, ValueError):
        return {}


def save_config(**updates):
    try:
        CONFIG.write_text(json.dumps(load_config() | updates, indent=2))
    except OSError:
        pass


def km2(x):
    return f"{x:,.0f} km²" if x >= 100 else f"{x:.1f} km²" if x >= 10 else f"{x:.2f} km²"


def size(mb):
    return f"{mb / 1000:.1f} GB" if mb >= 1000 else f"{mb:,.0f} MB"


def fmt_days(days):
    """'22 Jun, 28 Jun, 4 Jul 2026'"""
    days = sorted(days)
    return ", ".join(f"{d.day} {d:%b}" for d in days) + f" {days[-1].year}"


def short(day):
    return engine.short_date(str(day))


@st.cache_data(show_spinner="Searching…", ttl=3600)
def search_places(query):
    return inputs.search_places(query)


@st.cache_data(show_spinner="Looking for satellite images…", ttl=1800, max_entries=20)
def find_images(key, _params):
    return (ts.make_plan if isinstance(_params, ts.TSParams) else engine.make_plan)(_params)


@st.cache_data(show_spinner="Drawing map…", max_entries=8)
def change_map_html(key, det, _result):
    return render.build_map(_result, render.Detection(*det)).get_root().render()


@st.cache_data(show_spinner=False, max_entries=8)
def spot_list(key, det, _result):
    return export.spots_csv(_result, render.Detection(*det))


@st.cache_resource(show_spinner="Finding objects in every pass…", max_entries=4)
def analysis(key, sens, _result):
    return ts.analyse(_result, ts.Sensitivity(*sens))


@st.cache_data(show_spinner="Drawing map…", max_entries=8)
def activity_map_html(key, sens, pass_index, _result, _analysis):
    return render.build_activity_map(_result, _analysis, pass_index).get_root().render()


def describe(det):
    return (f"Shows spots where the radar signal changed by at least **{det.min_db:g} dB** over at "
            f"least **{det.min_area_m2:,.0f} m²**, with less than a **1 in {1 / det.alpha:,.0f}** "
            "chance per pixel of being noise.")


def describe_sens(sens, ships):
    what = "ships" if ships else "objects"
    return (f"Counts {what} at least **{sens.min_db:g} dB** brighter than usual, "
            f"**{sens.min_area_m2:,.0f}–{sens.max_area_m2:,.0f} m²** in size, with less than a "
            f"**1 in {1 / sens.alpha:,.0f}** chance per pixel and pass of being noise.")


# When a new box is drawn, remove the old one, so there is only ever one area.
# Rendered as a child of the drawing layer: _parent is the layer, its parent the map.
KEEP_NEWEST = """
{% macro script(this, kwargs) %}
{{ this._parent._parent.get_name() }}.on('draw:created', function(e) {
  var group = {{ this._parent.get_name() }};
  group.eachLayer(function(layer) { if (layer !== e.layer) { group.removeLayer(layer); } });
});
{% endmacro %}
"""


def area_picker(bounds):
    """Map where the area can be drawn (square tool) or resized (pencil tool).
    Returns the drawn shapes, or None."""
    m = folium.Map(location=[50.45, 30.52], zoom_start=10, tiles=None, control_scale=True)
    folium.TileLayer(SATELLITE, attr="Esri World Imagery", name="Satellite photo").add_to(m)
    folium.TileLayer("OpenStreetMap", name="Street map", show=False).add_to(m)
    group = folium.FeatureGroup(name="Area", control=False).add_to(m)
    if bounds:
        west, south, east, north = bounds
        folium.Rectangle([[south, west], [north, east]], color="#ffcc00", weight=3,
                         fill=True, fill_opacity=0.08).add_to(group)
        m.fit_bounds([[south, west], [north, east]], padding=(30, 30))
    Draw(
        feature_group=group, show_geometry_on_click=False,
        draw_options={"rectangle": {"shapeOptions": {"color": "#ffcc00", "weight": 3, "fillOpacity": 0.08}},
                      "polyline": False, "polygon": False,
                      "circle": False, "marker": False, "circlemarker": False},
        edit_options={"remove": False},
    ).add_to(m)
    folium.LayerControl(position="bottomright").add_to(m)
    keep = MacroElement()
    keep._template = Template(KEEP_NEWEST)
    group.add_child(keep)
    state = streamlit_folium.st_folium(m, key="area_picker", height=560, use_container_width=True,
                                       returned_objects=["all_drawings"])
    return (state or {}).get("all_drawings")


def open_result(result, key):
    ss.result = result
    ss.pending_view = RESULT
    save_config(last_result=key)
    st.rerun()


# --- Start-up state ------------------------------------------------------------------

cfg = load_config()

# Widget values set by the previous run (a widget's value can only be changed
# before it is drawn).
for key in [k for k in ss if str(k).startswith("pending_")]:
    ss[key.removeprefix("pending_")] = ss.pop(key)

# The last box drawn on the map is remembered between sessions.
if "drawn" not in ss and cfg.get("drawn"):
    ss.drawn = tuple(cfg["drawn"])

# Reopen the last result without recomputing.
if "result" not in ss and cfg.get("last_result"):
    path = engine.CACHE_DIR / f"{cfg['last_result']}.npz"
    if path.exists():
        try:
            ss.result = engine.load_result(path)
        except Exception:
            pass

# --- Sidebar ---------------------------------------------------------------------------

with st.sidebar:
    st.title("🛰️ Satellite change")
    st.caption("Free Sentinel-1 radar images: sees through cloud, day and night. No account needed.")

    mode = st.radio("What do you want to find?", list(MODES), key="mode",
                    format_func=lambda m: MODES[m][0], captions=[MODES[m][1] for m in MODES])

    # 1. Area
    st.subheader("1 · Area")
    presets = engine.CITIES if mode == "compare" else ts.SITES[TARGET[mode]]
    area_key = f"area_{mode}"
    area_choice = st.selectbox("Area", [*presets, SEARCH, DRAWN, COORDS], key=area_key,
                               label_visibility="collapsed")
    label, bounds, note = area_choice, None, None
    if area_choice in presets:
        bounds = presets[area_choice]
    elif area_choice == DRAWN:
        label = st.text_input("Name", "My area", key="drawn_name")
        if ss.get("drawn"):
            bounds, note = inputs.fit_area(ss.drawn)
        else:
            st.caption("Draw a box on the map →")
    elif area_choice == SEARCH:
        query = st.text_input("Place name", placeholder="e.g. Kharkiv, Ukraine")
        if query.strip():
            try:
                places = search_places(query.strip())
            except Exception:
                places = None
                st.error("Place search isn't reachable right now. Try again, or enter coordinates.")
            if places == []:
                st.warning("No places found. Try adding the country.")
            elif places:
                names = [name for name, _ in places]
                name = st.selectbox("Matches", names)
                label = name.split(",")[0]
                bounds, note = inputs.fit_area(dict(places)[name])
    else:
        label = st.text_input("Name", "My area")
        c1, c2 = st.columns(2)
        west = c1.number_input("West (lon)", value=30.15, format="%.4f")
        east = c2.number_input("East (lon)", value=30.40, format="%.4f")
        south = c1.number_input("South (lat)", value=50.48, format="%.4f")
        north = c2.number_input("North (lat)", value=50.62, format="%.4f")
        if west < east and south < north:
            bounds, note = inputs.fit_area((west, south, east, north))
        else:
            st.error("West must be less than east, and south less than north.")
    if bounds:
        st.caption(f"About {inputs.area_km2(bounds):,.0f} km². {note or ''}")

    # 2. Dates
    st.subheader("2 · Dates")
    latest = dt.date.today() - dt.timedelta(days=3)  # new images take a few days to appear
    span_days = 31 if mode == "compare" else 60
    c1, c2 = st.columns(2)
    start = c1.date_input("Start", latest - dt.timedelta(days=span_days), format="DD/MM/YYYY",
                          key=f"start_{mode}")
    end = c2.date_input("End", latest, format="DD/MM/YYYY", key=f"end_{mode}")
    if mode == "compare":
        st.caption("Compares the latest images up to the start date with the latest images up to "
                   "the end date.")
        images = st.select_slider(
            "Images to average on each side", [1, 2, 3, 4, 6], value=3,
            help="Averaging several images removes more noise, so smaller changes can be found. "
                 "More images mean a bigger download and reach further back from each date.")
    else:
        st.caption(f"Looks at every satellite pass between the dates (up to {ts.MAX_PASSES}, the "
                   "latest ones). Each pass is compared with the passes just before and after it.")

    with st.expander("Advanced settings"):
        default_smooth = 0 if mode == "ships" else 1
        smooth = SMOOTHING[st.selectbox(
            "Noise reduction", list(SMOOTHING), index=default_smooth, key=f"smooth_{mode}",
            help="Averages neighbouring pixels before comparing. Removes most radar speckle; "
                 "the smallest detail becomes about 30 m (Medium) or 50 m (Strong). Ships are "
                 "best found at full 10 m detail.")]
        if mode != "compare":
            all_orbits = st.checkbox(
                "Use every satellite track that covers the area", True,
                help="Combining tracks (e.g. morning and evening passes) roughly doubles how often "
                     "the site is seen.")
        orbit_pass = st.selectbox("Orbit direction", ["Any", "ASCENDING", "DESCENDING"])
        if mode == "compare":
            orbit = st.number_input("Relative orbit (0 = best available)", 0, 175, 0)
        auto_enl = st.checkbox("Measure speckle level from the data", True,
                               help="Equivalent number of looks (ENL) of one image after noise reduction.")
        enl = None if auto_enl else st.number_input("Equivalent number of looks", 1.0, 200.0, 4.4)
        bands = st.multiselect("Polarisations", ["VV", "VH"], ["VV", "VH"])

    # Check the inputs and look up the images before anything is downloaded.
    problem, params, plan, cached = None, None, None, False
    if start >= end:
        problem = "The end date must be after the start date."
    elif end > dt.date.today():
        problem = "The end date is in the future, so there are no images for it yet."
    elif start < dt.date(2014, 10, 1):
        problem = "Sentinel-1 images start in October 2014."
    elif not bands:
        problem = "Choose at least one polarisation."
    elif bounds:
        common = dict(label=label, bounds=tuple(float(b) for b in bounds), start=str(start), end=str(end),
                      smooth=smooth, orbit_pass=None if orbit_pass == "Any" else orbit_pass, enl=enl,
                      bands=tuple(b for b in ("VV", "VH") if b in bands))
        if mode == "compare":
            params = engine.Params(**common, images=int(images), orbit=int(orbit) or None)
        else:
            params = ts.TSParams(**common, target=TARGET[mode], all_orbits=all_orbits)
        cached = (engine.CACHE_DIR / f"{params.cache_key()}.npz").exists()
        if not cached:  # a saved result opens without going online
            try:
                plan = find_images(params.cache_key(), params)
            except (source.SourceError, ValueError) as e:
                problem = str(e)
            except Exception as e:
                problem = f"Couldn't look up images: {e}"
        if plan and mode != "compare" and not ts.fits_in_memory(engine.Grid.for_bounds(params.bounds),
                                                                len(plan.passes)):
            problem = ("That's too much to analyse at once. Choose a smaller area or a shorter "
                       "date range.")
            plan = None

    if problem:
        st.error(problem)
    elif cached:
        st.info("Already computed: opens instantly.")
    elif plan and mode == "compare":
        st.info(f"**Before:** {fmt_days(plan.before)}  \n**After:** {fmt_days(plan.after)}  \n"
                f"Orbit {plan.orbit} ({plan.orbit_state}) · download about **{size(plan.download_mb)}**")
        gaps = [(start - max(plan.before)).days, (end - max(plan.after)).days]
        if max(gaps) > 14:
            st.warning(f"The nearest images are up to {max(gaps)} days before your dates.")
    elif plan:
        tracks = sorted({o for _, o, _ in plan.passes})
        st.info(f"**{len(plan.passes)} passes**, {short(plan.days[0])} – {short(plan.days[-1])}, "
                f"from {len(tracks)} satellite track{'s' if len(tracks) > 1 else ''}  \n"
                f"Download about **{size(plan.download_mb)}**")
    if plan and plan.download_mb > BIG_DOWNLOAD_MB:
        st.warning("That's a big download. A smaller area or shorter range is faster.")

    run = st.button(RUN_LABEL[mode], type="primary", use_container_width=True,
                    disabled=not (plan or cached))

    saved = engine.saved_results()
    if saved:
        with st.expander(f"Saved results ({len(saved)})"):
            paths = {desc: path for path, desc in saved}
            pick = st.selectbox("Saved result", list(paths), label_visibility="collapsed")
            c1, c2 = st.columns(2)
            if c1.button("Open", use_container_width=True):
                open_result(engine.load_result(paths[pick]), paths[pick].stem)
            if c2.button("Delete all", use_container_width=True):
                engine.delete_saved()
                ss.pop("result", None)
                save_config(last_result=None)
                st.rerun()

# --- Run ---------------------------------------------------------------------------------

if run:
    bar = st.progress(0.0, "Starting…")
    runner = engine.run if mode == "compare" else ts.run
    try:
        result = runner(params, plan=plan, progress=lambda f, msg: bar.progress(min(f, 1.0), msg))
    except (source.SourceError, ValueError) as e:
        bar.empty()
        st.error(str(e))
    except Exception as e:
        bar.empty()
        st.error(f"Something went wrong: {e}")
    else:
        open_result(result, params.cache_key())


# --- Result views ------------------------------------------------------------------------

def download_box(name, settings_key, make_image, extra_files, numbers_help=None):
    """Image with legend and watermark, plus other files. ``settings_key``
    identifies what the image shows, so a stale preview isn't offered."""
    with st.container(border=True):
        st.markdown("#### ⬇️ Download")
        c1, c2, c3, c4 = st.columns([1.4, 1.2, 0.8, 0.8], vertical_alignment="bottom")
        background = c1.radio("Background", ["Satellite photo", "Radar image"], horizontal=True,
                              key="export_background")
        watermark = c2.text_input("Watermark", cfg.get("watermark", DEFAULT_WATERMARK), key="export_watermark")
        numbers = c3.checkbox("Numbers", True, key="export_numbers", help=numbers_help,
                              disabled=numbers_help is None)
        fmt = c4.selectbox("Format", ["PNG", "JPEG"], key="export_format")
        settings = (settings_key, background, watermark, numbers, fmt)
        if st.button("Create image", type="primary"):
            with st.spinner("Creating image…"):
                data, note = make_image("satellite" if background == "Satellite photo" else "radar",
                                        watermark, numbers, fmt)
            ss.export = (settings, data, note)
            save_config(watermark=watermark)
        made = ss.get("export")
        if made and made[0] == settings:
            _, data, note = made
            if note:
                st.warning(note)
            st.image(data, width=560)
            st.download_button(f"Download image ({fmt})", data, file_name=f"{name}.{fmt.lower()}",
                               mime=f"image/{fmt.lower()}", type="primary")
        elif made:
            st.caption("Settings changed: press **Create image** again.")
        cols = st.columns(3)
        for col, (label_, data, file_name, mime, help_) in zip(cols, extra_files):
            col.download_button(label_, data, file_name=file_name, mime=mime, use_container_width=True,
                                help=help_)


def timeline(result, a, ships, selected=None):
    """Objects per pass: one series, so the title names it and there's no legend."""
    df = pd.DataFrame({
        "pass": range(len(result.days)),
        "date": [short(d) for d in result.days],
        "label": [f"{dt.date.fromisoformat(d).day} {dt.date.fromisoformat(d):%b}" for d in result.days],
        "count": a.per_pass,
        "selected": [i == selected for i in range(len(result.days))],
    })
    noun = "Ships" if ships else "Objects"
    color = (alt.condition("datum.selected", alt.value(render.SHIP_COLOR), alt.value(render.ACTIVITY_RAMP[1]))
             if selected is not None else alt.value(render.ACTIVITY_RAMP[1]))
    chart = (
        alt.Chart(df, title=alt.TitleParams(f"{noun} per satellite pass", anchor="start", fontSize=14))
        .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
        .encode(
            x=alt.X("label:N", sort=None, title=None, axis=alt.Axis(labelAngle=0, labelOverlap=True)),
            y=alt.Y("count:Q", title=None, axis=alt.Axis(tickMinStep=1, gridColor="#e4e3df", domain=False)),
            color=color,
            tooltip=[alt.Tooltip("date:N", title="Pass"), alt.Tooltip("count:Q", title=noun)],
        )
        .properties(height=170)
        .configure_view(strokeWidth=0)
        .configure_axis(labelColor="#52514e", tickColor="#c3c2b7")
    )
    st.altair_chart(chart, use_container_width=True)


def show_activity(result):
    p = result.params
    ships = p.target == "water"
    n = len(result.days)
    st.header(f"{p.label}: {'ships' if ships else 'activity'}, {short(result.days[0])} → {short(result.days[-1])}")
    tracks = sorted(set(result.pass_orbits))
    st.caption(f"{n} satellite passes from {len(tracks)} track{'s' if len(tracks) > 1 else ''}. "
               "Each pass is compared with the passes just before and after it, so only things "
               "that come and go are counted.")

    c1, c2 = st.columns([2, 3])
    presets = ts.PRESETS[p.target]
    preset = c1.radio("Sensitivity", list(presets), index=1, horizontal=True, key="ts_preset",
                      help="Sensitive also counts smaller and fainter objects; Strict only big, clear ones.")
    sens = presets[preset]
    c2.markdown(describe_sens(sens, ships))
    a = analysis(p.cache_key(), tuple(vars(sens).values()), result)

    total = len(a.detections)
    busiest = max(range(n), key=lambda i: a.per_pass[i])
    c1, c2, c3, c4 = st.columns(4)
    noun = "ship" if ships else "object"
    most = a.per_pass[busiest]
    busiest_day = short(result.days[busiest]) if most else "–"
    if ships:
        c1.metric("Ship detections", f"{total:,}", help="A ship seen in two passes counts twice.")
        c2.metric("Average per pass", f"{total / n:.1f}")
    else:
        c1.metric("Objects seen", f"{total:,}", help="Counted in every pass; an aircraft parked for "
                                                    "three passes counts three times.")
        c2.metric("Busy spots", f"{len(a.hotspots):,}", help="Places where objects were seen at least once.")
    c3.metric("Busiest pass", busiest_day, help=f"{most} {noun}{'s' if most != 1 else ''} in that pass.")
    c4.metric("Passes", f"{n}")

    view = st.radio("Show", ["All passes", "One pass"], horizontal=True, key="ts_view")
    pass_index = None
    if view == "One pass":
        if ss.get("ts_pass") not in range(n):
            ss.ts_pass = busiest
        pass_index = st.select_slider(
            "Pass", list(range(n)), key="ts_pass",
            format_func=lambda i: f"{short(result.days[i])} · {a.per_pass[i]} {'ship' if ships else 'object'}"
                                  f"{'s' if a.per_pass[i] != 1 else ''}")
    timeline(result, a, ships, pass_index)

    html = activity_map_html(p.cache_key(), (preset,), pass_index, result, a)
    st.iframe(html, height=680)
    if pass_index is not None:
        st.caption("Objects seen in this pass. Turn on the radar image of the pass in the layer menu "
                   "(top right); hover over a dot for its size and brightness.")
    elif ships:
        st.caption("Each dot is a ship seen in one pass; hover for the date and size. Pick "
                   "**One pass** to step through the passes.")
    else:
        st.caption("Blue: places where objects came and went, darker = more passes. The label is "
                   "how many passes something was there out of the passes that covered it; hover "
                   "for the dates. Pick **One pass** to step through the passes.")

    name = f"{p.label.lower().replace(' ', '_')}_{'ships' if ships else 'activity'}_{p.start}_{p.end}"
    if pass_index is not None:
        name += f"_pass_{result.days[pass_index]}"
    files = [("Interactive map (HTML)", html, f"{name}.html", "text/html", None),
             ("Every detection (CSV)", export.detections_csv(a), f"{name}_detections.csv", "text/csv",
              "Date, location, length, size and brightness of every object seen.")]
    if not ships:
        files.append(("Busy spots (CSV)", export.hotspots_csv(a), f"{name}_spots.csv", "text/csv",
                      "Location, passes seen and dates of every busy spot."))
    download_box(
        name, (p.cache_key(), preset, pass_index),
        lambda bg, wm, num, fmt: export.export_activity_image(result, a, preset, pass_index, bg, wm, num, fmt),
        files, numbers_help=None if ships or pass_index is not None else
        "Show '3/12' labels: passes something was there, out of passes that covered it.")

    with st.expander("Details and method"):
        lo, hi = (min(t for pair in a.thresholds_db for t in pair if t is not None),
                  max(t for pair in a.thresholds_db for t in pair if t is not None))
        water_km2 = float(result.water.sum(axis=1) @ result.grid.row_area_km2())
        st.markdown(f"""
- **Images:** {n} Sentinel-1 passes (radiometrically terrain-corrected, from Microsoft Planetary
  Computer), tracks {', '.join(map(str, tracks))}: {fmt_days(dt.date.fromisoformat(d) for d in result.days)}.
- **Usual state:** for each pass and pixel, the median of up to {ts.REF_SIDE} passes before it and,
  separately, of up to {ts.REF_SIDE} after it (same track). An object counts only if it is brighter
  than both, so lasting changes (demolition, construction) and slow trends are not counted. Each
  pass's overall brightness (e.g. after rain) is levelled first.
- **Threshold:** exact for the measured speckle level (ENL **{result.enl:.1f}** per pass after
  {p.smooth} × {p.smooth} noise reduction), between {lo:.1f} and {hi:.1f} dB with these settings.
- **Objects:** connected patches of {sens.min_area_m2:,.0f}–{sens.max_area_m2:,.0f} m²; sizes are
  corrected for the noise filter's blur. Length is from the patch's shape.
- **Water:** {water_km2:.1f} km² found automatically (large areas that are usually dark).
  {'Ships are only counted on water.' if ships else 'Objects on water are not counted.'}
- **Limits:** objects there in most passes become the "usual" state and are not counted;
  very small boats and cars may be missed at 10 m.
""")


def show_compare(result):
    p = result.params
    st.header(f"{p.label}: {engine.short_date(p.start)} → {engine.short_date(p.end)}")
    st.caption(f"Before: {fmt_days(dt.date.fromisoformat(d) for d in result.before_days)} · "
               f"After: {fmt_days(dt.date.fromisoformat(d) for d in result.after_days)}")

    c1, c2 = st.columns([2, 3])
    preset = c1.radio("Detection", [*render.PRESETS, "Custom"], index=1, horizontal=True, key="preset",
                      help="Sensitive shows smaller and fainter changes, Strict only big, clear ones.")
    if preset == "Custom":
        with c2:
            base = render.PRESETS["Balanced"]
            det = render.Detection(
                alpha=st.select_slider("Certainty (chance a pixel is noise)", ALPHAS, value=base.alpha,
                                       format_func=lambda a: f"1 in {1 / a:,.0f}"),
                min_db=st.slider("Smallest change in radar signal (dB)", 0.5, 10.0, base.min_db, 0.5,
                                 help="3 dB = the signal halved or doubled."),
                min_area_m2=st.select_slider("Smallest patch (m²)", AREAS, value=base.min_area_m2,
                                             format_func=lambda a: f"{a:,}"),
            )
    else:
        det = render.PRESETS[preset]
        c2.markdown(describe(det))
    s = render.summarize(result, det)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Radar signal decreased", km2(s.decrease_km2),
              help="Often demolished or damaged buildings, cleared land, flooding, "
                   "vehicles or aircraft that left.")
    c2.metric("Radar signal increased", km2(s.increase_km2),
              help="Often new structures, rubble, vehicles or aircraft that arrived, "
                   "vegetation.")
    c3.metric("Changed spots", f"{s.spots:,}")
    if s.noise_km2 is not None:
        c4.metric("Noise check", km2(s.noise_km2),
                  help="The same detection comparing the before images with each other, where "
                       "nothing should have changed. Close to 0 means what's shown is real change.")

    det_key = (det.alpha, det.min_db, det.min_area_m2)
    html = change_map_html(p.cache_key(), det_key, result)
    st.iframe(html, height=680)
    st.caption("Red: radar signal decreased · Cyan: increased · The number next to each spot is "
               "how many times the satellite saw a change there between one pass and the next; "
               "hover over it for the dates. Use the layer menu (top right) for the before and "
               "after radar images. A change in radar signal means something changed on the "
               "ground, not necessarily damage.")

    name = f"{p.label.lower().replace(' ', '_')}_{p.start}_{p.end}"
    download_box(
        f"{name}_{preset.lower()}", (p.cache_key(), det_key),
        lambda bg, wm, num, fmt: export.export_image(result, det, preset, bg, wm, num, fmt),
        [("Interactive map (HTML)", html, f"{name}.html", "text/html", None),
         ("List of changed spots (CSV)", spot_list(p.cache_key(), det_key, result), f"{name}_spots.csv",
          "text/csv", "Location, size, change and dates of every detected spot.")],
        numbers_help="Show how many times a change was seen next to each spot.")

    with st.expander("Details and method"):
        st.markdown(f"""
- **Images:** Sentinel-1 (radiometrically terrain-corrected, from Microsoft Planetary
  Computer), relative orbit **{result.orbit}**: the latest {len(result.before_days)} up to the
  start date and the latest {len(result.after_days)} up to the end date, averaged on each side.
- **Noise reduction:** {p.smooth} × {p.smooth} pixel averaging. Speckle level after it:
  ENL = **{result.enl:.1f}** per image ({'measured from consecutive before images'
  if result.enl_estimated else 'set manually'}).
- **Test:** likelihood-ratio test for equal mean radar signal per pixel
  ({' + '.join(p.bands)}), with Bartlett correction, χ² with {len(p.bands)} dof.
- **Filters:** a pixel is shown only if the test is significant (chance of noise below
  1 in {1 / det.alpha:,.0f}), the signal changed by at least {det.min_db:g} dB, and it is part
  of a patch of at least {det.min_area_m2:,.0f} m².
- **Noise check:** the same test and filters comparing the before images with each other.
- **Grid:** {result.grid.width} × {result.grid.height} pixels of 10 m.
""")
        if result.orbits:
            st.caption("Orbits considered")
            st.dataframe(
                [{"Orbit": o["orbit"], "Area covered": f"{o['coverage']:.0%}",
                  "Before images": o["before"], "After images": o["after"]} for o in result.orbits],
                hide_index=True,
            )


# --- Main area -----------------------------------------------------------------------------

result = ss.get("result")
if result is not None:
    ss.setdefault("view", RESULT)
    view = st.radio("View", [CHOOSE, RESULT], key="view", horizontal=True, label_visibility="collapsed")
else:
    view = CHOOSE

if view == CHOOSE:
    if result is None:
        st.header("Radar change map")
        st.markdown("Find things on the ground with free radar satellite images. Pick what you're "
                    "looking for:")
        cols = st.columns(3)
        for col, (key, (title, text)) in zip(cols, MODES.items()):
            with col.container(border=True):
                st.markdown(f"**{title}**")
                st.caption(text)
                if st.button("Choose", key=f"pick_{key}", disabled=key == mode,
                             type="secondary", use_container_width=True):
                    ss.pending_mode = key
                    st.rerun()
    st.subheader("Choose an area")
    st.markdown(
        "**Drag a box on the map:** click **▢** (top left), then click and drag across the map. "
        "To adjust it, click **✎**, drag the corners or the middle, then click **Save**. "
        "Scroll to zoom, drag to move around. Or pick a place in the sidebar.")
    new = inputs.bounds_from_drawings(area_picker(bounds))
    if new and new != ss.get("drawn"):
        ss.drawn = new
        ss[f"pending_{area_key}"] = DRAWN
        save_config(drawn=list(new))
        st.rerun()
    if bounds:
        st.caption(f"**{label}**: about {inputs.area_km2(bounds):,.0f} km². {note or ''} "
                   f"Then choose the dates and press **{RUN_LABEL[mode]}** in the sidebar.")
elif isinstance(result, ts.TSResult):
    show_activity(result)
else:
    show_compare(result)
