"""Satellite Change Map: Streamlit app. Start it with launch.py or the
"Start Change Map" launcher for your system."""

import datetime as dt
import json
import os
from pathlib import Path

import folium
import streamlit as st
import streamlit_folium
from branca.element import MacroElement
from folium.plugins import Draw
from jinja2 import Template

from satchange import engine, export, inputs, render, source

ROOT = Path(__file__).resolve().parent
CONFIG = Path(os.environ.get("SATCHANGE_CONFIG", ROOT / "config.json"))
ALPHAS = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8]
AREAS = [100, 200, 400, 1000, 2000, 5000, 10000]
SMOOTHING = {"Off (10 m detail)": 1, "Medium (recommended)": 3, "Strong": 5}
SEARCH, DRAWN, COORDS = "Search for a place…", "Drawn on the map", "Enter coordinates…"
CHOOSE, RESULT = "✏️ Choose area", "🗺️ Change map"
SATELLITE = "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
BIG_DOWNLOAD_MB = 500
DEFAULT_WATERMARK = "@Kaldockhi"

st.set_page_config(page_title="Satellite Change Map", page_icon="🛰️", layout="wide")
ss = st.session_state


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


@st.cache_data(show_spinner="Searching…", ttl=3600)
def search_places(query):
    return inputs.search_places(query)


@st.cache_data(show_spinner="Looking for satellite images…", ttl=1800, max_entries=20)
def find_images(key, _params):
    return engine.make_plan(_params)


@st.cache_data(show_spinner="Drawing map…", max_entries=8)
def map_html(key, det, _result):
    return render.build_map(_result, render.Detection(*det)).get_root().render()


@st.cache_data(show_spinner=False, max_entries=8)
def spot_list(key, det, _result):
    return export.spots_csv(_result, render.Detection(*det))


def describe(det):
    return (f"Shows spots where the radar signal changed by at least **{det.min_db:g} dB** over at "
            f"least **{det.min_area_m2:,.0f} m²**, with less than a **1 in {1 / det.alpha:,.0f}** "
            "chance per pixel of being noise.")


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


cfg = load_config()

# Widget values set by the previous run (a widget's value can only be changed
# before it is drawn).
for key in ("area_choice", "view"):
    if f"pending_{key}" in ss:
        ss[key] = ss.pop(f"pending_{key}")

# The last box drawn on the map is remembered between sessions.
if "drawn" not in ss and cfg.get("drawn"):
    ss.drawn = tuple(cfg["drawn"])

# Reopen the last result without recomputing.
if "result" not in ss and cfg.get("last_result"):
    path = engine.CACHE_DIR / f"{cfg['last_result']}.npz"
    if path.exists():
        try:
            ss.result = engine.Result.load(path)
        except Exception:
            pass

# --- Sidebar -------------------------------------------------------------
with st.sidebar:
    st.title("🛰️ Satellite change")
    st.caption("Free Sentinel-1 radar images. No account needed.")

    # 1. Area
    st.subheader("1 · Area")
    area_choice = st.selectbox("Area", [*engine.CITIES, SEARCH, DRAWN, COORDS], key="area_choice",
                               label_visibility="collapsed")
    label, bounds, note = area_choice, None, None
    if area_choice in engine.CITIES:
        bounds = engine.CITIES[area_choice]
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
    c1, c2 = st.columns(2)
    start = c1.date_input("Start", latest - dt.timedelta(days=31), format="DD/MM/YYYY", key="start")
    end = c2.date_input("End", latest, format="DD/MM/YYYY", key="end")
    st.caption("Compares the latest images up to the start date with the latest images up to "
               "the end date.")
    images = st.select_slider(
        "Images to average on each side", [1, 2, 3, 4, 6], value=3,
        help="Averaging several images removes more noise, so smaller changes can be found. "
             "More images mean a bigger download and reach further back from each date.")

    with st.expander("Advanced settings"):
        smooth = SMOOTHING[st.selectbox(
            "Noise reduction", list(SMOOTHING), index=1,
            help="Averages neighbouring pixels before comparing. Removes most radar speckle; "
                 "the smallest detail becomes about 30 m (Medium) or 50 m (Strong).")]
        orbit_pass = st.selectbox("Orbit direction", ["Any", "ASCENDING", "DESCENDING"])
        orbit = st.number_input("Relative orbit (0 = best available)", 0, 175, 0)
        auto_enl = st.checkbox("Measure speckle level from the data", True,
                               help="Equivalent number of looks (ENL) of one image after noise reduction.")
        enl = None if auto_enl else st.number_input("Equivalent number of looks", 1.0, 200.0, 4.4)
        bands = st.multiselect("Polarisations", ["VV", "VH"], ["VV", "VH"])

    # Check the dates and look up the images before anything is downloaded.
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
        params = engine.Params(
            label=label, bounds=tuple(float(b) for b in bounds), start=str(start), end=str(end),
            images=int(images), smooth=smooth,
            orbit_pass=None if orbit_pass == "Any" else orbit_pass,
            orbit=int(orbit) or None, enl=enl, bands=tuple(b for b in ("VV", "VH") if b in bands),
        )
        cached = (engine.CACHE_DIR / f"{params.cache_key()}.npz").exists()
        if not cached:  # a saved result opens without going online
            try:
                plan = find_images(params.cache_key(), params)
            except (source.SourceError, ValueError) as e:
                problem = str(e)
            except Exception as e:
                problem = f"Couldn't look up images: {e}"

    if problem:
        st.error(problem)
    elif cached:
        st.info("Already computed: opens instantly.")
    elif plan:
        st.info(f"**Before:** {fmt_days(plan.before)}  \n**After:** {fmt_days(plan.after)}  \n"
                f"Orbit {plan.orbit} ({plan.orbit_state}) · download about **{size(plan.download_mb)}**")
        gaps = [(start - max(plan.before)).days, (end - max(plan.after)).days]
        if max(gaps) > 14:
            st.warning(f"The nearest images are up to {max(gaps)} days before your dates.")
        if plan.download_mb > BIG_DOWNLOAD_MB:
            st.warning("That's a big download. A smaller area or fewer images is faster.")

    run = st.button("Find changes", type="primary", use_container_width=True,
                    disabled=not (plan or cached))

    saved = engine.saved_results()
    if saved:
        with st.expander(f"Saved maps ({len(saved)})"):
            paths = {desc: path for path, desc in saved}
            pick = st.selectbox("Saved map", list(paths), label_visibility="collapsed")
            c1, c2 = st.columns(2)
            if c1.button("Open", use_container_width=True):
                ss.result = engine.Result.load(paths[pick])
                ss.pending_view = RESULT
                save_config(last_result=paths[pick].stem)
                st.rerun()
            if c2.button("Delete all", use_container_width=True):
                engine.delete_saved()
                ss.pop("result", None)
                save_config(last_result=None)
                st.rerun()

# --- Run -------------------------------------------------------------------
if run:
    bar = st.progress(0.0, "Starting…")
    try:
        result = engine.run(params, plan=plan, progress=lambda f, msg: bar.progress(min(f, 1.0), msg))
    except (source.SourceError, ValueError) as e:
        bar.empty()
        st.error(str(e))
    except Exception as e:
        bar.empty()
        st.error(f"Something went wrong: {e}")
    else:
        ss.result = result
        ss.pending_view = RESULT
        save_config(last_result=params.cache_key())
        st.rerun()  # redraw the sidebar, which now knows the result is saved

# --- Main area ------------------------------------------------------------
result = ss.get("result")
if result is not None:
    ss.setdefault("view", RESULT)
    view = st.radio("View", [CHOOSE, RESULT], key="view", horizontal=True, label_visibility="collapsed")
else:
    view = CHOOSE

if view == CHOOSE:
    if result is None:
        st.header("Radar change map")
        st.markdown(
            "Compares Sentinel-1 radar images from a start date and an end date, and marks spots "
            "whose radar signal clearly changed: demolished or new buildings, vehicles and "
            "aircraft coming and going, cleared land, flooding and so on. Radar sees through "
            "clouds and at night. Images are free from Microsoft Planetary Computer; no account "
            "is needed.")
    st.subheader("Choose an area")
    st.markdown(
        "**Drag a box on the map:** click **▢** (top left), then click and drag across the map. "
        "To adjust it, click **✎**, drag the corners or the middle, then click **Save**. "
        "Scroll to zoom, drag to move around. Or pick a place in the sidebar.")
    new = inputs.bounds_from_drawings(area_picker(bounds))
    if new and new != ss.get("drawn"):
        ss.drawn = new
        ss.pending_area_choice = DRAWN
        save_config(drawn=list(new))
        st.rerun()
    if bounds:
        st.caption(f"**{label}**: about {inputs.area_km2(bounds):,.0f} km². {note or ''} "
                   "Then choose the dates and press **Find changes** in the sidebar.")
else:
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

    html = map_html(p.cache_key(), (det.alpha, det.min_db, det.min_area_m2), result)
    st.iframe(html, height=680)
    st.caption("Red: radar signal decreased · Cyan: increased · The number next to each spot is "
               "how many times the satellite saw a change there between one pass and the next; "
               "hover over it for the dates. Use the layer menu (top right) for the before and "
               "after radar images. A change in radar signal means something changed on the "
               "ground, not necessarily damage.")

    with st.container(border=True):
        st.markdown("#### ⬇️ Download")
        c1, c2, c3, c4 = st.columns([1.4, 1.2, 0.8, 0.8], vertical_alignment="bottom")
        background = c1.radio("Background", ["Satellite photo", "Radar image"], horizontal=True,
                              key="export_background")
        watermark = c2.text_input("Watermark", cfg.get("watermark", DEFAULT_WATERMARK), key="export_watermark")
        numbers = c3.checkbox("Numbers", True, key="export_numbers",
                              help="Show how many times a change was seen next to each spot.")
        fmt = c4.selectbox("Format", ["PNG", "JPEG"], key="export_format")
        settings = (p.cache_key(), det, background, watermark, numbers, fmt)
        if st.button("Create image", type="primary"):
            with st.spinner("Creating image…"):
                data, note = export.export_image(
                    result, det, preset, "satellite" if background == "Satellite photo" else "radar",
                    watermark, numbers, fmt)
            ss.export = (settings, data, note)
            save_config(watermark=watermark)
        made = ss.get("export")
        name = f"{p.label.lower().replace(' ', '_')}_{p.start}_{p.end}"
        if made and made[0] == settings:
            _, data, note = made
            if note:
                st.warning(note)
            st.image(data, width=560)
            st.download_button(f"Download image ({fmt})", data, file_name=f"{name}.{fmt.lower()}",
                               mime=f"image/{fmt.lower()}", type="primary")
        elif made:
            st.caption("Settings changed: press **Create image** again.")

        c1, c2, _ = st.columns([1, 1, 1])
        c1.download_button("Interactive map (HTML)", html, file_name=f"{name}.html", mime="text/html",
                           use_container_width=True)
        c2.download_button("List of changed spots (CSV)",
                           spot_list(p.cache_key(), (det.alpha, det.min_db, det.min_area_m2), result),
                           file_name=f"{name}_spots.csv", mime="text/csv", use_container_width=True,
                           help="Location, size, change and dates of every detected spot.")

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
