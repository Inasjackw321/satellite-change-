"""Satellite Change Map: Streamlit app. Start it with launch.py or the
"Start Change Map" launcher for your system."""

import datetime as dt
import json
import os
from pathlib import Path

import folium
import streamlit as st

from satchange import engine, inputs, render, source

ROOT = Path(__file__).resolve().parent
CONFIG = Path(os.environ.get("SATCHANGE_CONFIG", ROOT / "config.json"))
ALPHAS = [0.1, 0.05, 0.01, 0.001, 1e-4, 1e-5, 1e-6]
SEARCH, COORDS = "Search for a place…", "Enter coordinates…"
BIG_DOWNLOAD_MB = 500

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


def fmt_period(period):
    start, end = period
    return f"{start:%d %b %Y} – {end:%d %b %Y}"


@st.cache_data(show_spinner="Searching…", ttl=3600)
def search_places(query):
    return inputs.search_places(query)


@st.cache_data(show_spinner="Looking for satellite images…", ttl=1800, max_entries=20)
def find_images(key, _params):
    return engine.make_plan(_params)


@st.cache_data(show_spinner="Drawing map…", max_entries=8)
def map_html(key, alpha, _result):
    return render.build_map(_result, alpha).get_root().render()


def area_preview(bounds):
    west, south, east, north = bounds
    m = folium.Map(tiles=None)
    folium.TileLayer(
        "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri World Imagery",
    ).add_to(m)
    folium.Rectangle([[south, west], [north, east]], color="#ffcc00", weight=2, fill=False).add_to(m)
    m.fit_bounds([[south, west], [north, east]])
    return m.get_root().render()


cfg = load_config()

# --- Sidebar -------------------------------------------------------------
with st.sidebar:
    st.title("🛰️ Satellite change")
    st.caption("Free Sentinel-1 radar images. No account needed.")

    # 1. Area
    st.subheader("1 · Area")
    area_choice = st.selectbox("Area", [*engine.CITIES, SEARCH, COORDS], label_visibility="collapsed")
    label, bounds, note = area_choice, None, None
    if area_choice in engine.CITIES:
        bounds = engine.CITIES[area_choice]
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
    if not st.toggle("Choose both periods myself"):
        after_start = st.date_input(
            "Look for changes that happened before", dt.date(2022, 4, 1), format="DD/MM/YYYY",
            help="The 'after' images start on this date.")
        weeks = st.select_slider("Length of each period", [4, 6, 8, 12, 16], value=8,
                                 format_func=lambda w: f"{w} weeks")
        before, after = inputs.periods_for_event(after_start, weeks)
        st.caption(f"**Before:** {fmt_period(before)}  \n**After:** {fmt_period(after)}  \n"
                   "Same season a year apart, so leaves, snow and crops don't count as change.")
    else:
        before = st.date_input("Before period", (dt.date(2021, 4, 1), dt.date(2021, 5, 31)),
                               format="DD/MM/YYYY")
        after = st.date_input("After period", (dt.date(2022, 4, 1), dt.date(2022, 5, 31)),
                              format="DD/MM/YYYY")

    max_images = st.select_slider(
        "Images per period", [1, 2, 3, 4, 6, 8, 12], value=4,
        help="More images find smaller changes but take longer to download. "
             "1 is the tutorial's single before/after pair.")

    with st.expander("Advanced settings"):
        orbit_pass = st.selectbox("Orbit direction", ["Any", "ASCENDING", "DESCENDING"])
        orbit = st.number_input("Relative orbit (0 = best available)", 0, 175, 0)
        auto_enl = st.checkbox("Estimate speckle level from the data", True,
                               help="Equivalent number of looks (ENL). Nominal Sentinel-1 value is 4.4.")
        enl = None if auto_enl else st.number_input("Equivalent number of looks", 1.0, 50.0, 4.4)
        bands = st.multiselect("Polarisations", ["VV", "VH"], ["VV", "VH"])

    # Check the dates and look up the images before anything is downloaded.
    problem, params, plan, cached = None, None, None, False
    if len(before) != 2 or len(after) != 2:
        problem = "Pick a start and an end date for both periods."
    elif before[1] >= after[0]:
        problem = "The before period must end before the after period starts."
    elif after[0] > dt.date.today():
        problem = "The after period starts in the future, so there are no images yet."
    elif not bands:
        problem = "Choose at least one polarisation."
    elif bounds:
        one_day = dt.timedelta(days=1)
        params = engine.Params(
            label=label, bounds=tuple(float(b) for b in bounds),
            before=(str(before[0]), str(before[1] + one_day)),  # end exclusive
            after=(str(after[0]), str(after[1] + one_day)),
            orbit_pass=None if orbit_pass == "Any" else orbit_pass,
            orbit=int(orbit) or None, max_images=int(max_images),
            enl=enl, bands=tuple(b for b in ("VV", "VH") if b in bands),
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
        st.info(f"Found **{len(plan.before)} before** and **{len(plan.after)} after** images "
                f"(orbit {plan.orbit}, {plan.orbit_state}).  \n"
                f"Download: about **{size(plan.download_mb)}**.")
        if plan.download_mb > BIG_DOWNLOAD_MB:
            st.warning("That's a big download. A smaller area or fewer images per period is faster.")

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
        save_config(last_result=params.cache_key())
        st.rerun()  # redraw the sidebar, which now knows the result is saved

# Reopen the last result without recomputing.
if "result" not in ss and cfg.get("last_result"):
    path = engine.CACHE_DIR / f"{cfg['last_result']}.npz"
    if path.exists():
        try:
            ss.result = engine.Result.load(path)
        except Exception:
            pass

# --- Main area ------------------------------------------------------------
result = ss.get("result")
if result is None:
    st.header("Radar change map")
    st.markdown(
        "Compares Sentinel-1 radar images from before and after a date, and marks every 10 m "
        "spot whose radar signal changed more than noise can explain: demolished or new "
        "buildings, cleared land, flooding and so on. Radar sees through clouds and at night.\n\n"
        "1. Choose an area\n2. Choose a date\n3. Press **Find changes**\n\n"
        "Images come free from Microsoft Planetary Computer; no account is needed. "
        "Results are saved on this computer and reopen instantly.")
    if bounds:
        st.caption(f"Area to analyse: {label}")
        st.iframe(area_preview(bounds), height=420)
else:
    p = result.params
    st.header(f"{p.label}: radar change")
    alpha = st.select_slider(
        "Sensitivity: how often an unchanged spot may be flagged by chance (α)",
        options=ALPHAS, value=0.01, format_func=lambda a: f"{a:g}",
        help="Smaller values show only the clearest changes.",
    )
    s = render.summarize(result, alpha)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Radar signal decreased", km2(s.decrease_km2),
              help="Often demolished or damaged buildings, cleared land, flooding.")
    c2.metric("Radar signal increased", km2(s.increase_km2),
              help="Often new structures, rubble, debris, vehicles, vegetation.")
    c3.metric("Expected by chance", km2(s.expected_km2),
              help=f"α × area analysed ({s.analysed_km2:,.0f} km²). Flagged area well above "
                   "this is real change.")
    if s.null_rate is not None:
        c4.metric("No-change check", f"{s.null_rate:.2%} flagged",
                  help="The same test comparing the before period with itself, where nothing "
                       f"should change. Close to α ({alpha:.2%}) means the statistics can be trusted.")

    html = map_html(p.cache_key(), alpha, result)
    st.iframe(html, height=680)
    st.caption("Red: radar signal decreased · Cyan: increased · Use the layer menu (top right) "
               "to see the before/after radar images. A flagged spot means something changed, not "
               "necessarily damage; look for clusters rather than scattered single spots.")

    c1, _ = st.columns([1, 3])
    c1.download_button("Download map (HTML)", html,
                       file_name=f"{p.label.lower().replace(' ', '_')}_change_{alpha:g}.html",
                       mime="text/html", use_container_width=True)

    with st.expander("Details and method"):
        st.markdown(f"""
- **Images:** Sentinel-1 (radiometrically terrain-corrected, from Microsoft Planetary
  Computer), relative orbit **{result.orbit}**, {len(result.before_days)} before
  acquisitions ({', '.join(result.before_days)}) and {len(result.after_days)} after
  ({', '.join(result.after_days)}).
- **Speckle level:** ENL = **{result.enl:.2f}**
  ({'measured from consecutive before images' if result.enl_estimated else 'set manually / nominal'};
  Sentinel-1 nominal 4.4).
- **Test:** likelihood-ratio test for equal mean backscatter per pixel
  ({' + '.join(p.bands)}), with Bartlett correction, χ² with {len(p.bands)} dof.
  Threshold at α = {alpha:g}: **{s.threshold:.2f}**.
- **Resolution:** computed at 10 m for every pixel ({result.grid.width} × {result.grid.height}).
""")
        if result.orbits:
            st.caption("Orbits considered")
            st.dataframe(
                [{"Orbit": o["orbit"], "Area covered": f"{o['coverage']:.0%}",
                  "Before images": o["before"], "After images": o["after"]} for o in result.orbits],
                hide_index=True,
            )
