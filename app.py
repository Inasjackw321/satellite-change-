"""Satellite Change Map: Streamlit app. Start it with launch.py or the
"Start Change Map" launcher for your system."""

import datetime as dt
import json
from pathlib import Path

import streamlit as st

from satchange import engine, render

ROOT = Path(__file__).resolve().parent
CONFIG = ROOT / "config.json"
ALPHAS = [0.1, 0.05, 0.01, 0.001, 1e-4, 1e-5, 1e-6]
MAX_PIXELS = 80e6  # ~8,000 km2 at 10 m
CUSTOM = "Custom area…"

st.set_page_config(page_title="Satellite Change Map", page_icon="🛰️", layout="wide")


def load_config():
    try:
        return json.loads(CONFIG.read_text())
    except (OSError, ValueError):
        return {}


def save_config(**updates):
    cfg = load_config() | updates
    try:
        CONFIG.write_text(json.dumps(cfg, indent=2))
    except OSError:
        pass


def try_connect(project):
    """Connect quietly; returns an error message or None."""
    try:
        engine.connect(project)
        return None
    except engine.NotSignedIn:
        return "Not signed in to Google Earth Engine."
    except Exception as e:  # e.g. project not registered for Earth Engine
        return str(e)


def km2(x):
    return f"{x:,.0f} km²" if x >= 100 else f"{x:.1f} km²" if x >= 10 else f"{x:.2f} km²"


@st.cache_data(show_spinner="Drawing map…", max_entries=8)
def map_html(key, alpha, layer_urls, _result):
    return render.build_map(_result, alpha, layer_urls).get_root().render()


cfg = load_config()

# --- Sidebar: inputs -------------------------------------------------------
with st.sidebar:
    st.title("🛰️ Satellite change")
    st.caption("Sentinel-1 radar change detection with Google Earth Engine")

    project = st.text_input(
        "Google Cloud project", value=cfg.get("project", ""),
        help="A Cloud project registered for Earth Engine (free for noncommercial use): "
             "https://code.earthengine.google.com/register",
    )
    area = st.selectbox("Area", [*engine.CITIES, CUSTOM])
    if area == CUSTOM:
        label = st.text_input("Name", "Custom area")
        c1, c2 = st.columns(2)
        west = c1.number_input("West (lon)", value=30.15, format="%.4f")
        east = c2.number_input("East (lon)", value=30.85, format="%.4f")
        south = c1.number_input("South (lat)", value=50.20, format="%.4f")
        north = c2.number_input("North (lat)", value=50.62, format="%.4f")
        bounds = (west, south, east, north)
    else:
        label, bounds = area, engine.CITIES[area]

    before = st.date_input("Before period", (dt.date(2021, 4, 1), dt.date(2021, 5, 31)),
                           help="Same season as the after period, so seasonal changes cancel out.")
    after = st.date_input("After period", (dt.date(2022, 4, 1), dt.date(2022, 5, 31)))

    with st.expander("Advanced"):
        orbit_pass = st.selectbox("Orbit direction", ["Any", "ASCENDING", "DESCENDING"])
        orbit = st.number_input("Relative orbit (0 = best available)", 0, 175, 0)
        max_images = st.number_input(
            "Images per period (0 = all)", 0, 60, 0,
            help="1 reproduces the tutorial's single before/after pair. More images "
                 "average out speckle and detect much smaller changes.")
        auto_enl = st.checkbox("Estimate speckle level from the data", True,
                               help="Equivalent number of looks (ENL). Nominal Sentinel-1 value is 4.4.")
        enl = None if auto_enl else st.number_input("Equivalent number of looks", 1.0, 50.0, 4.4)
        bands = st.multiselect("Polarisations", ["VV", "VH"], ["VV", "VH"])

    run = st.button("Run analysis", type="primary", use_container_width=True)

# --- Sign-in ---------------------------------------------------------------
if not engine.has_credentials():
    st.info("**First time?** Sign in to Google Earth Engine. A browser tab will open; "
            "come back here once it says you're signed in.")
    if st.button("Sign in with Google"):
        with st.spinner("Waiting for you to finish signing in in the browser…"):
            engine.sign_in()
        st.rerun()

# --- Run -------------------------------------------------------------------
if run:
    errors = []
    if len(before) != 2 or len(after) != 2:
        errors.append("Pick a start and an end date for both periods.")
    elif before[1] >= after[0]:
        errors.append("The before period must end before the after period starts.")
    if not (bounds[0] < bounds[2] and bounds[1] < bounds[3]):
        errors.append("West must be less than east and south less than north.")
    elif (g := engine.Grid.for_bounds(bounds)).width * g.height > MAX_PIXELS:
        errors.append(f"Area too large for a 10 m analysis ({g.width * g.height / 1e6:.0f} M pixels, "
                      f"max {MAX_PIXELS / 1e6:.0f} M). Choose a smaller area.")
    if not bands:
        errors.append("Choose at least one polarisation.")
    if not project.strip():
        errors.append("Enter your Google Cloud project.")
    for e in errors:
        st.error(e)

    if not errors:
        save_config(project=project.strip())
        one_day = dt.timedelta(days=1)
        params = engine.Params(
            label=label, bounds=tuple(float(b) for b in bounds),
            before=(str(before[0]), str(before[1] + one_day)),  # end exclusive
            after=(str(after[0]), str(after[1] + one_day)),
            orbit_pass=None if orbit_pass == "Any" else orbit_pass,
            orbit=int(orbit) or None, max_images=int(max_images),
            enl=enl, bands=tuple(b for b in ("VV", "VH") if b in bands),
        )
        error = try_connect(project.strip())
        if error:
            st.error(f"Could not connect to Earth Engine: {error}")
        else:
            bar = st.progress(0.0, "Starting…")
            try:
                result = engine.run(params, progress=lambda f, msg: bar.progress(min(f, 1.0), msg))
            except Exception as e:
                bar.empty()
                st.error(f"Analysis failed: {e}")
            else:
                bar.empty()
                st.session_state.result = result
                st.session_state.layers = None
                save_config(last_result=params.cache_key())

# Reopen the last result without recomputing.
if "result" not in st.session_state and cfg.get("last_result"):
    path = engine.CACHE_DIR / f"{cfg['last_result']}.npz"
    if path.exists():
        st.session_state.result = engine.Result.load(path)
        st.session_state.layers = None

result = st.session_state.get("result")
if result is None:
    st.header("Radar change map")
    st.markdown(
        "Compares Sentinel-1 radar images from a **before** and an **after** period and marks "
        "every 10 m pixel whose radar backscatter changed more than speckle noise can explain.\n\n"
        "1. Enter your Google Cloud project (registered for Earth Engine) in the sidebar.\n"
        "2. Pick an area and the two periods; the defaults compare spring 2021 with spring 2022 in Kyiv.\n"
        "3. Press **Run analysis**. The first run takes a few minutes; results are saved and reopen instantly."
    )
    st.stop()

# --- Result ------------------------------------------------------------------
p = result.params
if st.session_state.get("layers") is None:
    # Before/after backscatter tiles are optional; they need a live connection.
    ok = project.strip() and engine.has_credentials() and try_connect(project.strip()) is None
    try:
        st.session_state.layers = engine.display_layers(result) if ok else {}
    except Exception:
        st.session_state.layers = {}

st.header(f"{p.label}: radar change")
alpha = st.select_slider(
    "Significance level α: the share of *unchanged* pixels you accept being flagged by chance",
    options=ALPHAS, value=0.01, format_func=lambda a: f"{a:g}",
)
s = render.summarize(result, alpha)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Backscatter decrease", km2(s.decrease_km2),
          help="Often demolished or damaged buildings, cleared land, flooding.")
c2.metric("Backscatter increase", km2(s.increase_km2),
          help="Often new structures, rubble, debris, vehicles, vegetation.")
c3.metric("Expected by chance", km2(s.expected_km2),
          help=f"α × area analysed ({s.analysed_km2:,.0f} km²). Flagged area well above "
               "this is real change.")
if s.null_rate is not None:
    c4.metric("No-change check", f"{s.null_rate:.2%} flagged",
              help="The same test comparing the before period with itself, where nothing "
                   f"should change. Close to α ({alpha:.2%}) means the statistics are "
                   "well calibrated.")

layers = st.session_state.layers or {}
html = map_html((p.cache_key(), tuple(layers)), alpha, layers, result)
st.iframe(html, height=680)

st.download_button("Download this map (HTML)", html,
                   file_name=f"{p.label.lower().replace(' ', '_')}_change_{alpha:g}.html",
                   mime="text/html")

with st.expander("Details and method"):
    st.markdown(f"""
- **Images:** Sentinel-1 relative orbit **{result.orbit}**, {len(result.before_days)} before
  acquisitions ({', '.join(result.before_days)}) and {len(result.after_days)} after
  ({', '.join(result.after_days)}).
- **Speckle level:** ENL = **{result.enl:.2f}**
  ({'estimated from consecutive before images' if result.enl_estimated else 'set manually / nominal'};
  Sentinel-1 nominal 4.4).
- **Test:** likelihood-ratio test for equal mean backscatter per pixel
  ({' + '.join(p.bands)}), with Bartlett correction, χ² with {len(p.bands)} dof.
  Threshold at α = {alpha:g}: **{s.threshold:.2f}**.
- **Resolution:** computed at 10 m for every pixel ({result.grid.width} × {result.grid.height}).
- A flagged pixel means the radar signal changed, not necessarily damage. Look for
  clusters; single scattered pixels are mostly the expected chance hits.
""")
    if result.orbits:
        st.caption("Orbits considered")
        st.dataframe(
            [{"Orbit": o["orbit"], "Area covered": f"{o['coverage']:.0%}",
              "Before images": o["before"], "After images": o["after"]} for o in result.orbits],
            hide_index=True,
        )
