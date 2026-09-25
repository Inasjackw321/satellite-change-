"""Satellite Change Map: Streamlit app. Start it with launch.py or the
"Start Change Map" launcher for your system."""

import datetime as dt
import json
import os
import time
from pathlib import Path

import folium
import streamlit as st

from satchange import account, engine, inputs, render

ROOT = Path(__file__).resolve().parent
CONFIG = Path(os.environ.get("SATCHANGE_CONFIG", ROOT / "config.json"))
ALPHAS = [0.1, 0.05, 0.01, 0.001, 1e-4, 1e-5, 1e-6]
SEARCH, COORDS = "Search for a place…", "Enter coordinates…"
SIGN_IN_TIMEOUT = 300  # seconds

st.set_page_config(page_title="Satellite Change Map", page_icon="🛰️", layout="wide")
ss = st.session_state


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


def km2(x):
    return f"{x:,.0f} km²" if x >= 100 else f"{x:.1f} km²" if x >= 10 else f"{x:.2f} km²"


def fmt_period(period):
    start, end = period
    return f"{start:%d %b %Y} – {end:%d %b %Y}"


@st.cache_data(show_spinner="Searching…", ttl=3600)
def search_places(query):
    return inputs.search_places(query)


@st.cache_data(show_spinner="Drawing map…", max_entries=8)
def map_html(key, alpha, layer_urls, _result):
    return render.build_map(_result, alpha, layer_urls).get_root().render()


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


def forget_session(*keys):
    for key in keys:
        ss.pop(key, None)


cfg = load_config()
signed_in = account.is_signed_in()

# --- Sidebar -------------------------------------------------------------
with st.sidebar:
    st.title("🛰️ Satellite change")

    # 1. Account
    st.subheader("1 · Google account")
    if signed_in:
        c1, c2 = st.columns([3, 2], vertical_alignment="center")
        c1.markdown("✅ Signed in to Earth Engine")
        if c2.button("Sign out", use_container_width=True):
            ss.confirm_sign_out = True
        if ss.get("confirm_sign_out"):
            st.warning("Sign out of Google Earth Engine on this computer? Your saved maps are kept.")
            c1, c2 = st.columns(2)
            if c1.button("Yes, sign out", type="primary", use_container_width=True):
                account.sign_out()
                forget_session("confirm_sign_out", "projects", "layers")
                st.toast("Signed out")
                st.rerun()
            if c2.button("Cancel", use_container_width=True):
                forget_session("confirm_sign_out")
                st.rerun()
    elif ss.get("signing_in"):
        st.caption("Finishing sign-in in your browser…")
    else:
        st.caption("Free for noncommercial use. Opens Google's sign-in page in your browser.")
        if st.button("Sign in with Google", type="primary", use_container_width=True):
            account.start_sign_in()
            ss.signing_in = time.time()
            st.rerun()

    # 2. Project
    st.subheader("2 · Cloud project")
    if signed_in and "projects" not in ss:
        with st.spinner("Looking up your projects…"):
            ss.projects = account.list_projects()
    projects = ss.get("projects") or []
    saved_project = cfg.get("project", "")
    if projects:
        options = [*projects, "Other…"]
        choice = st.selectbox("Project", options,
                              index=projects.index(saved_project) if saved_project in projects else 0,
                              label_visibility="collapsed")
        project = st.text_input("Project ID", saved_project) if choice == "Other…" else choice
    else:
        project = st.text_input("Project ID", saved_project, placeholder="e.g. my-ee-project",
                                label_visibility="collapsed")
    project = project.strip()
    st.caption(f"Don't have one? [Create a free Earth Engine project]({account.REGISTER_URL}).")

    # 3. Area
    st.subheader("3 · Area")
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
        east = c2.number_input("East (lon)", value=30.85, format="%.4f")
        south = c1.number_input("South (lat)", value=50.20, format="%.4f")
        north = c2.number_input("North (lat)", value=50.62, format="%.4f")
        if west < east and south < north:
            bounds, note = inputs.fit_area((west, south, east, north))
        else:
            st.error("West must be less than east, and south less than north.")
    if bounds:
        st.caption(f"About {inputs.area_km2(bounds):,.0f} km². {note or ''}")

    # 4. Dates
    st.subheader("4 · Dates")
    if not st.toggle("Choose both periods myself"):
        after_start = st.date_input(
            "Look for changes that happened before", dt.date(2022, 4, 1), format="DD/MM/YYYY",
            help="The 'after' images start on this date.")
        weeks = st.select_slider("Length of each period", [4, 6, 8, 12, 16], value=8,
                                 format_func=lambda w: f"{w} weeks",
                                 help="Longer periods include more images, which finds smaller changes.")
        before, after = inputs.periods_for_event(after_start, weeks)
        st.caption(f"**Before:** {fmt_period(before)}  \n**After:** {fmt_period(after)}  \n"
                   "Same season a year apart, so leaves, snow and crops don't count as change.")
    else:
        before = st.date_input("Before period", (dt.date(2021, 4, 1), dt.date(2021, 5, 31)),
                               format="DD/MM/YYYY")
        after = st.date_input("After period", (dt.date(2022, 4, 1), dt.date(2022, 5, 31)),
                              format="DD/MM/YYYY")

    with st.expander("Advanced settings"):
        orbit_pass = st.selectbox("Orbit direction", ["Any", "ASCENDING", "DESCENDING"])
        orbit = st.number_input("Relative orbit (0 = best available)", 0, 175, 0)
        max_images = st.number_input(
            "Images per period (0 = all)", 0, 60, 0,
            help="1 reproduces the tutorial's single before/after pair. More images "
                 "average out speckle and find much smaller changes.")
        auto_enl = st.checkbox("Estimate speckle level from the data", True,
                               help="Equivalent number of looks (ENL). Nominal Sentinel-1 value is 4.4.")
        enl = None if auto_enl else st.number_input("Equivalent number of looks", 1.0, 50.0, 4.4)
        bands = st.multiselect("Polarisations", ["VV", "VH"], ["VV", "VH"])

    missing = [why for ok, why in [(signed_in, "sign in"), (project, "choose a project"),
                                   (bounds, "choose an area"), (bands, "pick a polarisation")] if not ok]
    run = st.button("Find changes", type="primary", use_container_width=True, disabled=bool(missing),
                    help=f"First {', '.join(missing)}." if missing else None)

    saved = engine.saved_results()
    if saved:
        with st.expander(f"Saved maps ({len(saved)})"):
            paths = dict((desc, path) for path, desc in saved)
            pick = st.selectbox("Saved map", list(paths), label_visibility="collapsed")
            c1, c2 = st.columns(2)
            if c1.button("Open", use_container_width=True):
                ss.result, ss.layers = engine.Result.load(paths[pick]), None
                save_config(last_result=paths[pick].stem)
                st.rerun()
            if c2.button("Delete all", use_container_width=True):
                engine.delete_saved()
                forget_session("result", "layers")
                save_config(last_result=None)
                st.rerun()

# --- Run -------------------------------------------------------------------
if run:
    errors = []
    if len(before) != 2 or len(after) != 2:
        errors.append("Pick a start and an end date for both periods.")
    elif before[1] >= after[0]:
        errors.append("The before period must end before the after period starts.")
    elif after[0] > dt.date.today():
        errors.append("The after period starts in the future, so there are no images yet.")
    for e in errors:
        st.error(e)

    if not errors:
        save_config(project=project)
        one_day = dt.timedelta(days=1)
        params = engine.Params(
            label=label, bounds=tuple(float(b) for b in bounds),
            before=(str(before[0]), str(before[1] + one_day)),  # end exclusive
            after=(str(after[0]), str(after[1] + one_day)),
            orbit_pass=None if orbit_pass == "Any" else orbit_pass,
            orbit=int(orbit) or None, max_images=int(max_images),
            enl=enl, bands=tuple(b for b in ("VV", "VH") if b in bands),
        )
        bar = st.progress(0.0, "Connecting to Earth Engine…")
        try:
            account.connect(project)
            result = engine.run(params, progress=lambda f, msg: bar.progress(min(f, 1.0), msg))
        except Exception as e:
            bar.empty()
            st.error(account.explain_error(e, project))
        else:
            bar.empty()
            ss.result, ss.layers = result, None
            save_config(last_result=params.cache_key())

# Reopen the last result without recomputing.
if "result" not in ss and cfg.get("last_result"):
    path = engine.CACHE_DIR / f"{cfg['last_result']}.npz"
    if path.exists():
        ss.result, ss.layers = engine.Result.load(path), None

# --- Main area ------------------------------------------------------------
result = ss.get("result")
if ss.get("signing_in"):
    st.info("**A Google sign-in page opened in your browser.** Choose your account and allow "
            "access, then come back here. This page updates by itself.")
    if st.button("Cancel sign-in"):
        forget_session("signing_in")
        st.rerun()

if result is None:
    st.header("Radar change map")
    st.markdown(
        "Compares Sentinel-1 radar images from before and after a date, and marks every 10 m "
        "spot whose radar signal changed more than noise can explain: demolished or new "
        "buildings, cleared land, flooding and so on. Radar sees through clouds and at night.")
    steps = [(signed_in, "Sign in with your Google account"),
             (bool(project), "Choose a Cloud project registered for Earth Engine"),
             (bool(bounds), "Choose an area and a date"),
             (False, "Press **Find changes**. The first run takes a few minutes; results are saved.")]
    st.markdown("\n\n".join(f"{'✅' if ok else '⬜'} {i}. {text}" for i, (ok, text) in enumerate(steps, 1)))
    if bounds:
        st.caption(f"Area to analyse: {label}")
        st.iframe(area_preview(bounds), height=420)
else:
    p = result.params
    if ss.get("layers") is None:
        # Before/after radar tiles are optional; they need a live connection.
        try:
            account.connect(project)
            ss.layers = engine.display_layers(result)
        except Exception:
            ss.layers = {}

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

    layers = ss.layers or {}
    html = map_html((p.cache_key(), tuple(layers)), alpha, layers, result)
    st.iframe(html, height=680)
    st.caption("Red: radar signal decreased · Cyan: increased · Use the layer menu (top right) "
               "for before/after radar images. A flagged spot means something changed, not "
               "necessarily damage; look for clusters rather than scattered single spots.")

    c1, c2 = st.columns([1, 3])
    c1.download_button("Download map (HTML)", html,
                       file_name=f"{p.label.lower().replace(' ', '_')}_change_{alpha:g}.html",
                       mime="text/html", use_container_width=True)

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
""")
        if result.orbits:
            st.caption("Orbits considered")
            st.dataframe(
                [{"Orbit": o["orbit"], "Area covered": f"{o['coverage']:.0%}",
                  "Before images": o["before"], "After images": o["after"]} for o in result.orbits],
                hide_index=True,
            )

# Wait for the browser sign-in last, so the rest of the page is already drawn.
if ss.get("signing_in"):
    status = st.empty()
    deadline = ss.signing_in + SIGN_IN_TIMEOUT
    while not account.is_signed_in() and time.time() < deadline:
        status.caption(f"Waiting for sign-in… ({int(deadline - time.time())} s)")
        time.sleep(1)
    forget_session("signing_in", "projects")
    if account.is_signed_in():
        st.toast("Signed in")
    else:
        st.session_state.sign_in_failed = True
    st.rerun()
if ss.pop("sign_in_failed", False):
    st.error("Sign-in didn't finish. Press **Sign in with Google** to try again.")
