"""Turn a downloaded change result into numbers and an interactive map."""

import base64
import datetime as dt
import html
import io
from dataclasses import dataclass

import folium
from branca.element import MacroElement
from jinja2 import Template
import numpy as np
from PIL import Image

from scipy import ndimage

from .engine import DISPLAY_FACTOR, GROUND_RES
from .stats import chi2_threshold, decode

DECREASE = "#ff3b30"
INCREASE = "#00c8ff"


@dataclass(frozen=True)
class Detection:
    """What counts as a change. A pixel must pass all three filters."""
    alpha: float = 1e-5  # statistical significance (chance of flagging an unchanged pixel)
    min_db: float = 3.0  # the radar signal must change by at least this much
    min_area_m2: float = 400  # and be part of a patch at least this big


PRESETS = {
    "Sensitive": Detection(alpha=1e-3, min_db=1.5, min_area_m2=100),
    "Balanced": Detection(),
    "Strict": Detection(alpha=1e-7, min_db=6.0, min_area_m2=2000),
}


@dataclass
class Summary:
    analysed_km2: float
    increase_km2: float
    decrease_km2: float
    spots: int
    noise_km2: float | None  # detected in the before-vs-before comparison, where nothing changed
    noise_spots: int | None


def detect(signed, change_db, det, dof):
    """(increased, decreased, number of spots) after all three filters."""
    stat, _, valid = decode(signed)
    db = change_db.astype(np.float32) / 100
    flag = valid & (stat > chi2_threshold(det.alpha, dof)) & (np.abs(db) >= det.min_db)
    labels, n = ndimage.label(flag, structure=np.ones((3, 3)))
    spots = 0
    if n:
        min_px = max(1, int(np.ceil(det.min_area_m2 / GROUND_RES ** 2)))
        keep = np.bincount(labels.ravel()) >= min_px
        keep[0] = False
        flag, spots = keep[labels], int(keep.sum())
    increased = db > 0
    return flag & increased, flag & ~increased, spots


def summarize(result, det):
    dof = len(result.params.bands)
    row_area = result.grid.row_area_km2()
    km2 = lambda mask: float(mask.sum(axis=1) @ row_area)
    inc, dec, spots = detect(result.signed, result.change_db, det, dof)
    valid = decode(result.signed)[2]
    noise_km2 = noise_spots = None
    if result.null_signed is not None:
        n_inc, n_dec, noise_spots = detect(result.null_signed, result.null_db, det, dof)
        noise_km2 = km2(n_inc | n_dec)
    return Summary(km2(valid), km2(inc), km2(dec), spots, noise_km2, noise_spots)


def _span(days):
    """'1 Apr – 19 May 2021' for a list of ISO dates."""
    first, last = (dt.date.fromisoformat(d) for d in (days[0], days[-1]))
    if first == last:
        return f"{first.day} {first:%b %Y}"
    head = f"{first.day} {first:%b}" if first.year == last.year else f"{first.day} {first:%b %Y}"
    return f"{head} – {last.day} {last:%b %Y}"


def _hex_rgb(color):
    return tuple(int(color[i:i + 2], 16) for i in (1, 3, 5))


def overlay_png(result, det):
    """RGBA PNG (as a data URL) of the detected changes; the rest transparent."""
    inc, dec, _ = detect(result.signed, result.change_db, det, len(result.params.bands))
    rgba = np.zeros(inc.shape + (4,), dtype=np.uint8)
    rgba[dec] = _hex_rgb(DECREASE) + (255,)
    rgba[inc] = _hex_rgb(INCREASE) + (255,)
    return _png(Image.fromarray(rgba, "RGBA"))


def _png(image):
    buf = io.BytesIO()
    image.save(buf, format="PNG", compress_level=6)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def backscatter_png(db):
    """Greyscale PNG of a stored uint8 backscatter layer; no data transparent."""
    return _png(Image.fromarray(np.dstack([db, db, db, np.where(db > 0, 255, 0).astype(np.uint8)]), "RGBA"))


def build_map(result, det):
    p = result.params
    (south, west), (north, east) = result.grid.latlon_bounds()
    m = folium.Map(location=[(south + north) / 2, (west + east) / 2], zoom_start=11,
                   tiles=None, control_scale=True)
    folium.TileLayer(
        "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri World Imagery", name="Satellite photo (Esri)",
    ).add_to(m)
    folium.TileLayer("OpenStreetMap", name="Street map", show=False).add_to(m)
    for name, db in (("Before", result.before_db), ("After", result.after_db)):
        if db is not None:
            f = DISPLAY_FACTOR
            folium.raster_layers.ImageOverlay(
                backscatter_png(db), bounds=result.grid.window_latlon(0, 0, db.shape[1] * f, db.shape[0] * f),
                name=f"{name}: radar image (VV)", show=False,
                attr="Contains modified Copernicus Sentinel data",
            ).add_to(m)

    folium.raster_layers.ImageOverlay(
        overlay_png(result, det), bounds=[[south, west], [north, east]],
        name="Detected changes",
        attr="Contains modified Copernicus Sentinel data (via Microsoft Planetary Computer)",
    ).add_to(m)
    w, s, e, n = p.bounds
    folium.Rectangle([[s, w], [n, e]], name="Area analysed", fill=False,
                     color="#ffffff", weight=1.5).add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)
    m.fit_bounds([[s, w], [n, e]])

    # Sharp 10 m pixels when zoomed in, smooth density when zoomed out.
    # A MacroElement child renders its script after the map is created.
    crisp = MacroElement()
    crisp._template = Template("""
        {% macro script(this, kwargs) %}
        (function(map) {
          function crisp() {
            document.querySelectorAll('.leaflet-image-layer').forEach(function(el) {
              el.style.imageRendering = map.getZoom() >= 14 ? 'pixelated' : 'auto';
            });
          }
          map.on('zoomend overlayadd', crisp);
          crisp();
        })({{ this._parent.get_name() }});
        {% endmacro %}
    """)
    m.add_child(crisp)
    m.get_root().html.add_child(folium.Element(f"""
    <div style="position:fixed;bottom:24px;left:12px;z-index:9999;max-width:300px;
                background:rgba(20,20,20,.85);color:#eee;padding:8px 10px;border-radius:6px;
                font:12px/1.5 system-ui,sans-serif">
      <b>{html.escape(p.label)}</b><br>
      Before: {_span(result.before_days)}<br>After: {_span(result.after_days)}<br>
      <span style="color:{DECREASE}">&#9632;</span> radar signal decreased &nbsp;
      <span style="color:{INCREASE}">&#9632;</span> increased
    </div>"""))
    return m
