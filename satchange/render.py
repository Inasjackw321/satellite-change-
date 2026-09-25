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

from .engine import NULL_BINS, NULL_MAX
from .stats import chi2_threshold, decode, exceedance

DECREASE = "#ff3b30"
INCREASE = "#00c8ff"


@dataclass
class Summary:
    alpha: float
    threshold: float
    analysed_km2: float
    increase_km2: float
    decrease_km2: float
    null_rate: float | None  # fraction flagged in the before-vs-before check

    @property
    def expected_km2(self):
        return self.alpha * self.analysed_km2


def _flags(result, alpha):
    threshold = chi2_threshold(alpha, dof=len(result.params.bands))
    stat, increased, valid = decode(result.signed)
    flagged = valid & (stat > threshold)
    return threshold, valid, flagged & increased, flagged & ~increased


def summarize(result, alpha):
    threshold, valid, inc, dec = _flags(result, alpha)
    row_area = result.grid.row_area_km2()
    km2 = lambda mask: float(mask.sum(axis=1) @ row_area)
    null_rate = None
    if result.null_hist is not None:
        null_rate = exceedance(result.null_hist, NULL_MAX / NULL_BINS, threshold)
    return Summary(alpha, threshold, km2(valid), km2(inc), km2(dec), null_rate)


def _span(days):
    """'1 Apr – 19 May 2021' for a list of ISO dates."""
    first, last = (dt.date.fromisoformat(d) for d in (days[0], days[-1]))
    if first == last:
        return f"{first.day} {first:%b %Y}"
    head = f"{first.day} {first:%b}" if first.year == last.year else f"{first.day} {first:%b %Y}"
    return f"{head} – {last.day} {last:%b %Y}"


def _hex_rgb(color):
    return tuple(int(color[i:i + 2], 16) for i in (1, 3, 5))


def overlay_png(result, alpha):
    """RGBA PNG (as a data URL) of the flagged pixels; the rest transparent."""
    _, _, inc, dec = _flags(result, alpha)
    rgba = np.zeros(inc.shape + (4,), dtype=np.uint8)
    rgba[dec] = _hex_rgb(DECREASE) + (255,)
    rgba[inc] = _hex_rgb(INCREASE) + (255,)
    buf = io.BytesIO()
    Image.fromarray(rgba, "RGBA").save(buf, format="PNG", compress_level=6)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def build_map(result, alpha, layers=None):
    p = result.params
    (south, west), (north, east) = result.grid.latlon_bounds()
    m = folium.Map(location=[(south + north) / 2, (west + east) / 2], zoom_start=11,
                   tiles=None, control_scale=True)
    folium.TileLayer(
        "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri World Imagery", name="Satellite photo (Esri)",
    ).add_to(m)
    folium.TileLayer("OpenStreetMap", name="Street map", show=False).add_to(m)
    for name, url in (layers or {}).items():
        folium.TileLayer(url, attr="Google Earth Engine, Copernicus Sentinel-1",
                         name=f"{name}: radar VV (dB)", overlay=True, show=False).add_to(m)

    folium.raster_layers.ImageOverlay(
        overlay_png(result, alpha), bounds=[[south, west], [north, east]],
        name=f"Significant change (α = {alpha:g})",
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
