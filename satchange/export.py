"""A picture of a result to save and share: map, spot numbers, legend and
watermark."""

import io
import math
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import requests
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage

from . import render
from .engine import DISPLAY_FACTOR, EARTH_RADIUS

ESRI_TILES = "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
ESRI_CREDIT = "Satellite photo: Esri, Maxar, Earthstar Geographics"
SENTINEL_CREDIT = "Contains modified Copernicus Sentinel data (Microsoft Planetary Computer)"
WORLD = 2 * math.pi * EARTH_RADIUS  # width of the Web Mercator world in metres
MAX_TILES = 144
LONG_SIDE = (1400, 2400)  # output map size range in pixels
MAX_LABELS = 150
DECREASE, INCREASE = render._hex_rgb(render.DECREASE), render._hex_rgb(render.INCREASE)


# Fonts with arrows, "≥" and "²", tried in order (Windows, macOS, Linux).
FONTS = {
    False: ["arial.ttf", "Arial.ttf", "/System/Library/Fonts/Supplemental/Arial.ttf",
            "/Library/Fonts/Arial.ttf", "DejaVuSans.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "LiberationSans-Regular.ttf"],
    True: ["arialbd.ttf", "Arial Bold.ttf", "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
           "/Library/Fonts/Arial Bold.ttf", "DejaVuSans-Bold.ttf",
           "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", "LiberationSans-Bold.ttf"],
}
# Pillow's built-in font lacks some symbols; used only if no system font is found.
ASCII = str.maketrans({"→": "->", "≥": ">=", "–": "-", "²": "2", "·": "|"})
_font_cache = {}


def _font(size, bold=False):
    size = max(8, int(size))
    if (size, bold) not in _font_cache:
        for name in FONTS[bold]:
            try:
                _font_cache[size, bold] = ImageFont.truetype(name, size)
                break
            except OSError:
                continue
        else:
            _font_cache[size, bold] = ImageFont.load_default(size=size)
    return _font_cache[size, bold]


def _text(font, text):
    """Text safe to draw with ``font`` (Pillow's built-in font is "Aileron")."""
    return text.translate(ASCII) if font.getname()[0] == "Aileron" else text


def map_size(grid):
    """Output map size (width, height) and the scale from grid pixels to it."""
    long_px = max(grid.width, grid.height)
    factor = min(max(long_px, LONG_SIDE[0]), LONG_SIDE[1]) / long_px
    return (round(grid.width * factor), round(grid.height * factor)), factor


# --- Backgrounds ---------------------------------------------------------------

def radar_background(result, size):
    """The after radar image (or before, if missing) in grey."""
    db = result.after_db if result.after_db is not None else result.before_db
    if db is None:
        return Image.new("RGB", size, (70, 70, 70))
    f = DISPLAY_FACTOR
    box = (0, 0, result.grid.width / f, result.grid.height / f)
    return Image.fromarray(db, "L").resize(size, Image.BILINEAR, box=box).convert("RGB")


def fetch_tile(session, z, x, y):
    r = session.get(ESRI_TILES.format(z=z, x=x, y=y), timeout=20,
                    headers={"User-Agent": "satellite-change-map/1.0"})
    r.raise_for_status()
    return Image.open(io.BytesIO(r.content)).convert("RGB")


def satellite_background(grid, size, fetch=fetch_tile):
    """Esri World Imagery for the grid, stitched from Web Mercator tiles (the
    grid is Web Mercator too, so they line up exactly)."""
    out_res = grid.scale * grid.width / size[0]  # metres per output pixel
    z = min(19, max(0, math.ceil(math.log2(WORLD / 256 / out_res))))
    x0, x1 = grid.x0, grid.x0 + grid.width * grid.scale
    y1, y0 = grid.y0, grid.y0 - grid.height * grid.scale
    while True:
        res = WORLD / 256 / 2 ** z
        px = lambda x: (x + WORLD / 2) / res
        py = lambda y: (WORLD / 2 - y) / res
        tx0, tx1 = int(px(x0) // 256), int((px(x1) - 1e-6) // 256)
        ty0, ty1 = int(py(y1) // 256), int((py(y0) - 1e-6) // 256)
        if (tx1 - tx0 + 1) * (ty1 - ty0 + 1) <= MAX_TILES or z == 0:
            break
        z -= 1
    mosaic = Image.new("RGB", ((tx1 - tx0 + 1) * 256, (ty1 - ty0 + 1) * 256))
    session = requests.Session()
    tiles = [(x, y) for y in range(ty0, ty1 + 1) for x in range(tx0, tx1 + 1)]
    with ThreadPoolExecutor(max_workers=8) as pool:
        for (x, y), tile in zip(tiles, pool.map(lambda t: fetch(session, z, *t), tiles)):
            mosaic.paste(tile, ((x - tx0) * 256, (y - ty0) * 256))
    box = (px(x0) - tx0 * 256, py(y1) - ty0 * 256, px(x1) - tx0 * 256, py(y0) - ty0 * 256)
    return mosaic.resize(size, Image.LANCZOS, box=box)


# --- Drawing -------------------------------------------------------------------

def _resize_mask(mask, size):
    """Any change inside an output pixel marks it, so small spots survive
    shrinking."""
    img = Image.fromarray(mask.astype(np.uint8) * 255, "L").resize(size, Image.BOX)
    return np.asarray(img) > 0


def draw_changes(base, found, size):
    out = np.asarray(base, dtype=np.float32).copy()
    for mask, color in ((found.decreased, DECREASE), (found.increased, INCREASE)):
        m = _resize_mask(mask, size)
        edge = m & ~ndimage.binary_erosion(m)
        out[m] = out[m] * 0.25 + np.array(color) * 0.75
        out[edge] = np.array(color) * 0.6  # a darker rim keeps small spots visible
    return Image.fromarray(out.clip(0, 255).astype(np.uint8))


def _badge(draw, x, y, text, color, font):
    l, t, r, b = draw.textbbox((0, 0), text, font=font)
    w, h = max(r - l, b - t) + 10, (b - t) + 8
    draw.rounded_rectangle((x, y - h / 2, x + w, y + h / 2), radius=h / 2, fill="white",
                           outline=color, width=max(2, int(h / 8)))
    draw.text((x + w / 2, y), text, fill=(17, 17, 17), font=font, anchor="mm")


def draw_numbers(img, spots, factor, max_labels=MAX_LABELS):
    """Number next to each of the largest spots; skipped where it would
    overlap another."""
    draw = ImageDraw.Draw(img)
    font = _font(img.width / 45, bold=True)
    placed = []
    for spot in spots[:max_labels]:
        x, y = (spot.col + 0.5) * factor + 6, (spot.row + 0.5) * factor
        if any(abs(x - px) < 2.2 * font.size and abs(y - py) < 1.6 * font.size for px, py in placed):
            continue
        placed.append((x, y))
        _badge(draw, x, y, str(spot.times), DECREASE if spot.change_db < 0 else INCREASE, font)


def draw_north_arrow(img):
    draw = ImageDraw.Draw(img)
    s = img.width / 60
    x, y = img.width - 2 * s, 1.6 * s
    draw.polygon([(x, y - s), (x - 0.6 * s, y + 0.6 * s), (x, y + 0.2 * s), (x + 0.6 * s, y + 0.6 * s)],
                 fill="white", outline="black")
    draw.text((x, y + 1.3 * s), "N", fill="white", font=_font(s, bold=True), anchor="mm",
              stroke_width=2, stroke_fill="black")


def draw_watermark(img, text):
    if not text.strip():
        return img
    layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    font = _font(img.width / 24, bold=True)
    pad = img.width / 60
    draw.text((img.width - pad, img.height - pad), text.strip(), font=font, anchor="rd",
              fill=(255, 255, 255, 200), stroke_width=max(2, int(img.width / 700)),
              stroke_fill=(0, 0, 0, 160))
    return Image.alpha_composite(img.convert("RGBA"), layer).convert("RGB")


def _nice_length(metres):
    for v in (10, 20, 50, 100, 200, 250, 500, 1000, 2000, 2500, 5000, 10000, 20000, 50000):
        if v >= metres:
            return v
    return 100000


def legend_panel(result, det_name, det, summary, width, metres_per_px, satellite):
    p = result.params
    k = width / 1400
    big, mid, small = _font(34 * k, bold=True), _font(20 * k), _font(16 * k)
    panel = Image.new("RGB", (width, int(250 * k)), "white")
    d = ImageDraw.Draw(panel)
    x, y = 30 * k, 22 * k
    d.text((x, y), p.label, fill="black", font=big)
    y += 48 * k
    d.text((x, y), _text(mid, f"Radar change {render._day(p.start)} → {render._day(p.end)}"),
           fill="black", font=mid)
    y += 32 * k
    d.text((x, y), _text(small, f"Before: {render._span(result.before_days)}   ·   "
                                f"After: {render._span(result.after_days)}"), fill=(70, 70, 70), font=small)
    y += 26 * k
    spots = f"{summary.spots:,} changed spot{'s' if summary.spots != 1 else ''}"
    d.text((x, y), _text(small, f"Detection: {det_name} (≥{det.min_db:g} dB, ≥{det.min_area_m2:,.0f} m², "
                                f"noise chance < 1 in {1 / det.alpha:,.0f})   ·   {spots}"),
           fill=(70, 70, 70), font=small)

    # Key
    x2, y2 = width * 0.60, 26 * k
    sq = 20 * k
    for color, text in ((DECREASE, "Radar signal decreased"), (INCREASE, "Radar signal increased")):
        d.rectangle((x2, y2, x2 + sq, y2 + sq), fill=color, outline=tuple(int(c * 0.6) for c in color))
        d.text((x2 + sq + 12 * k, y2 + sq / 2), text, fill="black", font=mid, anchor="lm")
        y2 += 34 * k
    _badge(d, x2 - 2 * k, y2 + sq / 2, "2", (60, 60, 60), _font(15 * k, bold=True))
    d.text((x2 + sq + 12 * k, y2 + sq / 2), "Times a change was seen between passes",
           fill="black", font=mid, anchor="lm")
    y2 += 44 * k

    # Scale bar
    length = _nice_length(width * 0.15 * metres_per_px)
    bar = length / metres_per_px
    d.rectangle((x2, y2, x2 + bar, y2 + 8 * k), fill="black")
    d.rectangle((x2 + bar / 2, y2 + 1, x2 + bar - 1, y2 + 8 * k - 1), fill="white")
    label = f"{length // 1000} km" if length >= 1000 else f"{length} m"
    d.text((x2 + bar + 10 * k, y2 + 4 * k), label, fill="black", font=small, anchor="lm")

    credits = SENTINEL_CREDIT + (f"   ·   {ESRI_CREDIT}" if satellite else "")
    tiny = _font(13 * k)
    d.text((30 * k, panel.height - 16 * k), _text(tiny, credits), fill=(110, 110, 110), font=tiny, anchor="ld")
    return panel


def export_image(result, det, det_name="Balanced", background="satellite", watermark="@Kaldockhi",
                 numbers=True, fmt="PNG", fetch=fetch_tile):
    """Returns (image bytes, note). The note says if the satellite photo
    couldn't be downloaded and the radar image was used instead."""
    grid = result.grid
    size, factor = map_size(grid)
    note = None
    if background == "satellite":
        try:
            base = satellite_background(grid, size, fetch)
        except Exception:
            base, note = radar_background(result, size), "Couldn't download the satellite photo, so the radar image is used."
    else:
        base = radar_background(result, size)
    satellite = background == "satellite" and note is None

    found = render.detect(result.signed, result.change_db, det, len(result.params.bands))
    img = draw_changes(base, found, size)
    if numbers:
        draw_numbers(img, render.find_spots(result, found), factor)
    draw_north_arrow(img)
    img = draw_watermark(img, watermark)

    lat = (grid.latlon_bounds()[0][0] + grid.latlon_bounds()[1][0]) / 2
    metres_per_px = grid.scale * math.cos(math.radians(lat)) / factor
    panel = legend_panel(result, det_name, det, render.summarize(result, det), size[0],
                         metres_per_px, satellite)
    out = Image.new("RGB", (size[0], size[1] + panel.height), "white")
    out.paste(img, (0, 0))
    out.paste(panel, (0, size[1]))

    buf = io.BytesIO()
    if fmt.upper() in ("JPG", "JPEG"):
        out.save(buf, format="JPEG", quality=90)
    else:
        out.save(buf, format="PNG", optimize=True)
    return buf.getvalue(), note


def spots_csv(result, det):
    """The detected spots as CSV text."""
    found = render.detect(result.signed, result.change_db, det, len(result.params.bands))
    lines = ["latitude,longitude,area_m2,change_db,direction,times_changed,when"]
    for s in render.find_spots(result, found):
        lines.append(f"{s.lat:.6f},{s.lon:.6f},{s.area_m2:.0f},{s.change_db:.2f},"
                     f"{'decrease' if s.change_db < 0 else 'increase'},{s.times},\"{'; '.join(s.when)}\"")
    return "\n".join(lines) + "\n"
