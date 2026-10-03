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


def _scale_bar(d, x, y, width, metres_per_px, k):
    length = _nice_length(width * 0.15 * metres_per_px)
    bar = length / metres_per_px
    d.rectangle((x, y, x + bar, y + 8 * k), fill="black")
    d.rectangle((x + bar / 2, y + 1, x + bar - 1, y + 8 * k - 1), fill="white")
    label = f"{length // 1000} km" if length >= 1000 else f"{length} m"
    d.text((x + bar + 10 * k, y + 4 * k), label, fill="black", font=_font(16 * k), anchor="lm")


def _credits(d, panel, satellite, k):
    credits = SENTINEL_CREDIT + (f"   ·   {ESRI_CREDIT}" if satellite else "")
    tiny = _font(13 * k)
    d.text((30 * k, panel.height - 16 * k), _text(tiny, credits), fill=(110, 110, 110), font=tiny, anchor="ld")


def _metres_per_px(grid, factor):
    lat = (grid.latlon_bounds()[0][0] + grid.latlon_bounds()[1][0]) / 2
    return grid.scale * math.cos(math.radians(lat)) / factor


def _background(result, grid, size, background, fetch, radar):
    """(image, note, satellite?)"""
    if background == "satellite":
        try:
            return satellite_background(grid, size, fetch), None, True
        except Exception:
            return radar(), "Couldn't download the satellite photo, so the radar image is used.", False
    return radar(), None, False


def _finish(img, panel, watermark, fmt):
    draw_north_arrow(img)
    img = draw_watermark(img, watermark)
    out = Image.new("RGB", (img.width, img.height + panel.height), "white")
    out.paste(img, (0, 0))
    out.paste(panel, (0, img.height))
    buf = io.BytesIO()
    if fmt.upper() in ("JPG", "JPEG"):
        out.save(buf, format="JPEG", quality=90)
    else:
        out.save(buf, format="PNG", optimize=True)
    return buf.getvalue()


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

    _scale_bar(d, x2, y2, width, metres_per_px, k)
    _credits(d, panel, satellite, k)
    return panel


def export_image(result, det, det_name="Balanced", background="satellite", watermark="@Kaldockhi",
                 numbers=True, fmt="PNG", fetch=fetch_tile):
    """Returns (image bytes, note). The note says if the satellite photo
    couldn't be downloaded and the radar image was used instead."""
    grid = result.grid
    size, factor = map_size(grid)
    base, note, satellite = _background(result, grid, size, background, fetch,
                                        lambda: radar_background(result, size))
    found = render.detect(result.signed, result.change_db, det, len(result.params.bands))
    img = draw_changes(base, found, size)
    if numbers:
        draw_numbers(img, render.find_spots(result, found), factor)
    panel = legend_panel(result, det_name, det, render.summarize(result, det), size[0],
                         _metres_per_px(grid, factor), satellite)
    return _finish(img, panel, watermark, fmt), note


def spots_csv(result, det):
    """The detected spots as CSV text."""
    found = render.detect(result.signed, result.change_db, det, len(result.params.bands))
    lines = ["latitude,longitude,area_m2,change_db,direction,times_changed,when"]
    for s in render.find_spots(result, found):
        lines.append(f"{s.lat:.6f},{s.lon:.6f},{s.area_m2:.0f},{s.change_db:.2f},"
                     f"{'decrease' if s.change_db < 0 else 'increase'},{s.times},\"{'; '.join(s.when)}\"")
    return "\n".join(lines) + "\n"


# --- Activity and ships ------------------------------------------------------------

def pass_background(result, size, pass_index=None):
    """Radar image of one pass, or the average of all passes."""
    imgs = result.pass_images
    if pass_index is not None:
        img = imgs[pass_index]
    else:
        valid = imgs > 0
        img = (imgs.sum(axis=0, dtype=np.float32) / np.maximum(valid.sum(axis=0), 1)).astype(np.uint8)
    f = 2  # pass images are at 20 m
    box = (0, 0, result.grid.width / f, result.grid.height / f)
    return Image.fromarray(img, "L").resize(size, Image.BILINEAR, box=box).convert("RGB")


def draw_activity(base, analysis, size, pass_index=None):
    out = np.asarray(base, dtype=np.float32).copy()
    if pass_index is None:
        cls = render.activity_class(analysis.count)
    else:
        cls = analysis.masks[pass_index].astype(np.uint8) * 3
    for k, color in enumerate(render.ACTIVITY_RAMP, 1):
        m = _resize_mask(cls == k, size)
        out[m] = out[m] * 0.15 + np.array(render._hex_rgb(color)) * 0.85
    any_ = _resize_mask(cls > 0, size)
    out[any_ & ~ndimage.binary_erosion(any_)] = 255  # light rim
    return Image.fromarray(out.clip(0, 255).astype(np.uint8))


def draw_ships(img, detections, factor):
    draw = ImageDraw.Draw(img)
    r = max(6, img.width / 150)
    color = render._hex_rgb(render.SHIP_COLOR)
    for d in detections:
        x, y = (d.col + 0.5) * factor, (d.row + 0.5) * factor
        draw.ellipse((x - r, y - r, x + r, y + r), fill=color, outline="white", width=max(2, int(r / 3)))


def draw_hotspot_labels(img, hotspots, factor, max_labels=MAX_LABELS):
    draw = ImageDraw.Draw(img)
    font = _font(img.width / 55, bold=True)
    placed = []
    for h in hotspots[:max_labels]:
        x, y = (h.col + 0.5) * factor + 8, (h.row + 0.5) * factor
        if any(abs(x - px) < 3 * font.size and abs(y - py) < 1.6 * font.size for px, py in placed):
            continue
        placed.append((x, y))
        _badge(draw, x, y, f"{h.seen}/{h.of}", render._hex_rgb(render.ACTIVITY_RAMP[2]), font)


def _timeline(d, x, y, w, h, result, analysis, k, highlight=None, noun="Objects"):
    """Objects per pass as small bars, first and last date under them."""
    counts = analysis.per_pass
    top = max(max(counts), 1)
    n = len(counts)
    gap = max(1, w / n * 0.25)
    bar_w = (w - gap * (n - 1)) / n
    color = render._hex_rgb(render.ACTIVITY_RAMP[1])
    for i, c in enumerate(counts):
        bx = x + i * (bar_w + gap)
        bh = h * c / top
        fill = render._hex_rgb(render.SHIP_COLOR) if i == highlight else color
        if bh:
            d.rectangle((bx, y + h - bh, bx + bar_w, y + h), fill=fill)
    d.line((x, y + h, x + w, y + h), fill=(170, 170, 170), width=1)
    small = _font(13 * k)
    d.text((x, y + h + 4 * k), _text(small, render._day(result.days[0])), fill=(90, 90, 90), font=small)
    d.text((x + w, y + h + 4 * k), _text(small, render._day(result.days[-1])), fill=(90, 90, 90),
           font=small, anchor="ra")
    d.text((x, y - 6 * k), _text(small, f"{noun} per pass (most: {max(counts)})"), fill=(90, 90, 90),
           font=small, anchor="ld")


def activity_legend(result, analysis, sens_name, width, metres_per_px, satellite, pass_index=None):
    p = result.params
    ships = p.target == "water"
    k = width / 1400
    big, mid, small = _font(34 * k, bold=True), _font(20 * k), _font(16 * k)
    panel = Image.new("RGB", (width, int(300 * k)), "white")
    d = ImageDraw.Draw(panel)
    x, y = 30 * k, 22 * k
    d.text((x, y), p.label, fill="black", font=big)
    y += 48 * k
    what = "Ships" if ships else "Activity"
    when = (render._day(result.days[pass_index]) if pass_index is not None
            else f"{render._day(result.days[0])} → {render._day(result.days[-1])}")
    d.text((x, y), _text(mid, f"{what} seen by radar, {when}"), fill="black", font=mid)
    y += 32 * k
    total = len(analysis.detections) if pass_index is None else analysis.per_pass[pass_index]
    noun = "ship detection" if ships else "object"
    line = f"{total:,} {noun}{'s' if total != 1 else ''}"
    if pass_index is None:
        line += f" in {len(result.days)} passes"
        if not ships:
            line += f"   ·   {len(analysis.hotspots):,} busy spots"
    d.text((x, y), _text(small, line), fill=(70, 70, 70), font=small)
    y += 26 * k
    d.text((x, y), _text(small, f"Detection: {sens_name}"), fill=(70, 70, 70), font=small)
    _timeline(d, x, y + 56 * k, width * 0.45, 56 * k, result, analysis, k, highlight=pass_index,
              noun="Ships" if ships else "Objects")

    x2, y2 = width * 0.56, 26 * k
    sq = 20 * k
    if ships or pass_index is not None:
        color = render._hex_rgb(render.SHIP_COLOR if ships else render.ACTIVITY_RAMP[2])
        d.ellipse((x2, y2, x2 + sq, y2 + sq), fill=color, outline="white")
        d.text((x2 + sq + 12 * k, y2 + sq / 2),
               "A ship in one pass" if ships else "Object present in this pass", fill="black",
               font=mid, anchor="lm")
        y2 += 40 * k
    else:
        d.text((x2, y2 + sq / 2), "Something there in:", fill="black", font=mid, anchor="lm")
        y2 += 32 * k
        for color, label in zip(render.ACTIVITY_RAMP, render.ACTIVITY_CLASSES):
            d.rectangle((x2, y2, x2 + sq, y2 + sq), fill=render._hex_rgb(color), outline="white")
            d.text((x2 + sq + 10 * k, y2 + sq / 2), _text(small, label), fill="black", font=small, anchor="lm")
            x2 += 140 * k
        x2 = width * 0.56
        y2 += 36 * k
        _badge(d, x2, y2 + sq / 2, "3/12", render._hex_rgb(render.ACTIVITY_RAMP[2]), _font(15 * k, bold=True))
        d.text((x2 + 62 * k, y2 + sq / 2), _text(mid, "= seen in 3 of 12 passes"), fill="black",
               font=mid, anchor="lm")
        y2 += 44 * k
    _scale_bar(d, x2, y2, width, metres_per_px, k)
    _credits(d, panel, satellite, k)
    return panel


def export_activity_image(result, analysis, sens_name="Balanced", pass_index=None, background="satellite",
                          watermark="@Kaldockhi", numbers=True, fmt="PNG", fetch=fetch_tile):
    grid = result.grid
    size, factor = map_size(grid)
    base, note, satellite = _background(result, grid, size, background, fetch,
                                        lambda: pass_background(result, size, pass_index))
    ships = result.params.target == "water"
    img = base if ships and pass_index is None else draw_activity(base, analysis, size, pass_index)
    if ships or pass_index is not None:
        draw_ships(img, [d for d in analysis.detections if pass_index is None or d.pass_index == pass_index],
                   factor)
    elif numbers:
        draw_hotspot_labels(img, analysis.hotspots, factor)
    panel = activity_legend(result, analysis, sens_name, size[0], _metres_per_px(grid, factor),
                            satellite, pass_index)
    return _finish(img, panel, watermark, fmt), note


def detections_csv(analysis):
    lines = ["date,latitude,longitude,length_m,area_m2,brightness_db,on_water"]
    for d in analysis.detections:
        lines.append(f"{d.day},{d.lat:.6f},{d.lon:.6f},{d.length_m:.0f},{d.area_m2:.0f},"
                     f"{d.brightness_db:.1f},{'yes' if d.on_water else 'no'}")
    return "\n".join(lines) + "\n"


def hotspots_csv(analysis):
    lines = ["latitude,longitude,passes_seen,passes_covered,typical_area_m2,dates"]
    for h in analysis.hotspots:
        lines.append(f"{h.lat:.6f},{h.lon:.6f},{h.seen},{h.of},{h.area_m2:.0f},\"{'; '.join(h.days)}\"")
    return "\n".join(lines) + "\n"
