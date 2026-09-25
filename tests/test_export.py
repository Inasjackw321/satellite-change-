import io
import math

import numpy as np
import pytest
from PIL import Image

from satchange import engine, export, render, source
from tests import conftest

PARAMS = engine.Params(label="Test", bounds=conftest.AOI, start="2026-07-04", end="2026-08-04")
BALANCED = render.PRESETS["Balanced"]


@pytest.fixture
def result(offline):
    return engine.run(PARAMS, signer=source.Signer())


def image(data):
    return np.asarray(Image.open(io.BytesIO(data)).convert("RGB")).astype(int)


def test_image_has_map_legend_and_watermark(result):
    data, note = export.export_image(result, BALANCED, background="radar")
    img = Image.open(io.BytesIO(data))
    (w, h), _ = export.map_size(result.grid)
    assert img.format == "PNG" and note is None
    assert img.width == w and img.height > h + 100  # legend panel below the map
    assert max(w, h) >= export.LONG_SIDE[0]

    plain, _ = export.export_image(result, BALANCED, background="radar", watermark="")
    a, b = image(data), image(plain)
    corner = (slice(h - h // 8, h), slice(w - w // 3, w))
    assert np.abs(a[corner] - b[corner]).mean() > 1  # the watermark is in the bottom-right corner
    assert np.abs(a[: h // 2] - b[: h // 2]).max() == 0  # and nowhere near the top


def test_numbers_are_drawn_next_to_spots(result):
    with_numbers, _ = export.export_image(result, BALANCED, background="radar", watermark="")
    without, _ = export.export_image(result, BALANCED, background="radar", watermark="", numbers=False)
    diff = np.abs(image(with_numbers) - image(without)).sum(axis=2) > 0
    rows, cols = np.nonzero(diff)
    _, factor = export.map_size(result.grid)
    spots = render.find_spots(result, render.detect(result.signed, result.change_db, BALANCED, 2))
    for spot in spots:  # a badge starts just right of each spot's centre
        x, y = (spot.col + 0.5) * factor, (spot.row + 0.5) * factor
        near = (abs(cols - x - 20) < 30) & (abs(rows - y) < 20)
        assert near.any()


def test_jpeg(result):
    data, _ = export.export_image(result, BALANCED, background="radar", fmt="JPEG")
    assert Image.open(io.BytesIO(data)).format == "JPEG"


def test_satellite_tiles_cover_the_area_and_line_up(result):
    requested = []

    def fake_fetch(session, z, x, y):
        requested.append((z, x, y))
        return Image.new("RGB", (256, 256), (10, 200, 30))

    size, _ = export.map_size(result.grid)
    background = export.satellite_background(result.grid, size, fetch=fake_fetch)
    assert background.size == size and background.getpixel((5, 5)) == (10, 200, 30)

    z = requested[0][0]
    xs, ys = {x for _, x, _ in requested}, {y for _, _, y in requested}

    def tile(lon, lat):  # standard slippy-map tile numbers
        n = 2 ** z
        return (int((lon + 180) / 360 * n),
                int((1 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2 * n))

    (south, west), (north, east) = result.grid.latlon_bounds()
    nw, se = tile(west + 1e-7, north - 1e-7), tile(east - 1e-7, south + 1e-7)
    assert (min(xs), min(ys)) == nw and (max(xs), max(ys)) == se
    # Tile pixels are no coarser than output pixels.
    out_res = result.grid.scale * result.grid.width / size[0]
    assert export.WORLD / 256 / 2 ** z <= out_res


def test_satellite_failure_falls_back_to_radar(result):
    def offline(session, z, x, y):
        raise OSError("no internet")

    data, note = export.export_image(result, BALANCED, background="satellite", fetch=offline)
    assert "radar image is used" in note and Image.open(io.BytesIO(data)).width > 0


def test_spot_list_csv(result):
    lines = export.spots_csv(result, BALANCED).strip().splitlines()
    assert lines[0].startswith("latitude,longitude,area_m2,change_db,direction,times_changed")
    assert len(lines) == 3
    assert ",decrease,1," in lines[1] and ",increase,3," in lines[2]
