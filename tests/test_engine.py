import math

import numpy as np
import pytest

from satchange import engine, render, stats
from tests import conftest

KYIV = engine.Params(label="Kyiv", bounds=engine.CITIES["Kyiv"],
                     before=("2021-04-01", "2021-06-01"), after=("2022-04-01", "2022-06-01"))


def test_grid_has_10m_ground_pixels_and_covers_bounds():
    g = engine.Grid.for_bounds(KYIV.bounds)
    (south, west), (north, east) = g.latlon_bounds()
    w, s, e, n = KYIV.bounds
    assert west == pytest.approx(w) and north == pytest.approx(n)
    assert e <= east < e + 0.001 and s - 0.001 < south <= s
    side = np.sqrt(g.row_area_km2() * 1e6)
    assert side.min() == pytest.approx(10, abs=0.06) and side.max() == pytest.approx(10, abs=0.06)
    # ~50 km x 47 km at 10 m
    assert 4900 < g.width < 5000 and 4600 < g.height < 4750


def _run(fake_ee, params=KYIV):
    g = engine.Grid.for_bounds(params.bounds)
    conftest.FIRST_TILE_X[0], conftest.FIRST_TILE_Y[0] = g.x0, g.y0
    return engine.run(params)


def test_run_end_to_end(fake_ee):
    result = _run(fake_ee)
    assert result.orbit == 36  # full coverage beats more images elsewhere
    assert result.before_days == ["2021-04-01", "2021-04-13", "2021-04-25", "2021-05-07", "2021-05-19"]
    assert result.enl_estimated and result.enl == pytest.approx(conftest.TRUE_ENL, rel=0.08)
    assert result.signed.shape == (result.grid.height, result.grid.width)
    assert result.null_hist is not None and result.null_hist.sum() == 200_000

    stat, increased, valid = stats.decode(result.signed)
    block = conftest.CHANGED_BLOCK
    assert (stat[block] > 90).all() and not increased[block].any() and valid[block].all()

    s = render.summarize(result, 0.01)
    # Background is chi2(2) noise, so chance flags are ~alpha of the area.
    assert s.increase_km2 / s.analysed_km2 == pytest.approx(0.005, rel=0.1)
    assert s.null_rate == pytest.approx(0.01, rel=0.1)

    html = render.build_map(result, 0.01, engine.display_layers(result)).get_root().render()
    assert "data:image/png;base64," in html and "Before: radar VV" in html


def test_every_tile_is_downloaded_once(fake_ee):
    result = _run(fake_ee)
    tiles = math.ceil(result.grid.width / engine.TILE) * math.ceil(result.grid.height / engine.TILE)
    assert fake_ee.calls.count("computePixels") == tiles
    # The fake marks each tile's last row no-data: tiles landed in place.
    tile_rows = math.ceil(result.grid.height / engine.TILE)
    assert (result.signed == stats.NODATA).sum() == tile_rows * result.grid.width
    for r in range(engine.TILE, result.grid.height, engine.TILE):
        assert (result.signed[r - 1] == stats.NODATA).all()
    assert (result.signed[-1] == stats.NODATA).all()


def test_second_run_is_served_from_cache(fake_ee):
    first = _run(fake_ee)
    n_calls = len(fake_ee.calls)
    second = _run(fake_ee)
    assert len(fake_ee.calls) == n_calls
    np.testing.assert_array_equal(first.signed, second.signed)
    assert second.params == first.params and second.grid == first.grid


def test_single_pair_mode_uses_nominal_enl_when_asked(fake_ee):
    params = engine.Params(**{**KYIV.__dict__, "max_images": 1, "enl": 4.4, "bands": ("VV",)})
    result = _run(fake_ee, params)
    assert result.before_days == ["2021-05-19"] and result.after_days == ["2022-04-01"]
    assert result.enl == 4.4 and not result.enl_estimated
    assert result.null_hist is None  # needs two before images
    assert not any("Image.sample" in c for c in fake_ee.calls)
