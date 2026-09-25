import math

import numpy as np
import pytest

from satchange import engine, render, source, stats
from tests import conftest

PARAMS = engine.Params(label="Test", bounds=conftest.AOI,
                       before=("2021-04-01", "2021-06-01"), after=("2022-04-01", "2022-06-01"))


def grid_index(grid, lon, lat):
    x, y = engine._merc(lon, lat)
    return int((grid.y0 - y) / grid.scale), int((x - grid.x0) / grid.scale)


def patch_mask(grid, shrink=2):
    """Output pixels inside the changed patch (a little inside its edge)."""
    w, s, e, n = conftest.PATCH
    r0, c0 = grid_index(grid, w, n)
    r1, c1 = grid_index(grid, e, s)
    mask = np.zeros((grid.height, grid.width), bool)
    mask[r0 + shrink:r1 - shrink, c0 + shrink:c1 - shrink] = True
    ring = np.zeros_like(mask)
    ring[r0 - 5:r1 + 5, c0 - 5:c1 + 5] = True
    return mask, ring


@pytest.fixture
def result(offline):
    return engine.run(PARAMS, signer=source.Signer())


def test_grid_has_10m_ground_pixels_and_covers_bounds():
    bounds = engine.CITIES["Kyiv – whole city (large download)"]
    g = engine.Grid.for_bounds(bounds)
    (south, west), (north, east) = g.latlon_bounds()
    w, s, e, n = bounds
    assert west == pytest.approx(w) and north == pytest.approx(n)
    assert e <= east < e + 0.001 and s - 0.001 < south <= s
    side = np.sqrt(g.row_area_km2() * 1e6)
    assert side.min() == pytest.approx(10, abs=0.06) and side.max() == pytest.approx(10, abs=0.06)


def test_plan_picks_full_coverage_orbit_and_estimates_download(offline):
    plan = engine.make_plan(PARAMS)
    assert plan.orbit == 36  # orbit 109 covers only a third of the area
    assert {o["orbit"]: round(o["coverage"], 1) for o in plan.orbits}[109] < 0.5
    assert list(plan.before) == conftest.BEFORE_DAYS and list(plan.after) == conftest.AFTER_DAYS
    assert len(plan.before[conftest.BEFORE_DAYS[0]]) == 2  # split pass
    grid = engine.Grid.for_bounds(PARAMS.bounds)
    assert plan.download_mb == pytest.approx(grid.pixels * 2 * 9 * engine.BYTES_PER_PIXEL / 1e6)


def test_detects_the_changed_patch_where_it_is(result):
    grid = result.grid
    stat, increased, valid = stats.decode(result.signed)
    assert valid.all()  # the split first pass left no holes
    inside, ring = patch_mask(grid)
    flagged = stat > stats.chi2_threshold(0.01, 2)
    assert flagged[inside].mean() > 0.97 and not increased[inside & flagged].any()
    # Outside the patch only chance hits remain: ~alpha of unchanged pixels.
    outside = ~ring
    assert flagged[outside].mean() == pytest.approx(0.01, rel=0.25)


def test_speckle_estimate_and_no_change_check(result):
    assert result.enl_estimated and result.enl == pytest.approx(conftest.TRUE_ENL, rel=0.08)
    s = render.summarize(result, 0.01)
    assert s.null_rate == pytest.approx(0.01, rel=0.25)
    patch_km2 = 0.010 * 111.32 * math.cos(math.radians(50.52)) * 0.007 * 110.57
    assert s.decrease_km2 - s.increase_km2 == pytest.approx(patch_km2, rel=0.15)


def test_map_and_background_layers(result):
    assert result.before_db.shape == (-(-result.grid.height // 4), -(-result.grid.width // 4))
    assert result.before_db.min() > 0  # full coverage
    html = render.build_map(result, 0.01).get_root().render()
    assert html.count("data:image/png;base64,") == 3 and "Before: radar image" in html


def test_second_run_is_served_from_cache(offline, monkeypatch):
    first = engine.run(PARAMS, signer=source.Signer())
    monkeypatch.setattr(source, "read_day", lambda *a, **k: pytest.fail("should not download"))
    second = engine.run(PARAMS, signer=source.Signer())
    np.testing.assert_array_equal(first.signed, second.signed)
    np.testing.assert_array_equal(first.before_db, second.before_db)
    assert second.params == first.params and second.grid == first.grid


def test_single_pair_vv_only_with_manual_enl(offline):
    params = engine.Params(**{**PARAMS.__dict__, "max_images": 1, "enl": 5.0, "bands": ("VV",)})
    result = engine.run(params, signer=source.Signer())
    assert result.before_days == [str(conftest.BEFORE_DAYS[-1])]
    assert result.after_days == [str(conftest.AFTER_DAYS[0])]
    assert result.enl == 5.0 and not result.enl_estimated and result.null_hist is None
    stat, _, _ = stats.decode(result.signed)
    inside, ring = patch_mask(result.grid)
    flagged = stat > stats.chi2_threshold(0.01, 1)
    assert flagged[~ring].mean() == pytest.approx(0.01, rel=0.25)
    # One pair is far less sensitive: theory gives 40.6% detection of a
    # 7 dB drop at 5 looks and alpha = 0.01 (vs >97% with 5 + 4 images).
    assert flagged[inside].mean() == pytest.approx(0.406, abs=0.03)


def test_no_common_orbit_is_explained(offline):
    params = engine.Params(**{**PARAMS.__dict__, "after": ("2023-01-01", "2023-02-01")})
    with pytest.raises(ValueError, match="same orbit in both periods"):
        engine.make_plan(params)
