import datetime as dt
import math

import numpy as np
import pytest

from satchange import engine, render, source, stats
from tests import conftest

PARAMS = engine.Params(label="Test", bounds=conftest.AOI, start="2026-07-04", end="2026-08-04")
D = dt.date


def grid_index(grid, lon, lat):
    x, y = engine._merc(lon, lat)
    return int((grid.y0 - y) / grid.scale), int((x - grid.x0) / grid.scale)


def patch_mask(grid, shrink=3):
    """Output pixels inside the changed patch (a little inside its edge), and a
    wider ring around it that excludes the patch's blurred edge."""
    w, s, e, n = conftest.PATCH
    r0, c0 = grid_index(grid, w, n)
    r1, c1 = grid_index(grid, e, s)
    inside = np.zeros((grid.height, grid.width), bool)
    inside[r0 + shrink:r1 - shrink, c0 + shrink:c1 - shrink] = True
    ring = np.zeros_like(inside)
    ring[r0 - 6:r1 + 6, c0 - 6:c1 + 6] = True
    return inside, ring


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


def test_date_windows():
    assert PARAMS.before_window == ("2026-04-05", "2026-07-05")  # up to and including the start
    assert PARAMS.after_window == ("2026-07-05", "2026-08-05")  # after the start, up to the end


def test_plan_uses_latest_images_up_to_each_date(offline):
    plan = engine.make_plan(PARAMS)
    assert plan.orbit == 36  # orbit 109 covers only a third of the area
    assert list(plan.before) == [D(2026, 6, 22), D(2026, 6, 28), D(2026, 7, 4)]
    assert list(plan.after) == [D(2026, 7, 22), D(2026, 7, 28), D(2026, 8, 3)]
    assert len(plan.before[D(2026, 6, 22)]) == 2  # split pass
    grid = engine.Grid.for_bounds(PARAMS.bounds)
    assert plan.download_mb == pytest.approx(grid.pixels * 2 * 6 * engine.BYTES_PER_PIXEL / 1e6)


def test_balanced_detection_finds_the_patch_and_nothing_else(result):
    inc, dec, spots = render.detect(result.signed, result.change_db, render.PRESETS["Balanced"], 2)
    inside, ring = patch_mask(result.grid)
    assert dec[inside].mean() > 0.97 and not inc[inside].any()
    assert not (inc | dec)[~ring].any()  # no noise anywhere else
    assert spots == 1


def test_change_size_is_measured(result):
    db = result.change_db.astype(float) / 100
    inside, ring = patch_mask(result.grid)
    assert np.median(db[inside]) == pytest.approx(-7, abs=0.3)  # the patch lost 7 dB
    assert abs(np.median(db[~ring])) < 0.2


def test_speckle_filter_raises_looks_and_noise_check_is_clean(result):
    # 3x3 filtering of 5-look images: more looks, but less than 9 x 5 because
    # nearest-neighbour resampling duplicates some pixels.
    assert result.enl_estimated and 20 < result.enl < 45
    s = render.summarize(result, render.PRESETS["Balanced"])
    assert s.noise_km2 == 0 and s.noise_spots == 0
    patch_km2 = 0.010 * 111.32 * math.cos(math.radians(50.52)) * 0.007 * 110.57
    assert s.decrease_km2 == pytest.approx(patch_km2, rel=0.15) and s.increase_km2 == 0


def test_sensitive_setting_shows_more_noise_than_balanced(result):
    sensitive = render.summarize(result, render.PRESETS["Sensitive"])
    balanced = render.summarize(result, render.PRESETS["Balanced"])
    assert sensitive.increase_km2 + sensitive.decrease_km2 >= balanced.increase_km2 + balanced.decrease_km2


def test_unfiltered_statistics_still_match_alpha(offline):
    """Without speckle filter or extra filters the per-pixel test keeps its
    advertised false-alarm rate."""
    params = engine.Params(**{**PARAMS.__dict__, "smooth": 1})
    result = engine.run(params, signer=source.Signer())
    assert result.enl == pytest.approx(conftest.TRUE_ENL, rel=0.08)
    stat, _, _ = stats.decode(result.signed)
    _, ring = patch_mask(result.grid)
    assert (stat > stats.chi2_threshold(0.01, 2))[~ring].mean() == pytest.approx(0.01, rel=0.25)


def test_map_and_background_layers(result):
    assert result.before_db.shape == (-(-result.grid.height // 4), -(-result.grid.width // 4))
    assert result.before_db.min() > 0  # full coverage
    html = render.build_map(result, render.Detection()).get_root().render()
    assert html.count("data:image/png;base64,") == 3 and "Before: radar image" in html


def test_second_run_is_served_from_cache(offline, monkeypatch):
    first = engine.run(PARAMS, signer=source.Signer())
    monkeypatch.setattr(source, "read_day", lambda *a, **k: pytest.fail("should not download"))
    second = engine.run(PARAMS, signer=source.Signer())
    for name in engine.Result.ARRAYS:
        np.testing.assert_array_equal(getattr(first, name), getattr(second, name))
    assert second.params == first.params and second.grid == first.grid


def test_old_saved_results_are_ignored(offline):
    result = engine.run(PARAMS, signer=source.Signer())
    result.version = 2
    result.save(engine.CACHE_DIR / "old.npz")
    with pytest.raises(ValueError):
        engine.Result.load(engine.CACHE_DIR / "old.npz")
    assert [p.name for p, _ in engine.saved_results()] == [f"{PARAMS.cache_key()}.npz"]


def test_single_image_each_side(offline):
    params = engine.Params(**{**PARAMS.__dict__, "images": 1})
    result = engine.run(params, signer=source.Signer())
    assert result.before_days == ["2026-07-04"] and result.after_days == ["2026-08-03"]
    assert result.null_signed is None
    inc, dec, _ = render.detect(result.signed, result.change_db, render.Detection(), 2)
    inside, _ = patch_mask(result.grid)
    assert dec[inside].mean() > 0.9  # the 3x3 filter makes even one pair usable


def test_range_without_images_is_explained(offline):
    params = engine.Params(**{**PARAMS.__dict__, "start": "2026-08-04", "end": "2026-08-10"})
    with pytest.raises(ValueError, match="Try a longer date range"):
        engine.make_plan(params)
