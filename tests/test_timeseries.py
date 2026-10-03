import datetime as dt

import numpy as np
import pytest

from satchange import engine, source, stats, timeseries as ts
from tests import conftest

LAND = ts.TSParams(label="Test", bounds=conftest.AOI, start="2026-05-29", end="2026-08-04")
SHIPS = ts.TSParams(**{**LAND.__dict__, "target": "water", "smooth": 1})


def pixel(grid, lat, lon):
    x, y = engine._merc(lon, lat)
    return int((grid.y0 - y) / grid.scale), int((x - grid.x0) / grid.scale)


def box_mask(grid, bounds, pad=0):
    w, s, e, n = bounds
    r0, c0 = pixel(grid, n, w)
    r1, c1 = pixel(grid, s, e)
    mask = np.zeros((grid.height, grid.width), bool)
    mask[max(0, r0 - pad):r1 + pad, max(0, c0 - pad):c1 + pad] = True
    return mask


@pytest.fixture
def land(offline):
    return ts.run(LAND, signer=source.Signer())


@pytest.fixture
def ships(offline):
    return ts.run(SHIPS, signer=source.Signer())


def test_plan_uses_every_pass_of_full_coverage_orbits(offline):
    plan = ts.make_plan(LAND)
    assert plan.days == conftest.DAYS  # all 12 passes, 29 May .. 3 Aug
    assert {o for _, o, _ in plan.passes} == {36}  # orbit 109 covers only a third


def test_plan_combines_orbits_for_more_passes(monkeypatch):
    full = {"type": "Polygon", "coordinates": [[[30, 50], [31, 50], [31, 51], [30, 51], [30, 50]]]}
    scenes = [source.Scene(id=f"{o}_{d}", day=d, orbit=o, orbit_state=state, geometry=full, hrefs={})
              for o, state, first in ((36, "descending", 1), (65, "ascending", 4), (7, "ascending", 2))
              for d in (dt.date(2026, 7, first) + dt.timedelta(days=12 * k)
                        for k in range(4 if o != 7 else 2))]
    monkeypatch.setattr(source, "search", lambda *a, **k: scenes)
    both = ts.make_plan(LAND)
    assert [o for _, o, _ in both.passes] == [36, 65, 36, 65, 36, 65, 36, 65]  # orbit 7: too few
    one = ts.make_plan(ts.TSParams(**{**LAND.__dict__, "all_orbits": False}))
    assert {o for _, o, _ in one.passes} == {36} and len(one.passes) == 4


def test_aircraft_stands_counted_by_pass(land):
    a = ts.analyse(land, ts.PRESETS["land"]["Balanced"])
    assert len(a.hotspots) == 2
    busy, quiet = a.hotspots  # most often seen first
    for spot, name in ((busy, "A"), (quiet, "B")):
        (lat, lon), days = conftest.STANDS[name]
        assert abs(spot.lat - lat) * 111_000 < 20 and abs(spot.lon - lon) * 71_000 < 20
        assert (spot.seen, spot.of) == (len(days), 12) and spot.days == [str(d) for d in days]
        assert 1500 < spot.area_m2 < 4000  # 50 m x 50 m
    assert a.per_pass == [1, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0]
    # The patch that went dark after 4 July is not an object that appeared.
    assert not a.count[box_mask(land.grid, conftest.PATCH)].any()


def test_size_limit_rejects_area_wide_brightening(land):
    """The 350 m x 330 m patch that brightens on two passes is far bigger than
    any aircraft or vehicle, so it is not counted, unless the limit is raised."""
    busy = box_mask(land.grid, conftest.ACTIVITY)
    assert not ts.analyse(land, ts.PRESETS["land"]["Balanced"]).count[busy].any()
    loose = ts.Sensitivity(1e-5, 6.0, 300, max_area_m2=500_000)
    spot = next(h for h in ts.analyse(land, loose).hotspots if h.area_m2 > 50_000)
    assert spot.days == ["2026-07-22", "2026-08-03"]


def test_water_is_found(land):
    water = land.water
    strip = box_mask(land.grid, conftest.WATER)
    inside = strip.copy()
    inside[:, :] &= ~box_mask(land.grid, (conftest.WATER[2] - 0.001, *conftest.WATER[1:3], conftest.WATER[3]))
    assert water[strip].mean() > 0.97
    east = box_mask(land.grid, (conftest.WATER[2] + 0.002, conftest.AOI[1], conftest.AOI[2], conftest.AOI[3]))
    assert water[east].mean() < 0.01  # dark land pixels and the darkened patch are not water


def test_ships_detected_where_and_when_they_were(ships):
    a = ts.analyse(ships, ts.PRESETS["water"]["Balanced"])
    found = {(d.day, round(d.lat, 3)) for d in a.detections}
    expected = {(str(day), round(conftest.ship_position(day)[0], 3)) for day in conftest.SHIP_DAYS}
    assert len(a.detections) == 6
    for d in a.detections:
        lat, lon = conftest.ship_position(dt.date.fromisoformat(d.day))
        assert abs(d.lat - lat) * 111_000 < 30 and abs(d.lon - lon) * 71_000 < 30  # within 30 m
        assert d.on_water and 60 <= d.length_m <= 110  # an 80 m ship
        assert d.brightness_db > 15
    assert len(found) == len(expected)
    assert sum(a.per_pass) == 6 and a.per_pass[ships.days.index("2026-07-10")] == 1


def test_ship_mode_ignores_land_and_land_mode_ignores_ships(land, ships):
    on_land = ts.analyse(ships, ts.PRESETS["water"]["Balanced"], target="land")
    assert all(not d.on_water for d in on_land.detections)
    assert not any(d.on_water for d in ts.analyse(land, ts.PRESETS["land"]["Balanced"]).detections)


def test_false_alarm_rate_on_unchanged_land_is_at_most_alpha(land):
    """Raw per-pixel test, no size or dB filters, on a middle pass with 4
    passes each side: speckle alone exceeds one side's threshold about alpha
    of the time (a bit less, as the ENL is measured conservatively), and both
    sides together less often."""
    alpha, i = 1e-2, 6
    level = stats.pass_threshold_db(alpha, land.enl, 4) * ts.DB_SCALE
    quiet = ~land.water
    for bounds in (conftest.PATCH, conftest.ACTIVITY):
        quiet &= ~box_mask(land.grid, bounds, 6)
    for (lat, lon), _ in conftest.STANDS.values():
        quiet &= ~box_mask(land.grid, (lon - 0.001, lat - 0.001, lon + 0.001, lat + 0.001))
    before, after = land.before_db[i][quiet], land.after_db[i][quiet]
    one = (before >= level).mean()
    both = ((before >= level) & (after >= level)).mean()
    assert 0.3 * alpha < one < 1.2 * alpha and both < one


def test_results_are_cached(offline, monkeypatch):
    first = ts.run(LAND, signer=source.Signer())
    monkeypatch.setattr(source, "read_day", lambda *a, **k: pytest.fail("should not download"))
    second = ts.run(LAND, signer=source.Signer())
    np.testing.assert_array_equal(first.before_db, second.before_db)
    np.testing.assert_array_equal(first.after_db, second.after_db)
    np.testing.assert_array_equal(first.water, second.water)
    assert second.days == first.days and second.params == first.params


def test_too_few_passes_is_explained(offline):
    with pytest.raises(ValueError, match="at least 3 passes"):
        ts.make_plan(ts.TSParams(**{**LAND.__dict__, "start": "2026-07-25", "end": "2026-08-04"}))


def test_saved_activity_results_are_listed_and_reopen(land):
    path, desc = engine.saved_results()[0]
    assert desc == "Test: activity, 29 May 2026 → 4 Aug 2026"
    again = engine.load_result(path)
    assert isinstance(again, ts.TSResult) and again.days == land.days
