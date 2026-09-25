import datetime as dt

import pytest

from satchange import engine, inputs


# --- Inputs -----------------------------------------------------------------

def test_fit_area_keeps_normal_city():
    kyiv = engine.CITIES["Kyiv – whole city (large download)"]
    bounds, note = inputs.fit_area(kyiv)
    assert bounds == pytest.approx(kyiv) and note is None


def test_fit_area_enlarges_a_point():
    bounds, note = inputs.fit_area((30.5, 50.4, 30.5, 50.4))
    assert inputs.area_km2(bounds) == pytest.approx(16, rel=0.02)
    assert "Enlarged" in note


def test_fit_area_trims_huge_area_to_limit():
    bounds, note = inputs.fit_area((22.0, 44.0, 40.0, 52.0))  # most of Ukraine
    g = engine.Grid.for_bounds(bounds)
    assert g.pixels <= engine.MAX_PIXELS and "trimmed" in note
    assert (bounds[0] + bounds[2]) / 2 == pytest.approx(31.0)


def test_periods_for_event_is_same_season_a_year_apart():
    before, after = inputs.periods_for_event(dt.date(2022, 4, 1), weeks=8)
    assert after == (dt.date(2022, 4, 1), dt.date(2022, 5, 26))
    assert before == (dt.date(2021, 4, 1), dt.date(2021, 5, 26))


def test_periods_for_event_on_leap_day():
    before, _ = inputs.periods_for_event(dt.date(2024, 2, 29), weeks=4)
    assert before[0] == dt.date(2023, 2, 28)


def test_search_places_parses_nominatim(monkeypatch):
    class Response:
        def raise_for_status(self):
            pass

        def json(self):
            return [{"display_name": "Kharkiv, Ukraine",
                     "boundingbox": ["49.88", "50.10", "36.10", "36.46"]}]

    calls = []
    monkeypatch.setattr(inputs.requests, "get", lambda url, **kw: calls.append(kw) or Response())
    assert inputs.search_places("Kharkiv") == [("Kharkiv, Ukraine", (36.10, 49.88, 36.46, 50.10))]
    assert "User-Agent" in calls[0]["headers"]


# --- Saved results ------------------------------------------------------------

def test_saved_results_lists_and_deletes(monkeypatch, tmp_path):
    from tests.test_app import synthetic_result

    monkeypatch.setattr(engine, "CACHE_DIR", tmp_path)
    r = synthetic_result()
    r.save(tmp_path / "a.npz")
    (tmp_path / "broken.npz").write_bytes(b"not a zip")
    saved = engine.saved_results()
    assert [p.name for p, _ in saved] == ["a.npz"]
    assert saved[0][1].startswith("Testville: 2021-04 → 2022-04")
    engine.delete_saved()
    assert engine.saved_results() == []


def test_legend_date_span():
    from satchange.render import _span
    assert _span(["2021-04-01", "2021-05-19"]) == "1 Apr – 19 May 2021"
    assert _span(["2021-12-20", "2022-01-05"]) == "20 Dec 2021 – 5 Jan 2022"
    assert _span(["2022-04-01"]) == "1 Apr 2022"


def rectangle(w, s, e, n):
    return {"type": "Feature", "properties": {},
            "geometry": {"type": "Polygon", "coordinates": [[[w, s], [w, n], [e, n], [e, s], [w, s]]]}}


def test_bounds_from_drawings_uses_the_newest_shape():
    old, new = rectangle(30.1, 50.4, 30.2, 50.5), rectangle(30.3, 50.45, 30.41, 50.52)
    assert inputs.bounds_from_drawings([old, new]) == (30.3, 50.45, 30.41, 50.52)
    assert inputs.bounds_from_drawings([]) is None and inputs.bounds_from_drawings(None) is None


def test_bounds_from_drawings_ignores_points_and_flat_shapes():
    point = {"type": "Feature", "geometry": {"type": "Point", "coordinates": [30.1, 50.4]}}
    assert inputs.bounds_from_drawings([point]) is None
    assert inputs.bounds_from_drawings([rectangle(30.1, 50.4, 30.1, 50.5)]) is None
