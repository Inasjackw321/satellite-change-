from pathlib import Path

import numpy as np
import pytest
from streamlit.testing.v1 import AppTest

from satchange import engine, source, stats
from tests import conftest

APP = str(Path(__file__).resolve().parent.parent / "app.py")


def synthetic_result():
    params = engine.Params(label="Testville", bounds=(30.0, 50.0, 30.02, 50.02),
                           before=("2021-04-01", "2021-06-01"), after=("2022-04-01", "2022-06-01"))
    grid = engine.Grid.for_bounds(params.bounds)
    rng = np.random.default_rng(0)
    stat = rng.chisquare(2, (grid.height, grid.width))
    stat[:20, :20] = 60  # a 200 m x 200 m block of strong change
    signed = np.round(stat * stats.STAT_SCALE * -1).astype(np.int16)
    null = sum(rng.chisquare(1, 100_000) for _ in range(2))
    hist, _ = np.histogram(np.minimum(null, 49.99), bins=engine.NULL_BINS, range=(0, engine.NULL_MAX))
    return engine.Result(params=params, grid=grid, signed=signed, orbit=36,
                         before_days=["2021-04-01", "2021-04-13"], after_days=["2022-04-01"],
                         enl=4.8, enl_estimated=True, null_hist=hist, orbits=[])


_ARCHIVE = []


def app_archive():
    return _ARCHIVE[0]


@pytest.fixture
def app(offline, monkeypatch, tmp_path):
    _ARCHIVE[:] = [offline]
    import streamlit as st
    st.cache_data.clear()
    monkeypatch.setenv("SATCHANGE_CONFIG", str(tmp_path / "config.json"))
    return AppTest.from_file(APP, default_timeout=60)


def button(at, label):
    return next(b for b in at.button if b.label == label)


def choose_test_area(at):
    at.sidebar.selectbox[0].set_value("Enter coordinates…").run()
    w, s, e, n = conftest.AOI
    for label, v in (("West", w), ("East", e), ("South", s), ("North", n)):
        next(x for x in at.sidebar.number_input if x.label.startswith(label)).set_value(v)
    at.run()


def test_first_open_needs_no_account(app):
    app.run()
    assert not app.exception
    labels = [b.label for b in app.button]
    assert "Find changes" in labels and not any("Sign" in l for l in labels)
    info = " ".join(i.value for i in app.sidebar.info)
    assert "Found **4 before** and **4 after** images" in info and "Download: about" in info
    assert not button(app, "Find changes").disabled


def test_find_changes_end_to_end(app):
    app.run()
    choose_test_area(app)
    button(app, "Find changes").click().run()
    assert not app.exception and not app.error
    assert "My area" in app.header[0].value
    metrics = {m.label: m.value for m in app.metric}
    patch = float(metrics["Radar signal decreased"].split()[0])
    assert 0.4 < patch < 0.8  # the ~0.55 km2 patch plus chance hits
    assert metrics["No-change check"].endswith("flagged")
    # Now saved: it reopens instantly, even offline.
    assert any("Already computed" in i.value for i in app.sidebar.info)
    source_search = source.search
    try:
        source.search = lambda *a, **k: pytest.fail("should not go online")
        app.run()
        assert not app.exception and not app.sidebar.error
        assert not button(app, "Find changes").disabled
    finally:
        source.search = source_search


def test_area_without_images_is_explained(app, monkeypatch):
    far_away = [source.Scene(**{**s.__dict__, "geometry": {
        "type": "Polygon", "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]]}})
        for s in conftest.Archive.search(app_archive(), None, "2021-01-01", "2023-01-01")]
    monkeypatch.setattr(source, "search", lambda b, start, end, session=None: [
        s for s in far_away if str(start) <= str(s.day) < str(end)])
    app.run()
    choose_test_area(app)
    button(app, "Find changes").click().run()
    assert any("don't cover this area" in e.value for e in app.error)
    assert "Radar change map" in app.header[0].value  # no empty result shown


def test_archive_problems_are_shown(app, monkeypatch):
    def unreachable(*a, **k):
        raise source.SourceError("Can't reach Microsoft Planetary Computer. Check your internet connection.")
    monkeypatch.setattr(source, "search", unreachable)
    app.run()
    assert any("Can't reach Microsoft Planetary Computer" in e.value for e in app.sidebar.error)
    assert button(app, "Find changes").disabled


def test_dates_in_wrong_order_are_explained(app):
    app.run()
    app.sidebar.toggle[0].set_value(True).run()
    app.sidebar.date_input[0].set_value(("2022-04-01", "2022-05-01")).run()
    assert any("before period must end before" in e.value for e in app.sidebar.error)


def test_simple_dates_show_both_periods(app):
    app.run()
    text = " ".join(c.value for c in app.sidebar.caption)
    assert "01 Apr 2021 – 26 May 2021" in text and "01 Apr 2022 – 26 May 2022" in text


def test_result_view_updates_with_alpha(app):
    app.session_state["result"] = synthetic_result()
    app.run()
    assert not app.exception
    metrics = {m.label: m.value for m in app.metric}
    assert float(metrics["Radar signal decreased"].split()[0]) > 0.04

    app.select_slider[0].set_value(1e-6)
    app.run()
    metrics = {m.label: m.value for m in app.metric}
    # At alpha = 1e-6 essentially only the 0.04 km2 block remains.
    assert float(metrics["Radar signal decreased"].split()[0]) == pytest.approx(0.04, abs=0.002)
    assert metrics["No-change check"].startswith("0.00")


def test_saved_maps_can_be_reopened_and_deleted(app, tmp_path):
    engine.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    synthetic_result().save(engine.CACHE_DIR / "saved.npz")
    app.run()
    button(app, "Open").click().run()
    assert not app.exception and "Testville" in app.header[0].value
    button(app, "Delete all").click().run()
    assert engine.saved_results() == [] and "Radar change map" in app.header[0].value


def test_drawing_a_box_sets_the_area(app, monkeypatch):
    import streamlit_folium
    w, s_, e, n = conftest.AOI
    box = {"type": "Feature", "properties": {}, "geometry": {
        "type": "Polygon", "coordinates": [[[w, s_], [w, n], [e, n], [e, s_], [w, s_]]]}}
    drawn = []
    monkeypatch.setattr(streamlit_folium, "st_folium",
                        lambda m, **kw: drawn.append(m) or {"all_drawings": [box]})
    app.run()
    assert not app.exception
    assert app.sidebar.selectbox[0].value == "Drawn on the map"
    assert any("Found **4 before**" in i.value for i in app.sidebar.info)
    # The picker map now shows the drawn box, with the draw tool.
    html = drawn[-1].get_root().render()
    assert "L.Control.Draw" in html and f"{n}" in html
    button(app, "Find changes").click().run()
    assert not app.exception and "My area" in app.header[0].value
    assert app.radio[0].value == "🗺️ Change map"


def test_view_switch_returns_to_the_area_picker(app, monkeypatch):
    import streamlit_folium
    calls = []
    monkeypatch.setattr(streamlit_folium, "st_folium", lambda m, **kw: calls.append(kw) or None)
    app.session_state["result"] = synthetic_result()
    app.run()
    assert app.radio[0].value == "🗺️ Change map" and not calls
    app.radio[0].set_value("✏️ Choose area").run()
    assert not app.exception and calls and calls[-1]["key"] == "area_picker"


def test_drawn_box_is_remembered_next_time(app, monkeypatch, tmp_path):
    import streamlit_folium
    box = {"type": "Feature", "properties": {}, "geometry": {
        "type": "Polygon", "coordinates": [[[30.2, 50.5], [30.2, 50.54], [30.26, 50.54], [30.26, 50.5], [30.2, 50.5]]]}}
    monkeypatch.setattr(streamlit_folium, "st_folium", lambda m, **kw: {"all_drawings": [box]})
    app.run()
    monkeypatch.setattr(streamlit_folium, "st_folium", lambda m, **kw: None)
    fresh = AppTest.from_file(APP, default_timeout=60)
    fresh.run()
    fresh.sidebar.selectbox[0].set_value("Drawn on the map").run()
    assert any("About 19 km²" in c.value for c in fresh.sidebar.caption)
