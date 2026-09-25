from pathlib import Path

import numpy as np
import pytest
from streamlit.testing.v1 import AppTest

from satchange import engine, source, stats
from tests import conftest

APP = str(Path(__file__).resolve().parent.parent / "app.py")


def synthetic_result():
    """A small result with a 200 m x 200 m block that lost 7 dB, plus noise."""
    params = engine.Params(label="Testville", bounds=(30.0, 50.0, 30.02, 50.02),
                           start="2026-07-04", end="2026-08-04")
    grid = engine.Grid.for_bounds(params.bounds)
    rng = np.random.default_rng(0)
    stat = rng.chisquare(2, (grid.height, grid.width))
    db = rng.normal(0, 0.5, stat.shape)
    stat[:20, :20], db[:20, :20] = 60, -7
    encode = lambda st, d: (np.round(st * stats.STAT_SCALE * np.sign(d)).astype(np.int16),
                            np.round(d * 100).astype(np.int16))
    signed, change_db = encode(stat, db)
    null_signed, null_db = encode(rng.chisquare(2, stat.shape), rng.normal(0, 0.5, stat.shape))
    return engine.Result(params=params, grid=grid, signed=signed, change_db=change_db, orbit=36,
                         before_days=["2026-06-28", "2026-07-04"], after_days=["2026-08-03"],
                         enl=40.0, enl_estimated=True, null_signed=null_signed, null_db=null_db)


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


def set_dates(at, start="2026-07-04", end="2026-08-04"):
    at.sidebar.date_input(key="start").set_value(start)
    at.sidebar.date_input(key="end").set_value(end)
    at.run()


def choose_test_area(at):
    set_dates(at)
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


def test_images_found_for_the_dates_are_listed(app):
    app.run()
    set_dates(app)
    info = " ".join(i.value for i in app.sidebar.info)
    assert "**Before:** 22 Jun, 28 Jun, 4 Jul 2026" in info
    assert "**After:** 22 Jul, 28 Jul, 3 Aug 2026" in info and "download about" in info
    assert not button(app, "Find changes").disabled


def test_find_changes_end_to_end(app):
    app.run()
    choose_test_area(app)
    button(app, "Find changes").click().run()
    assert not app.exception and not app.error
    assert "My area: 4 Jul 2026 → 4 Aug 2026" in app.header[0].value
    metrics = {m.label: m.value for m in app.metric}
    assert 0.45 < float(metrics["Radar signal decreased"].split()[0]) < 0.65  # the ~0.55 km2 patch
    assert 0.10 < float(metrics["Radar signal increased"].split()[0]) < 0.18  # the "aircraft" patch
    assert metrics["Changed spots"] == "2" and metrics["Noise check"] == "0.00 km²"
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
        for s in conftest.Archive.search(app_archive(), None, "2026-01-01", "2027-01-01")]
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
    set_dates(app, "2026-08-04", "2026-07-04")
    assert any("end date must be after the start date" in e.value for e in app.sidebar.error)
    assert button(app, "Find changes").disabled


def test_range_without_images_is_explained(app):
    app.run()
    set_dates(app, "2026-08-04", "2026-08-10")
    assert any("Try a longer date range" in e.value for e in app.sidebar.error)


def test_detection_settings(app):
    app.session_state["result"] = synthetic_result()
    app.run()
    assert not app.exception
    metrics = lambda: {m.label: m.value for m in app.metric}
    # Balanced: only the 0.04 km2 block, and the noise check is clean.
    assert metrics()["Radar signal decreased"] == "0.04 km²" and metrics()["Changed spots"] == "1"
    assert metrics()["Noise check"] == "0.00 km²"

    # Custom with loose settings lets noise through, and the noise check shows it.
    app.radio(key="preset").set_value("Custom").run()
    app.select_slider[0].set_value(1e-2)
    app.slider[0].set_value(0.5)
    app.select_slider[1].set_value(100)
    app.run()
    assert float(metrics()["Noise check"].split()[0]) > 0 and int(metrics()["Changed spots"]) > 1

    app.radio(key="preset").set_value("Strict").run()
    assert metrics()["Changed spots"] == "1" and metrics()["Noise check"] == "0.00 km²"


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
    set_dates(app)
    assert not app.exception
    assert app.sidebar.selectbox[0].value == "Drawn on the map"
    assert any("**Before:** 22 Jun" in i.value for i in app.sidebar.info)
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


def test_download_box_makes_a_watermarked_image(app):
    app.run()
    choose_test_area(app)
    button(app, "Find changes").click().run()
    assert app.text_input(key="export_watermark").value == "@Kaldockhi"
    app.radio(key="export_background").set_value("Radar image").run()  # no internet in tests
    button(app, "Create image").click().run()
    assert not app.exception and not app.warning
    labels = [b.label for b in app.get("download_button")]
    assert "Download image (PNG)" in labels
    assert "List of changed spots (CSV)" in labels and "Interactive map (HTML)" in labels

    # Changing a setting asks for a new image.
    app.text_input(key="export_watermark").set_value("@someone").run()
    assert any("press **Create image** again" in c.value for c in app.caption)
