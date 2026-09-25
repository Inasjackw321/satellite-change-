from pathlib import Path

import numpy as np
import pytest
from streamlit.testing.v1 import AppTest

from satchange import engine, stats

APP = str(Path(__file__).resolve().parent.parent / "app.py")


def synthetic_result():
    params = engine.Params(label="Testville", bounds=(30.0, 50.0, 30.02, 50.02),
                           before=("2021-04-01", "2021-06-01"), after=("2022-04-01", "2022-06-01"))
    grid = engine.Grid.for_bounds(params.bounds)
    rng = np.random.default_rng(0)
    stat = rng.chisquare(2, (grid.height, grid.width))
    stat[:20, :20] = 60  # a 200 m x 200 m block of strong change
    signed = np.round(stat * stats.STAT_SCALE * -1).astype(np.int16)
    L = 15
    null = sum(rng.chisquare(1, 100_000) for _ in range(2))
    hist, _ = np.histogram(np.minimum(null, 49.99), bins=engine.NULL_BINS, range=(0, engine.NULL_MAX))
    return engine.Result(params=params, grid=grid, signed=signed, orbit=36,
                         before_days=["2021-04-01", "2021-04-13"], after_days=["2022-04-01"],
                         enl=4.8, enl_estimated=True, null_hist=hist, orbits=[])


@pytest.fixture
def app(monkeypatch, tmp_path):
    monkeypatch.setattr(engine, "has_credentials", lambda: False)
    monkeypatch.setattr(engine, "CACHE_DIR", tmp_path)
    return AppTest.from_file(APP, default_timeout=30)


def test_first_open_explains_and_offers_sign_in(app):
    app.run()
    assert not app.exception
    assert any("Sign in" in b.label for b in app.button)
    assert "Radar change map" in app.header[0].value


def test_run_without_project_shows_error(app):
    app.run()
    app.sidebar.text_input[0].set_value("")
    app.sidebar.button[0].click()
    app.run()
    assert any("Google Cloud project" in e.value for e in app.error)


def test_result_view_updates_with_alpha(app):
    app.session_state["result"] = synthetic_result()
    app.session_state["layers"] = {}
    app.run()
    assert not app.exception
    metrics = {m.label: m.value for m in app.metric}
    decrease = float(metrics["Backscatter decrease"].split()[0])
    assert decrease > 0.04  # the 0.04 km2 block plus chance hits

    app.select_slider[0].set_value(1e-6)
    app.run()
    metrics = {m.label: m.value for m in app.metric}
    # At alpha = 1e-6 essentially only the block remains.
    assert float(metrics["Backscatter decrease"].split()[0]) == pytest.approx(0.04, abs=0.002)
    assert metrics["No-change check"].startswith("0.00")
