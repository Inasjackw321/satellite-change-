from pathlib import Path

import numpy as np
import pytest
from streamlit.testing.v1 import AppTest

from satchange import account, engine, stats

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


class FakeAccount:
    def __init__(self, signed_in):
        self.signed_in = signed_in
        self.sign_ins = self.sign_outs = 0

    def install(self, monkeypatch):
        monkeypatch.setattr(account, "is_signed_in", lambda: self.signed_in)
        monkeypatch.setattr(account, "list_projects", lambda: ["alpha-project", "beta-project"])
        monkeypatch.setattr(account, "start_sign_in", self.start_sign_in)
        monkeypatch.setattr(account, "sign_out", self.sign_out)
        monkeypatch.setattr(account, "connect", self.connect)

    def start_sign_in(self):
        self.sign_ins += 1
        self.signed_in = True  # as if the browser flow finished instantly

    def sign_out(self):
        self.sign_outs += 1
        self.signed_in = False

    def connect(self, project):
        raise RuntimeError("offline in tests")


def make_app(monkeypatch, tmp_path, signed_in):
    fake = FakeAccount(signed_in)
    fake.install(monkeypatch)
    monkeypatch.setattr(engine, "CACHE_DIR", tmp_path)
    monkeypatch.setenv("SATCHANGE_CONFIG", str(tmp_path / "config.json"))
    at = AppTest.from_file(APP, default_timeout=30)
    return at, fake


def button(at, label):
    return next(b for b in at.button if b.label == label)


def test_signed_out_first_open(monkeypatch, tmp_path):
    at, _ = make_app(monkeypatch, tmp_path, signed_in=False)
    at.run()
    assert not at.exception
    assert button(at, "Find changes").disabled
    assert any("Sign in with your Google account" in m.value for m in at.markdown)


def test_sign_in_then_out(monkeypatch, tmp_path):
    at, fake = make_app(monkeypatch, tmp_path, signed_in=False)
    at.run()
    button(at, "Sign in with Google").click().run()
    at.run()
    assert not at.exception and fake.sign_ins == 1
    assert any("Signed in" in m.value for m in at.sidebar.markdown)
    # Projects are offered in a dropdown once signed in.
    assert at.sidebar.selectbox[0].options[:2] == ["alpha-project", "beta-project"]
    assert not button(at, "Find changes").disabled

    button(at, "Sign out").click().run()
    assert any("Sign out of Google Earth Engine" in w.value for w in at.sidebar.warning)
    button(at, "Yes, sign out").click().run()
    assert fake.sign_outs == 1
    assert any(b.label == "Sign in with Google" for b in at.button)
    assert button(at, "Find changes").disabled


def test_cancel_sign_out(monkeypatch, tmp_path):
    at, fake = make_app(monkeypatch, tmp_path, signed_in=True)
    at.run()
    button(at, "Sign out").click().run()
    button(at, "Cancel").click().run()
    assert fake.sign_outs == 0 and not at.sidebar.warning


def test_connection_error_is_explained(monkeypatch, tmp_path):
    at, fake = make_app(monkeypatch, tmp_path, signed_in=True)
    monkeypatch.setattr(account, "connect", lambda p: (_ for _ in ()).throw(
        Exception("Not signed up for Earth Engine or project is not registered.")))
    at.run()
    button(at, "Find changes").click().run()
    assert any("isn't registered" in e.value for e in at.error)


def test_simple_dates_show_both_periods(monkeypatch, tmp_path):
    at, _ = make_app(monkeypatch, tmp_path, signed_in=True)
    at.run()
    text = " ".join(c.value for c in at.sidebar.caption)
    assert "01 Apr 2021 – 26 May 2021" in text and "01 Apr 2022 – 26 May 2022" in text


def test_result_view_updates_with_alpha(monkeypatch, tmp_path):
    at, _ = make_app(monkeypatch, tmp_path, signed_in=True)
    at.session_state["result"] = synthetic_result()
    at.session_state["layers"] = {}
    at.run()
    assert not at.exception
    metrics = {m.label: m.value for m in at.metric}
    assert float(metrics["Radar signal decreased"].split()[0]) > 0.04

    at.select_slider[0].set_value(1e-6)
    at.run()
    metrics = {m.label: m.value for m in at.metric}
    # At alpha = 1e-6 essentially only the 0.04 km2 block remains.
    assert float(metrics["Radar signal decreased"].split()[0]) == pytest.approx(0.04, abs=0.002)
    assert metrics["No-change check"].startswith("0.00")


def test_saved_maps_can_be_reopened(monkeypatch, tmp_path):
    at, _ = make_app(monkeypatch, tmp_path, signed_in=True)
    synthetic_result().save(tmp_path / "saved.npz")
    at.run()
    button(at, "Open").click().run()
    assert not at.exception
    assert "Testville" in at.header[0].value
