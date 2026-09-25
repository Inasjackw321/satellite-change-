"""A fake Earth Engine server, so the pipeline can be tested offline.

ee's bundled algorithm signatures are loaded, so building the computation
graph checks every function and argument name against the real API. Server
round-trips return synthetic Sentinel-1-like data.
"""

import datetime as dt

import ee
import numpy as np
import pytest
from ee import apitestcase

from satchange import stats

TRUE_ENL = 5.0
CHANGED_BLOCK = (slice(100, 200), slice(300, 400))  # rows, cols in the first tile


def _millis(day):
    return int(dt.datetime(day.year, day.month, day.day, 4, tzinfo=dt.timezone.utc).timestamp() * 1000)


def _rows(start):
    """Orbit 36 fully covers the area every 12 days (split in two scenes on the
    first day), orbit 109 partly covers it."""
    rows = [[36, _millis(start)], [36, _millis(start) + 25_000]]
    rows += [[36, _millis(start + dt.timedelta(days=12 * k))] for k in range(1, 5)]
    rows += [[109, _millis(start + dt.timedelta(days=3 + 12 * k))] for k in range(2)]
    return rows


class FakeServer:
    def __init__(self):
        self.rng = np.random.default_rng(0)
        self.calls = []

    def compute_value(self, obj):
        expr = ee.serializer.toJSON(obj)
        self.calls.append(expr)
        if "reduceColumns" in expr:
            return _rows(dt.date(2021, 4, 1) if "2021" in expr else dt.date(2022, 4, 1))
        if "intersection" in expr:
            return {"36": 1.0, "109": 0.6}
        if "Image.sample" in expr:
            n = 6000
            ratio = lambda: np.log(self.rng.gamma(TRUE_ENL, 1 / TRUE_ENL, n)
                                   / self.rng.gamma(TRUE_ENL, 1 / TRUE_ENL, n))
            return {"VV": ratio().tolist(), "VH": ratio().tolist()}
        if "fixedHistogram" in expr:
            L = TRUE_ENL * 3
            stat = sum(
                stats_lrt(self.rng.gamma(L, 1 / L, 200_000), self.rng.gamma(L, 1 / L, 200_000), L, L)
                for _ in range(2)
            )
            counts, edges = np.histogram(np.minimum(stat, 49.99), bins=500, range=(0, 50))
            return [[float(e), int(c)] for e, c in zip(edges, counts)]
        raise AssertionError("unexpected getInfo: " + expr[:300])

    def compute_pixels(self, params):
        self.calls.append("computePixels")
        grid = params["grid"]
        w, h = grid["dimensions"]["width"], grid["dimensions"]["height"]
        stat = self.rng.chisquare(2, (h, w))
        sign = np.where(self.rng.random((h, w)) < 0.5, -1, 1)
        if grid["affineTransform"]["translateX"] == FIRST_TILE_X[0]:
            if grid["affineTransform"]["translateY"] == FIRST_TILE_Y[0]:
                stat[CHANGED_BLOCK] = 100
                sign[CHANGED_BLOCK] = -1
        signed = np.round(np.minimum(stat, stats.STAT_MAX) * stats.STAT_SCALE * sign).astype(np.int16)
        signed[-1, :] = stats.NODATA
        out = np.zeros((h, w), dtype=[("signed", np.int16)])
        out["signed"] = signed
        return out

    def get_map_id(self, params):
        return {"tile_fetcher": type("TF", (), {"url_format": "https://example.test/{z}/{x}/{y}"})()}


FIRST_TILE_X, FIRST_TILE_Y = [None], [None]


def stats_lrt(s1, s2, L1, L2):
    return eval(stats.LRT_EXPRESSION, {"log": np.log}, {"s1": s1, "s2": s2, "L1": L1, "L2": L2})


@pytest.fixture
def fake_ee(monkeypatch, tmp_path):
    from satchange import engine

    server = FakeServer()
    ee.Reset()
    monkeypatch.setattr(ee.data, "_install_cloud_api_resource", lambda: None)
    monkeypatch.setattr(ee.data, "getAlgorithms", apitestcase.GetAlgorithms)
    monkeypatch.setattr(ee.deprecation, "_FetchDataCatalogStac", lambda: {})
    monkeypatch.setattr(ee.data, "computeValue", server.compute_value)
    monkeypatch.setattr(ee.data, "computePixels", server.compute_pixels)
    monkeypatch.setattr(ee.data, "getMapId", server.get_map_id)
    ee.Initialize(None, "", project="test")
    monkeypatch.setattr(engine, "CACHE_DIR", tmp_path / "cache")
    yield server
    ee.Reset()
