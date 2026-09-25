"""Synthetic Sentinel-1 scenes as real GeoTIFFs, so the whole pipeline
(search results -> reading -> reprojection -> mosaic -> test) runs offline.

Scenes are in UTM zone 36N like the real Kyiv data, with gamma speckle of a
known number of looks and a patch that loses 7 dB after the event.
"""

import datetime as dt

import numpy as np
import pytest
import rasterio
from affine import Affine
from rasterio.warp import transform_bounds

from satchange import engine, source

TRUE_ENL = 5.0
AOI = (30.20, 50.50, 30.26, 50.54)  # ~4.3 x 4.4 km near Irpin
PATCH = (30.225, 50.515, 30.235, 50.522)  # changed area (lon/lat), ~700 x 780 m
UTM = "EPSG:32636"
BEFORE_DAYS = [dt.date(2021, 4, 1) + dt.timedelta(days=12 * k) for k in range(5)]
AFTER_DAYS = [dt.date(2022, 4, 1) + dt.timedelta(days=12 * k) for k in range(4)]


def _footprint(bounds_utm):
    w, s, e, n = transform_bounds(UTM, "EPSG:4326", *bounds_utm)
    return {"type": "Polygon", "coordinates": [[[w, s], [e, s], [e, n], [w, n], [w, s]]]}


class Archive:
    """Writes scenes to disk and answers source.search like the real catalogue."""

    def __init__(self, root):
        self.root = root
        self.rng = np.random.default_rng(42)
        w, s, e, n = transform_bounds("EPSG:4326", UTM, *AOI)
        pad = 300
        self.transform = Affine(10, 0, np.floor(w - pad), 0, -10, np.ceil(n + pad))
        self.width = int((e - w + 2 * pad) / 10)
        self.height = int((n - s + 2 * pad) / 10)
        # Varied land cover: mean backscatter between about -20 and 0 dB.
        self.mean = {"VV": 10 ** self.rng.uniform(-2, 0, (self.height, self.width)),
                     "VH": 10 ** self.rng.uniform(-2.8, -0.8, (self.height, self.width))}
        pw, ps, pe, pn = transform_bounds("EPSG:4326", UTM, *PATCH)
        cols = ((pw - self.transform.c) / 10, (pe - self.transform.c) / 10)
        rows = ((self.transform.f - pn) / 10, (self.transform.f - ps) / 10)
        self.patch = (slice(int(rows[0]), int(rows[1])), slice(int(cols[0]), int(cols[1])))
        self.scenes = []
        for day in BEFORE_DAYS + AFTER_DAYS:
            # The first pass is split into two scenes, like real slices.
            halves = [(0, self.height // 2), (self.height // 2, self.height)] if day == BEFORE_DAYS[0] else [(0, self.height)]
            images = self._acquire(changed=day >= AFTER_DAYS[0])
            for k, (r0, r1) in enumerate(halves):
                self.scenes.append(self._write(day, k, images, r0, r1, orbit=36))
            # A second orbit that only covers the western third.
            self.scenes.append(self._partial(day))

    def _acquire(self, changed):
        out = {}
        for b, mean in self.mean.items():
            m = mean.copy()
            if changed:
                m[self.patch] *= 10 ** -0.7  # -7 dB
            out[b] = (self.rng.gamma(TRUE_ENL, 1 / TRUE_ENL, m.shape) * m).astype(np.float32)
        return out

    def _write(self, day, part, images, r0, r1, orbit):
        hrefs = {}
        t = self.transform * Affine.translation(0, r0)
        for b, img in images.items():
            path = self.root / f"{day}_{orbit}_{part}_{b}.tif"
            with rasterio.open(path, "w", driver="GTiff", width=self.width, height=r1 - r0, count=1,
                               dtype="float32", crs=UTM, transform=t, nodata=0) as dst:
                dst.write(img[r0:r1], 1)
            hrefs[b] = str(path)
        bounds = (t.c, t.f - 10 * (r1 - r0), t.c + 10 * self.width, t.f)
        return source.Scene(id=f"S1_{day}_{orbit}_{part}", day=day, orbit=orbit,
                            orbit_state="descending", geometry=_footprint(bounds), hrefs=hrefs)

    def _partial(self, day):
        w = self.transform.c
        bounds = (w, self.transform.f - 10 * self.height, w + 10 * self.width / 3, self.transform.f)
        return source.Scene(id=f"S1_{day}_109", day=day, orbit=109, orbit_state="ascending",
                            geometry=_footprint(bounds), hrefs={"VV": "missing", "VH": "missing"})

    def search(self, bounds, start, end, session=None):
        start, end = dt.date.fromisoformat(str(start)), dt.date.fromisoformat(str(end))
        return [s for s in self.scenes if start <= s.day < end]


@pytest.fixture(scope="session")
def archive(tmp_path_factory):
    return Archive(tmp_path_factory.mktemp("s1"))


@pytest.fixture
def offline(archive, monkeypatch, tmp_path):
    monkeypatch.setattr(source, "search", archive.search)
    monkeypatch.setattr(engine, "CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(engine, "TILE", 256)  # several tiles even for the small test area
    return archive
