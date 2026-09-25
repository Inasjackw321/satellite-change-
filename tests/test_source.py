import datetime as dt

import pytest
import requests

from satchange import source


def item(id_, day, orbit=36, mode="IW", bands=("vv", "vh")):
    return {
        "id": id_,
        "geometry": {"type": "Polygon", "coordinates": [[[30, 50], [31, 50], [31, 51], [30, 51], [30, 50]]]},
        "properties": {"datetime": f"{day}T04:05:06.123Z", "sat:relative_orbit": orbit,
                       "sat:orbit_state": "descending", "sar:instrument_mode": mode},
        "assets": {b: {"href": f"https://example.test/{id_}/{b}.tif"} for b in bands}
                  | {"thumbnail": {"href": "https://example.test/thumb.png"}},
    }


class Response:
    def __init__(self, data, status=200):
        self.data, self.status_code = data, status

    def json(self):
        return self.data

    def raise_for_status(self):
        if self.status_code >= 400:
            err = requests.HTTPError(f"{self.status_code}")
            err.response = self
            raise err


class FakeSession:
    def __init__(self, responses):
        self.responses, self.calls = list(responses), []

    def post(self, url, json=None, timeout=None):
        self.calls.append(("POST", url, json))
        return self.responses.pop(0)

    def get(self, url, timeout=None):
        self.calls.append(("GET", url, None))
        return self.responses.pop(0)


def test_search_follows_pages_and_keeps_usable_scenes():
    page1 = {"features": [item("a", "2021-04-01"), item("ew", "2021-04-02", mode="EW")],
             "links": [{"rel": "next", "href": source.STAC_URL, "method": "POST", "body": {"token": "p2"}}]}
    page2 = {"features": [item("b", "2021-04-13", orbit=109), item("vvonly", "2021-04-14", bands=("vv",))],
             "links": []}
    session = FakeSession([Response(page1), Response(page2)])
    scenes = source.search((30.1, 50.4, 30.4, 50.6), "2021-04-01", "2021-06-01", session=session)

    assert [(s.id, s.day, s.orbit) for s in scenes] == [
        ("a", dt.date(2021, 4, 1), 36), ("b", dt.date(2021, 4, 13), 109)]
    assert scenes[0].hrefs == {"VV": "https://example.test/a/vv.tif", "VH": "https://example.test/a/vh.tif"}
    first = session.calls[0][2]
    assert first["collections"] == ["sentinel-1-rtc"] and first["bbox"] == [30.1, 50.4, 30.4, 50.6]
    assert first["datetime"] == "2021-04-01T00:00:00Z/2021-05-31T23:59:59Z"  # end is exclusive
    assert session.calls[1] == ("POST", source.STAC_URL, {"token": "p2"})


def test_search_offline_is_explained():
    class Offline:
        def post(self, *a, **k):
            raise requests.ConnectionError("no route")

    with pytest.raises(source.SourceError, match="internet connection"):
        source.search((30, 50, 31, 51), "2021-04-01", "2021-05-01", session=Offline())


def test_signer_fetches_token_once_and_refreshes_when_expiring():
    soon = (dt.datetime.now(dt.timezone.utc) + dt.timedelta(hours=1)).isoformat().replace("+00:00", "Z")
    later = (dt.datetime.now(dt.timezone.utc) + dt.timedelta(minutes=2)).isoformat().replace("+00:00", "Z")
    session = FakeSession([Response({"token": "sv=1&sig=abc", "msft:expiry": soon})])
    signer = source.Signer(session)
    assert signer.sign("https://x.test/a.tif") == "https://x.test/a.tif?sv=1&sig=abc"
    assert signer.sign("https://x.test/b.tif?x=1") == "https://x.test/b.tif?x=1&sv=1&sig=abc"
    assert len(session.calls) == 1
    assert session.calls[0][1] == "https://planetarycomputer.microsoft.com/api/sas/v1/token/sentinel-1-rtc"

    signer.expires = dt.datetime.fromisoformat(later.replace("Z", "+00:00"))
    session.responses.append(Response({"token": "new", "msft:expiry": soon}))
    assert signer.sign("https://x.test/a.tif").endswith("?new")


def test_signer_refusal_is_explained():
    signer = source.Signer(FakeSession([Response({}, status=403)]))
    with pytest.raises(source.SourceError, match="refused access"):
        signer.sign("https://x.test/a.tif")


def test_local_paths_are_not_signed():
    assert source.Signer(FakeSession([])).sign("/tmp/a.tif") == "/tmp/a.tif"
