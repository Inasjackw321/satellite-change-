import datetime as dt
import json

import pytest

from satchange import account, engine, inputs


# --- Sign out ---------------------------------------------------------------

@pytest.fixture
def credentials(monkeypatch, tmp_path):
    path = tmp_path / "credentials"
    path.write_text(json.dumps({"refresh_token": "secret-token"}))
    monkeypatch.setattr(account, "credentials_path", lambda: str(path))
    return path


def test_sign_out_revokes_token_and_deletes_credentials(credentials, monkeypatch):
    posted = []
    monkeypatch.setattr(account.requests, "post", lambda url, **kw: posted.append((url, kw)))
    assert account.is_signed_in()
    account.sign_out()
    assert not account.is_signed_in() and not credentials.exists()
    assert posted == [(account.REVOKE_URL, {"params": {"token": "secret-token"}, "timeout": 10})]


def test_sign_out_still_signs_out_when_offline(credentials, monkeypatch):
    def fail(*a, **kw):
        raise account.requests.ConnectionError("offline")
    monkeypatch.setattr(account.requests, "post", fail)
    account.sign_out()
    assert not credentials.exists()


def test_sign_out_when_already_signed_out(credentials):
    credentials.unlink()
    account.sign_out()  # no error


def test_connect_requires_sign_in(credentials):
    credentials.unlink()
    with pytest.raises(account.NotSignedIn):
        account.connect("p")


@pytest.mark.parametrize("message,expected", [
    ("Not signed up for Earth Engine or project is not registered.", "isn't registered"),
    ("Earth Engine API has not been used in project 123 before or it is disabled.", "switched off"),
    ("Caller does not have required permission to use project p.", "doesn't have access"),
    ("invalid_grant: Token has been expired or revoked.", "sign-in has expired"),
    ("Too many concurrent aggregations.", "busy"),
    ("Something unusual", "Something unusual"),
])
def test_explain_error(message, expected):
    assert expected in account.explain_error(Exception(message), "p")


# --- Inputs -----------------------------------------------------------------

def test_fit_area_keeps_normal_city():
    bounds, note = inputs.fit_area(engine.CITIES["Kyiv"])
    assert bounds == pytest.approx(engine.CITIES["Kyiv"]) and note is None


def test_fit_area_enlarges_a_point():
    bounds, note = inputs.fit_area((30.5, 50.4, 30.5, 50.4))
    assert inputs.area_km2(bounds) == pytest.approx(16, rel=0.02)
    assert "Enlarged" in note


def test_fit_area_trims_huge_area_to_limit():
    bounds, note = inputs.fit_area((22.0, 44.0, 40.0, 52.0))  # most of Ukraine
    g = engine.Grid.for_bounds(bounds)
    assert g.width * g.height <= inputs.MAX_PIXELS and "trimmed" in note
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
