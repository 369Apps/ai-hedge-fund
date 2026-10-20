"""FDClient contract tests — mocked HTTP, no API key required.

Pins the two Phase 0 guarantees:

1. Fail-loud: infrastructure failures RAISE FDClientError instead of
   silently returning empty (silent empties poison backtests — missing
   data reads as "no signal").
2. Point-in-time: get_financial_metrics returns only periods that were
   public by end_date. The query uses report_period so archive rows with
   no filing_date are not dropped by the server; the client then drops
   anything not yet filed. A fiscal period end alone is not public.
"""

import pytest
import requests

from hedge_fund.data import FDClient, FDClientError
from hedge_fund.features.snapshot import build_snapshot, InsufficientData


class _FakeResponse:
    def __init__(self, status_code=200, payload=None, text=""):
        self.status_code = status_code
        self._payload = payload or {}
        self.text = text

    def json(self):
        return self._payload


@pytest.fixture
def client():
    c = FDClient(api_key="test-key")
    yield c
    c.close()


def _stub(client, responses):
    """Replace the session's request method; each call pops one response.

    A response that is an Exception instance is raised instead.
    """
    calls = []

    def fake_request(method, url, **kwargs):
        calls.append({"method": method, "url": url, **kwargs})
        r = responses.pop(0)
        if isinstance(r, Exception):
            raise r
        return r

    client._session.request = fake_request
    return calls


# ---------------------------------------------------------------------------
# Fail-loud contract
# ---------------------------------------------------------------------------

def test_http_500_raises(client):
    _stub(client, [_FakeResponse(500, text="internal error")])
    with pytest.raises(FDClientError) as exc_info:
        client.get_prices("AAPL", "2024-01-01", "2024-12-31")
    assert exc_info.value.status_code == 500


def test_http_401_raises(client):
    _stub(client, [_FakeResponse(401, text="bad key")])
    with pytest.raises(FDClientError) as exc_info:
        client.get_financial_metrics("AAPL", "2024-12-31")
    assert exc_info.value.status_code == 401


def test_network_error_raises(client):
    _stub(client, [requests.ConnectionError("boom")])
    with pytest.raises(FDClientError):
        client.get_prices("AAPL", "2024-01-01", "2024-12-31")


def test_404_means_no_data_not_failure(client):
    """404 is 'this data does not exist' — a data fact, not a failure."""
    _stub(client, [_FakeResponse(404)])
    assert client.get_financial_metrics("ZZZZ", "2024-12-31") == []


def test_429_retries_then_raises_when_exhausted(client, monkeypatch):
    monkeypatch.setattr("hedge_fund.data.client.time.sleep", lambda s: None)
    _stub(client, [_FakeResponse(429)] * (len(FDClient._RETRY_DELAYS) + 1))
    with pytest.raises(FDClientError) as exc_info:
        client.get_prices("AAPL", "2024-01-01", "2024-12-31")
    assert exc_info.value.status_code == 429


def test_429_then_success_recovers(client, monkeypatch):
    monkeypatch.setattr("hedge_fund.data.client.time.sleep", lambda s: None)
    _stub(client, [
        _FakeResponse(429),
        _FakeResponse(200, {"prices": [{
            "open": 1.0, "close": 2.0, "high": 2.0, "low": 1.0,
            "volume": 100, "time": "2024-01-02",
        }]}),
    ])
    prices = client.get_prices("AAPL", "2024-01-01", "2024-12-31")
    assert len(prices) == 1


# ---------------------------------------------------------------------------
# Point-in-time contract
# ---------------------------------------------------------------------------

def test_financial_metrics_query_keeps_undated_archive_rows(client):
    """filing_date_lte drops every row whose filing was never linked, which
    is most of a long TTM history. Ask by report period instead, and ask
    for a few extra rows so dropping one unfiled quarter still fills `limit`."""
    calls = _stub(client, [_FakeResponse(200, {"financial_metrics": []})])

    client.get_financial_metrics("AAPL", "2024-06-30", period="ttm", limit=4)

    params = calls[0]["params"]
    assert params["report_period_lte"] == "2024-06-30"
    assert params["period"] == "ttm"
    assert params["limit"] > 4
    assert "filing_date_lte" not in params


def _metric_row(report_period, filing_date, market_cap):
    return {
        "ticker": "MU",
        "report_period": report_period,
        "period": "ttm",
        "filing_date": filing_date,
        "market_cap": market_cap,
    }


class _SnapshotClient:
    """FDClient for metrics; facts stay empty so no second HTTP call."""

    def __init__(self, fd: FDClient):
        self._fd = fd

    def get_financial_metrics(self, ticker, end_date, period="ttm", limit=10):
        return self._fd.get_financial_metrics(ticker, end_date, period, limit)

    def get_company_facts(self, ticker):
        return None


def test_undated_archive_periods_count_and_unfiled_ones_do_not(client):
    """The MU failure: three recent TTM rows carry a filing_date and the
    rest of a long history does not. As of 2026-09-19 that used to be
    "only 3 filed periods (need 4)". The undated older rows are public
    history. A filing after as-of, and an undated period that just ended,
    still do not count. Two rows for one period count once, and the later
    public filing wins."""
    rows = [
        _metric_row("2026-08-31", "2026-09-25", 1),  # filed after as-of
        _metric_row("2026-08-01", None, 2),          # ended too recently to assume filed
        _metric_row("2026-05-31", "2026-06-10", 3),  # earlier filing of this period
        _metric_row("2026-05-31", "2026-06-25", 4),  # later public filing wins
        _metric_row("2026-02-28", "2026-03-20", 5),
        _metric_row("2025-11-30", "2025-12-20", 6),
        _metric_row("2025-08-31", None, 7),
        _metric_row("2025-05-31", None, 8),
        _metric_row("2025-05-31", None, 9),          # same undated period, keep the first
    ]
    payload = {"financial_metrics": rows}
    _stub(client, [_FakeResponse(200, payload), _FakeResponse(200, payload)])

    metrics = client.get_financial_metrics("MU", "2026-09-19", period="ttm", limit=20)

    assert [(m.report_period, m.market_cap, m.filing_date) for m in metrics] == [
        ("2026-05-31", 4, "2026-06-25"),
        ("2026-02-28", 5, "2026-03-20"),
        ("2025-11-30", 6, "2025-12-20"),
        ("2025-08-31", 7, None),
        ("2025-05-31", 8, None),
    ]
    snap = build_snapshot("MU", "2026-09-19", _SnapshotClient(client))
    assert [p.report_period for p in snap.periods] == [m.report_period for m in metrics]


def test_three_public_periods_still_raise(client):
    """A name that really has only three public periods still abstains.
    The minimum did not move; the count of what is public did."""
    rows = [
        _metric_row("2026-08-31", "2026-10-01", 1),
        _metric_row("2026-08-01", None, 2),
        _metric_row("2026-05-31", "2026-06-25", 3),
        _metric_row("2026-02-28", "2026-03-20", 4),
        _metric_row("2025-11-30", "2025-12-20", 5),
    ]
    _stub(client, [_FakeResponse(200, {"financial_metrics": rows})])

    with pytest.raises(InsufficientData, match="only 3 filed periods"):
        build_snapshot("MU", "2026-09-19", _SnapshotClient(client))


# ---------------------------------------------------------------------------
# Pagination contract
# ---------------------------------------------------------------------------

def _price_row(day):
    return {
        "open": 1.0, "close": 2.0, "high": 2.0, "low": 1.0,
        "volume": 100, "time": f"2024-01-{day:02d}",
    }


def test_follows_next_page_url_to_the_end(client):
    """The API caps list responses at a fixed page size; the client must
    reassemble the full result by following next_page_url until absent."""
    calls = _stub(client, [
        _FakeResponse(200, {
            "prices": [_price_row(1), _price_row(2)],
            "next_page_url": "https://api.financialdatasets.ai/prices/?cursor=page2",
        }),
        _FakeResponse(200, {
            "prices": [_price_row(3), _price_row(4)],
            "next_page_url": "https://api.financialdatasets.ai/prices/?cursor=page3",
        }),
        _FakeResponse(200, {"prices": [_price_row(5)]}),
    ])

    prices = client.get_prices("AAPL", "2024-01-01", "2024-12-31")

    assert [p.time for p in prices] == [f"2024-01-0{d}" for d in (1, 2, 3, 4, 5)]
    # Pages 2+ request the next_page_url verbatim — no re-derived params.
    assert calls[1]["url"] == "https://api.financialdatasets.ai/prices/?cursor=page2"
    assert calls[2]["url"] == "https://api.financialdatasets.ai/prices/?cursor=page3"
    assert "params" not in calls[1]


def test_no_next_page_url_means_single_request(client):
    calls = _stub(client, [_FakeResponse(200, {"prices": [_price_row(1)]})])
    prices = client.get_prices("AAPL", "2024-01-01", "2024-12-31")
    assert len(prices) == 1
    assert len(calls) == 1


def test_mid_walk_404_keeps_accumulated_rows(client):
    """A 404 on page 2+ ends the stream; rows already fetched are kept."""
    _stub(client, [
        _FakeResponse(200, {
            "prices": [_price_row(1)],
            "next_page_url": "https://api.financialdatasets.ai/prices/?cursor=page2",
        }),
        _FakeResponse(404),
    ])
    prices = client.get_prices("AAPL", "2024-01-01", "2024-12-31")
    assert len(prices) == 1


def test_mid_walk_500_still_fails_loud(client):
    """The fail-loud contract survives pagination: a real failure on any
    page raises instead of silently returning a partial series."""
    _stub(client, [
        _FakeResponse(200, {
            "prices": [_price_row(1)],
            "next_page_url": "https://api.financialdatasets.ai/prices/?cursor=page2",
        }),
        _FakeResponse(500, text="internal error"),
    ])
    with pytest.raises(FDClientError):
        client.get_prices("AAPL", "2024-01-01", "2024-12-31")


def test_snapshot_is_a_single_quote_not_a_list(client):
    calls = _stub(client, [_FakeResponse(200, {"snapshot": {
        "ticker": "BRK.A", "price": 756000.0, "day_change": 1210.0,
        "day_change_percent": 0.16, "time": "2026-10-05T14:28:39Z",
        "time_milliseconds": 1791210519441,
    }})])

    snap = client.get_snapshot("BRK.A")

    assert calls[0]["url"].endswith("/prices/snapshot/")
    assert calls[0]["params"] == {"ticker": "BRK.A"}
    assert snap.price == 756000.0
    assert snap.time == "2026-10-05T14:28:39Z"


def test_snapshot_404_means_unknown_ticker(client):
    _stub(client, [_FakeResponse(404)])
    assert client.get_snapshot("ZZZZ") is None


def test_snapshot_500_fails_loud(client):
    _stub(client, [_FakeResponse(500, text="internal error")])
    with pytest.raises(FDClientError):
        client.get_snapshot("AAPL")


def test_financial_metrics_parses_filing_metadata(client):
    _stub(client, [_FakeResponse(200, {"financial_metrics": [{
        "ticker": "AAPL",
        "report_period": "2024-03-30",
        "period": "quarterly",
        "filing_date": "2024-05-02",
        "filing_datetime": "2024-05-02T16:31:00-04:00",
        "market_cap": 3.0e12,
    }]})])

    m = client.get_financial_metrics("AAPL", "2024-06-30")[0]

    assert m.filing_date == "2024-05-02"
    assert m.filing_datetime == "2024-05-02T16:31:00-04:00"
    assert m.report_period == "2024-03-30"
