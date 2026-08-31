"""
Unit tests for the positioning modality (positioning.py)
Uses real MyFxBook API calls where credentials are available.
Tests requiring API access are skipped gracefully if .env is not configured.

Usage - python -m pytest tests/test_timeseries.py -v --disable-warnings
"""

import os
import pytest
from dotenv import load_dotenv
from modalities.positioning import login, logout, fetch_community_outlook, extract_pair_data, apply_contrarian_rule, get_positioning_signal
from infrastructure.config import CONTRARIAN_THRESHOLD

load_dotenv()

# Skip markers - used on any test that requires live API credentials
credentials_available = bool(
    os.getenv("MYFXBOOK_EMAIL") and os.getenv("MYFXBOOK_PASSWORD")
)

requires_api = pytest.mark.skipif(
    not credentials_available,
    reason="MYFXBOOK_EMAIL and MYFXBOOK_PASSWORD not set in .env. Skipping live API tests."
)


# Module-level fixture
# * log in once and reuse session across API tests
@pytest.fixture(scope="module")
def session():
    if not credentials_available:
        pytest.skip("No credentials.")
    sid = login()
    if not sid:
        pytest.skip("Login failed - check credentials in .env.")
    yield sid
    logout(sid)

@pytest.fixture(scope="module")
def all_symbols(session):
    """Fetch full community outlook once and reuse across extraction tests."""
    symbols = fetch_community_outlook(session)
    if not symbols:
        pytest.skip("No symbols returned from MyFxBook API.")
    return symbols



# apply_contrarian_rule() function test
# * pure logic, no API needed
class TestApplyContrarianRule:
    """Fully deterministic, no API calls."""
    def test_65_short_produces_bullish_signal(self):
        """65% short exceeds threshold -> contrarian bullish -> signal +1."""
        result = apply_contrarian_rule({"shortPercentage": 65, "longPercentage": 35})
        assert result["signal"] == 1

    def test_65_long_produces_bearish_signal(self):
        """65% long exceeds threshold -> contrarian bearish -> signal -1."""
        result = apply_contrarian_rule({"shortPercentage": 35, "longPercentage": 65})
        assert result["signal"] == -1

    def test_at_threshold_short_is_bullish(self):
        """Exactly at CONTRARIAN_THRESHOLD (60%) should trigger the signal."""
        result = apply_contrarian_rule({"shortPercentage": CONTRARIAN_THRESHOLD, "longPercentage": 40})
        assert result["signal"] == 1

    def test_below_threshold_is_neutral(self):
        """Both sides below threshold -> no extreme -> signal 0."""
        result = apply_contrarian_rule({"shortPercentage": 55, "longPercentage": 45})
        assert result["signal"] == 0

    def test_balanced_50_50_is_neutral(self):
        result = apply_contrarian_rule({"shortPercentage": 50, "longPercentage": 50})
        assert result["signal"] == 0

    def test_confidence_scales_with_percentage(self):
        """Confidence is percentage/100 - higher extreme = higher confidence."""
        low = apply_contrarian_rule({"shortPercentage": 61, "longPercentage": 39})
        high = apply_contrarian_rule({"shortPercentage": 85, "longPercentage": 15})
        assert high["confidence"] > low["confidence"]

    def test_confidence_in_range(self):
        result = apply_contrarian_rule({"shortPercentage": 70, "longPercentage": 30})
        assert 0.0 <= result["confidence"] <= 1.0

    def test_returns_required_keys(self):
        result = apply_contrarian_rule({"shortPercentage": 65, "longPercentage": 35})
        for key in ["signal", "confidence", "long_pct", "short_pct", "note"]:
            assert key in result

    def test_note_is_string(self):
        result = apply_contrarian_rule({"shortPercentage": 65, "longPercentage": 35})
        assert isinstance(result["note"], str)
        assert len(result["note"]) > 0

    def test_missing_keys_use_defaults(self):
        """If percentages missing from API response, should default to 50/50 -> neutral."""
        result = apply_contrarian_rule({})
        assert result["signal"] == 0


# extract_pair_data() function testing
# * pure logic on a mock symbols list, no API needed
class TestExtractPairData:
    MOCK_SYMBOLS = [
        {"name": "EURUSD", "shortPercentage": 62, "longPercentage": 38},
        {"name": "GBPUSD", "shortPercentage": 45, "longPercentage": 55},
        {"name": "USDJPY", "shortPercentage": 70, "longPercentage": 30},
    ]

    def test_extracts_correct_pair(self):
        result = extract_pair_data(self.MOCK_SYMBOLS, "EURUSD")
        assert result is not None
        assert result["name"] == "EURUSD"

    def test_case_insensitive_match(self):
        """MyFxBook may return names in mixed case - matching should be case-insensitive."""
        symbols = [{"name": "eurusd", "shortPercentage": 60, "longPercentage": 40}]
        result = extract_pair_data(symbols, "EURUSD")
        assert result is not None

    def test_missing_pair_returns_none(self):
        result = extract_pair_data(self.MOCK_SYMBOLS, "AUDUSD")
        assert result is None

    def test_empty_symbols_returns_none(self):
        result = extract_pair_data([], "EURUSD")
        assert result is None


# Live API tests 
# * if credential not set just skip
class TestLoginLogout:
    @requires_api
    def test_login_returns_session_string(self):
        sid = login()
        assert isinstance(sid, str)
        assert len(sid) > 0
        logout(sid)

    @requires_api
    def test_logout_does_not_raise(self, session):
        # Session fixture handles login/logout - just confirm no exception
        assert session is not None


# fetch_community_outlook() function testing
class TestFetchCommunityOutlook:
    @requires_api
    def test_returns_list(self, session):
        symbols = fetch_community_outlook(session)
        assert isinstance(symbols, list)

    @requires_api
    def test_non_empty(self, all_symbols):
        assert len(all_symbols) > 0

    @requires_api
    def test_each_symbol_has_percentages(self, all_symbols):
        """Every symbol dict should have shortPercentage and longPercentage."""
        for symbol in all_symbols:
            assert "shortPercentage" in symbol, f"Missing shortPercentage in {symbol}"
            assert "longPercentage" in symbol, f"Missing longPercentage in {symbol}"

    @requires_api
    def test_percentages_sum_to_100(self, all_symbols):
        """For each symbol, long + short should sum to 100 (or very close due to rounding)."""
        for symbol in all_symbols:
            total = symbol["shortPercentage"] + symbol["longPercentage"]
            if total == 0:
                continue # no positioning data available
            assert abs(total - 100) <= 1, (
                f"{symbol['name']}: long+short={total}, expected ~100"
            )



# get_positioning_signal() Integration testing
# * end-to-end for the modality
class TestGetPositioningSignal:
    @requires_api
    def test_returns_required_keys(self):
        result = get_positioning_signal("EURUSD")
        for key in ["signal", "confidence", "long_pct", "short_pct", "note"]:
            assert key in result

    @requires_api
    def test_signal_is_valid(self):
        result = get_positioning_signal("EURUSD")
        assert result["signal"] in {-1, 0, 1}

    @requires_api
    def test_confidence_in_range(self):
        result = get_positioning_signal("EURUSD")
        assert 0.0 <= result["confidence"] <= 1.0

    @requires_api
    def test_contrarian_signal_correct_for_three_pairs(self):
        """confirm contrarian signal matches expected direction, 
        for at least 3 different pairs based on live data.
        """
        for pair in ["EURUSD", "GBPUSD", "USDJPY"]:
            result = get_positioning_signal(pair)
            long_pct = result["long_pct"]
            short_pct = result["short_pct"]

            if short_pct is None or long_pct is None:
                continue # API fallback - skip this pair

            if short_pct >= CONTRARIAN_THRESHOLD:
                assert result["signal"] == 1, (
                    f"{pair}: {short_pct}% short should produce bullish signal"
                )
            elif long_pct >= CONTRARIAN_THRESHOLD:
                assert result["signal"] == -1, (
                    f"{pair}: {long_pct}% long should produce bearish signal"
                )
            else:
                assert result["signal"] == 0, (
                    f"{pair}: no extreme positioning should produce neutral signal"
                )

    def test_fallback_on_bad_credentials(self, monkeypatch):
        """If env vars are missing, should return neutral fallback without crashing."""
        monkeypatch.delenv("MYFXBOOK_EMAIL", raising=False)
        monkeypatch.delenv("MYFXBOOK_PASSWORD", raising=False)
        result = get_positioning_signal("EURUSD")
        assert result["signal"] == 0
        assert result["confidence"] == 0.0