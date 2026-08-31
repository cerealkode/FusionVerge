"""
Unit tests for the sentiment modality (sentiment.py)
Uses real FinBERT inference and real RSS feeds.
FinBERT is loaded once at module level to avoid reloading per test.

Usage - python -m pytest tests/test_sentiment.py -v --disable-warnings
"""

import pytest
from infrastructure.config import CURRENCY_KEYWORDS
from modalities.sentiment import load_sentiment_model, fetch_headlines, classify_headlines, detect_currency_focus, apply_currency_adjustment, aggregate_signal, get_sentiment_signal

pytestmark = pytest.mark.filterwarnings(
    "ignore:.*WordPiece.__init__.*:DeprecationWarning"
)

# Module-level fixture
# * load FinBERT model ONCE for the entire test session,
#   instead of repeated loading for each tests
@pytest.fixture(scope="module")
def finbert():
    return load_sentiment_model()


# fetch_headlines() function testing
class TestFetchHeadlines:
    def test_returns_list_for_active_pair(self):
        """EURUSD is an active pair - should return at least some headlines."""
        headlines = fetch_headlines("EURUSD")
        assert isinstance(headlines, list)

    def test_headlines_are_strings(self):
        """Headlines should be in String format"""
        headlines = fetch_headlines("EURUSD")
        for h in headlines:
            assert isinstance(h, str)
            assert len(h) > 0

    def test_filtered_headlines_contain_keyword(self):
        """Every returned headline must contain at least one configured keyword."""
        pair = "EURUSD"
        keywords = [kw.lower() for kw in CURRENCY_KEYWORDS["EUR"] + CURRENCY_KEYWORDS["USD"]]
        headlines = fetch_headlines(pair)
        for h in headlines:
            assert any(kw in h.lower() for kw in keywords), (
                f"Headline did not match any keyword: '{h}'"
            )

    def test_unknown_pair_returns_empty(self):
        """A pair with no configured keywords should return empty list."""
        headlines = fetch_headlines("XAUUSD")
        assert headlines == []

    def test_max_headlines_caps_result(self):
        """max_headlines should hard-cap the returned list length."""
        headlines = fetch_headlines("EURUSD", max_headlines=5)
        assert len(headlines) <= 5

    def test_max_headlines_zero_returns_empty(self):
        """Requesting max_headlines=0 should return empty list."""
        headlines = fetch_headlines("EURUSD", max_headlines=0)
        assert headlines == []


# classify_headlines() function testing (FinBERT inference)
class TestClassifyHeadlines:
    def test_output_structure(self, finbert):
        """Each result must have headline, label, confidence keys."""
        headlines = ["EUR strengthens as ECB signals rate hike"]
        results = classify_headlines(finbert, headlines)
        assert len(results) == 1
        r = results[0]
        assert "headline" in r
        assert "label" in r
        assert "confidence" in r

    def test_label_is_valid(self, finbert):
        """Label must be one of the three FinBERT classes."""
        headlines = [
            "Dollar rallies on strong US jobs data",
            "Euro falls amid eurozone recession fears",
            "Fed holds rates steady at policy meeting",
        ]
        results = classify_headlines(finbert, headlines)
        for r in results:
            assert r["label"] in {"positive", "negative", "neutral"}, (
                f"Unexpected label: {r['label']}"
            )

    def test_confidence_in_range(self, finbert):
        """Confidence must be a float between 0 and 1 inclusive."""
        headlines = ["USD gains ground after strong retail sales"]
        results = classify_headlines(finbert, headlines)
        for r in results:
            assert isinstance(r["confidence"], float)
            assert 0.0 <= r["confidence"] <= 1.0

    def test_strongly_positive_headline(self, finbert):
        """A clearly positive financial headline should be labelled positive."""
        headlines = ["Markets surge to record highs on strong economic growth"]
        results = classify_headlines(finbert, headlines)
        assert results[0]["label"] == "positive"

    def test_strongly_negative_headline(self, finbert):
        """A clearly negative financial headline should be labelled negative."""
        headlines = ["Economy crashes into recession as unemployment spikes"]
        results = classify_headlines(finbert, headlines)
        assert results[0]["label"] == "negative"

    def test_empty_input_returns_empty(self, finbert):
        results = classify_headlines(finbert, [])
        assert results == []

    def test_one_result_per_headline(self, finbert):
        headlines = ["EUR rises", "USD falls", "JPY steady"]
        results = classify_headlines(finbert, headlines)
        assert len(results) == len(headlines)


# detect_currency_focus() function testing
class TestDetectCurrencyFocus:
    def test_eur_headline_detected_as_eur(self):
        focus = detect_currency_focus("ECB raises rates as eurozone inflation rises", "EURUSD")
        assert focus == "EUR"

    def test_usd_headline_detected_as_usd(self):
        focus = detect_currency_focus("Fed holds rates, dollar strengthens on greenback demand", "EURUSD")
        assert focus == "USD"

    def test_both_currencies_resolves_neutral(self):
        # Equal mentions of EUR and USD keywords should resolve to neutral
        focus = detect_currency_focus("EUR and USD both react to global trade data", "EURUSD")
        assert focus == "neutral"

    def test_unrelated_headline_is_neutral(self):
        focus = detect_currency_focus("Gold prices hit all-time high on safe haven demand", "EURUSD")
        assert focus == "neutral"

    def test_gbp_headline_for_gbpusd(self):
        focus = detect_currency_focus("Bank of England raises rates, pound sterling surges", "GBPUSD")
        assert focus == "GBP"

    def test_jpy_headline_for_usdjpy(self):
        focus = detect_currency_focus("BOJ intervenes to support yen amid sharp selloff", "USDJPY")
        assert focus == "JPY"

    def test_cad_headline_for_usdcad(self):
        focus = detect_currency_focus("Oil prices surge, loonie gains against dollar", "USDCAD")
        assert focus == "CAD"


# apply_currency_adjustment() function testing
class TestApplyCurrencyAdjustment:
    """
    Tests the directional mapping: FinBERT label + currency focus -> pair signal.
    Core logic: positive base = +1, positive quote = -1, negatives flip.
    """
    def _make_classified(self, headline, label, confidence=0.9):
        return [{"headline": headline, "label": label, "confidence": confidence}]

    def test_positive_usd_is_bearish_eurusd(self):
        """Positive USD sentiment -> USD strengthens -> EURUSD falls -> signal -1."""
        classified = self._make_classified(
            "Fed raises rates, dollar surges on greenback demand", "positive"
        )
        adjusted = apply_currency_adjustment(classified, "EURUSD")
        assert adjusted[0]["signal"] == -1

    def test_positive_eur_is_bullish_eurusd(self):
        """Positive EUR sentiment -> EUR strengthens -> EURUSD rises -> signal +1."""
        classified = self._make_classified(
            "ECB signals aggressive rate hikes, euro surges", "positive"
        )
        adjusted = apply_currency_adjustment(classified, "EURUSD")
        assert adjusted[0]["signal"] == 1

    def test_negative_eur_is_bearish_eurusd(self):
        """Negative EUR sentiment -> EUR weakens -> EURUSD falls -> signal -1."""
        classified = self._make_classified(
            "Eurozone economy contracts sharply, euro falls", "negative"
        )
        adjusted = apply_currency_adjustment(classified, "EURUSD")
        assert adjusted[0]["signal"] == -1

    def test_positive_gbp_is_bullish_gbpusd(self):
        """Positive GBP -> cable rises -> signal +1."""
        classified = self._make_classified(
            "Bank of England raises rates, pound sterling surges", "positive"
        )
        adjusted = apply_currency_adjustment(classified, "GBPUSD")
        assert adjusted[0]["signal"] == 1

    def test_positive_jpy_is_bearish_usdjpy(self):
        """Positive JPY -> yen strengthens -> USDJPY falls -> signal -1."""
        classified = self._make_classified(
            "BOJ tightens policy, yen strengthens sharply", "positive"
        )
        adjusted = apply_currency_adjustment(classified, "USDJPY")
        assert adjusted[0]["signal"] == -1

    def test_positive_usd_is_bullish_usdjpy(self):
        """Positive USD -> USD strengthens -> USDJPY rises -> signal +1."""
        classified = self._make_classified(
            "Fed raises rates, dollar surges on greenback demand", "positive"
        )
        adjusted = apply_currency_adjustment(classified, "USDJPY")
        assert adjusted[0]["signal"] == 1

    def test_neutral_label_gives_zero_signal(self):
        classified = self._make_classified(
            "Fed holds rates steady, markets await guidance", "neutral"
        )
        adjusted = apply_currency_adjustment(classified, "EURUSD")
        assert adjusted[0]["signal"] == 0

    def test_unrelated_headline_gives_zero_signal(self):
        """Headline about gold has no EUR/USD keywords - should resolve neutral focus -> 0."""
        classified = self._make_classified(
            "Gold prices hit record high on safe haven demand", "positive"
        )
        adjusted = apply_currency_adjustment(classified, "EURUSD")
        assert adjusted[0]["signal"] == 0

    def test_signal_key_present_on_all_items(self):
        classified = [
            {"headline": "ECB hikes rates", "label": "positive", "confidence": 0.85},
            {"headline": "Dollar steady", "label": "neutral", "confidence": 0.72},
        ]
        adjusted = apply_currency_adjustment(classified, "EURUSD")
        for item in adjusted:
            assert "signal" in item


# aggregate_signal() function testing
class TestAggregateSignal:
    def test_empty_input_returns_neutral(self):
        result = aggregate_signal([])
        assert result["signal"] == 0
        assert result["confidence"] == 0.0

    def test_bullish_majority_returns_positive_signal(self):
        adjusted = [
            {"label": "positive", "confidence": 0.9, "signal": 1},
            {"label": "positive", "confidence": 0.85, "signal": 1},
            {"label": "negative", "confidence": 0.5, "signal": -1},
        ]
        result = aggregate_signal(adjusted)
        assert result["signal"] == 1

    def test_bearish_majority_returns_negative_signal(self):
        adjusted = [
            {"label": "negative", "confidence": 0.9, "signal": -1},
            {"label": "negative", "confidence": 0.88, "signal": -1},
            {"label": "positive", "confidence": 0.5, "signal": 1},
        ]
        result = aggregate_signal(adjusted)
        assert result["signal"] == -1

    def test_balanced_signals_return_neutral(self):
        """Equal-weight bull and bear should produce weighted_avg near 0 -> signal 0."""
        adjusted = [
            {"label": "positive", "confidence": 0.8, "signal": 1},
            {"label": "negative", "confidence": 0.8, "signal": -1},
        ]
        result = aggregate_signal(adjusted)
        assert result["signal"] == 0

    def test_distribution_counts_labels(self):
        adjusted = [
            {"label": "positive", "confidence": 0.9, "signal": 1},
            {"label": "positive", "confidence": 0.8, "signal": 1},
            {"label": "negative", "confidence": 0.7, "signal": -1},
            {"label": "neutral", "confidence": 0.6, "signal": 0},
        ]
        result = aggregate_signal(adjusted)
        assert result["distribution"]["positive"] == 2
        assert result["distribution"]["negative"] == 1
        assert result["distribution"]["neutral"] == 1

    def test_confidence_is_float_in_range(self):
        adjusted = [{"label": "positive", "confidence": 0.75, "signal": 1}]
        result = aggregate_signal(adjusted)
        assert isinstance(result["confidence"], float)
        assert 0.0 <= result["confidence"] <= 1.0

    def test_output_has_required_keys(self):
        adjusted = [{"label": "positive", "confidence": 0.8, "signal": 1}]
        result = aggregate_signal(adjusted)
        assert "signal" in result
        assert "confidence" in result
        assert "distribution" in result


# get_sentiment_signal() Integration testing
# * end-to-end for the modality
class TestGetSentimentSignal:
    def test_returns_required_keys(self, finbert):
        result = get_sentiment_signal("EURUSD", pipe=finbert)
        assert "signal" in result
        assert "confidence" in result
        assert "distribution" in result

    def test_signal_is_valid(self, finbert):
        result = get_sentiment_signal("EURUSD", pipe=finbert)
        assert result["signal"] in {-1, 0, 1}

    def test_confidence_in_range(self, finbert):
        result = get_sentiment_signal("EURUSD", pipe=finbert)
        assert 0.0 <= result["confidence"] <= 1.0

    def test_distribution_has_all_labels(self, finbert):
        result = get_sentiment_signal("EURUSD", pipe=finbert)
        assert "positive" in result["distribution"]
        assert "negative" in result["distribution"]
        assert "neutral" in result["distribution"]

    def test_unknown_pair_returns_neutral(self, finbert):
        """Pair with no keywords should return neutral fallback without crashing."""
        result = get_sentiment_signal("XAUUSD", pipe=finbert)
        assert result["signal"] == 0
        assert result["confidence"] == 0.0

    def test_max_headlines_respected(self, finbert):
        """Passing max_headlines=5 should still return a valid signal dict."""
        result = get_sentiment_signal("EURUSD", pipe=finbert, max_headlines=5)
        assert result["signal"] in {-1, 0, 1}