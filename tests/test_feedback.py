"""
Unit tests for the feedback module (core/feedback.py)
Uses mocked DB connections (MagicMock) - no real Postgres required.

Usage - python -m pytest tests/test_feedback.py -v --disable-warnings
"""
import json
import pytest
from unittest.mock import MagicMock
from core.feedback import load_weights, load_weight_history, apply_weights, compute_weight_update, save_weights, weights_as_pct
from infrastructure.config import DEFAULT_WEIGHT, LEARNING_RATE, WEIGHT_FLOOR, WEIGHT_CEILING, MODALITIES


# Shared fixtures
@pytest.fixture
def default_weights():
    return {m: DEFAULT_WEIGHT for m in MODALITIES}

@pytest.fixture
def skewed_weights():
    return {"news": 1.2, "chart": 0.8, "timeseries": 1.0, "positioning": 1.1} # custom weights for test

@pytest.fixture
def mock_conn():
    """Returns (conn, cursor) where both are MagicMocks that mimic psycopg2's
    connection/cursor context manager protocol

    - conn.cursor() returns a context manager whose __enter__ yields cursor
    - source and usage https://docs.python.org/3/library/unittest.mock.html
    """
    conn = MagicMock()
    cursor = MagicMock()
    conn.cursor.return_value.__enter__ = MagicMock(return_value=cursor)
    conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
    return conn, cursor

@pytest.fixture
def all_bullish_outputs():
    return {m: {"signal": 1, "confidence": 0.70} for m in MODALITIES}

@pytest.fixture
def mixed_outputs():
    """news+chart bullish, timeseries+positioning bearish."""
    return {
        "news": {"signal": 1, "confidence": 0.78},
        "chart": {"signal": 1, "confidence": 0.65},
        "timeseries": {"signal": -1, "confidence": 0.62},
        "positioning": {"signal": -1, "confidence": 0.60},
    }


# load_weights() function testing
class TestLoadWeights:
    def test_returns_default_when_no_history(self, mock_conn):
        conn, cursor = mock_conn
        cursor.fetchone.return_value = None
        weights = load_weights(conn, "EURUSD")
        assert weights == {m: DEFAULT_WEIGHT for m in MODALITIES}

    def test_returns_stored_weights_when_history_exists(self, mock_conn):
        conn, cursor = mock_conn
        stored = {"news": 1.1, "chart": 0.9, "timeseries": 1.0, "positioning": 1.0}
        cursor.fetchone.return_value = (stored,)
        weights = load_weights(conn, "EURUSD")
        assert weights == stored

    def test_missing_modality_key_backfilled_with_default(self, mock_conn):
        """If a new modality added after weight history written,
        missing key must be filled with DEFAULT_WEIGHT isntead of KeyError.
        """
        conn, cursor = mock_conn
        stored = {"news": 1.1, "chart": 0.9, "timeseries": 1.0} # missing positioning
        cursor.fetchone.return_value = (stored,)
        weights = load_weights(conn, "EURUSD")
        assert "positioning" in weights
        assert weights["positioning"] == DEFAULT_WEIGHT

    def test_all_modality_keys_always_present(self, mock_conn):
        conn, cursor = mock_conn
        cursor.fetchone.return_value = None
        weights = load_weights(conn, "EURUSD")
        for m in MODALITIES:
            assert m in weights

    def test_queries_correct_pair(self, mock_conn):
        """load_weights must pass the pair to the SQL query."""
        conn, cursor = mock_conn
        cursor.fetchone.return_value = None
        load_weights(conn, "GBPUSD")
        execute_args = cursor.execute.call_args
        assert "GBPUSD" in execute_args.args[1] or "GBPUSD" in str(execute_args)


# apply_weights() function testing
class TestApplyWeights:
    def test_returns_new_dict_does_not_mutate(self, default_weights):
        """apply_weights must not mutate the input modality_outputs dict."""
        outputs = {"news": {"signal": 1, "confidence": 0.72}}
        original_conf = outputs["news"]["confidence"]
        apply_weights(outputs, default_weights)
        assert outputs["news"]["confidence"] == original_conf

    def test_effective_confidence_equals_raw_times_weight(self):
        outputs = {"news": {"signal": 1, "confidence": 0.80}}
        weights = {"news": 1.25}
        result = apply_weights(outputs, weights)
        expected = round(min(0.80 * 1.25, 1.0), 4)
        assert result["news"]["confidence"] == expected

    def test_effective_confidence_clamped_to_one(self):
        outputs = {"news": {"signal": 1, "confidence": 0.95}}
        weights = {"news": 2.0}
        result = apply_weights(outputs, weights)
        assert result["news"]["confidence"] <= 1.0

    def test_raw_confidence_preserved(self):
        outputs = {"news": {"signal": 1, "confidence": 0.72}}
        weights = {"news": 0.85}
        result = apply_weights(outputs, weights)
        assert result["news"]["raw_confidence"] == 0.72

    def test_weight_key_added_to_output(self):
        outputs = {"news": {"signal": 1, "confidence": 0.72}}
        weights = {"news": 0.85}
        result = apply_weights(outputs, weights)
        assert result["news"]["weight"] == 0.85

    def test_default_weight_is_transparent(self, default_weights):
        """With all weights at 1.0, effective confidence equals raw confidence."""
        outputs = {
            "news": {"signal": 1, "confidence": 0.72},
            "chart": {"signal": -1, "confidence": 0.55},
            "timeseries": {"signal": 1, "confidence": 0.60},
            "positioning": {"signal": 1, "confidence": 0.62},
        }
        result = apply_weights(outputs, default_weights)
        for mod, data in outputs.items():
            assert result[mod]["confidence"] == data["confidence"]

    def test_all_modalities_processed(self, default_weights):
        outputs = {
            "news": {"signal": 1, "confidence": 0.72},
            "chart": {"signal": -1, "confidence": 0.35},
            "timeseries": {"signal": 1, "confidence": 0.61},
            "positioning": {"signal": 1, "confidence": 0.62},
        }
        result = apply_weights(outputs, default_weights)
        assert set(result.keys()) == set(outputs.keys())

    def test_signal_unchanged_by_weighting(self):
        """Weighting must only affect confidence, not the signal direction."""
        outputs = {"news": {"signal": -1, "confidence": 0.72}}
        weights = {"news": 0.5}
        result = apply_weights(outputs, weights)
        assert result["news"]["signal"] == -1

    def test_missing_weight_uses_default(self):
        """If a modality has no entry in weights dict, DEFAULT_WEIGHT is used."""
        outputs = {"news": {"signal": 1, "confidence": 0.72}}
        result = apply_weights(outputs, {})
        assert result["news"]["confidence"] == round(0.72 * DEFAULT_WEIGHT, 4)


# compute_weight_update() function testing
class TestComputeWeightUpdate:
    # No update cases / neutral
    def test_uncertain_outcome_returns_none(self, default_weights, all_bullish_outputs):
        result = compute_weight_update(default_weights, all_bullish_outputs, "uncertain", "bullish")
        assert result is None

    def test_unanimous_wrong_returns_none(self, default_weights, all_bullish_outputs):
        """All modalities agreed bullish, outcome incorrect - no update."""
        result = compute_weight_update(default_weights, all_bullish_outputs, "incorrect", "bullish")
        assert result is None

    def test_all_neutral_signals_preserves_default_weights(self, default_weights):
        """All neutral signals - nothing to attribute, jsut preserve same existing weights."""
        neutral_outputs = {m: {"signal": 0, "confidence": 0.5} for m in MODALITIES}
        result = compute_weight_update(default_weights, neutral_outputs, "correct", "neutral")
        assert result == default_weights

    # Correct outcome result
    def test_correct_outcome_agreeing_modality_boosted(self, default_weights):
        """All bullish + correct outcome - all four should be boosted."""
        outputs = {m: {"signal": 1, "confidence": 0.7} for m in MODALITIES}
        # Need at least one dissenter so its not the 'unanimous' case, but for correct outcome unanimous is fine (so all boosted)
        result = compute_weight_update(default_weights, outputs, "correct", "bullish")
        assert result is not None
        for mod in MODALITIES:
            assert result[mod] == round(DEFAULT_WEIGHT + LEARNING_RATE, 3)

    def test_correct_outcome_dissenter_unchanged(self, default_weights, mixed_outputs):
        """news+chart bullish, ts+pos bearish, stance bullish, outcome correct.
        
        - agreers (news, chart) boosted, dissenters (ts, pos) unchanged
        """
        result = compute_weight_update(default_weights, mixed_outputs, "correct", "bullish")
        assert result is not None
        assert result["news"] == round(DEFAULT_WEIGHT + LEARNING_RATE, 3)
        assert result["chart"] == round(DEFAULT_WEIGHT + LEARNING_RATE, 3)
        assert result["timeseries"] == DEFAULT_WEIGHT
        assert result["positioning"] == DEFAULT_WEIGHT

    # Incorrect outcome
    def test_incorrect_outcome_agreeing_modality_penalised(self, default_weights):
        """3 bullish agree, 1 bearish dissents, stance bullish, outcome incorrect."""
        outputs = {
            "news": {"signal": 1, "confidence": 0.78},
            "chart": {"signal": 1, "confidence": 0.65},
            "timeseries": {"signal": 1, "confidence": 0.60},
            "positioning": {"signal": -1, "confidence": 0.62},
        }
        result = compute_weight_update(default_weights, outputs, "incorrect", "bullish")
        assert result is not None
        assert result["news"] == round(DEFAULT_WEIGHT - LEARNING_RATE, 3)
        assert result["chart"] == round(DEFAULT_WEIGHT - LEARNING_RATE, 3)
        assert result["timeseries"] == round(DEFAULT_WEIGHT - LEARNING_RATE, 3)
        assert result["positioning"] == round(DEFAULT_WEIGHT + LEARNING_RATE, 3)

    def test_floor_enforced_on_penalty(self):
        """Weights near floor must not drop below WEIGHT_FLOOR."""
        near_floor = {m: WEIGHT_FLOOR + 0.05 for m in MODALITIES}
        outputs = {
            "news": {"signal": 1, "confidence": 0.7},
            "chart": {"signal": 1, "confidence": 0.7},
            "timeseries": {"signal": 1, "confidence": 0.7},
            "positioning": {"signal": -1, "confidence": 0.7}, # dissenter
        }
        result = compute_weight_update(near_floor, outputs, "incorrect", "bullish")
        assert result is not None
        for mod in ["news", "chart", "timeseries"]:
            assert result[mod] >= WEIGHT_FLOOR

    def test_ceiling_enforced_on_boost(self):
        """Weights near ceiling must not exceed WEIGHT_CEILING."""
        near_ceil = {m: WEIGHT_CEILING - 0.05 for m in MODALITIES}
        outputs = {m: {"signal": 1, "confidence": 0.7} for m in MODALITIES}
        result = compute_weight_update(near_ceil, outputs, "correct", "bullish")
        assert result is not None
        for mod in MODALITIES:
            assert result[mod] <= WEIGHT_CEILING

    def test_neutral_signal_modality_unchanged(self, default_weights):
        """A modality with signal=0 must not be updated regardless of outcome."""
        outputs = {
            "news": {"signal": 1, "confidence": 0.78},
            "chart": {"signal": 1, "confidence": 0.65},
            "timeseries": {"signal": 0, "confidence": 0.50},
            "positioning": {"signal": -1, "confidence": 0.62},
        }
        result = compute_weight_update(
            default_weights, outputs, "incorrect", "bullish"
        )
        assert result is not None
        assert result["timeseries"] == DEFAULT_WEIGHT

    def test_returns_dict_with_all_modality_keys(self, default_weights):
        outputs = {
            "news": {"signal": 1, "confidence": 0.78},
            "chart": {"signal": 1, "confidence": 0.65},
            "timeseries": {"signal": 1, "confidence": 0.60},
            "positioning": {"signal": -1, "confidence": 0.62},
        }
        result = compute_weight_update(
            default_weights, outputs, "correct", "bullish"
        )
        assert result is not None
        for m in MODALITIES:
            assert m in result

    def test_does_not_mutate_input_weights(self, default_weights):
        """compute_weight_update must not modify the weights dict it receives."""
        original = dict(default_weights)
        outputs = {
            "news": {"signal": 1, "confidence": 0.78},
            "chart": {"signal": 1, "confidence": 0.65},
            "timeseries": {"signal": 1, "confidence": 0.60},
            "positioning": {"signal": -1, "confidence": 0.62},
        }
        compute_weight_update(default_weights, outputs, "correct", "bullish")
        assert default_weights == original

    def test_learning_rate_applied_correctly(self, default_weights):
        """Delta must equal exactly LEARNING_RATE, not an approximation."""
        outputs = {
            "news": {"signal": 1, "confidence": 0.78},
            "chart": {"signal": 1, "confidence": 0.65},
            "timeseries": {"signal": 1, "confidence": 0.60},
            "positioning": {"signal": -1, "confidence": 0.62},
        }
        result = compute_weight_update(default_weights, outputs, "correct", "bullish")
        assert result is not None
        delta = round(result["news"] - default_weights["news"], 6)
        assert abs(delta - LEARNING_RATE) < 1e-6


# save_weights() function testing
class TestSaveWeights:
    def test_inserts_row_with_correct_feedback_id(self, mock_conn, default_weights):
        conn, cursor = mock_conn
        save_weights(conn, feedback_id=42, weights=default_weights)
        execute_args = cursor.execute.call_args
        assert 42 in execute_args.args[1] or 42 in str(execute_args)

    def test_weights_serialised_as_json(self, mock_conn, skewed_weights):
        """Weights must be JSON-serialised for insertion into the JSONB column."""
        conn, cursor = mock_conn
        save_weights(conn, feedback_id=1, weights=skewed_weights)
        execute_args = cursor.execute.call_args
        params = execute_args.args[1]
        json_param = params[1]
        parsed = json.loads(json_param)
        assert parsed == skewed_weights

    def test_execute_called_once(self, mock_conn, default_weights):
        conn, cursor = mock_conn
        save_weights(conn, feedback_id=1, weights=default_weights)
        cursor.execute.assert_called_once()


# load_weight_history() funciton testing
class TestLoadWeightHistory:
    def test_returns_list(self, mock_conn):
        conn, cursor = mock_conn
        cursor.fetchall.return_value = []
        history = load_weight_history(conn, "EURUSD")
        assert isinstance(history, list)

    def test_returns_empty_list_when_no_history(self, mock_conn):
        conn, cursor = mock_conn
        cursor.fetchall.return_value = []
        history = load_weight_history(conn, "EURUSD")
        assert history == []

    def test_each_entry_has_expected_keys(self, mock_conn):
        from datetime import datetime, timezone
        conn, cursor = mock_conn
        fake_weights = {"news": 1.1, "chart": 0.9, "timeseries": 1.0, "positioning": 1.0}
        cursor.fetchall.return_value = [
            (datetime(2025, 6, 1, tzinfo=timezone.utc), fake_weights, 7),
            (datetime(2025, 6, 2, tzinfo=timezone.utc), fake_weights, 8),
        ]
        history = load_weight_history(conn, "EURUSD")
        assert len(history) == 2
        for entry in history:
            assert "logged_at" in entry
            assert "weights" in entry
            assert "feedback_id" in entry

    def test_limit_passed_to_query(self, mock_conn):
        conn, cursor = mock_conn
        cursor.fetchall.return_value = []
        load_weight_history(conn, "EURUSD", limit=5)
        execute_args = cursor.execute.call_args
        assert 5 in execute_args.args[1] or 5 in str(execute_args)

    def test_queries_correct_pair(self, mock_conn):
        conn, cursor = mock_conn
        cursor.fetchall.return_value = []
        load_weight_history(conn, "GBPUSD", limit=10)
        execute_args = cursor.execute.call_args
        assert "GBPUSD" in execute_args.args[1] or "GBPUSD" in str(execute_args)


# weights_as_pct() function testing
class TestWeightsAsPct:
    def test_equal_weights_give_25_percent_each(self, default_weights):
        pcts = weights_as_pct(default_weights)
        for mod in MODALITIES:
            assert pcts[mod] == 25

    def test_all_keys_present(self, default_weights):
        pcts = weights_as_pct(default_weights)
        for mod in MODALITIES:
            assert mod in pcts

    def test_percentages_sum_to_100(self, skewed_weights):
        pcts = weights_as_pct(skewed_weights)
        assert abs(sum(pcts.values()) - 100) <= 1

    def test_higher_weight_gives_higher_percentage(self):
        weights = {"news": 2.0, "chart": 1.0, "timeseries": 1.0, "positioning": 1.0}
        pcts = weights_as_pct(weights)
        assert pcts["news"] > pcts["chart"]

    def test_zero_total_returns_equal_split(self):
        """Guards against division by zero if all weights are 0."""
        zero_weights = {m: 0.0 for m in MODALITIES}
        pcts = weights_as_pct(zero_weights)
        for mod in MODALITIES:
            assert pcts[mod] == 25

    def test_returns_integer_values(self, default_weights):
        pcts = weights_as_pct(default_weights)
        for v in pcts.values():
            assert isinstance(v, int)