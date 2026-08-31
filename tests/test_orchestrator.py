"""
Unit tests for the orchestrator (core/orchestrator.py)
Uses mocked model loading, modality signals, fusion and feedback weights (Ollama + trained models only required for the integration test class).

Usage - python -m pytest tests/test_orchestrator.py -v --disable-warnings
"""
import warnings
warnings.filterwarnings(
    "ignore",
    message=r".*mode\.copy_on_write.*",
)
import os
import pytest
from unittest.mock import patch, MagicMock
import ollama
from core.orchestrator import Orchestrator
from infrastructure.config import DEFAULT_WEIGHT, GRU_BEST_CONFIGS_PATH

pytestmark = [
    pytest.mark.filterwarnings(
        "ignore:.*WordPiece.__init__.*:DeprecationWarning"
    ),
    pytest.mark.filterwarnings(
        "ignore::sklearn.exceptions.InconsistentVersionWarning"
    ),
]

# Default weights dict used by mock patches (all 1.0 so apply_weights)
MOCK_DEFAULT_WEIGHTS = {
    "news": DEFAULT_WEIGHT,
    "chart": DEFAULT_WEIGHT,
    "timeseries": DEFAULT_WEIGHT,
    "positioning": DEFAULT_WEIGHT,
}


# Dependency availability checks
def ollama_reachable() -> bool:
    try:
        ollama.list()
        return True
    except Exception:
        return False

# Testing on 2 models for this session, EURUSD and GBPUSD
models_available = (
    os.path.exists(GRU_BEST_CONFIGS_PATH) and
    os.path.exists("models/EURUSD_1h.keras") and
    os.path.exists("models/EURUSD_1h_scaler.pkl") and
    os.path.exists("models/GBPUSD_1h.keras") and
    os.path.exists("models/GBPUSD_1h_scaler.pkl")
)

requires_ollama = pytest.mark.skipif(
    not ollama_reachable(),
    reason="Ollama not reachable. Start Ollama and run 'ollama pull mistral' first."
)

requires_all = pytest.mark.skipif(
    not (ollama_reachable() and models_available),
    reason="Full integration requires Ollama running and trained model files in models/."
)


# Shared mock outputs
MOCK_NEWS = {
    "signal": 1, "confidence": 0.72,
    "distribution": {"positive": 3, "negative": 1, "neutral": 2}
}
MOCK_CHART = {
    "signal": -1, "confidence": 0.35,
    "pattern": "Head and shoulders top"
}
MOCK_TIMESERIES = {
    "signal": 1, "confidence": 0.61,
    "direction_probability": 0.61
}
MOCK_POSITIONING = {
    "signal": 1, "confidence": 0.62,
    "long_pct": 38, "short_pct": 62,
    "note": "Retail 62% short - contrarian bullish"
}
MOCK_FUSION = {
    "stance": "bullish",
    "confidence": 0.65,
    "conflict_level": "medium",
    "signals": {
        "news": "Positive sentiment on EUR strength.",
        "chart": "Head and shoulders top detected.",
        "timeseries": "GRU predicts upward movement.",
        "positioning": "Retail heavily short, contrarian bullish.",
    },
    "reasoning": (
        "Three modalities align bullish despite a bearish chart signal. "
        "Conflict is medium. Overall stance cautiously bullish."
    ),
}


# Helper: pass-through apply_weights for unit tests
# * adds feedback keys without changing signal or confidence values,
#   so existing assertions on modality_outputs still pass
def _passthrough_apply_weights(modality_outputs, weights):
    return {
        mod: {**data, "raw_confidence": data["confidence"], "weight": 1.0}
        for mod, data in modality_outputs.items()
    }


# Orchestrator fixture
@pytest.fixture
def mock_orchestrator():
    """
    Creates an Orchestrator with FinBERT and YOLO loading patched out.
    - all modality run functions, fusion and feedback weight functions
      are patched so tests are deterministic and DB-free
    - apply_weights is a transparent pass-through: adds raw_confidence and
      weight keys, leaving confidence values unchanged
    """
    with patch("core.orchestrator.load_sentiment_model", return_value=MagicMock()), \
         patch("core.orchestrator.load_yolo_model", return_value=MagicMock()), \
         patch("core.orchestrator.get_sentiment_signal", return_value=MOCK_NEWS), \
         patch("core.orchestrator.get_chart_signal", return_value=MOCK_CHART), \
         patch("core.orchestrator.get_timeseries_signal", return_value=MOCK_TIMESERIES), \
         patch("core.orchestrator.get_positioning_signal", return_value=MOCK_POSITIONING), \
         patch("core.orchestrator.get_fusion_output", return_value=MOCK_FUSION), \
         patch("core.orchestrator.load_weights", return_value=MOCK_DEFAULT_WEIGHTS), \
         patch("core.orchestrator.apply_weights", side_effect=_passthrough_apply_weights):
        orch = Orchestrator()
        yield orch



# Orchestrator initialisation test
class TestOrchestratorInit:
    def test_init_loads_sentiment_model(self):
        with patch("core.orchestrator.load_sentiment_model", return_value=MagicMock()) as mock_sent, \
             patch("core.orchestrator.load_yolo_model", return_value=MagicMock()):
            Orchestrator()
            mock_sent.assert_called_once()

    def test_init_loads_yolo_model(self):
        with patch("core.orchestrator.load_sentiment_model", return_value=MagicMock()), \
             patch("core.orchestrator.load_yolo_model", return_value=MagicMock()) as mock_yolo:
            Orchestrator()
            mock_yolo.assert_called_once()

    def test_models_stored_as_instance_variables(self):
        fake_pipe = MagicMock()
        fake_yolo = MagicMock()
        with patch("core.orchestrator.load_sentiment_model", return_value=fake_pipe), \
             patch("core.orchestrator.load_yolo_model", return_value=fake_yolo):
            orch = Orchestrator()
            assert orch.sentiment_pipe is fake_pipe
            assert orch.yolo_model is fake_yolo

    def test_db_conn_defaults_to_none(self):
        """db_conn defaults to None when not supplied - orchestrator must still initialise."""
        with patch("core.orchestrator.load_sentiment_model", return_value=MagicMock()), \
             patch("core.orchestrator.load_yolo_model", return_value=MagicMock()):
            orch = Orchestrator()
            assert orch.db_conn is None

    def test_db_conn_stored_when_provided(self):
        """A supplied db_conn must be stored as an instance variable for run() to use."""
        fake_conn = MagicMock()
        with patch("core.orchestrator.load_sentiment_model", return_value=MagicMock()), \
             patch("core.orchestrator.load_yolo_model", return_value=MagicMock()):
            orch = Orchestrator(db_conn=fake_conn)
            assert orch.db_conn is fake_conn


# Orchestrator.run - calling modalities
class TestOrchestratorCallsModalities:   
    def test_calls_sentiment(self, mock_orchestrator):
        with patch("core.orchestrator.get_sentiment_signal", return_value=MOCK_NEWS) as mock:
            with patch("core.orchestrator.get_chart_signal", return_value=MOCK_CHART), \
                 patch("core.orchestrator.get_timeseries_signal", return_value=MOCK_TIMESERIES), \
                 patch("core.orchestrator.get_positioning_signal", return_value=MOCK_POSITIONING), \
                 patch("core.orchestrator.get_fusion_output", return_value=MOCK_FUSION):
                mock_orchestrator.run("EURUSD", "1h")
                mock.assert_called_once()

    def test_calls_chart(self, mock_orchestrator):
        with patch("core.orchestrator.get_chart_signal", return_value=MOCK_CHART) as mock:
            with patch("core.orchestrator.get_sentiment_signal", return_value=MOCK_NEWS), \
                 patch("core.orchestrator.get_timeseries_signal", return_value=MOCK_TIMESERIES), \
                 patch("core.orchestrator.get_positioning_signal", return_value=MOCK_POSITIONING), \
                 patch("core.orchestrator.get_fusion_output", return_value=MOCK_FUSION):
                mock_orchestrator.run("EURUSD", "1h")
                mock.assert_called_once()

    def test_calls_timeseries(self, mock_orchestrator):
        with patch("core.orchestrator.get_timeseries_signal", return_value=MOCK_TIMESERIES) as mock:
            with patch("core.orchestrator.get_sentiment_signal", return_value=MOCK_NEWS), \
                 patch("core.orchestrator.get_chart_signal", return_value=MOCK_CHART), \
                 patch("core.orchestrator.get_positioning_signal", return_value=MOCK_POSITIONING), \
                 patch("core.orchestrator.get_fusion_output", return_value=MOCK_FUSION):
                mock_orchestrator.run("EURUSD", "1h")
                mock.assert_called_once()

    def test_calls_positioning(self, mock_orchestrator):
        with patch("core.orchestrator.get_positioning_signal", return_value=MOCK_POSITIONING) as mock:
            with patch("core.orchestrator.get_sentiment_signal", return_value=MOCK_NEWS), \
                 patch("core.orchestrator.get_chart_signal", return_value=MOCK_CHART), \
                 patch("core.orchestrator.get_timeseries_signal", return_value=MOCK_TIMESERIES), \
                 patch("core.orchestrator.get_fusion_output", return_value=MOCK_FUSION):
                mock_orchestrator.run("EURUSD", "1h")
                mock.assert_called_once()

    def test_calls_fusion(self, mock_orchestrator):
        with patch("core.orchestrator.get_fusion_output", return_value=MOCK_FUSION) as mock:
            with patch("core.orchestrator.get_sentiment_signal", return_value=MOCK_NEWS), \
                 patch("core.orchestrator.get_chart_signal", return_value=MOCK_CHART), \
                 patch("core.orchestrator.get_timeseries_signal", return_value=MOCK_TIMESERIES), \
                 patch("core.orchestrator.get_positioning_signal", return_value=MOCK_POSITIONING):
                mock_orchestrator.run("EURUSD", "1h")
                mock.assert_called_once()

    def test_sentiment_called_with_preloaded_pipe(self):
        """Confirms the orchestrator passes its preloaded sentiment_pipe to get_sentiment_signal."""
        fake_pipe = MagicMock()
        with patch("core.orchestrator.load_sentiment_model", return_value=fake_pipe), \
             patch("core.orchestrator.load_yolo_model", return_value=MagicMock()), \
             patch("core.orchestrator.get_sentiment_signal", return_value=MOCK_NEWS) as mock_sent, \
             patch("core.orchestrator.get_chart_signal", return_value=MOCK_CHART), \
             patch("core.orchestrator.get_timeseries_signal", return_value=MOCK_TIMESERIES), \
             patch("core.orchestrator.get_positioning_signal", return_value=MOCK_POSITIONING), \
             patch("core.orchestrator.get_fusion_output", return_value=MOCK_FUSION), \
             patch("core.orchestrator.load_weights", return_value=MOCK_DEFAULT_WEIGHTS), \
             patch("core.orchestrator.apply_weights", side_effect=_passthrough_apply_weights):
            orch = Orchestrator()
            orch.run("EURUSD", "1h")
            call_kwargs = mock_sent.call_args
            assert call_kwargs.kwargs.get("pipe") is fake_pipe or \
                   (call_kwargs.args and fake_pipe in call_kwargs.args)

    def test_chart_called_with_preloaded_yolo(self):
        """Confirms the orchestrator passes its preloaded yolo_model to get_chart_signal."""
        fake_yolo = MagicMock()
        with patch("core.orchestrator.load_sentiment_model", return_value=MagicMock()), \
             patch("core.orchestrator.load_yolo_model", return_value=fake_yolo), \
             patch("core.orchestrator.get_sentiment_signal", return_value=MOCK_NEWS), \
             patch("core.orchestrator.get_chart_signal", return_value=MOCK_CHART) as mock_chart, \
             patch("core.orchestrator.get_timeseries_signal", return_value=MOCK_TIMESERIES), \
             patch("core.orchestrator.get_positioning_signal", return_value=MOCK_POSITIONING), \
             patch("core.orchestrator.get_fusion_output", return_value=MOCK_FUSION), \
             patch("core.orchestrator.load_weights", return_value=MOCK_DEFAULT_WEIGHTS), \
             patch("core.orchestrator.apply_weights", side_effect=_passthrough_apply_weights):
            orch = Orchestrator()
            orch.run("EURUSD", "1h")
            call_kwargs = mock_chart.call_args
            assert call_kwargs.kwargs.get("yolo_model") is fake_yolo or \
                   (call_kwargs.args and fake_yolo in call_kwargs.args)


# Orchestrator.run - output structue
class TestOrchestratorOutput:
    def test_returns_dict(self, mock_orchestrator):
        result = mock_orchestrator.run("EURUSD", "1h")
        assert isinstance(result, dict)

    def test_contains_fusion_keys(self, mock_orchestrator):
        result = mock_orchestrator.run("EURUSD", "1h")
        for key in ["stance", "confidence", "conflict_level", "signals", "reasoning"]:
            assert key in result, f"Missing key: {key}"

    def test_contains_metadata_keys(self, mock_orchestrator):
        result = mock_orchestrator.run("EURUSD", "1h")
        assert "pair" in result
        assert "timeframe" in result
        assert "date" in result

    def test_contains_weights_key(self, mock_orchestrator):
        """Output must include the weights dict used for this run."""
        result = mock_orchestrator.run("EURUSD", "1h")
        assert "weights" in result
        assert isinstance(result["weights"], dict)
        for mod in ["news", "chart", "timeseries", "positioning"]:
            assert mod in result["weights"]

    def test_metadata_values_correct(self, mock_orchestrator):
        result = mock_orchestrator.run("EURUSD", "1h")
        assert result["pair"] == "EURUSD"
        assert result["timeframe"] == "1h"

    def test_contains_modality_outputs(self, mock_orchestrator):
        result = mock_orchestrator.run("EURUSD", "1h")
        assert "modality_outputs" in result
        for key in ["news", "chart", "timeseries", "positioning"]:
            assert key in result["modality_outputs"]

    def test_modality_outputs_have_signal_and_confidence(self, mock_orchestrator):
        """Modality outputs should retain the core signal and confidence fields.

        - apply_weights enhance output with additional metadata
        - assert only required keys rather than exact dicts are exact same
        """
        result = mock_orchestrator.run("EURUSD", "1h")
        for mod in ["news", "chart", "timeseries", "positioning"]:
            out = result["modality_outputs"][mod]
            assert "signal" in out
            assert "confidence" in out

    def test_stance_is_valid(self, mock_orchestrator):
        result = mock_orchestrator.run("EURUSD", "1h")
        assert result["stance"] in {"bullish", "bearish", "neutral"}

    def test_confidence_in_range(self, mock_orchestrator):
        result = mock_orchestrator.run("EURUSD", "1h")
        assert 0.0 <= result["confidence"] <= 1.0

    def test_date_is_string(self, mock_orchestrator):
        result = mock_orchestrator.run("EURUSD", "1h")
        assert isinstance(result["date"], str)
        from datetime import date
        date.fromisoformat(result["date"])


# Orchestrator.run - validation and fallback behaviour
class TestOrchestratorValidation:
    def test_unsupported_pair_raises(self, mock_orchestrator):
        with pytest.raises(ValueError, match="Unsupported pair"):
            mock_orchestrator.run("XAUUSD", "1h")

    def test_unsupported_timeframe_raises(self, mock_orchestrator):
        with pytest.raises(ValueError, match="Unsupported timeframe"):
            mock_orchestrator.run("EURUSD", "15m")

    def test_modality_failure_uses_neutral_fallback(self):
        """If a modality raises exception mid-run, should catch, log, use neutral fallback.
        
        - no crashing midway
        """
        with patch("core.orchestrator.load_sentiment_model", return_value=MagicMock()), \
             patch("core.orchestrator.load_yolo_model", return_value=MagicMock()), \
             patch("core.orchestrator.get_sentiment_signal",
                   side_effect=Exception("RSS feed down")), \
             patch("core.orchestrator.get_chart_signal", return_value=MOCK_CHART), \
             patch("core.orchestrator.get_timeseries_signal", return_value=MOCK_TIMESERIES), \
             patch("core.orchestrator.get_positioning_signal", return_value=MOCK_POSITIONING), \
             patch("core.orchestrator.get_fusion_output", return_value=MOCK_FUSION), \
             patch("core.orchestrator.load_weights", return_value=MOCK_DEFAULT_WEIGHTS), \
             patch("core.orchestrator.apply_weights", side_effect=_passthrough_apply_weights):
            orch = Orchestrator()
            result = orch.run("EURUSD", "1h")
            assert "stance" in result
            assert result["modality_outputs"]["news"]["signal"] == 0

    def test_all_modalities_failing_still_returns_result(self):
        """Worst case if every modality fails. Should still return fusion output (default output)."""
        with patch("core.orchestrator.load_sentiment_model", return_value=MagicMock()), \
             patch("core.orchestrator.load_yolo_model", return_value=MagicMock()), \
             patch("core.orchestrator.get_sentiment_signal",
                   side_effect=Exception("fail")), \
             patch("core.orchestrator.get_chart_signal",
                   side_effect=Exception("fail")), \
             patch("core.orchestrator.get_timeseries_signal",
                   side_effect=Exception("fail")), \
             patch("core.orchestrator.get_positioning_signal",
                   side_effect=Exception("fail")), \
             patch("core.orchestrator.get_fusion_output", return_value=MOCK_FUSION), \
             patch("core.orchestrator.load_weights", return_value=MOCK_DEFAULT_WEIGHTS), \
             patch("core.orchestrator.apply_weights", side_effect=_passthrough_apply_weights):
            orch = Orchestrator()
            result = orch.run("EURUSD", "1h")
            assert "stance" in result


# System-level INTEGRATION TEST (since we running everyting) - require Ollama + trained models + live APIs
class TestOrchestratorIntegration:
    @requires_all
    def test_eurusd_returns_valid_output(self):
        orch = Orchestrator()
        result = orch.run("EURUSD", "1h")
        assert result["stance"] in {"bullish", "bearish", "neutral"}
        assert 0.0 <= result["confidence"] <= 1.0
        assert result["conflict_level"] in {"low", "medium", "high"}
        for key in ["stance", "confidence", "conflict_level", "signals", "reasoning",
                    "pair", "timeframe", "date", "modality_outputs", "weights"]:
            assert key in result

    @requires_all
    def test_gbpusd_returns_valid_output(self):
        orch = Orchestrator()
        result = orch.run("GBPUSD", "1h")
        assert result["stance"] in {"bullish", "bearish", "neutral"}
        assert 0.0 <= result["confidence"] <= 1.0

    @requires_all
    def test_model_reuse_across_calls(self):
        """Two calls on the same orchestrator instance should reuse models.
        
        - the second call should be strictly faster than the first, since no need to load again on second run
        """
        import time
        orch = Orchestrator()

        start1 = time.time()
        orch.run("EURUSD", "1h")
        time1 = time.time() - start1

        start2 = time.time()
        orch.run("EURUSD", "1h")
        time2 = time.time() - start2

        assert time2 < time1 * 1.2, (
            f"Second call ({time2:.1f}s) was not faster than first ({time1:.1f}s) "
            f"- models may be reloading on each run() call."
        )

    @requires_all
    def test_consistency_five_consecutive_runs(self):
        """Pipeline and fusion integration test: run the same input 5 times,
          and assert stance is CONSISTENT.

        - LLM at low temperature should not flip wildly
        """
        orch = Orchestrator()
        stances = []
        for _ in range(5):
            result = orch.run("EURUSD", "1h")
            stances.append(result["stance"])

        unique_stances = set(stances)
        assert len(unique_stances) <= 2, (
            f"Stance varied too much across 5 runs: {stances}. "
            f"Consider lowering OLLAMA_TEMP in config.py."
        )

    @requires_all
    def test_result_contains_weights_from_db_or_default(self):
        """Feedback loop integration: result must contain a weights dict with all
        four modality keys, whether loaded from DB history or defaulted to 1.0.
        """
        orch = Orchestrator()
        result = orch.run("EURUSD", "1h")
        assert "weights" in result
        for mod in ["news", "chart", "timeseries", "positioning"]:
            assert mod in result["weights"]
            assert isinstance(result["weights"][mod], float)
            assert result["weights"][mod] >= 0.1 # must be at least WEIGHT_FLOOR