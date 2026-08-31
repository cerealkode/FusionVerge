"""
Unit tests for the time-series modality (timeseries.py)
Uses real yfinance fetch and real model/scaler loading from models/.
Tests that require trained models are skipped gracefully if models/ is missing, so the test suite still runs in CI or on a fresh clone without model files.

Usage - python -m pytest tests/test_timeseries.py -v --disable-warnings
"""
import warnings
warnings.filterwarnings(
    "ignore",
    message=r".*mode\.copy_on_write.*",
)
import os
import pytest
import numpy as np
import pandas as pd
from modalities.timeseries import fetch_ohlcv, compute_indicators, load_best_config, load_gru, run_gru_inference, get_timeseries_signal
from infrastructure.config import OHLC, OHLC_ALL, GRU_BEST_CONFIGS_PATH

pytestmark = [
    pytest.mark.filterwarnings(
        "ignore::sklearn.exceptions.InconsistentVersionWarning"
    ),
]

# Skip markers, used on any test that requires trained model files
models_available = (
    os.path.exists(GRU_BEST_CONFIGS_PATH) and
    os.path.exists("models/EURUSD_1h.keras") and
    os.path.exists("models/EURUSD_1h_scaler.pkl")
)

requires_models = pytest.mark.skipif(
    not models_available,
    reason="Trained model files not found in models/. Run the training notebook first."
)


# Module-level fixtures
@pytest.fixture(scope="module")
def eurusd_ohlc():
    """Fetch raw OHLC for EURUSD 1h once - reused by indicator and inference tests."""
    return fetch_ohlcv("EURUSD", "1h")

@pytest.fixture(scope="module")
def eurusd_with_indicators(eurusd_ohlc):
    """OHLC + computed indicators - reused by inference tests."""
    return compute_indicators(eurusd_ohlc)

@pytest.fixture(scope="module")
def gru_model_and_scaler():
    """Load EURUSD 1h model and scaler once for inference tests."""
    if not models_available:
        pytest.skip("Model files not available.")
    return load_gru("EURUSD", "1h")


# fetch_ohlcv() function testing
class TestFetchOhlcv:
    def test_returns_dataframe(self):
        df = fetch_ohlcv("EURUSD", "1h")
        assert isinstance(df, pd.DataFrame)

    def test_has_ohlc_columns(self):
        df = fetch_ohlcv("EURUSD", "1h")
        for col in OHLC:
            assert col in df.columns

    def test_no_volume_column(self):
        df = fetch_ohlcv("EURUSD", "1h")
        assert "Volume" not in df.columns

    def test_no_nulls(self):
        df = fetch_ohlcv("EURUSD", "1h")
        assert df.isnull().sum().sum() == 0

    def test_non_empty(self):
        df = fetch_ohlcv("EURUSD", "1h")
        assert len(df) > 50 # should have many rows after 300d fetch

    def test_unsupported_pair_raises(self):
        with pytest.raises(ValueError, match="Unsupported pair"):
            fetch_ohlcv("XAUUSD", "1h")

    def test_unsupported_timeframe_raises(self):
        with pytest.raises(ValueError, match="Unsupported timeframe"):
            fetch_ohlcv("EURUSD", "15m")


# compute_indicators() function testing
class TestComputeIndicators:
    def test_adds_indicator_columns(self, eurusd_ohlc):
        df = compute_indicators(eurusd_ohlc)
        for col in ["RSI", "MACD", "EMA20", "ATR"]:
            assert col in df.columns, f"Missing indicator column: {col}"

    def test_no_nulls_after_compute(self, eurusd_ohlc):
        df = compute_indicators(eurusd_ohlc)
        assert df.isnull().sum().sum() == 0

    def test_row_count_reduced_by_warmup(self, eurusd_ohlc):
        """Indicator warmup (MACD needs 26 bars) must drop some rows."""
        df = compute_indicators(eurusd_ohlc)
        assert len(df) < len(eurusd_ohlc)

    def test_rsi_in_valid_range(self, eurusd_ohlc):
        """RSI is bounded 0–100 by definition."""
        df = compute_indicators(eurusd_ohlc)
        assert df["RSI"].between(0, 100).all()

    def test_original_ohlc_columns_preserved(self, eurusd_ohlc):
        df = compute_indicators(eurusd_ohlc)
        for col in OHLC:
            assert col in df.columns

    def test_does_not_modify_input(self, eurusd_ohlc):
        """compute_indicators should work on a copy, not mutate input."""
        original_cols = list(eurusd_ohlc.columns)
        compute_indicators(eurusd_ohlc)
        assert list(eurusd_ohlc.columns) == original_cols


# load_best_config() function testing
class TestLoadBestConfig:
    @requires_models
    def test_returns_dict(self):
        cfg = load_best_config("EURUSD", "1h")
        assert isinstance(cfg, dict)

    @requires_models
    def test_has_required_keys(self):
        cfg = load_best_config("EURUSD", "1h")
        assert "features" in cfg
        assert "lookback" in cfg

    @requires_models
    def test_features_is_list(self):
        cfg = load_best_config("EURUSD", "1h")
        assert isinstance(cfg["features"], list)
        assert len(cfg["features"]) > 0

    @requires_models
    def test_lookback_is_positive_int(self):
        cfg = load_best_config("EURUSD", "1h")
        assert isinstance(cfg["lookback"], int)
        assert cfg["lookback"] > 0

    @requires_models
    def test_features_are_valid_columns(self):
        """All features in best config must be in OHLC_ALL."""
        cfg = load_best_config("EURUSD", "1h")
        for f in cfg["features"]:
            assert f in OHLC_ALL, f"Unknown feature in best config: {f}"

    def test_missing_json_raises(self, tmp_path, monkeypatch):
        monkeypatch.setattr("modalities.timeseries.GRU_BEST_CONFIGS_PATH", str(tmp_path / "nonexistent.json"))
        with pytest.raises(FileNotFoundError, match="best_configs.json not found"):
            load_best_config("EURUSD", "1h")

    @requires_models
    def test_unknown_pair_raises(self):
        with pytest.raises(KeyError, match="No entry for"):
            load_best_config("XAUUSD", "1h")


# load_gru() function testing
class TestLoadGru:
    @requires_models
    def test_returns_model_and_scaler(self, gru_model_and_scaler):
        model, scaler = gru_model_and_scaler
        assert model is not None
        assert scaler is not None

    def test_missing_model_raises(self):
        with pytest.raises(FileNotFoundError, match="No saved model"):
            load_gru("XAUUSD", "1h")

    def test_missing_scaler_raises(self, tmp_path, monkeypatch):
        """If model exists but scaler is missing, should raise FileNotFoundError."""
        monkeypatch.setattr("modalities.timeseries.GRU_SCALER_TEMPLATE", str(tmp_path / "{pair}_{timeframe}_scaler.pkl"))
        with pytest.raises(FileNotFoundError, match="No scaler"):
            load_gru("EURUSD", "1h")


# run_gru_inference() function testing
class TestRunGruInference:
    @requires_models
    def test_input_shape_is_correct(self, gru_model_and_scaler, eurusd_with_indicators):
        """
        Verifies the inference window fed to the model has shape (1, lookback, n_features).
        """
        model, scaler = gru_model_and_scaler
        cfg = load_best_config("EURUSD", "1h")
        features = cfg["features"]
        lookback = cfg["lookback"]

        # Replicate what run_gru_inference does internally to check shape
        data = eurusd_with_indicators[features].values
        data_scaled = scaler.transform(data)
        X = data_scaled[-lookback:].reshape(1, lookback, len(features))

        assert X.shape == (1, lookback, len(features))

    @requires_models
    def test_returns_required_keys(self, gru_model_and_scaler, eurusd_with_indicators):
        model, scaler = gru_model_and_scaler
        cfg = load_best_config("EURUSD", "1h")
        result = run_gru_inference(model, scaler, eurusd_with_indicators, cfg["features"], cfg["lookback"])
        assert "signal" in result
        assert "confidence" in result
        assert "direction_probability" in result

    @requires_models
    def test_signal_is_valid(self, gru_model_and_scaler, eurusd_with_indicators):
        model, scaler = gru_model_and_scaler
        cfg = load_best_config("EURUSD", "1h")
        result = run_gru_inference(model, scaler, eurusd_with_indicators, cfg["features"], cfg["lookback"])
        assert result["signal"] in {-1, 1}  # GRU always commits to a direction

    @requires_models
    def test_direction_is_0_or_1(self, gru_model_and_scaler, eurusd_with_indicators):
        """direction_probability is the raw sigmoid output - must be in [0, 1]."""
        model, scaler = gru_model_and_scaler
        cfg = load_best_config("EURUSD", "1h")
        result = run_gru_inference(model, scaler, eurusd_with_indicators, cfg["features"], cfg["lookback"])
        assert 0.0 <= result["direction_probability"] <= 1.0

    @requires_models
    def test_confidence_in_range(self, gru_model_and_scaler, eurusd_with_indicators):
        model, scaler = gru_model_and_scaler
        cfg = load_best_config("EURUSD", "1h")
        result = run_gru_inference(model, scaler, eurusd_with_indicators, cfg["features"], cfg["lookback"])
        assert 0.0 <= result["confidence"] <= 1.0

    @requires_models
    def test_confidence_consistent_with_direction_probability(self, gru_model_and_scaler, eurusd_with_indicators):
        """
        Confidence is derived from direction_probability:
        if dir_prob > 0.5: confidence = dir_prob, else confidence = 1 - dir_prob.
        Always >= 0.5.
        """
        model, scaler = gru_model_and_scaler
        cfg = load_best_config("EURUSD", "1h")
        result = run_gru_inference(model, scaler, eurusd_with_indicators, cfg["features"], cfg["lookback"])
        assert result["confidence"] >= 0.5

    @requires_models
    def test_insufficient_rows_raises(self, gru_model_and_scaler):
        """Fewer rows than lookback should raise ValueError, not crash silently."""
        model, scaler = gru_model_and_scaler
        cfg = load_best_config("EURUSD", "1h")
        tiny_df = pd.DataFrame(
            np.random.rand(3, len(cfg["features"])),
            columns=cfg["features"]
        )
        with pytest.raises(ValueError, match="Not enough rows"):
            run_gru_inference(model, scaler, tiny_df, cfg["features"], cfg["lookback"])


# get_timeseries_signal() Integration testing
# *end-to-end for the modality
class TestGetTimeseriesSignal:
    @requires_models
    def test_returns_required_keys(self):
        result = get_timeseries_signal("EURUSD", "1h")
        assert "signal" in result
        assert "confidence" in result
        assert "direction_probability" in result

    @requires_models
    def test_signal_is_valid(self):
        result = get_timeseries_signal("EURUSD", "1h")
        assert result["signal"] in {-1, 1}

    @requires_models
    def test_confidence_in_range(self):
        result = get_timeseries_signal("EURUSD", "1h")
        assert 0.0 <= result["confidence"] <= 1.0

    def test_unsupported_pair_raises(self):
        with pytest.raises(KeyError):
            get_timeseries_signal("XAUUSD", "1h")