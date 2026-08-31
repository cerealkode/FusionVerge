"""
Unit tests for the chart modality (chart.py)
Uses real yfinance fetch and real YOLO inference.
YOLO model is loaded once at module level to avoid repeated downloads.

Usage - python -m pytest tests/test_chart.py -v --disable-warnings
"""
import os
import pytest
import pandas as pd
from modalities.chart import fetch_ohlcv, generate_chart, load_yolo_model, run_yolo_inference, get_chart_signal
from infrastructure.config import YOLO_SIGNAL_MAP

# Module-level fixture
# * load FinBERT model ONCE for the entire test session,
#   instead of repeated loading for each tests
@pytest.fixture(scope="module")
def yolo():
    return load_yolo_model()

@pytest.fixture(scope="module")
def eurusd_df():
    """Fetch EURUSD 1h OHLC once and reuse across chart tests."""
    return fetch_ohlcv("EURUSD", "1h")


# fetch_ohlcv() function testing
class TestFetchOhlcv:
    def test_returns_dataframe(self):
        df = fetch_ohlcv("EURUSD", "1h")
        assert isinstance(df, pd.DataFrame)

    def test_has_ohlc_columns(self):
        df = fetch_ohlcv("EURUSD", "1h")
        for col in ["Open", "High", "Low", "Close"]:
            assert col in df.columns

    def test_no_volume_column(self):
        """Volume should not be present - chart modality uses OHLC only."""
        df = fetch_ohlcv("EURUSD", "1h")
        assert "Volume" not in df.columns

    def test_no_nulls(self):
        df = fetch_ohlcv("EURUSD", "1h")
        assert df.isnull().sum().sum() == 0

    def test_non_empty(self):
        df = fetch_ohlcv("EURUSD", "1h")
        assert len(df) > 0

    def test_index_is_datetime(self):
        df = fetch_ohlcv("EURUSD", "1h")
        assert isinstance(df.index, pd.DatetimeIndex)

    def test_unsupported_pair_raises(self):
        with pytest.raises(ValueError, match="Unsupported pair"):
            fetch_ohlcv("XAUUSD", "1h")


# generate_chart() function testing
class TestGenerateChart:
    def test_creates_image_file(self, eurusd_df, tmp_path):
        savepath = str(tmp_path / "test_chart.png")
        result_path = generate_chart(eurusd_df, savepath=savepath)
        assert os.path.exists(result_path)

    def test_returns_savepath(self, eurusd_df, tmp_path):
        savepath = str(tmp_path / "test_chart.png")
        result_path = generate_chart(eurusd_df, savepath=savepath)
        assert result_path == savepath

    def test_image_has_nonzero_size(self, eurusd_df, tmp_path):
        savepath = str(tmp_path / "test_chart.png")
        generate_chart(eurusd_df, savepath=savepath)
        assert os.path.getsize(savepath) > 0

    def test_missing_column_raises(self, tmp_path):
        """DataFrame missing a required OHLC column should raise ValueError."""
        bad_df = pd.DataFrame({"Open": [1.1], "High": [1.2], "Low": [1.0]})
        bad_df.index = pd.to_datetime(["2024-01-01"])
        bad_df.index.name = "Date"
        savepath = str(tmp_path / "bad_chart.png")
        with pytest.raises(ValueError, match="missing required columns"):
            generate_chart(bad_df, savepath=savepath)


# run_yolo_inference() function testing
class TestRunYoloInference:
    def test_returns_required_keys(self, yolo, eurusd_df, tmp_path):
        savepath = str(tmp_path / "yolo_test.png")
        generate_chart(eurusd_df, savepath=savepath)
        result = run_yolo_inference(yolo, savepath)
        assert "signal" in result
        assert "confidence" in result
        assert "pattern" in result

    def test_signal_is_valid(self, yolo, eurusd_df, tmp_path):
        savepath = str(tmp_path / "yolo_signal.png")
        generate_chart(eurusd_df, savepath=savepath)
        result = run_yolo_inference(yolo, savepath)
        assert result["signal"] in {-1, 0, 1}

    def test_confidence_in_range(self, yolo, eurusd_df, tmp_path):
        savepath = str(tmp_path / "yolo_conf.png")
        generate_chart(eurusd_df, savepath=savepath)
        result = run_yolo_inference(yolo, savepath)
        assert 0.0 <= result["confidence"] <= 1.0

    def test_pattern_is_known_label(self, yolo, eurusd_df, tmp_path):
        """Pattern must be either a known YOLO class or 'none' for no detection."""
        from infrastructure.config import YOLO_CLASSES
        savepath = str(tmp_path / "yolo_pattern.png")
        generate_chart(eurusd_df, savepath=savepath)
        result = run_yolo_inference(yolo, savepath)
        valid_patterns = set(YOLO_CLASSES) | {"none"}
        assert result["pattern"] in valid_patterns

    def test_no_detection_returns_neutral(self, yolo, tmp_path):
        """A blank white image should produce no detections -> signal 0, pattern 'none'."""
        from PIL import Image
        savepath = str(tmp_path / "blank.png")
        Image.new("RGB", (640, 480), color=(255, 255, 255)).save(savepath)
        result = run_yolo_inference(yolo, savepath)
        assert result["signal"] == 0
        assert result["pattern"] == "none"
        assert result["confidence"] == 0.0

    def test_signal_matches_pattern(self, yolo, eurusd_df, tmp_path):
        """Signal must be consistent with YOLO_SIGNAL_MAP for the detected pattern."""
        savepath = str(tmp_path / "yolo_map.png")
        generate_chart(eurusd_df, savepath=savepath)
        result = run_yolo_inference(yolo, savepath)
        expected_signal = YOLO_SIGNAL_MAP.get(result["pattern"], 0)
        assert result["signal"] == expected_signal


# get_chart_signal() Integration testing
# * end-to-end for the modality
class TestGetChartSignal:
    def test_returns_required_keys(self, yolo):
        result = get_chart_signal("EURUSD", "1h", yolo_model=yolo)
        assert "signal" in result
        assert "confidence" in result
        assert "pattern" in result

    def test_signal_is_valid(self, yolo):
        result = get_chart_signal("EURUSD", "1h", yolo_model=yolo)
        assert result["signal"] in {-1, 0, 1}

    def test_confidence_in_range(self, yolo):
        result = get_chart_signal("EURUSD", "1h", yolo_model=yolo)
        assert 0.0 <= result["confidence"] <= 1.0

    def test_unsupported_pair_raises(self, yolo):
        with pytest.raises(ValueError, match="Unsupported pair"):
            get_chart_signal("XAUUSD", "1h", yolo_model=yolo)