from infrastructure.config import SUPPORTED_ASSETS, TIMEFRAME_CONFIGS, CONFLICT_LOW_THRESHOLD, CONFLICT_MEDIUM_THRESHOLD

# Asset/timeframe lists are derived from config.py to maintain a single source of truth
ASSETS = list(SUPPORTED_ASSETS.keys())
TIMEFRAMES = list(TIMEFRAME_CONFIGS.keys())

SIGNAL_LABELS = {1: "BULLISH", -1: "BEARISH", 0: "NEUTRAL"}
SIGNAL_CLASSES = {1: "sig-bull", -1: "sig-bear", 0: "sig-neut"}
STANCE_CLASSES = {"bullish": "stance-bull", "bearish": "stance-bear", "neutral": "stance-neutral"}
CONFLICT_CLASSES = {"low": "conflict-low", "medium": "conflict-medium", "high": "conflict-high"}
MODALITY_NAMES = {
    "news": "News / Sentiment",
    "chart": "Chart Pattern",
    "timeseries": "Time-Series GRU",
    "positioning": "Retail Positioning",
}
OUTCOME_LABELS = {"correct": "✓ CORRECT", "incorrect": "✗ INCORRECT", "uncertain": "? UNCERTAIN"}
OUTCOME_CLASSES = {"correct": "outcome-correct", "incorrect": "outcome-incorrect", "uncertain": "outcome-pending"}

# Format some of the values from YOLO model into human readable format for display
PATTERN_LABELS = {
    "Head and shoulders top": "Head & Shoulders Top",
    "Head and shoulders bottom": "Head & Shoulders Bottom",
    "M_Head": "Double Top (M-Head)",
    "W_Bottom": "Double Bottom (W-Bottom)",
    "Triangle": "Triangle Consolidation",
    "StockLine": "Trendline",
    "none": "No Pattern Detected",
}